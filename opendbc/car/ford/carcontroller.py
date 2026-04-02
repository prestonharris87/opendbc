import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford import fordcan
from opendbc.car.ford.icbm import ICBMController, GapController
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR, FordSafetyFlags
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_ford_angle(desired_angle: float, current_angle: float) -> float:
  """Compute the steering angle delta for LKA-based steering, clipped to PSCM limits."""
  delta = desired_angle - current_angle
  return float(np.clip(delta, -5.8, 5.8))


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.apply_angle_last = 0.
    self.last_direction = 2
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    # ICBM controllers for LKA cars (stock ACC speed + gap management)
    self.icbm = ICBMController()
    self.gap_controller = GapController()
    self.icbm_speed_limit_mph = 0

    self._icbm_speed_limit_file = "/data/openpilot/icbm_speed_limit"

  def update(self, CC, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    # Skip for LKA cars (Bronco) — they don't have TJA and this interferes with stock ACC
    elif not (self.CP.flags & FordFlags.LKA_STEERING) and CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### ICBM: automatic cruise speed + follow distance for LKA cars (stock ACC) ###
    if (self.CP.flags & FordFlags.LKA_STEERING) and not CC.cruiseControl.cancel and not CC.cruiseControl.resume:
      # Read speed limit from file every ~1 second
      if self.frame % 100 == 0:
        try:
          with open(self._icbm_speed_limit_file) as f:
            self.icbm_speed_limit_mph = int(f.read().strip())
        except Exception:
          self.icbm_speed_limit_mph = 0

      cruise_set_mph = CS.out.cruiseState.speed * CV.MS_TO_MPH
      v_ego_mph = CS.out.vEgo * CV.MS_TO_MPH
      driver_speed_btn = bool(CS.speed_inc_button or CS.speed_dec_button)
      driver_gap_btn = bool(CS.distance_button)
      current_gap = int(CS.acc_tja_status_stock_values["AccTGap_D_Dsply"])

      speed_inc, speed_dec = self.icbm.update(
        self.frame, CS.out.cruiseState.enabled, CC.enabled,
        cruise_set_mph, self.icbm_speed_limit_mph, driver_speed_btn)

      gap_toggle = self.gap_controller.update(
        self.frame, CS.out.cruiseState.enabled, CC.enabled,
        current_gap, v_ego_mph, driver_gap_btn)

      if speed_inc or speed_dec or gap_toggle:
        can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values,
                                                   speed_inc=speed_inc, speed_dec=speed_dec, gap_toggle=gap_toggle))
        can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values,
                                                   speed_inc=speed_inc, speed_dec=speed_dec, gap_toggle=gap_toggle))

    ### lateral control ###
    if self.CP.flags & FordFlags.LKA_STEERING:
      # LKA angle-based steering (full-size Bronco and similar)
      # Note: no LateralMotionControl sent — camera's messages pass through relay unblocked

      # Send active LKA steering commands at 33Hz
      if (self.frame % CarControllerParams.LKA_STEP) == 0:
        if CC.latActive and CS.lkas_available:
          apply_angle = apply_ford_angle(actuators.steeringAngleDeg, CS.out.steeringAngleDeg)
          self.apply_angle_last = apply_angle
          # Direction: 2=left intervention, 4=right intervention
          direction = 2 if CS.out.steeringAngleDeg > 0 else 4
          ramp_type = 1 if abs(apply_angle) >= 5.0 else 0
          can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN, True, apply_angle, 0., direction, ramp_type))
        else:
          self.apply_angle_last = 0.
          can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))
    else:
      # Standard curvature-based steering (LCA/TJA)
      # send steer msg at 20Hz
      if (self.frame % CarControllerParams.STEER_STEP) == 0:
        # Bronco Sport and some other cars consistently overshoot curv requests
        # Apply some deadzone + smoothing convergence to avoid oscillations
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          apply_curvature = self.anti_overshoot_curvature_last
        else:
          apply_curvature = actuators.curvature

        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

        self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP)

        if self.CP.flags & FordFlags.CANFD:
          mode = 1 if CC.latActive else 0
          counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
          can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -self.apply_curvature_last, 0., counter))
        else:
          can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

      # send lka msg at 33Hz
      if (self.frame % CarControllerParams.LKA_STEP) == 0:
        can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    # Skip for LKA cars — native ACC handles longitudinal; FORD_LKA_TX_MSGS blocks ACCDATA anyway
    if self.CP.openpilotLongitudinalControl and not (self.CP.flags & FordFlags.LKA_STEERING) and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    if self.CP.flags & FordFlags.LKA_STEERING:
      # Full-size Bronco: pass through IPMA_Data and ACCDATA_3 completely unmodified from camera.
      # Modifying these messages causes pre-collision assist and ACC failures because the Bronco
      # doesn't have TJA/LCA and its safety systems don't expect modified ADAS messages.
      if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0:
        can_sends.append(self.packer.make_can_msg("IPMA_Data", self.CAN.main, CS.lkas_status_stock_values))
      if (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
        can_sends.append(self.packer.make_can_msg("ACCDATA_3", self.CAN.main, CS.acc_tja_status_stock_values))
    else:
      send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
      # send lkas ui msg at 1Hz or if ui state changes
      if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
        can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

      # send acc ui msg at 5Hz or if ui state changes
      if hud_control.leadDistanceBars != self.lead_distance_bars_last:
        send_ui = True
        self.distance_bar_frame = self.frame

      if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
        show_distance_bars = self.frame - self.distance_bar_frame < 400
        can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                   fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                   hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    if self.CP.flags & FordFlags.LKA_STEERING:
      new_actuators.steeringAngleDeg = self.apply_angle_last + CS.out.steeringAngleDeg
    else:
      new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
