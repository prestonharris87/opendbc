from enum import IntEnum

from opendbc.car.ford.values import CarControllerParams


class SpeedState(IntEnum):
  INACTIVE = 0
  PRE_ACTIVE = 1
  INCREASING = 2
  DECREASING = 3
  HOLDING = 4


class GapState(IntEnum):
  INACTIVE = 0
  WAITING = 1
  TOGGLING = 2


class ICBMController:
  """Intelligent Cruise Button Management — cruise speed controller.

  Tracks speed limits and user offsets:
  - On first ACC activation: sets cruise to current speed limit
  - User manual +/- adjustments: captures offset from speed limit
  - On speed limit change: applies user offset to new limit
  - Between changes: holds current cruise speed
  """

  def __init__(self):
    self.state = SpeedState.INACTIVE
    self.state_frame = 0
    self.driver_override_frames = 0

    # Speed limit tracking — offset is a percentage (e.g., 0.10 = +10% over limit)
    self.user_offset_pct = 0.0      # user's percentage offset from speed limit
    self.cruise_was_enabled = False  # track cruise activation edge

    # Convert time-based constants to frame counts (100 Hz control loop)
    self.pre_active_frames = int(CarControllerParams.ICBM_PRE_ACTIVE_DELAY * 100)
    self.driver_override_cooldown = int(CarControllerParams.ICBM_DRIVER_OVERRIDE_COOLDOWN * 100)

  def update(self, frame: int, cruise_enabled: bool, controls_allowed: bool,
             cruise_set_speed_mph: float, speed_limit_mph: float,
             driver_speed_button_pressed: bool) -> tuple[bool, bool]:
    """Returns (speed_inc, speed_dec) booleans for this frame."""

    # Driver manual +/-: recalculate percentage offset and pause
    if driver_speed_button_pressed:
      if speed_limit_mph > 0:
        self.user_offset_pct = (cruise_set_speed_mph - speed_limit_mph) / speed_limit_mph
      self.driver_override_frames = self.driver_override_cooldown
      self._set_state(SpeedState.INACTIVE, frame)
      return False, False

    if self.driver_override_frames > 0:
      self.driver_override_frames -= 1
      # Keep recalculating offset as driver adjusts (buttons may be held/repeated)
      if speed_limit_mph > 0:
        self.user_offset_pct = (cruise_set_speed_mph - speed_limit_mph) / speed_limit_mph
      return False, False

    # Cruise activated: reset offset, target = speed limit
    # Also resets on every re-activation (ACC off then on)
    if cruise_enabled and not self.cruise_was_enabled:
      self.cruise_was_enabled = True
      self.user_offset_pct = 0.0
      self._set_state(SpeedState.INACTIVE, frame)
    elif not cruise_enabled:
      self.cruise_was_enabled = False
      self.user_offset_pct = 0.0
      self._set_state(SpeedState.INACTIVE, frame)
      return False, False

    # No speed limit data or cruise not active: do nothing
    if speed_limit_mph <= 0 or not cruise_enabled or not controls_allowed:
      self._set_state(SpeedState.INACTIVE, frame)
      return False, False

    # Target is always speed_limit * (1 + offset_pct), rounded to nearest mph
    target = min(round(speed_limit_mph * (1.0 + self.user_offset_pct)),
                 CarControllerParams.ICBM_MAX_SPEED)
    speed_delta = target - cruise_set_speed_mph
    deadband = CarControllerParams.ICBM_SPEED_DEADBAND

    # State machine
    if self.state == SpeedState.INACTIVE:
      if abs(speed_delta) > deadband:
        self._set_state(SpeedState.PRE_ACTIVE, frame)
      return False, False

    elif self.state == SpeedState.PRE_ACTIVE:
      if abs(speed_delta) <= deadband:
        self._set_state(SpeedState.HOLDING, frame)
        return False, False
      if (frame - self.state_frame) >= self.pre_active_frames:
        if speed_delta > 0:
          self._set_state(SpeedState.INCREASING, frame)
        else:
          self._set_state(SpeedState.DECREASING, frame)
      return False, False

    elif self.state == SpeedState.INCREASING:
      if speed_delta <= 0:
        self._set_state(SpeedState.HOLDING, frame)
        return False, False
      if cruise_set_speed_mph >= CarControllerParams.ICBM_MAX_SPEED:
        self._set_state(SpeedState.HOLDING, frame)
        return False, False
      if (frame % CarControllerParams.BUTTONS_STEP) == 0:
        return True, False
      return False, False

    elif self.state == SpeedState.DECREASING:
      if speed_delta >= 0:
        self._set_state(SpeedState.HOLDING, frame)
        return False, False
      if (frame % CarControllerParams.BUTTONS_STEP) == 0:
        return False, True
      return False, False

    elif self.state == SpeedState.HOLDING:
      if abs(speed_delta) > deadband:
        self._set_state(SpeedState.PRE_ACTIVE, frame)
      return False, False

    return False, False

  def _set_state(self, new_state: SpeedState, frame: int):
    if self.state != new_state:
      self.state = new_state
      self.state_frame = frame


class GapController:
  """Automatic follow distance controller.

  Synthesizes AccButtnGapTogglePress to cycle the stock Ford ACC gap
  setting based on vehicle speed.
  """

  def __init__(self):
    self.state = GapState.INACTIVE
    self.wait_frames = 0
    self.driver_override_frames = 0
    self.driver_override_cooldown = int(CarControllerParams.ICBM_DRIVER_OVERRIDE_COOLDOWN * 100)
    self.gap_toggle_cooldown = int(CarControllerParams.ICBM_GAP_TOGGLE_COOLDOWN * 100)

  def update(self, frame: int, cruise_enabled: bool, controls_allowed: bool,
             current_gap: int, v_ego_mph: float,
             driver_gap_button_pressed: bool) -> bool:
    """Returns gap_toggle boolean for this frame."""

    # Driver override
    if driver_gap_button_pressed:
      self.driver_override_frames = self.driver_override_cooldown
      self.state = GapState.INACTIVE
      return False

    if self.driver_override_frames > 0:
      self.driver_override_frames -= 1
      return False

    if not cruise_enabled or not controls_allowed:
      self.state = GapState.INACTIVE
      return False

    # Compute target gap from speed thresholds
    target_gap = self._get_target_gap(v_ego_mph)

    # Unknown gap or already at target
    if current_gap == 0 or current_gap == target_gap:
      self.state = GapState.INACTIVE
      return False

    if self.state == GapState.INACTIVE:
      # Need to change gap — start waiting period before first toggle
      self.state = GapState.WAITING
      self.wait_frames = self.gap_toggle_cooldown
      return False

    elif self.state == GapState.WAITING:
      self.wait_frames -= 1
      if self.wait_frames <= 0:
        self.state = GapState.TOGGLING
        return False
      return False

    elif self.state == GapState.TOGGLING:
      # Send one toggle, then wait for CAN feedback
      self.state = GapState.WAITING
      self.wait_frames = self.gap_toggle_cooldown
      return True

    return False

  @staticmethod
  def _get_target_gap(v_ego_mph: float) -> int:
    for max_speed, gap in CarControllerParams.ICBM_GAP_THRESHOLDS:
      if v_ego_mph < max_speed:
        return gap
    return CarControllerParams.ICBM_GAP_THRESHOLDS[-1][1]
