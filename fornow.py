class CloseHandDetector:
    def __init__(
            self,
            velocity_threshold_x: float = 0.2, # Minimum velocity to detect movement in x direction
            velocity_threshold_y: float = 0.2, # Minimum velocity to detect movement in y direction
            ewma_alpha: float = 0.3, # smoothing factor (0=no smooth, 1=too smooth)
            hold_frames: int = 5, # frames to wait before confirming static gesture
            confidence_ramp_up: float = 0.3, # How fast confidence increases
            confidence_decay: float = 0.2, # how fast confidence decay
            confidence_threshold: float = 0.6 # minimun confidence to trigger acction as true

        ) -> None:
        self.velocity_threshold_x = velocity_threshold_x
        self.velocity_threshold_y = velocity_threshold_y
        self.ewma_alpha = ewma_alpha
        self.hold_frames = hold_frames
        self.confidence_ramp_up = confidence_ramp_up
        self.confidence_decay = confidence_decay
        self.confidence_threshold = confidence_threshold

        self._hands_sate = {
            'left' : self._create_hand_state(),
            'right' : self._create_hand_state()
            }
    def _create_hand_state(self):
        """
        Docstring for _create_hand_state
        
        :param self: Description
        """
        return {
            'ewma_velocity': EWMA(alpha=self.ewma_alpha),
            'move_confidence': 0.0,
            'current_move_direction': None,
            'static_hold_count': 0,
            'last_static_gesture': None

        }
    def _get_state_(self, hand_label:str):
        """
        Docstring for _get_state_
        
        :param self: Description
        :param hand_label: Description
        :type hand_label: str
        """
        if hand_label not in self._hands_sate:
            self._hands_sate[hand_label] = self._create_hand_state()
        return self._hands_sate[hand_label]
    def detect(self, metrics: HandMetrics,hand_label:str = 'right') -> GestureResult:
        
        # Get current hand state
        state = self._get_state_(hand_label)
        # Check if all fingers are closed
        extended = metrics.fingers_extended
        hand_closed = not any([extended['thumb'],extended['index'], extended['middle'], extended['ring'], extended['pinky']])
        base_metadata = {
            'hand_closed': hand_closed,
            'velocity': metrics.velocity,
            'ewma_velocity_y': 0.0,
            'move_confidence': state['move_confidence'],
            'static_hold_count': state['static_hold_count'],
            'hold_frames_needed': self.hold_frames,
            'hand_label': hand_label,
            'reason': None
        }

        if not hand_closed:
            state['static_hold_count'] = 0
            state['last_static_gesture'] = None
            state['move_confidence'] = max(0.0,state['move_confidence']-self.confidence_decay)
            if state['move_confidence'] == 0.0:
                state['current_move_direction'] = None
            base_metadata['reason'] = 'not_all_fingers_are_closed'
            return GestureResult(detected=False,gesture_name='none',metadata=base_metadata)
        
        # determinne velocity of hand
        vx, vy = metrics.velocity
        smoothed = state['ewma_velocity'].update([vx,vy])
        ewma_vx = float(smoothed[0])
        ewma_vy = float(smoothed[1])

        # if x velocity is dominant then choose the direction of x 
        if ewma_vx > ewma_vy:
            detected_move_direction = 'right' if ewma_vx > self.velocity_threshold_x else 'left'
        # if y velocity is dominant then choose the direction of y 
        if ewma_vx < ewma_vy:
            detected_move_direction = 'up' if ewma_vy < self.velocity_threshold_y else 'down'
        
        # if direction detected
        if detected_move_direction is not None:
            # if 1st time detected and have no direction history
            if state['current_move_direction'] is None:
                state['current_move_direction'] = detected_move_direction
                state['move_confidence'] = min(1.0,state['move_confidence'] + self.confidence_ramp_up)
            #if direction is in the same way as previous
            elif state['current_move_direction'] == detected_move_direction:
                state['move_confidence'] = min(1.0,state['move_confidence'] + self.confidence_ramp_up)
            # direction changed
            else:
                state['move_confidence'] = max(0.0,state['move_confidence'] - self.confidence_decay)
                if state['move_confidence'] == 0:
                    state['move_confidence'] = min(1.0,state['move_confidence'] + self.confidence_ramp_up)
                    state['current_move_direction'] = detected_move_direction
        
        # confirm movement if the confidence is high
        movement_confirmed = state['move_confidence'] > self.confidence_threshold

        if hand_closed:
            base_gesture = 'closed_hand'
            current_static = 'hand_closed'
            if movement_confirmed and state['current_move_direction']:
                gesture_name = f'{base_gesture}_moving_{state["current_move_direction"]}'
                detected = True
                state['static_hold_count'] = 0
                state['last_static_gesture'] = None
                base_metadata['reason'] = 'movement_confirmed'
            else:
                if state['last_static_gesture'] == current_static:
                    state['static_hold_count'] += 1
                else:
                    state['last_static_gesture'] = current_static
                    state['static_hold_count'] = 1

                base_metadata['static_hold_count'] = state['static_hold_count']
                
                if state['static_hold_count'] >= self.hold_frames:
                    gesture_name = base_gesture
                    detected = True
                    base_metadata['reason'] = 'static_confirmed'
                else:
                    base_metadata['reason'] = 'waiting_for_hold'
        # neither condition met
        else:
            state['static_hold_count'] = 0 
            state['last_static_gesture'] = None
            base_metadata['reason'] = 'not_closed_hand_pose'
        return GestureResult(
            detected= detected,
            gesture_name= gesture_name,
            confidence=state['move_confidence'] if movement_confirmed else (1.0 if detected else 0.0),
            metadata=base_metadata
        )