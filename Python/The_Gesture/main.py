import cv2
import pyautogui
import mediapipe as mp
import time
import numpy as np

class HandGestureControl:
    """
    A class to control computer actions using hand gestures,
    including wave-to-close and sequenced gestures (fist x2 + swipe).
    Modified to perform mouse swipe instead of hotkey after fist x2.
    """
    # --- Constants ---
    # Wave Detection
    WAVE_TIMEOUT = 1.5
    WAVE_BOUNDARY_RATIO = 0.25
    # Fist Detection
    FIST_THRESHOLD = 55 # Max distance for tips to MCPs for fist detection
    FIST_SEQUENCE_TIMEOUT = 1.5 # Max time between two fist closures
    FIST_DEBOUNCE_TIME = 0.4 # Min time between fist detections
    # Swipe Detection
    SWIPE_ARM_TIMEOUT = 2.0 # Max time after arming to start/complete swipe
    SWIPE_MIN_DISTANCE = 70 # Min vertical distance for swipe in pixels (camera frame)
    SWIPE_FINGER_EXTEND_THRESHOLD = 70 # Min distance tip-to-mcp to be considered extended
    MOUSE_SWIPE_DISTANCE = 150 # Pixels to move the mouse cursor during a simulated swipe
    MOUSE_SWIPE_DURATION = 0.2 # Duration of the simulated mouse swipe in seconds

    def __init__(self):
        self.window_status_closed = False
        self.cap = None
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothening = 7
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.gesture_timeout = 0.5 # Timeout for simple discrete actions like clicks
        self.last_gesture_time = 0 # Time of last simple action
        self.active_gesture = None # Last simple action performed
        self.frame_width = 0
        self.frame_height = 0
        self.frame_padding = 100

        # --- Wave Detection Variables ---
        self.wave_state = 'none'
        self.last_wave_side_time = 0

        # --- Sequence Gesture State Variables ---
        self.sequence_state = 'idle' # 'idle', 'fist_1_detected', 'armed_for_swipe'
        self.last_fist_time = 0
        self.fist_count = 0
        self.was_fist_last_frame = False # To detect transition *into* fist
        self.arm_time = 0 # Time when state becomes 'armed_for_swipe'
        self.swipe_start_y = None # Starting Y position for swipe detection in camera frame
        self.was_swiping_fingers_extended = False # Track if fingers were extended in previous frame during swipe check

    def initialize(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        success, frame = self.cap.read()
        if success:
            self.frame_height, self.frame_width, _ = frame.shape
            self.plocX, self.plocY = self.screen_width / 2, self.screen_height / 2
            print(f"Camera Frame Dimensions: Width={self.frame_width}, Height={self.frame_height}")
        else:
             raise IOError("Cannot read from webcam")
        # Reset states on init
        self._reset_sequence_state()
        self.wave_state = 'none'

    def detect_hands(self):
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def get_hand_landmarks(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        landmarks = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hand.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * self.frame_width), int(lm.y * self.frame_height)
                landmarks.append([id, cx, cy])
        return landmarks if landmarks else None

    def is_gesture_active(self, gesture_name):
        """Checks if a simple discrete gesture can be performed based on timeout."""
        current_time = time.time()
        # Only allow a new simple gesture if enough time has passed since the last one
        if current_time - self.last_gesture_time > self.gesture_timeout:
             self.last_gesture_time = current_time
             # self.active_gesture = gesture_name # Optionally track active simple gesture
             return True
        return False

    def _reset_sequence_state(self):
        """Resets the sequence gesture state machine."""
        # print("Resetting sequence state") # Debug
        self.sequence_state = 'idle'
        self.fist_count = 0
        self.last_fist_time = 0
        self.was_fist_last_frame = False
        self.arm_time = 0
        self.swipe_start_y = None
        self.was_swiping_fingers_extended = False

    def perform_action(self, landmarks, frame) -> bool:
        """
        Performs actions based on gestures. Includes wave-to-close and sequenced gestures.
        Returns True if the application should exit.
        """
        if not landmarks or self.frame_width == 0:
            # self.active_gesture = None # Keep mouse movement active if hand is temporarily lost?
            self.wave_state = 'none'
            # self._reset_sequence_state() # Keep sequence state? No, reset on hand lost is safer.
            self._reset_sequence_state()
            return False

        current_time = time.time()

        # --- Extract Key Landmark Coordinates ---
        # (Assuming landmarks list has 21 elements if hand is detected)
        try:
            wrist_coord = landmarks[0][1:]
            thumb_tip_coord = landmarks[4][1:]
            index_tip_coord = landmarks[8][1:]
            middle_tip_coord = landmarks[12][1:]
            ring_tip_coord = landmarks[16][1:]
            # pinky_tip_coord = landmarks[20][1:] # Not used often

            index_mcp_coord = landmarks[5][1:]
            middle_mcp_coord = landmarks[9][1:]
            ring_mcp_coord = landmarks[13][1:]
            # pinky_mcp_coord = landmarks[17][1:]
        except IndexError:
            print("Warning: Landmark list incomplete.")
            self._reset_sequence_state() # Reset sequence if landmark data is bad
            self.active_gesture = None
            return False # Cannot process gestures

        # --- Utility Function ---
        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        # --- 1. Wave Detection (Highest Priority for Exit) ---
        left_boundary = self.frame_width * self.WAVE_BOUNDARY_RATIO
        right_boundary = self.frame_width * (1 - self.WAVE_BOUNDARY_RATIO)
        wrist_x = wrist_coord[0]
        # (Wave logic remains the same as previous version - check wrist_x against boundaries and state)
        if wrist_x < left_boundary:
            if self.wave_state == 'seen_right' and (current_time - self.last_wave_side_time < self.WAVE_TIMEOUT):
                print("Wave detected (Right -> Left)! Closing...")
                return True # Signal exit
            elif self.wave_state != 'seen_left':
                 self.wave_state = 'seen_left'
                 self.last_wave_side_time = current_time
        elif wrist_x > right_boundary:
            if self.wave_state == 'seen_left' and (current_time - self.last_wave_side_time < self.WAVE_TIMEOUT):
                print("Wave detected (Left -> Right)! Closing...")
                return True # Signal exit
            elif self.wave_state != 'seen_right':
                self.wave_state = 'seen_right'
                self.last_wave_side_time = current_time
        else:
             if self.wave_state != 'none' and (current_time - self.last_wave_side_time > self.WAVE_TIMEOUT):
                 self.wave_state = 'none'

        # --- Gesture Calculations (Needed for multiple checks) ---
        dist_index_thumb = distance(index_tip_coord, thumb_tip_coord)
        dist_middle_thumb = distance(middle_tip_coord, thumb_tip_coord)

        # Fist Check: Check if Index, Middle, Ring fingers are curled
        is_fist = (distance(index_tip_coord, index_mcp_coord) < self.FIST_THRESHOLD and
                   distance(middle_tip_coord, middle_mcp_coord) < self.FIST_THRESHOLD and
                   distance(ring_tip_coord, ring_mcp_coord) < self.FIST_THRESHOLD)

        # Three Finger Extend Check (Index, Middle, Ring)
        are_fingers_extended = (distance(index_tip_coord, index_mcp_coord) > self.SWIPE_FINGER_EXTEND_THRESHOLD and
                                distance(middle_tip_coord, middle_mcp_coord) > self.SWIPE_FINGER_EXTEND_THRESHOLD and
                                distance(ring_tip_coord, ring_mcp_coord) > self.SWIPE_FINGER_EXTEND_THRESHOLD)


        # --- 2. Sequence Gesture State Machine ---
        reset_sequence = False # Flag to reset sequence if conflicting action happens
        performed_sequence_action = False

        # State: idle
        if self.sequence_state == 'idle':
            if is_fist and not self.was_fist_last_frame and current_time - self.last_fist_time > self.FIST_DEBOUNCE_TIME:
                print("Sequence: Fist 1 detected")
                self.sequence_state = 'fist_1_detected'
                self.last_fist_time = current_time
                self.fist_count = 1

        # State: fist_1_detected
        elif self.sequence_state == 'fist_1_detected':
            if is_fist and not self.was_fist_last_frame and current_time - self.last_fist_time > self.FIST_DEBOUNCE_TIME:
                if current_time - self.last_fist_time < self.FIST_SEQUENCE_TIMEOUT:
                    print("Sequence: Fist 2 detected - ARMED FOR MOUSE SWIPE")
                    self.sequence_state = 'armed_for_swipe'
                    self.last_fist_time = current_time
                    self.arm_time = current_time
                    self.fist_count = 2
                    self.swipe_start_y = None # Reset swipe start pos
                    self.was_swiping_fingers_extended = False
                    if(self.window_status_closed):
                        print("--- Action: Three Finger Swipe Up Detected (Mouse Swipe) ---")
                        pyautogui.hotkey('alt', 'tab')
                        self.window_status_closed = False
                    else:
                        print("--- Action: Three Finger Swipe Down Detected (Mouse Swipe) ---")
                        pyautogui.hotkey('win', 'd')
                        self.window_status_closed = True

                    self._reset_sequence_state()
                    performed_sequence_action = True
                    self.last_gesture_time = current_time # Record time
                else:
                    # Too long since first fist, reset and treat this as fist 1 again
                    print("Sequence: Timeout since Fist 1, resetting.")
                    self._reset_sequence_state()
                    # Re-trigger fist 1 detection for the current fist
                    print("Sequence: Fist 1 detected")
                    self.sequence_state = 'fist_1_detected'
                    self.last_fist_time = current_time
                    self.fist_count = 1
            elif not is_fist and current_time - self.last_fist_time > self.FIST_SEQUENCE_TIMEOUT:
                # If fist is released and timeout passed, reset
                # print("Sequence: Resetting (Fist 1 released, timeout)") # Debug
                self._reset_sequence_state()

        # State: armed_for_swipe
        # elif self.sequence_state == 'armed_for_swipe':
        #     if current_time - self.arm_time > self.SWIPE_ARM_TIMEOUT:
        #          print("Sequence: Swipe arm timeout, resetting.")
        #          self._reset_sequence_state()
        #     elif are_fingers_extended:
        #         # Calculate average Y of the three fingertips
        #         current_swipe_y = (index_tip_coord[1] + middle_tip_coord[1] + ring_tip_coord[1]) / 3.0
        #
        #         if not self.was_swiping_fingers_extended:
        #             # Fingers just became extended, record start Y
        #             print(f"Sequence: Swipe motion started at Y={current_swipe_y:.0f}")
        #             self.swipe_start_y = current_swipe_y
        #             self.was_swiping_fingers_extended = True
        #         elif self.swipe_start_y is not None:
        #             # Fingers were already extended, check for movement
        #             delta_y = current_swipe_y - self.swipe_start_y
        #             cv2.putText(frame, f"Swipe dY: {delta_y:.0f}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # Show delta
        #
        #             # Check for Swipe Down
        #             if delta_y > self.SWIPE_MIN_DISTANCE:
        #                 print("--- Action: Three Finger Swipe Down Detected (Mouse Swipe) ---")
        #                 cv2.putText(frame, "MOUSE SWIPE DOWN!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        #                 pyautogui.move(0, self.MOUSE_SWIPE_DISTANCE, duration=self.MOUSE_SWIPE_DURATION) # Simulate mouse swipe down
        #                 self._reset_sequence_state()
        #                 performed_sequence_action = True
        #                 self.last_gesture_time = current_time # Record time
        #
        #             # Check for Swipe Up
        #             elif delta_y < -self.SWIPE_MIN_DISTANCE:
        #                 print("--- Action: Three Finger Swipe Up Detected (Mouse Swipe) ---")
        #                 cv2.putText(frame, "MOUSE SWIPE UP!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        #                 pyautogui.move(0, -self.MOUSE_SWIPE_DISTANCE, duration=self.MOUSE_SWIPE_DURATION) # Simulate mouse swipe up
        #                 self._reset_sequence_state()
        #                 performed_sequence_action = True
        #                 self.last_gesture_time = current_time # Record time
        #     else:
        #         # Fingers are not extended, reset swipe tracking within armed state
        #         self.was_swiping_fingers_extended = False
        #         self.swipe_start_y = None

        # Update was_fist_last_frame for next iteration
        self.was_fist_last_frame = is_fist

        # --- 3. Simple Gestures (Only if sequence action wasn't performed) ---
        # Simple gestures like clicks and scrolls should interrupt/prevent sequence actions
        # Mouse movement is continuous and has higher priority
        action_performed_simple = False # Track if any simple action happened this frame

        # Mouse Movement (Index finger up, away from thumb) - High Priority
        is_index_extended = distance(index_tip_coord, index_mcp_coord) > self.SWIPE_FINGER_EXTEND_THRESHOLD
        if is_index_extended and dist_index_thumb > 60:
            # (Mouse movement logic remains the same - np.interp, smoothening)
            cam_x, cam_y = index_tip_coord
            # Interpolate camera coordinates to screen coordinates, reversing X for natural feel
            target_screen_x = np.interp(cam_x, (self.frame_padding, self.frame_width - self.frame_padding), (self.screen_width, 0))
            target_screen_y = np.interp(cam_y, (self.frame_padding, self.frame_height - self.frame_padding), (0, self.screen_height))
            # Smoothen movement
            self.clocX = self.plocX + (target_screen_x - self.plocX) / self.smoothening
            self.clocY = self.plocY + (target_screen_y - self.plocY) / self.smoothening
            # Ensure within screen bounds
            final_x = min(max(int(self.clocX), 0), self.screen_width - 1)
            final_y = min(max(int(self.clocY), 0), self.screen_height - 1)
            pyautogui.moveTo(final_x, final_y)
            self.plocX, self.plocY = self.clocX, self.clocY
            cv2.circle(frame, index_tip_coord, 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Moving Mouse", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            self.active_gesture = "move" # Indicate continuous mouse movement
            action_performed_simple = True
            reset_sequence = True # Mouse movement should interrupt sequence
            # Prevent sequence action from overriding mouse movement even if sequence was completed
            performed_sequence_action = False # This will prevent the check below from being skipped

        # Simple actions (Clicks, Scrolls) only if not actively moving the mouse cursor
        elif not self.active_gesture == "move":
            # Left Click (Index+Thumb close)
            if dist_index_thumb < 30:
                if self.is_gesture_active("click"):
                    pyautogui.click()
                    print("Left Click")
                    cv2.putText(frame, "Left Click", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    action_performed_simple = True
                    reset_sequence = True # Click interrupts sequence
                    self.active_gesture = "click"

            # Right Click (Middle+Thumb close)
            elif dist_middle_thumb < 30:
                 if self.is_gesture_active("right_click"):
                    pyautogui.click(button='right')
                    print("Right Click")
                    cv2.putText(frame, "Right Click", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    action_performed_simple = True
                    reset_sequence = True # Click interrupts sequence
                    self.active_gesture = "right_click"

            # Scroll Gestures (using fist check + thumb position)
            # Note: This fist check might conflict with sequence start, but sequence has priority now.
            # We only allow scroll if the sequence state is 'idle' or if scroll is performed *before* fist 1 is detected.
            # Let's simplify: Only allow scroll if sequence state is 'idle'.
            elif is_fist and self.sequence_state == 'idle': # Check if fist is formed for scrolling
                thumb_is_up = thumb_tip_coord[1] < middle_mcp_coord[1] - 15
                thumb_is_down = thumb_tip_coord[1] > middle_mcp_coord[1] + 15

                if thumb_is_up:
                    if self.is_gesture_active("scroll_up"):
                        pyautogui.scroll(100)
                        print("Scroll Up")
                        cv2.putText(frame, "Scroll Up", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                        action_performed_simple = True
                        self.active_gesture = "scroll_up"

                elif thumb_is_down:
                    if self.is_gesture_active("scroll_down"):
                        pyautogui.scroll(-100)
                        print("Scroll Down")
                        cv2.putText(frame, "Scroll Down", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                        action_performed_simple = True
                        self.active_gesture = "scroll_down"

            # If no simple action performed (clicks/scrolls), clear simple action state after timeout
            if not action_performed_simple and self.active_gesture in ["click", "right_click", "scroll_up", "scroll_down"]:
                 if current_time - self.last_gesture_time > self.gesture_timeout:
                     self.active_gesture = None
            # If not moving mouse and no simple action occurred, ensure active_gesture is None
            elif not action_performed_simple and self.active_gesture == "move":
                 # If we stopped moving mouse, clear active gesture after a brief moment
                 if current_time - self.last_gesture_time > 0.1: # Small delay before clearing "move" state
                      self.active_gesture = None
            elif not action_performed_simple and self.active_gesture is not None:
                 # If a simple gesture was active but didn't trigger this frame, clear it after timeout
                 if current_time - self.last_gesture_time > self.gesture_timeout:
                      self.active_gesture = None


        # Reset sequence state if a conflicting simple action occurred
        # This ensures a click or mouse movement cancels an ongoing sequence
        if reset_sequence and self.sequence_state != 'idle':
            print("Sequence reset due to conflicting action.")
            self._reset_sequence_state()

        # Display current sequence state on screen
        cv2.putText(frame, f"Seq State: {self.sequence_state}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        # Display active gesture state
        # cv2.putText(frame, f"Active: {self.active_gesture}", (20, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 100, 255), 2)


        return False # Return False as default (no exit signal from these gestures)


    def run(self):
        """ Main loop """
        try:
            self.initialize()
            print(f"Screen Dimensions: Width={self.screen_width}, Height={self.screen_height}")
            print("Gestures: Wave(close), Fistx2+SwipeUp/Down(Mouse Scroll), Index(move), Index+Thumb(LClick), Middle+Thumb(RClick), Fist+Thumb(scroll)")
            pyautogui.FAILSAFE = False # Disable failsafe for smoother control
            while True:
                success, frame = self.detect_hands()
                if not success:
                    print("Failed to read frame from camera, exiting.")
                    break # Exit if frame reading fails

                should_exit = False
                landmarks = self.get_hand_landmarks(frame)

                # Perform actions only if hand landmarks are detected
                if landmarks:
                    should_exit = self.perform_action(landmarks, frame)
                    if should_exit:
                        cv2.putText(frame, "WAVE DETECTED - CLOSING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.imshow("Hand Gesture Control", frame)
                        cv2.waitKey(1000) # Show message for a moment
                        break
                else:
                    # No hand detected, reset states that rely on continuous hand presence
                    # self.active_gesture = None # Keep mouse movement "active" state? No, clear it.
                    if self.active_gesture == "move": # If mouse was moving, clear the state quickly
                         if time.time() - self.last_gesture_time > 0.1:
                              self.active_gesture = None
                    else: # Clear other simple gesture states after timeout
                         if time.time() - self.last_gesture_time > self.gesture_timeout:
                              self.active_gesture = None

                    self.wave_state = 'none'
                    self._reset_sequence_state() # Reset sequence state if hand lost

                cv2.imshow("Hand Gesture Control", frame)
                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting via 'q' key...")
                    break

        except IOError as e: print(f"Initialization Error: {e}")
        except Exception as e: import traceback; print(f"Runtime error: {e}"); traceback.print_exc()
        finally: self.close()

    def close(self):
        """ Releases resources """
        if self.cap and self.cap.isOpened(): self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    gesture_control = HandGestureControl()
    gesture_control.run()