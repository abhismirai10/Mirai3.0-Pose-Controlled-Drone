import cv2
import mediapipe as mp

# Create a VideoCapture object for webcam feed
webcam = cv2.VideoCapture(0)

# If webcam fails to open, raise an error
if not webcam.isOpened():
    raise IOError("Cannot open webcam")

# Standard size for webcam
webcam_width = 1280
webcam_height = 720

# Set webcam resolution and FPS
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
webcam.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second to 30
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set MJPG as video codec

# Class for handling pose detection
class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=static_image_mode, 
                                               model_complexity=model_complexity, 
                                               min_detection_confidence=min_detection_confidence, 
                                               min_tracking_confidence=min_tracking_confidence)
        self.landmark_drawer = mp.solutions.drawing_utils

    # Method to process a frame, detect pose and draw landmarks
    def detect_and_draw_landmarks(self, frame):
        pose_positions = []

        # Convert BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find pose landmarks
        pose_landmarks_result = self.pose_detector.process(rgb_frame)

        # If pose landmarks are detected, draw them on the frame
        if pose_landmarks_result.pose_landmarks:
            self.landmark_drawer.draw_landmarks(frame, pose_landmarks_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks_coordinates = [(int(landmark.x * webcam_width), int(landmark.y * webcam_height)) for landmark in pose_landmarks_result.pose_landmarks.landmark]
            pose_positions.append(landmarks_coordinates)
        return pose_positions

# Create a PoseDetector object
pose_detector = PoseDetector()

# Start webcam feed and continue until 'q' is pressed
while True:
    # Capture frame from webcam
    ret, frame = webcam.read()

    # Flip the frame horizontally for a later selfie-view display
    flipped_frame = cv2.flip(frame, 1)

    # Get pose landmarks from the frame
    pose_landmarks = pose_detector.detect_and_draw_landmarks(flipped_frame)    

    # use landMark positions (example usage, modify as needed)
    # Here you can use the landmark positions as needed for your application

    # Verify that frame has been successfully read
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Pose Detection Webcam Feed', flipped_frame)

    # If 'q' is pressed on the keyboard, break the loop and end the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the webcam object
webcam.release()

# Destroy all the windows
cv2.destroyAllWindows()
