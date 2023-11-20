import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# Load the entire model
# Benchmark model
class BenchmarkModel(nn.Module):
    def __init__(self):
        super(BenchmarkModel, self).__init__()
        self.fc = nn.Linear(33 * 3, 8)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

# Initialize the model and hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BenchmarkModel().to(device)

# Replace this with your actual model file path
model_file_path = '/Users/abhishekchothani/01 Study/04 EE CS/04.03 Artificial Intelligence/02 Code/00 Mirai Code/Mirai3_Drone_Gesture_Control/Models/model_weights.pth'

# Attempt to load the model file
try:
    model_state_dict = torch.load(model_file_path)
    model.load_state_dict(model_state_dict)
    print("Model file loaded successfully.")
except Exception as e:
    print(f"Error loading model file: {e}")

model.eval()

class_names = [
    'backward', 'do nothing', 'forward', 'land', 
    'left', 'right', 'takeoff', 'up' 
]

def process_landmarks_for_model(landmarks):
    # Assuming landmarks is a list of (33, 3) shape
    landmarks_array = np.array(landmarks, dtype=np.float32)  # Explicitly specify dtype as float32
    # Reshape to (1, 33, 3) - adding an extra dimension for batch size
    return landmarks_array.reshape(1, 33, 3)

# Class for handling pose detection
class PoseDetector:
    def __init__(self, static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
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
            landmarks_coordinates = [(landmark.x, landmark.y, landmark.z) for landmark in pose_landmarks_result.pose_landmarks.landmark]
            pose_positions.append(landmarks_coordinates)
        return pose_positions, frame


# Load and process a single image
image_path = f'Raw Dataset/up100/up98.jpg'
frame = cv2.imread(image_path)

pose_detector = PoseDetector()
pose_landmarks, pose_frame = pose_detector.detect_and_draw_landmarks(frame)

if pose_landmarks:
    processed_landmarks = process_landmarks_for_model(pose_landmarks[0])
    input_tensor = torch.tensor(processed_landmarks, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor.to(device))

    predicted_class = torch.argmax(predictions, dim=1)
    predicted_label = class_names[predicted_class.item()]

    cv2.putText(pose_frame, f'Predicted Pose: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Display the resulting frame with the prediction
cv2.imshow('Predicted Pose', pose_frame)
cv2.waitKey(0)  # Wait for a key press to close

# Release resources
cv2.destroyAllWindows()