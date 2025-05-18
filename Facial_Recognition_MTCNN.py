import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pyttsx3
from facenet_pytorch import InceptionResnetV1, MTCNN

# 1. Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Parameters
img_size = 160
labels_map = ['Anshul', 'Harsh', 'Om']  # Update based on your dataset
confidence_threshold = 0.75
min_face_area = 6000  # Area in pixels

# 3. Load FaceNet Feature Extractor
feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad = False

# 4. Define Custom Classifier
class CustomFaceNet(nn.Module):
    def __init__(self, feature_extractor, num_classes):  # Fixed: __init_ instead of init
        super(CustomFaceNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return self.classifier(features)

# 5. Initialize Model
num_classes = len(labels_map)
model = CustomFaceNet(feature_extractor, num_classes).to(device)
model.load_state_dict(torch.load(r'C:\Users\mehka\Desktop\Smart Office Assistant\Facial_Recognition\Facial_Recognition_Trained_File.pth', map_location=device))
model.eval()
print("Custom model loaded!")

# 6. TTS Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 7. MTCNN for Face Detection
mtcnn = MTCNN(keep_all=True, device=device)

# 8. Transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 9. Initialize IP Webcam
ip_webcam_url = 'http://10.30.80.152:8080/video'  # Replace with your IP
cap = cv2.VideoCapture(ip_webcam_url)
prev_prediction = None

print("Starting webcam...")

# 10. Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!!!")
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            if w * h < min_face_area:
                continue

            # Clamp coordinates to stay inside frame
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            # Only process if box is valid
            if y2 > y1 and x2 > x1:
                face = frame[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (img_size, img_size))
                face = transforms.ToPILImage()(face)
                face = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(face)
                    probabilities = torch.softmax(outputs, dim=1)
                    max_prob, preds = torch.max(probabilities, 1)

                predicted_label = labels_map[preds.item()]

                if max_prob.item() < confidence_threshold:
                    detected_person = "Unknown Person"
                else:
                    detected_person = predicted_label

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_person, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if detected_person != prev_prediction:
                    engine.say(f"{detected_person} detected")
                    engine.runAndWait()
                    prev_prediction = detected_person

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()