import cv2
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from datetime import datetime
import sys

import firebase_admin
from firebase_admin import credentials, firestore, messaging

if len(sys.argv) < 1:
    print("Please provide the IP address as a command-line argument.")
    sys.exit(1)
ip = sys.argv[1]
# Load YOLOv5 model from a .pt file
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model
cred = credentials.Certificate('./bird.json')  # Replace with your service account key file
firebase_admin.initialize_app(cred)
db = firestore.client()

model_path = "./chicken.pt"  # Replace this with the path to your custom YOLOv5 .pt file
model = load_model(model_path)
acceptable_confidence = 0.5
# Set the model to evaluation mode
model.eval()

def save_detection(image_name):
    print("Saving detection ...")
    detectionLogs = {}
    detectionLogs['ip'] = ip 
    detectionLogs['imagePath'] = f"./results/{image_name}" 
    detectionLogs['createdAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detectionLogs['status'] = "Unread"
    ref = db.collection('detections')
    ref.add(detectionLogs)
    print("Successfully saved detection")
    
    
    
def save_image_with_boxes(frame, detections):
    detected_objects = []
    for index, detection in detections.iterrows():
        if detection['confidence'] >= acceptable_confidence:
            box = [
                int(detection['xmin']),
                int(detection['ymin']),
                int(detection['xmax']),
                int(detection['ymax'])
            ]
            # Draw bounding box on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{detection['name']} {detection['confidence']:.2f}",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            detected_objects.append({
                'name': detection['name'],
                'confidence': detection['confidence'],
                'bbox': box
            })

    if detected_objects:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"detected_{timestamp}.jpg"
        cv2.imwrite(f"./results/{image_name}", frame)
        return image_name, detected_objects

    return None, None
    

# Open a connection to the camera (0 represents the default camera, change it if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform inference on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    frame = results.render()[0]

    detections = results.pandas().xyxy[0]
        
    for index, detection in detections.iterrows():
        if (detection['confidence'] >= acceptable_confidence):
            print(f"Confidence: {detection['confidence']}, Name: {detection['name']}")
            im,s = save_image_with_boxes(frame,detections)
            save_detection(im)
    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
