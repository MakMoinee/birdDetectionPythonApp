import cv2
import torch

ip = ""
rtsp_url = f"rtsp://{ip}/live/ch00_0"

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

model_path = "./best.pt"  # Replace this with the path to your custom YOLOv5 .pt file
model = load_model(model_path)
updateOnce = False
# Set the model to evaluation mode
model.eval()

stream = cv2.VideoCapture(rtsp_url)
detectedCount = 0
acceptable_confidence = 0.6

try:
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            break

        if (updateOnce is not True):
            updateOnce = True
        
        # Perform inference
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        for index, detection in detections.iterrows():
            if (detection['confidence'] >= acceptable_confidence):
                print(f"Confidence: {detection['confidence']}, Name: {detection['name']}")
                
        
        cv2.imshow('Real-time Detection', results.render()[0])

        if cv2.waitKey(1) == ord('q'):
            break

except cv2.error as e:
    print(f"OpenCV error: {e}")
except KeyboardInterrupt:
    print("Keyboard Interrupt detected. Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Release resources
    # updateStatus(userID,ip,"Inactive")
    stream.release()
    cv2.destroyAllWindows()

