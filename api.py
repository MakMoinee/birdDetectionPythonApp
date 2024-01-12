import io
import os
import mysql.connector
import torch
from flask import Flask, request, jsonify
import random
import requests
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from yolov5 import detect
import pandas as pd
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from PIL import Image
from PIL import ImageDraw

weights = "chicken.pt"
source = "./results"
data = "coco128.yaml"
imgsz = (640, 640)
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
device = ""
view_img = False
save_txt = False
save_conf = False
save_crop = False
nosave = False
classes = None
agnostic_nms = False
augment = False
visualize = False
update = False
project = "C://Users//Mak Moinee//Documents//birdDetectionPythonApp"
name = "exp"
exist_ok = False
line_thickness = 3
hide_labels = False
hide_conf = False
half = False
dnn = False
vid_stride = 1

app = Flask(__name__)

# Load the YOLOv5 model

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor()

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model


# Initialize the model
model_path = "./chicken.pt"  # Replace this with the path to your custom YOLOv5 .pt file
model = load_model(model_path)
# Set the model to evaluation mode
model.eval()

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello World"


@app.route('/detect', methods=['POST'])
def detect_objects():
    # Check if request has an id field
    if 'id' not in request.form:
        logger.error("Missing id field")
        return jsonify(error="Missing id field"), 400

    # Get the id from the form data
    id = request.form['id']

    storagePath = request.form['storagePath']
    rawImage = request.form.get('image_url', '')
    print(storagePath)
    # Get the image URL from the request
    # image_url = "http://localhost:8443" + rawImage
    image_url = rawImage
    if not image_url:
        logger.error("Missing image URL field")
        return jsonify(error="Missing image URL field"), 400

    rand_num = random.randint(1, 3)

    print(image_url)

    # Perform object detection asynchronously
    executor.submit(do_object_detection, id, image_url, rand_num,storagePath, rawImage)

    logger.info("Object detection process started in the background.")
    print()
    print()
    return 'Object detection process started.'

def do_object_detection(id, image_url, rand_num,storagePath, rawImage):
    # Fetch the image from the URL
    # response = requests.get(image_url)
    # if response.status_code != 200:
    #     logger.error("Failed to fetch image")
    #     return

    # Read the image content and convert to bytes
    # image_bytes = io.BytesIO(response.content)

    # Detect objects in the image
    #loading_bar(100, prefix='Progress:', suffix='Complete', length=30, fill='█', empty='─')
    results = model(image_url)
    image = Image.open(io.BytesIO(requests.get(image_url).content))
    numberOfInstance = 0
    
    try:
        # Convert results to a Pandas DataFrame
        df = results.pandas().xyxy[0]

        # Get the number of detected objects
        numberOfInstance = df.shape[0]
        print(numberOfInstance)


    except Exception as e:
        print("Error:", str(e))

    for idx, result in enumerate(results.xyxy):
        logger.info(result)
        try:
            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for box in result:
                # Convert box values to integers
                box = [int(coord) for coord in box[0:4]]
                draw.rectangle(box, outline="red", width=2)
            
            # Generate a unique filename based on id and rand_num
            filename = f"{id}_{rand_num}_{idx + 1}.jpg"
            image_save_path = os.path.join(storagePath, filename)
            
            # Save the image with bounding boxes to the specified path
            image.save(image_save_path)
            
            logger.info("File saved successfully!")
        except Exception as e:
            logger.info("Error saving file:", str(e))
        
            
    
def loading_bar(total, prefix='', suffix='', length=30, fill='█', empty='─'):
    progress = 0
    while progress <= total:
        percent = progress / total
        filled_length = int(length * percent)
        bar = fill * filled_length + empty * (length - filled_length)
        if progress<=1:
            print()
            progress += 1
            continue
        print(f'\r{prefix} [{bar}] {progress}/{total} {suffix}', end='', flush=True)
        time.sleep(0.1)
        progress += 1
    print()

def runData(imageUrl):
    detect.run(weights=weights, source=imageUrl, data=data, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres,
    max_det=max_det, device=device, view_img=view_img, save_txt=save_txt, save_conf=save_conf, save_crop=save_crop,
    nosave=nosave, classes=classes, agnostic_nms=agnostic_nms, augment=augment, visualize=visualize, update=update,
    project=project, name=name, exist_ok=exist_ok, line_thickness=line_thickness, hide_labels=hide_labels,
    hide_conf=hide_conf, half=half, dnn=dnn, vid_stride=vid_stride)


if __name__ == '__main__':
    app.run()
