from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
import io
from pydantic import BaseModel
import base64
from PIL import Image
import os

app = FastAPI()

###

class ImageData(BaseModel):
    ImageBase64: str

def apply_greyscale(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert image to grayscale
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        hue = i / num_classes  # Distribute hues evenly across the spectrum
        color = np.array(cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        colors.append(color)
    return colors

COLORS = generate_colors(len(CLASSES))

def apply_object_detection(image_data):
    try:
        # image data to a numpy array
        image = np.array(Image.open(io.BytesIO(image_data)))
        # Convert to OpenCV format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Use a pre-trained model MobileNetSSD
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prototxt_path = os.path.join(base_dir, "deploy.prototxt")
        model_path = os.path.join(base_dir, "mobilenet_iter_73000.caffemodel")
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]

        # Draw  boxes and labels on the detected objects
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # confidence threshold
                
                idx = int(detections[0, 0, i, 1])  #class index
                label = CLASSES[idx]  
                color = COLORS[idx] 

                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), color.tolist(), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during object detection: {str(e)}")
    


@app.post("/test")
async def upload_image(image_data: ImageData):
    try:
        image_base64 = image_data.ImageBase64
        image_data = base64.b64decode(image_base64)
        
        # Apply the greyscale filter
        greyscale_image_data = apply_greyscale(image_data)
        # Convert  image to base64
        greyscale_image_base64 = base64.b64encode(greyscale_image_data).decode('utf-8')
        
        return {"ImageBase64": greyscale_image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload")
async def detect_objects(image_data: ImageData):
    try:
        image_base64 = image_data.ImageBase64
        image_data = base64.b64decode(image_base64)
        
        #object detection
        detected_image_data = apply_object_detection(image_data)
        detected_image_base64 = base64.b64encode(detected_image_data).decode('utf-8')
        
        return {"ImageBase64": detected_image_base64}
    except cv2.error as e:
        raise HTTPException(status_code=500, detail=f"OpenCV error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"General error: {str(e)}")

    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
