from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import gdown

app = FastAPI()

# Allow CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive model file ID and local model path
FILE_ID = '1_2oIx2AegYXiROo3JGKjXWod2QmewpYq'
MODEL_PATH = "./model.keras"

def fetch_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)

@app.on_event("startup")
async def load_model():
    fetch_model()
    global MODEL
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    global CLASS_NAMES
    CLASS_NAMES = ["Early blight", "Late blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, World"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {'class': predicted_class, 'confidence': float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
