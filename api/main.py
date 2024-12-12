from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

CONV_MODEL = tf.keras.models.load_model("./trained_models/conv_model.h5", compile=False)

with open('./trained_models/LR_model.pkl', 'rb') as f:
    LR_MODEL = pickle.load(f)

CLASS_NAMES = ['benign', 'malignant']

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/ping')
async def ping():
    return "Hello, I still alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)

    image_features = CONV_MODEL.predict(img_batch)
    image_features_flat = image_features.reshape(image_features.shape[0], -1)

    y_pred = LR_MODEL.predict_proba(image_features_flat)

    benign_con = y_pred[0][0]
    malignant_con = y_pred[0][1]

    if (benign_con > malignant_con):
        predicted_class = CLASS_NAMES[0]
        confidence = benign_con
    else:
        predicted_class = CLASS_NAMES[1]
        confidence = malignant_con

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)