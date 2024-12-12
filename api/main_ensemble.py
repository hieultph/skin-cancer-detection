from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

num_models = 3

MODEL_1 = tf.keras.models.load_model("./trained_models/model_scd_cnn-os_3_v2.h5", compile=False)

MODEL_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

MODEL_2 = tf.keras.models.load_model("./trained_models/model_scd_cnn-os_1_v2.h5", compile=False)

MODEL_3 = tf.keras.models.load_model("./trained_models/model_scd_cnn-os-lstm_1.h5", compile=False)

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
        
    y_pred = MODEL_1.predict(img_batch)

    average_benign = MODEL_1.predict(img_batch)[0][0] + MODEL_2.predict(img_batch)[0][0] + MODEL_3.predict(img_batch)[0][0]
    average_malignant = MODEL_1.predict(img_batch)[0][1] + MODEL_2.predict(img_batch)[0][1] + MODEL_3.predict(img_batch)[0][1]

    if (average_benign > average_malignant):
        predicted_class = CLASS_NAMES[0]
        confidence = average_benign / num_models
    else:
        predicted_class = CLASS_NAMES[1]
        confidence = average_malignant / num_models

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)