import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def load_model(model_path):
    return load_model(model_path)

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    return predicted_class