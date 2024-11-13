import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import gdown
import os

# Descargar el modelo desde Google Drive
def download_model():
    url = "https://drive.google.com/file/d/1u-wYYXEme2mqzh9mza5FGvWFWYWZTAJK/view?usp=sharing"  # Reemplaza con el ID de tu archivo de Google Drive
    output = '/tmp/cnn_custom.h5'
    gdown.download(url, output, quiet=False)
    return output

# Cargar el modelo
model_path = download_model()  # Descarga el modelo en tiempo de ejecución
model = tf.keras.models.load_model(model_path)


def preprocess_for_prediction(img_data):
    img = image.load_img(img_data, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para recibir las solicitudes
def predict(request):
    if request.method == 'POST':
        # Recibir archivo de imagen
        file = request.files['file']
        img_path = '/tmp/temp_image.jpg'  # Guardamos temporalmente la imagen
        file.save(img_path)

        # Preprocesar la imagen y realizar la predicción
        img_array = preprocess_for_prediction(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Devolver la predicción
        return jsonify({"prediction": class_labels[predicted_class]})

    return jsonify({"error": "Invalid request method"}), 400
