import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

# Cargar el modelo
model = tf.keras.models.load_model('cnn_custom.h5')

# Clases de ejemplo
class_labels = {
    0: 'camiseta_amarilla',
    1: 'camiseta_azul',
    2: 'camiseta_blanca',
    3: 'camiseta_roja',
    4: 'vestido_rojo',
    5: 'vestido_azul',
    6: 'vestido_amarillo',
    7: 'vestido_blanco'
}

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
