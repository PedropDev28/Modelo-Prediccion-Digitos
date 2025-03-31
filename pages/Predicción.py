import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle

# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Create a canvas to draw the digit
canvas = st_canvas(
    fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
    stroke_width=20,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)


def preprocess_image(image):
    # Convertir a escala de grises
    image = image.convert("L")

    # Cambiar tamaño a 8x8
    image = image.resize((8, 8))

    image_array = np.array(image)

    image_array = 16 * (image_array / 255)  # Escalar a [0, 16]

    image_array = image_array.flatten().reshape(1, -1)

    image_array = scaler.transform(image_array)

    return image_array


def preprocesar_canvas_para_svm(image_data):
    if image_data is None:
        return None

    imagen_scaled = preprocess_image(Image.fromarray(image_data.astype("uint8")))
    return imagen_scaled


# Predice mediante la imagen
def predict(image):
    image_array = preprocess_image(image)

    prediccion = clf.predict(image_array)

    return prediccion[0]


# Predice con la imagen
if st.button("Predict"):
    if canvas.image_data is not None:
        img_processed = preprocesar_canvas_para_svm(canvas.image_data)
        prediction = clf.predict(img_processed)
        st.subheader("Predicción")
        st.write(f"El modelo predice que el número es: **{prediction}**")

archivo_subido = st.file_uploader(
    "Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"]
)

if archivo_subido is not None:
    # Mostrar imagen con PIL
    image = Image.open(archivo_subido)
    st.image(image, caption="Imagen subida", width=150)
    st.write("")

    # Make a prediction
    prediction = predict(image)
    # prediccion = predecir_digito(archivo_subido)
    st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")

st.write(
    "Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos."
)
