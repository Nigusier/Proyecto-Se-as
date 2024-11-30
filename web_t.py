import pickle

import numpy as np
import streamlit as st
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


st.title("Enseñanza del Lenguaje de Señas")
st.write("el uso de la inteligencia artifical para enseñar el lenguaje de señas.")

imagen_1 = Image.open("./señas.jpg")
imagen_2 = Image.open("./fixis.png")
imagen_3 = Image.open("./matriz.jpg")
#texto de descripcion 


with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)

    with left_column:
        st.header("Lenguaje de señas")
        st.write(
            """
            
            ¿deseas aprender el lenguaje de señas ?

Abstract

El proyecto "Enseñanza del Lenguaje de Señas" tiene como objetivo desarrollar una herramienta interactiva y educativa que facilite el aprendizaje del lenguaje de señas. La iniciativa utiliza modelos avanzados de machine learning para reconocer y traducir gestos manuales en tiempo real, proporcionando una plataforma accesible para aprendices de todas las edades.

El núcleo del proyecto se basa en un sistema de visión por computadora implementado con OpenCV y MediaPipe, que captura y procesa imágenes de video de los gestos de las manos. Los datos recolectados se utilizan para entrenar un modelo de aprendizaje supervisado, específicamente un clasificador de regresión logística, que identifica y predice los signos representados. El modelo ha sido entrenado con un conjunto de datos robusto que incluye las cinco vocales y una palabra del lenguaje de señas para asegurar alta precisión y confiabilidad.

            """
        )

    with right_column:
        st.header("Sign Language")
        st.write(
            """

            Do you want to learn sign language?

Abstract

The project "Sign Language Teaching" aims to develop an interactive and educational tool that facilitates the learning of sign language. The initiative uses advanced machine learning models to recognize and translate manual gestures in real time, providing an accessible platform for learners of all ages.

The core of the project is based on a computer vision system implemented with OpenCV and MediaPipe, which captures and processes video images of hand gestures. The collected data is used to train a supervised learning model, specifically a logistic regression classifier, that identifies and predicts the represented signs. The model has been trained with a robust dataset that includes the five vowels and one word from sign language to ensure high accuracy and reliability.            """
        )
st.title("Descripción")

st.write("""
    Este proyecto es un detector de lenguaje de señas desarrollado con Python, OpenCV y Mediapipe. El objetivo principal es detectar y reconocer signos de lenguaje de señas utilizando técnicas de visión por computadora.

    **Características Principales:**

    - **Lenguaje de Programación:** Python
    - **Bibliotecas Utilizadas:** OpenCV, Mediapipe

    **Funcionalidades:**

    - **Recolección de Imágenes:** Utiliza un script `collect_imgs.py` para recolectar imágenes de signos de lenguaje de señas.
    - **Creación de Conjunto de Datos:** Un script `create_dataset.py` para crear un conjunto de datos a partir de las imágenes recolectadas.
    - **Entrenamiento del Clasificador:** Un script `train_classifier.py` para entrenar un clasificador utilizando el conjunto de datos creado.
    - **Inferencia del Clasificador:** Un script `inference_classifier.py` para realizar predicciones en tiempo real.

 El modelo utilizado para este proyecto es un árbol de decisión. Este algoritmo compara rápidamente las entradas con una base de datos de aprendizaje creada a partir de un conjunto de datos guardados. La siguiente matriz de confusión muestra las validaciones realizadas por el algoritmo para encontrar similitudes entre la captura en tiempo real de una seña y el modelo entrenado


""")

st.image(imagen_3, caption='Matriz de confusión', use_container_width=True)


st.title("¡Vamos a la práctica!")

st.write("A continuación veras dos imágenes que te servirán de guía para poner en práctica el lenguaje de señas,  mira las imágenes e intenta imitar los signos con  tu mano derecha frente a la cámara, el programa detectará cada una de las señas y te dirá que letra o palabra estás haciendo ")


# Usar columnas para centrar las imágenes una al lado de la otra
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.image(imagen_2, caption='Fixis', use_container_width=True)

    with col2:
        st.image(imagen_1, caption='Vocales A,E,I,O,U', use_container_width=True)

#solicitud de video 

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)


