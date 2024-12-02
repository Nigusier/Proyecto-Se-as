import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image
import os

# Cargar el modelo
if os.path.exists('./model.p'):
    try:
        with open('./model.p', 'rb') as model_file:
            model_dict = pickle.load(model_file)
    except Exception as e:
        st.error(f"Error al cargar el archivo de modelo: {e}")
else:
    st.error("El archivo 'model.p' no se encontró en el directorio especificado.")

model = model_dict['model']

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'E', 2: 'i', 3: 'U', 4: 'O', 5: 'Fixis'}

st.title("Enseñanza del Lenguaje de Señas")
st.write("El uso de la inteligencia artificial para enseñar el lenguaje de señas.")

imagen_1 = Image.open("./señas.jpg")
imagen_2 = Image.open("./fixis.png")
imagen_3 = Image.open("./matriz.jpg")

st.image(imagen_3, caption='Matriz de confusión', use_container_width=True)

# Clase de procesamiento de video
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        return img

# Streamlit-webrtc para capturar video en tiempo real
webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

# Imágenes de referencia
st.write("A continuación verás dos imágenes que te servirán de guía para poner en práctica el lenguaje de señas.")
col1, col2 = st.columns(2)

with col1:
    st.image(imagen_2, caption='Fixis', use_container_width=True)

with col2:
    st.image(imagen_1, caption='Vocales A,E,I,O,U', use_container_width=True)
