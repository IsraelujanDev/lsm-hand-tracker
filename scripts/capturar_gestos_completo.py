"""
Script para capturar gestos estáticos y dinámicos utilizando MediaPipe, OpenCV y pandas.
Este script está diseñado para un proyecto de reconocimiento de la Lengua de Señas Mexicana (LSM).

Características principales:
- Captura de gestos estáticos: imagen + 21 puntos (landmarks)
- Captura de gestos dinámicos: video + 21 puntos por cada frame
- Guarda todos los datos por separado en:
    • CSV y JSON para gestos estáticos
    • CSV y JSON para gestos dinámicos (incluyendo los puntos por frame)
    • Almacena las imágenes y videos capturados

Todas las líneas están comentadas detalladamente para comprender completamente su funcionamiento.
"""

# ========== Importación de bibliotecas necesarias ==========

import cv2                    # OpenCV: para captura de video, imágenes, y procesamiento de cámara
import mediapipe as mp        # MediaPipe: para detectar los 21 puntos de la mano (x, y, z)
import pandas as pd           # Pandas: para crear, manipular y guardar datasets en formato CSV
import os                     # OS: para crear carpetas y manipular rutas
import uuid                   # UUID: para generar nombres únicos en archivos (evita sobreescritura)
from datetime import datetime # Para registrar la fecha y hora de captura
import json                   # JSON: para guardar los datos estructurados en formato .json

# ========== Inicialización del modelo MediaPipe Hands ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,         # False: procesa en tiempo real (ideal para cámara en vivo)
    max_num_hands=1,                 # Detecta una sola mano por frame (suficiente para LSM)
    min_detection_confidence=0.5     # Umbral mínimo de confianza para aceptar una detección
)

# Herramienta de dibujo de MediaPipe para visualizar landmarks
mp_drawing = mp.solutions.drawing_utils

# ========== Configuración de carpetas y archivos de salida ==========
carpetas = ["datos", "datos/imagenes", "datos/videos", "datos/json", "datos/landmarks_csv"]
for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)  # Crea la carpeta si no existe

# Rutas de los archivos CSV y JSON por separado
csv_estaticos_path = "datos/gestos_estaticos.csv"
json_estaticos_path = "datos/json/gestos_estaticos.json"
csv_dinamicos_path = "datos/gestos_dinamicos.csv"
json_dinamicos_path = "datos/json/gestos_dinamicos.json"

# ========== Cargar datos existentes si ya hay archivos, si no iniciar vacíos ==========
df_estaticos = pd.read_csv(csv_estaticos_path) if os.path.exists(csv_estaticos_path) else pd.DataFrame()
df_dinamicos = pd.read_csv(csv_dinamicos_path) if os.path.exists(csv_dinamicos_path) else pd.DataFrame()
json_estaticos = json.load(open(json_estaticos_path)) if os.path.exists(json_estaticos_path) else []
json_dinamicos = json.load(open(json_dinamicos_path)) if os.path.exists(json_dinamicos_path) else []

# ========== Solicitar datos generales para esta sesión ==========
persona_id = input("ID de la persona (ej. P01): ")          # ID de quien está haciendo los gestos
mano = input("¿Qué mano se usó? (derecha/izquierda): ")    # Mano utilizada

# ========== Iniciar cámara (0 por defecto = cámara web principal) ==========
cap = cv2.VideoCapture(0)

# Instrucciones al usuario\print("Presiona 'c' para capturar gesto estático")
print("Presiona 'v' para capturar gesto dinámico (video)")
print("Presiona 'q' para salir")

# ========== Bucle principal del programa ==========
while True:
    ret, frame = cap.read()                      # Captura un frame de la cámara
    if not ret:
        break                                    # Si no se pudo capturar, salir del bucle

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Convertimos de BGR a RGB (formato requerido por MediaPipe)
    result = hands.process(frame_rgb)                     # Procesamos el frame para detectar landmarks de la mano

    # Si se detecta una mano, la dibujamos (visualización opcional)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostramos la ventana con la cámara activa
    cv2.imshow("Captura de Gestos", frame)
    key = cv2.waitKey(1)  # Esperamos 1 milisegundo por una tecla

    # ========== Gesto Estático ==========
    if key == ord('c') and result.multi_hand_landmarks:
        etiqueta = input("Letra del gesto estático: ")   # Por ejemplo, A, B, C, etc.

        # Extraemos los puntos x, y, z de los 21 landmarks
        puntos = [coord for lm in result.multi_hand_landmarks[0].landmark for coord in (lm.x, lm.y, lm.z)]

        # Guardamos imagen en carpeta /imagenes con nombre único
        nombre_img = f"{etiqueta}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(os.path.join("datos/imagenes", nombre_img), frame)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Guardamos en CSV
        fila = puntos + [etiqueta, nombre_img, persona_id, mano, timestamp]
        if df_estaticos.empty:
            columnas = [f"{eje}{i}" for i in range(21) for eje in ("x", "y", "z")]
            columnas += ["etiqueta", "imagen", "persona_id", "mano", "timestamp"]
            df_estaticos = pd.DataFrame(columns=columnas)
        df_estaticos.loc[len(df_estaticos)] = fila

        # Guardamos en JSON estructurado
        entrada_json = {
            "tipo": "estatico",
            "etiqueta": etiqueta,
            "imagen": nombre_img,
            "persona_id": persona_id,
            "mano": mano,
            "timestamp": timestamp,
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in result.multi_hand_landmarks[0].landmark
            ]
        }
        json_estaticos.append(entrada_json)
        print(f"Gesto estático '{etiqueta}' guardado correctamente.")

    # ========== Gesto Dinámico ==========
    elif key == ord('v'):
        etiqueta = input("Letra del gesto dinámico: ")
        duracion = int(input("Duración del video (segundos): "))

        # Definimos nombre único para el video
        nombre_video = f"{etiqueta}_dyn_{uuid.uuid4().hex[:8]}.mp4"
        ruta_video = os.path.join("datos/videos", nombre_video)

        # Definimos parámetros del video
        fps = 20                                     # Cuadros por segundo
        w, h = int(cap.get(3)), int(cap.get(4))      # Ancho y alto del frame actual
        out = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Preparamos para guardar landmarks
        landmark_seq = []
        frames_max = fps * duracion
        frame_count = 0

        print(f"Grabando gesto dinámico '{etiqueta}'...")

        while frame_count < frames_max:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            fila = []
            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks[0].landmark:
                    fila.extend([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            else:
                fila = [None] * (21 * 3)  # Si no hay mano, se guardan valores nulos

            fila.append(frame_count)  # Agregamos número de frame
            landmark_seq.append(fila)

            out.write(frame)          # Escribimos frame al archivo de video
            frame_count += 1
            cv2.imshow("Grabando Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        out.release()  # Finaliza la escritura del video
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Guardamos los landmarks por frame en CSV
        columnas = [f"{eje}{i}" for i in range(21) for eje in ("x", "y", "z")] + ["frame"]
        df_landmarks = pd.DataFrame(landmark_seq, columns=columnas)
        nombre_csv_landmarks = f"{etiqueta}_dyn_{uuid.uuid4().hex[:8]}.csv"
        ruta_csv_landmarks = os.path.join("datos/landmarks_csv", nombre_csv_landmarks)
        df_landmarks.to_csv(ruta_csv_landmarks, index=False)

        # Guardamos en CSV resumen
        fila_resumen = [etiqueta, nombre_video, nombre_csv_landmarks, fps, frame_count, persona_id, mano, timestamp]
        if df_dinamicos.empty:
            df_dinamicos = pd.DataFrame(columns=["etiqueta", "video", "landmarks_csv", "fps", "frames_totales", "persona_id", "mano", "timestamp"])
        df_dinamicos.loc[len(df_dinamicos)] = fila_resumen

        # Guardamos en JSON enriquecido con landmarks por frame
        entrada_json_dinamico = {
            "tipo": "dinamico",
            "etiqueta": etiqueta,
            "video": nombre_video,
            "landmarks_csv": nombre_csv_landmarks,
            "fps": fps,
            "frames_totales": frame_count,
            "persona_id": persona_id,
            "mano": mano,
            "timestamp": timestamp,
            "landmarks": df_landmarks.to_dict(orient="records")  # Cada frame con sus puntos
        }
        json_dinamicos.append(entrada_json_dinamico)
        print(f"Gesto dinámico '{etiqueta}' guardado correctamente.")

    # Salir del programa
    elif key == ord('q'):
        break

# ========== Guardado final de todos los datos ==========
df_estaticos.to_csv(csv_estaticos_path, index=False)
df_dinamicos.to_csv(csv_dinamicos_path, index=False)
with open(json_estaticos_path, "w") as f:
    json.dump(json_estaticos, f, indent=4)
with open(json_dinamicos_path, "w") as f:
    json.dump(json_dinamicos, f, indent=4)

# Liberamos la cámara y cerramos la ventana
cap.release()
cv2.destroyAllWindows()
print("✔ Todos los datos han sido guardados correctamente en sus archivos separados.")
