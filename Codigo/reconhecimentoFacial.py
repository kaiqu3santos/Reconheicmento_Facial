import cv2
import dlib
import numpy as np
import face_recognition
import pickle
import os
import threading

# Carregar o modelo de pontos de referência faciais do dlib
shape_predictor = dlib.shape_predictor(r".\modelos\shape_predictor_68_face_landmarks.dat")

# Carregar o modelo de reconhecimento facial do dlib
face_recognition_model = dlib.face_recognition_model_v1(r".\modelos\dlib_face_recognition_resnet_model_v1.dat")

# Função para carregar o modelo treinado
def load_trained_model(model_file):
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['encodings'], model_data['ids']

# Carregar o modelo treinado com as codificações conhecidas e IDs correspondentes
trained_model_file = r'.\modelos\trained_model.pkl'
known_face_encodings, known_face_names = load_trained_model(trained_model_file)
face_names = ['Desconhecido', 'Kaique', 'Velma']

# Variável compartilhada para armazenar o último frame capturado
last_frame = None
lock = threading.Lock()

# Função para capturar frames da câmera
def capture_frames():
    global last_frame
    cap = cv2.VideoCapture(0)  # Abrir a câmera padrão (webcam)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            last_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()

# Função para reconhecer rostos
def recognize_faces():
    global last_frame
    while True:
        with lock:
            frame = last_frame
        if frame is None:
            continue
        
        # Converter o frame para RGB (dlib usa imagens RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar rostos na imagem
        face_locations = face_recognition.face_locations(rgb_frame)

        # Para cada rosto detectado, calcular o encoding
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Obter os pontos de referência faciais (shape)
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            shape = shape_predictor(rgb_frame, dlib_rect)

            # Calcular o encoding do rosto atual
            face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)

            # Converter o encoding do rosto atual para numpy array
            face_encoding = np.array(face_encoding)
            print(face_encoding)

            # Comparar o encoding do rosto atual com os encodings conhecidos
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            # Definir uma tolerância de reconhecimento (pode ajustar conforme necessário)
            tolerance = 0.4
            if min_distance <= tolerance:
                name = face_names[int(known_face_names[min_distance_index])]
            else:
                name = "Desconhecido"

            # Desenhar um retângulo em torno do rosto detectado
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Escrever o nome ou "Desconhecido" acima do retângulo
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Escrever a precisão (distância) abaixo do retângulo
            cv2.putText(frame, f'{min_distance:.2f}', (left + 6, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar o frame com os rostos detectados
        cv2.imshow('Video', frame)

        # Pressione 'q' para sair do loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Chamada das funções principais em threads separadas
capture_thread = threading.Thread(target=capture_frames)
recognize_thread = threading.Thread(target=recognize_faces)

capture_thread.start()
recognize_thread.start()

capture_thread.join()
recognize_thread.join()
