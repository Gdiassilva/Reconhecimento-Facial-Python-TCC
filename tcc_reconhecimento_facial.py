import cv2
import face_recognition
import os
import pickle
import numpy as np
import dlib
from scipy.spatial import distance as dist

# Banco de dados de operadores
database_path = "database/operadores.pkl"
os.makedirs("database", exist_ok=True)

# Função para salvar no banco de dados
def salvar_banco_dados(nome, cargo, encodings, foto_paths):
    try:
        with open(database_path, "rb") as file:
            data = pickle.load(file)
    except FileNotFoundError:
        data = {}
    if nome in data:
        print("Erro: Operador já registrado.")
        return False
    data[nome] = {"cargo": cargo, "encodings": encodings, "foto_paths": foto_paths}
    with open(database_path, "wb") as file:
        pickle.dump(data, file)
    return True

# Função para salvar múltiplas imagens originais do operador
def salvar_fotos_operador(nome, frames):
    foto_paths = []
    foto_dir = f"database/{nome}"
    os.makedirs(foto_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = f"{foto_dir}/{nome}_foto_{i + 1}.jpg"
        cv2.imwrite(path, frame)
        foto_paths.append(path)
    return foto_paths

# Função para calcular o Eye Aspect Ratio (EAR)
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Função para determinar a direção do olhar
def calculate_gaze_direction(eye, gray_frame):
    eye_region = gray_frame[min(eye[:, 1]):max(eye[:, 1]), min(eye[:, 0]):max(eye[:, 0])]
    _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            width = threshold_eye.shape[1]
            if cx < width // 3:
                return "Esquerda"
            elif cx > 2 * (width // 3):
                return "Direita"
            else:
                return "Centro"
    return "Desconhecida"

# Inicializa o detector de rosto e o preditor de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Índices dos landmarks para os olhos e nariz
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
NOSE_POINT = 30  # Ponto do nariz

# Pergunta inicial ao usuário
while True:
    resposta = input("Você já está registrado? (Sim/Não): ").strip().lower()
    if resposta == "sim":
        print("Iniciando Login...")
        break
    elif resposta == "não":
        print("Iniciando Registro...")
        break
    else:
        print("Resposta inválida. Por favor, responda com 'Sim' ou 'Não'.")

if resposta == "não":
    # ==================== REGISTRO ====================
    # Captura do nome e cargo
    while True:
        nome = input("Digite seu nome: ").strip()
        if not nome:
            print("Erro: O nome não pode estar em branco. Por favor, digite novamente.\n")
        else:
            break

    while True:
        cargo = input("Digite seu cargo (pressione Enter para definir como 'operador'): ").strip() or "operador"
        if not cargo:
            print("Erro: O cargo não pode estar em branco. Por favor, digite novamente.\n")
        else:
            break

    # Inicia o processo de registro
    print(f"Iniciando processo de registro para {nome}...")

    video_capture = cv2.VideoCapture(0)

    blink_counter = 0
    total_blinks = 0
    gaze_status = "Desconhecida"
    face_size_ok = False
    encodings = []
    captured_frames = []

    # Frequência de atualização dos landmarks (a cada N frames)
    frequencia_recalculo_landmarks = 2  # Atualiza a cada 2 frames
    frame_counter = 0  # Contador de frames

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta as faces no frame atual
        faces = detector(gray)
        distance_message = "Rosto nao detectado"

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Avalia distância
            if 80 <= w <= 200:
                face_size_ok = True
                distance_message = "Distancia correta"
            elif w < 80:
                distance_message = "Aproxime o rosto da câmera"
            else:
                distance_message = "Afaste o rosto da câmera"

            # Atualiza os landmarks faciais a cada 'frequencia_recalculo_landmarks' frames
            if frame_counter % frequencia_recalculo_landmarks == 0:
                landmarks = predictor(gray, face)
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_POINTS]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_POINTS]
                nose = landmarks.part(NOSE_POINT)

                # Desenha os olhos e o ponto do nariz
                cv2.polylines(frame, [np.array(left_eye)], isClosed=True, color=(0, 255, 255), thickness=1)
                cv2.polylines(frame, [np.array(right_eye)], isClosed=True, color=(0, 255, 255), thickness=1)
                cv2.circle(frame, (nose.x, nose.y), 5, (0, 0, 255), -1)

            # Verifica piscadas
            ear_left = calculate_EAR(np.array(left_eye))
            ear_right = calculate_EAR(np.array(right_eye))
            average_ear = (ear_left + ear_right) / 2.0

            if average_ear < 0.2:
                blink_counter += 1
            else:
                if blink_counter >= 0.5:  # 1 frame
                    total_blinks += 1
                blink_counter = 0

            # Direção do olhar
            gaze_status = calculate_gaze_direction(np.array(left_eye), gray)

            # Gera encodings faciais usando face_recognition
            face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
            if face_encodings and face_size_ok and total_blinks >= 2 and gaze_status == "Centro":
                encodings.append(face_encodings[0])
                captured_frames.append(frame)

            if len(captured_frames) >= 5:
                print("Captura concluída.")
                video_capture.release()
                cv2.destroyAllWindows()

                # Salva as imagens e encodings no banco de dados
                foto_paths = salvar_fotos_operador(nome, captured_frames)
                if salvar_banco_dados(nome, cargo, encodings, foto_paths):
                    print(f"Registro concluído para {nome}.")
                else:
                    print("Erro ao registrar. Operador já existente.")
                break

        status_frame = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(status_frame, f"Piscadas: {total_blinks}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        cv2.putText(status_frame, f"Olhar: {gaze_status}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(status_frame, distance_message, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registro Biometrico", frame)
        cv2.imshow("Status", status_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Registro cancelado.")
            break

        # Atualiza o contador de frames
        frame_counter += 1

    video_capture.release()
    cv2.destroyAllWindows()

# ==================== LOGIN ===========================
# Início do Login
print("Iniciando Login...")

# Carrega o banco de dados
try:
    with open(database_path, "rb") as file:
        data = pickle.load(file)
except FileNotFoundError:
    print("Erro: Banco de dados não encontrado. Por favor, realize o registro primeiro.")
    exit()

# Captura do nome do operador
while True:
    nome_login = input("Digite seu nome: ").strip()
    if not nome_login:
        print("Erro: O nome não pode estar em branco. Por favor, digite novamente.\n")
    elif nome_login not in data:
        print(f"Erro: O operador '{nome_login}' não está registrado. Por favor, realize o registro primeiro.\n")
    else:
        break

print(f"Iniciando autenticação para {nome_login}...")

# Recupera os dados do operador
operador_data = data[nome_login]
encodings_registrados = operador_data["encodings"]

# Inicia a captura de vídeo
video_capture = cv2.VideoCapture(0)
blink_counter = 0
total_blinks = 0
gaze_status = "Desconhecida"
face_size_ok = False
autenticado = False

# Frequência de atualização dos landmarks (a cada N frames)
frequencia_recalculo_landmarks = 2  # Atualiza a cada 2 frames
frame_counter = 0  # Contador de frames

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta as faces no frame atual
    faces = detector(gray)
    distance_message = "Rosto nao detectado"

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Avalia distância
        if 80 <= w <= 200:
            face_size_ok = True
            distance_message = "Distancia correta"
        elif w < 80:
            distance_message = "Aproxime o rosto da câmera"
        else:
            distance_message = "Afaste o rosto da câmera"

        # Atualiza os landmarks faciais a cada 'frequencia_recalculo_landmarks' frames
        if frame_counter % frequencia_recalculo_landmarks == 0:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_POINTS]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_POINTS]
            nose = landmarks.part(NOSE_POINT)

            # Desenha os olhos e o ponto do nariz
            cv2.polylines(frame, [np.array(left_eye)], isClosed=True, color=(0, 255, 255), thickness=1)
            cv2.polylines(frame, [np.array(right_eye)], isClosed=True, color=(0, 255, 255), thickness=1)
            cv2.circle(frame, (nose.x, nose.y), 5, (0, 0, 255), -1)

        # Verifica piscadas
        ear_left = calculate_EAR(np.array(left_eye))
        ear_right = calculate_EAR(np.array(right_eye))
        average_ear = (ear_left + ear_right) / 2.0

        if average_ear < 0.2:
            blink_counter += 1
        else:
            if blink_counter >= 0.5:  # 1 frame
                total_blinks += 1
            blink_counter = 0

        # Direção do olhar
        gaze_status = calculate_gaze_direction(np.array(left_eye), gray)

        # Gera encoding facial usando face_recognition
        face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
        if face_encodings and face_size_ok and total_blinks >= 2 and gaze_status == "Centro":
            for encoding in encodings_registrados:
                match = face_recognition.compare_faces([encoding], face_encodings[0], tolerance=0.9)
                if match[0]:
                    autenticado = True
                    break

    # Atualiza o contador de frames
    frame_counter += 1

    # Exibe mensagens na tela
    status_frame = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(status_frame, f"Piscadas: {total_blinks}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(status_frame, f"Olhar: {gaze_status}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(status_frame, distance_message, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if autenticado:
        cv2.putText(status_frame, "RECONHECIDO", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"Bem-vindo, {nome_login}! Cargo: {operador_data['cargo']}")

        # Janela de Operação após autenticação
        operation_window = np.zeros((600, 1000, 3), dtype=np.uint8)  # Cria uma janela grande

        cv2.putText(operation_window, "Janela de Operacao", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        while True:
            cv2.imshow("Ambiente de Operacao", operation_window)

            if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
                print("Saindo do Ambiente de Operação...")
                break

        cv2.destroyAllWindows()

        exit()
    else:
        cv2.putText(status_frame, "NAO RECONHECIDO", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Login Biometrico", frame)
    cv2.imshow("Status", status_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para cancelar
        print("Login cancelado.")
        break

video_capture.release()
cv2.destroyAllWindows()
