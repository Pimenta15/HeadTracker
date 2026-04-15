from argparse import ArgumentParser
import cv2
import pyautogui
import numpy as np
from collections import deque

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

# Configuração do PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()

def run():
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    tm = cv2.TickMeter()

    # --- PARÂMETROS DE ESTABILIZAÇÃO ---
    deadzone = 0.8
    max_yaw_deg = 18.0
    max_pitch_deg = 12.0
    acceleration_curve = 1.4

    history_pitch = deque(maxlen=5) 
    history_yaw = deque(maxlen=5)

    last_face_box = None
    frames_since_detection = 0

    neutral_pitch = 0.0
    neutral_yaw = 0.0
    is_calibrated = False

    curr_mouse_x, curr_mouse_y = screen_w / 2, screen_h / 2

    print("=== CONTROLE INICIADO ===")
    print("Pressione 'c' na janela do vídeo para CALIBRAR O CENTRO.")
    print("Pressione 'q' ou 'ESC' para sair.")

    while True:
        frame_got, frame = cap.read()
        if not frame_got: break

        if video_src == 0:
            frame = cv2.flip(frame, 1)

        tm.start()

        # O detector pesado só corre se o rosto for perdido
        if last_face_box is None or frames_since_detection > 30:
            faces, _ = face_detector.detect(frame, 0.7)
            if len(faces) > 0:
                face = refine(faces, frame_width, frame_height, 0.15)[0]
                last_face_box = face[:4].astype(int)
                frames_since_detection = 0
            else:
                last_face_box = None

        if last_face_box is not None:
            frames_since_detection += 1
            x1, y1, x2, y2 = last_face_box
            
            # Corta os limites para não ultrapassar o tamanho do ecrã
            px1 = max(0, x1)
            py1 = max(0, y1)
            px2 = min(frame_width, x2)
            py2 = min(frame_height, y2)
            
            patch = frame[py1:py2, px1:px2]

            marks_raw = mark_detector.detect([patch])
            
            if marks_raw is None or len(marks_raw) == 0:
                last_face_box = None
                continue

            marks = marks_raw[0].reshape([68, 2])
            
            # Converte os pontos normalizados (0-1) para as coordenadas totais da imagem
            marks[:, 0] *= (px2 - px1)
            marks[:, 1] *= (py2 - py1)
            marks[:, 0] += px1
            marks[:, 1] += py1

            # --- CORREÇÃO MATEMÁTICA: FIM DO EFEITO "BURACO NEGRO" ---
            # 1. Encontra a largura real do rosto em pixéis (esta medida é independente do tamanho da caixa)
            w_marks = np.max(marks[:, 0]) - np.min(marks[:, 0])
            
            # 2. Força a nova caixa a ser sempre perfeitamente quadrada e proporcional à largura real do rosto (1.5x)
            box_side = w_marks * 1.5 
            
            # 3. Encontra o centro geográfico do rosto atual
            cx = np.mean(marks[:, 0])
            cy = np.mean(marks[:, 1])
            
            # 4. Desloca o centro um pouco para cima (compensando a falta de pontos na testa)
            cy -= (box_side * 0.1)
            
            # 5. Cria a caixa blindada para o próximo frame
            half = int(box_side / 2.0)
            last_face_box = [int(cx - half), int(cy - half), int(cx + half), int(cy + half)]

            # Resolve a Pose
            pose = pose_estimator.solve(marks)
            r_vec, t_vec = pose

            # Extração dos ângulos Euler
            rmat, _ = cv2.Rodrigues(r_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            history_pitch.append(angles[0])
            history_yaw.append(angles[1])
            
            raw_pitch = np.mean(history_pitch)
            raw_yaw = np.mean(history_yaw)

            # Calibração
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or not is_calibrated:
                neutral_pitch = raw_pitch
                neutral_yaw = raw_yaw
                is_calibrated = True
                print("Centro Calibrado!")

            if key == 27 or key == ord('q'):
                break

            # HCI e Movimento do Rato
            delta_pitch = raw_pitch - neutral_pitch
            delta_yaw = raw_yaw - neutral_yaw

            if abs(delta_yaw) < deadzone: delta_yaw = 0.0
            if abs(delta_pitch) < deadzone: delta_pitch = 0.0

            norm_x = max(-1.0, min(1.0, delta_yaw / max_yaw_deg))
            norm_y = max(-1.0, min(1.0, delta_pitch / max_pitch_deg))

            mapped_x = np.sign(norm_x) * (abs(norm_x) ** acceleration_curve)
            mapped_y = np.sign(norm_y) * (abs(norm_y) ** acceleration_curve)

            target_mouse_x = (screen_w / 2) + (mapped_x * (screen_w / 2))
            target_mouse_y = (screen_h / 2) - (mapped_y * (screen_h / 2)) 

            dist = np.hypot(target_mouse_x - curr_mouse_x, target_mouse_y - curr_mouse_y)
            dynamic_smooth = np.clip(dist / 200.0, 0.1, 0.6) 

            curr_mouse_x += (target_mouse_x - curr_mouse_x) * dynamic_smooth
            curr_mouse_y += (target_mouse_y - curr_mouse_y) * dynamic_smooth

            pyautogui.moveTo(int(curr_mouse_x), int(curr_mouse_y))

            # Desenhos na Janela
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2) # Caixa Antiga
            cv2.rectangle(frame, (last_face_box[0], last_face_box[1]), (last_face_box[2], last_face_box[3]), (0, 255, 255), 1) # Nova Caixa Blindada

        tm.stop()
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Head Tracker - HCI Controller", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == '__main__':
    run()