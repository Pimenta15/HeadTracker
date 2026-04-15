from argparse import ArgumentParser
import cv2
import pyautogui
import numpy as np

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
args = args = parser.parse_args()

def run():
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    tm = cv2.TickMeter()

    # --- PARÂMETROS DE ESTABILIZAÇÃO E HCI ---
    smooth_factor = 0.25      # Quão macio é o movimento (0.1 = muito atraso/suave, 0.9 = instantâneo/tremido)
    deadzone = 0.8            # Graus de movimento ignorados (evita tremedeira ao tentar parar o mouse)
    
    # Limites da cabeça (Girar X graus mapeia para as bordas da tela)
    max_yaw_deg = 18.0        # Esquerda/Direita
    max_pitch_deg = 12.0      # Cima/Baixo
    
    # Curva de aceleração (1.0 = linear. 1.5 = lento no meio, rápido nas bordas)
    acceleration_curve = 1.4  

    # Estado de Calibração
    neutral_pitch = 0.0
    neutral_yaw = 0.0
    is_calibrated = False

    # Posição atual filtrada do mouse
    curr_mouse_x, curr_mouse_y = screen_w / 2, screen_h / 2

    print("=== CONTROLE INICIADO ===")
    print("Pressione 'c' na janela do vídeo para CALIBRAR O CENTRO.")
    print("Pressione 'q' ou 'ESC' para sair.")

    while True:
        frame_got, frame = cap.read()
        if not frame_got: break

        if video_src == 0:
            frame = cv2.flip(frame, 1)

        faces, _ = face_detector.detect(frame, 0.7)

        if len(faces) > 0:
            tm.start()
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Resolve Pose (Tracker Otimizado 6DoF)
            pose = pose_estimator.solve(marks)
            r_vec, t_vec = pose

            # --- EXTRAÇÃO EULER ---
            rmat, _ = cv2.Rodrigues(r_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            raw_pitch = angles[0]
            raw_yaw = angles[1]

            # --- CALIBRAÇÃO (Define o marco zero) ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or not is_calibrated:
                neutral_pitch = raw_pitch
                neutral_yaw = raw_yaw
                is_calibrated = True
                print("Centro Calibrado!")

            if key == 27 or key == ord('q'):
                break

            # Calcula o delta em relação ao centro calibrado
            delta_pitch = raw_pitch - neutral_pitch
            delta_yaw = raw_yaw - neutral_yaw

            # --- ZONA MORTA ---
            if abs(delta_yaw) < deadzone: delta_yaw = 0.0
            if abs(delta_pitch) < deadzone: delta_pitch = 0.0

            # --- NORMALIZAÇÃO E ACELERAÇÃO NÃO-LINEAR ---
            # Transforma os graus em uma escala de -1.0 a 1.0
            norm_x = delta_yaw / max_yaw_deg
            norm_y = delta_pitch / max_pitch_deg

            # Trava os valores caso o usuário gire o pescoço além do limite
            norm_x = max(-1.0, min(1.0, norm_x))
            norm_y = max(-1.0, min(1.0, norm_y))

            # Aplica a curva geométrica (mantém o sinal)
            mapped_x = np.sign(norm_x) * (abs(norm_x) ** acceleration_curve)
            mapped_y = np.sign(norm_y) * (abs(norm_y) ** acceleration_curve)

            # --- MAPEAMENTO ABSOLUTO DE TELA ---
            target_mouse_x = (screen_w / 2) + (mapped_x * (screen_w / 2))
            # O Y é invertido (cabeça pra cima = mouse pra cima)
            target_mouse_y = (screen_h / 2) - (mapped_y * (screen_h / 2)) 

            # --- FILTRO PASSA-BAIXA (Inércia / Suavização) ---
            curr_mouse_x += (target_mouse_x - curr_mouse_x) * smooth_factor
            curr_mouse_y += (target_mouse_y - curr_mouse_y) * smooth_factor

            # Mover o ponteiro
            pyautogui.moveTo(int(curr_mouse_x), int(curr_mouse_y))

            tm.stop()

            # --- VISUAL FEEDBACK ---
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))
            cv2.putText(frame, f"Pitch: {raw_pitch:.1f} (Delta: {delta_pitch:.1f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Yaw:   {raw_yaw:.1f} (Delta: {delta_yaw:.1f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if abs(delta_yaw) == 0.0 and abs(delta_pitch) == 0.0:
                cv2.putText(frame, "DEADZONE (PARADO)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Head Tracker - HCI Controller", frame)

        # Prevenção extra caso o loop caia aqui sem ler as teclas
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == '__main__':
    run()