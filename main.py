"q""HeadTracker — entry point."""
from argparse import ArgumentParser
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import pyautogui

from headtracker.tracking.face_detection import FaceDetector
from headtracker.tracking.mark_detection import MarkDetector
from headtracker.tracking.pose_estimation import PoseEstimator
from headtracker.tracking.tracker import HeadTracker
from headtracker.control.filters import OneEuroFilter
from headtracker.control.cursor import CursorController
from headtracker.voice.engine import VoiceCommandEngine
from headtracker.voice.commands import COMANDOS
from headtracker import calibration

ASSETS_DIR = Path(__file__).parent / "assets"
CALIB_FILE = "calibration.json"
MAX_ANGLE_DELTA = 2.5
WINDOW_TITLE = "Head Tracker - HCI Controller"


def _draw_hud(frame, tracker, raw_pitch, raw_yaw, delta_pitch, delta_yaw, fps):
    cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    if tracker.is_active:
        mode_color = (0, 255, 0) if not tracker.just_reinitialized else (0, 140, 255)
        mode_txt = f"3D-TRACK {tracker.frames_tracked}" if not tracker.just_reinitialized else "3D-REINIT"
        cv2.putText(frame, f"Pitch: {raw_pitch:+.1f} (d {delta_pitch:+.1f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yaw:   {raw_yaw:+.1f} (d {delta_yaw:+.1f})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, mode_txt,
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        if delta_yaw == 0.0 and delta_pitch == 0.0:
            cv2.putText(frame, "DEADZONE", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    else:
        cv2.putText(frame, "Procurando rosto...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def run(args):
    # --- Fila de ações vindas dos comandos de voz ---
    action_queue: queue.Queue[str] = queue.Queue()

    def cmd_zero():
        action_queue.put("zero")

    todos_comandos = {**COMANDOS, "zero": cmd_zero}

    # --- Voice engine (background thread) ---
    try:
        engine = VoiceCommandEngine(
            model_path=str(ASSETS_DIR / "vosk-model-small-pt-0.3"),
            comandos=todos_comandos,
        )
        threading.Thread(target=engine.iniciar, daemon=True).start()
    except Exception as e:
        print(f"[!] Voz não iniciada: {e}")

    # --- Video capture ---
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Components ---
    face_detector = FaceDetector(str(ASSETS_DIR / "face_detector.onnx"))
    mark_detector = MarkDetector(str(ASSETS_DIR / "face_landmarks.onnx"))
    pose_estimator = PoseEstimator(frame_w, frame_h, str(ASSETS_DIR / "model.txt"))
    tracker = HeadTracker(face_detector, mark_detector, pose_estimator)

    print(f"Face detector providers: {face_detector.session.get_providers()}")
    print(f"Mark detector providers: {mark_detector.model.get_providers()}")

    screen_w, screen_h = pyautogui.size()
    cursor = CursorController(screen_w, screen_h)

    # --- Load calibration ---
    saved = calibration.load(CALIB_FILE)
    if saved:
        cursor.calibrate(saved.get("neutral_pitch", 0.0), saved.get("neutral_yaw", 0.0))
        cursor.max_yaw_deg = saved.get("max_yaw_deg", cursor.max_yaw_deg)
        cursor.max_pitch_deg = saved.get("max_pitch_deg", cursor.max_pitch_deg)
        print(f"Calibração carregada: yaw±{cursor.max_yaw_deg:.1f}° pitch±{cursor.max_pitch_deg:.1f}°")

    # --- Angle filters and state ---
    filt_pitch = OneEuroFilter(min_cutoff=0.08, beta=0.15)
    filt_yaw = OneEuroFilter(min_cutoff=0.08, beta=0.15)
    prev_raw_pitch = None
    prev_raw_yaw = None
    have_sample = False

    tm = cv2.TickMeter()

    print("=== CONTROLE INICIADO ===")
    if args.show:
        print("'c' → calibrar centro  |  'r' → reiniciar  |  'q'/ESC → sair")
    else:
        print("Janela desativada (use --show para exibir). Voz: 'zero' recalibra | 'encerrar programa' sai | Ctrl+C sai")

    while True:
        frame_got, frame = cap.read()
        if not frame_got:
            break
        if video_src == 0:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tm.start()

        raw_pitch = raw_yaw = 0.0
        delta_pitch = delta_yaw = 0.0

        if tracker.update(frame, gray):
            if tracker.just_reinitialized:
                cursor.reset_warmup()
                prev_raw_pitch = None
                prev_raw_yaw = None

            rmat, _ = cv2.Rodrigues(tracker.r_vec)
            angles, *_ = cv2.RQDecomp3x3(rmat)

            t_now = time.perf_counter()
            raw_pitch = float(filt_pitch(angles[0], t_now))
            raw_yaw = float(filt_yaw(angles[1], t_now))

            if prev_raw_pitch is not None:
                raw_pitch = np.clip(raw_pitch,
                                    prev_raw_pitch - MAX_ANGLE_DELTA,
                                    prev_raw_pitch + MAX_ANGLE_DELTA)
                raw_yaw = np.clip(raw_yaw,
                                  prev_raw_yaw - MAX_ANGLE_DELTA,
                                  prev_raw_yaw + MAX_ANGLE_DELTA)
            prev_raw_pitch = raw_pitch
            prev_raw_yaw = raw_yaw
            have_sample = True

            if not cursor.is_calibrated:
                cursor.calibrate(raw_pitch, raw_yaw)
                print("Centro calibrado automaticamente!")

            delta_pitch, delta_yaw = cursor.update(raw_pitch, raw_yaw)

            pose_estimator.visualize(frame, (tracker.r_vec, tracker.t_vec), color=(0, 255, 0))
            if tracker.track_pts_2d is not None:
                for pt in tracker.track_pts_2d.reshape(-1, 2).astype(int):
                    if 0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h:
                        cv2.circle(frame, tuple(pt), 2, (0, 220, 255), -1)

        tm.stop()

        # --- Ações vindas da fila de voz ---
        while not action_queue.empty():
            action = action_queue.get_nowait()
            if action == "zero":
                tracker.reset()
                filt_pitch.reset()
                filt_yaw.reset()
                prev_raw_pitch = None
                prev_raw_yaw = None
                cursor.is_calibrated = False
                print("[voz] Zero: rastreamento e centro reiniciados!")

        # --- Janela (apenas com --show) ---
        if args.show:
            _draw_hud(frame, tracker, raw_pitch, raw_yaw, delta_pitch, delta_yaw, tm.getFPS())
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break

            if key == ord('c') and have_sample:
                p = float(filt_pitch.x_prev) if filt_pitch.x_prev is not None else raw_pitch
                y = float(filt_yaw.x_prev) if filt_yaw.x_prev is not None else raw_yaw
                cursor.calibrate(p, y)
                print("Centro recalibrado!")

            if key == ord('r'):
                tracker.reset()
                filt_pitch.reset()
                filt_yaw.reset()
                prev_raw_pitch = None
                prev_raw_yaw = None
                print("Rastreamento reinicializado!")
        else:
            time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--show", action="store_true",
                        help="Exibe a janela com a imagem da webcam e debug overlay")
    run(parser.parse_args())
