from argparse import ArgumentParser
import json
import os
import time
import threading
import queue
import subprocess
import sys
import cv2
import pyautogui
import numpy as np

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

import pyaudio
from vosk import Model, KaldiRecognizer

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
CORNER_MARGIN = 2

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()

CALIB_FILE = "calibration.json"

# =============================================================================
# COMANDOS DE VOZ — VOSK + PYAUTOGUI
# =============================================================================

MODEL_PATH  = r"assets\vosk-model-small-pt-0.3"
SAMPLE_RATE = 16000
CHUNK_SIZE  = 2000

def segurar_mouse():
    pyautogui.mouseDown()

def soltar_mouse():
    pyautogui.mouseUp()

def alternar_windows():
    pyautogui.hotkey("win", "tab")

def teclado_windows():
    pyautogui.hotkey("win", "2")

def abrir_navegador():
    import webbrowser
    webbrowser.open("https://www.google.com")

def fechar_janela():
    if sys.platform == "darwin":
        pyautogui.hotkey("command", "w")
    else:
        pyautogui.hotkey("alt", "F4")

def clique_mouse():
    pyautogui.leftClick()

def cliquedireito_mouse():
    pyautogui.rightClick()

def aumentar_zoom():
    pyautogui.hotkey("ctrl", "+")

def diminuir_zoom():
    pyautogui.hotkey("ctrl", "-")

def tirar_screenshot():
    screenshot = pyautogui.screenshot()
    caminho = os.path.join(os.path.expanduser("~"), "Desktop",
                           f"screenshot_{int(time.time())}.png")
    screenshot.save(caminho)
    print(f"[✓] Screenshot salvo em: {caminho}")

def volume_aumentar():
    for _ in range(5):
        pyautogui.press("volumeup")

def volume_diminuir():
    for _ in range(5):
        pyautogui.press("volumedown")

def mutar():
    pyautogui.press("volumemute")

def copiar():
    pyautogui.hotkey("ctrl", "c")

def colar():
    pyautogui.hotkey("ctrl", "v")

def desfazer():
    pyautogui.hotkey("ctrl", "z")

def rolar_cima():
    pyautogui.scroll(5)

def rolar_baixo():
    pyautogui.scroll(-5)

def minimizar():
    if sys.platform == "darwin":
        pyautogui.hotkey("command", "m")
    else:
        pyautogui.hotkey("win", "down")

def abrir_terminal():
    if sys.platform == "win32":
        subprocess.Popen(["cmd.exe"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "Terminal"])
    else:
        for terminal in ["gnome-terminal", "xterm", "konsole", "xfce4-terminal"]:
            try:
                subprocess.Popen([terminal])
                break
            except FileNotFoundError:
                continue

def encerrar_programa():
    print("\n[!] Comando 'encerrar' detectado. Encerrando programa...")
    os._exit(0)


COMANDOS = {
    "abre navegador":    abrir_navegador,
    "fechar janela":     fechar_janela,
    "screenshot":        tirar_screenshot,
    "som":               volume_aumentar,
    "abaixa":            volume_diminuir,
    "março":             mutar,
    "copiar":            copiar,
    "colar":             colar,
    "desfazer":          desfazer,
    "rolar cima":        rolar_cima,
    "rolar baixo":       rolar_baixo,
    "minimizar":         minimizar,
    "abrir terminal":    abrir_terminal,
    "encerrar programa": encerrar_programa,
    "show":              clique_mouse,
    "sou":               clique_mouse,
    "aumenta":           aumentar_zoom,
    "diminui":           diminuir_zoom,
    "troca":             alternar_windows,
    "quadro":            teclado_windows,
    "fato":              cliquedireito_mouse,
    "colo":              segurar_mouse,
    "joia":              soltar_mouse,
}


class VoiceCommandEngine:
    def __init__(self, model_path, comandos, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Pasta do modelo não encontrada: '{model_path}'\n"
                "Baixe em https://alphacephei.com/vosk/models e extraia como 'model/'"
            )
        print(f"[*] Carregando modelo Vosk de '{model_path}'...")
        self.model      = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)

        palavras = []
        for frase in comandos.keys():
            palavras.extend(frase.split())
        vocab = json.dumps(list(set(palavras)) + ["[unk]"])
        self.recognizer.SetGrammar(vocab)
        print(f"[*] Vocabulário restrito a {len(set(palavras))} palavras dos comandos.")

        self.comandos        = comandos
        self.sample_rate     = sample_rate
        self.chunk_size      = chunk_size
        self._audio_q        = queue.Queue()
        self._running        = False
        self._ultimo_comando   = ""
        self._ultimo_exec_time = 0.0
        self._cooldown         = 1.2

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self._audio_q.put(in_data)
        return (None, pyaudio.paContinue)

    def _processar_texto(self, texto):
        texto = texto.lower().strip()
        if not texto:
            self._ultimo_comando = ""
            return
        for frase, acao in self.comandos.items():
            if frase in texto:
                agora = time.time()
                mesmo_comando   = self._ultimo_comando == frase
                dentro_cooldown = (agora - self._ultimo_exec_time) < self._cooldown
                if mesmo_comando and dentro_cooldown:
                    return
                print(f"[flash] Executando: {frase} -> {acao.__name__}")
                self._ultimo_comando   = frase
                self._ultimo_exec_time = agora
                try:
                    acao()
                except Exception as e:
                    print(f"[X] Erro ao executar '{frase}': {e}")
                return
        self._ultimo_comando = ""

    def _recognition_loop(self):
        while self._running:
            try:
                data = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.recognizer.AcceptWaveform(data):
                resultado = json.loads(self.recognizer.Result())
                texto = resultado.get("text", "")
                if texto:
                    print(f"[mic] Final: \"{texto}\"")
                self._processar_texto(texto)
            else:
                parcial = json.loads(self.recognizer.PartialResult())
                texto_parcial = parcial.get("partial", "")
                if texto_parcial:
                    print(f"[...] Parcial: \"{texto_parcial}\"", end="\r")
                    self._processar_texto(texto_parcial)

    def iniciar(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._running = True
        thread = threading.Thread(target=self._recognition_loop, daemon=True)
        thread.start()
        stream.start_stream()
        print("[OK] Reconhecimento de voz ativo!")
        print("Comandos disponíveis:")
        for cmd in self.comandos:
            print(f"   * \"{cmd}\"")


# =============================================================================
# RASTREAMENTO 3D POR MODELO — MÉTODO DE NEWTON
#
# Pipeline por frame:
#   1. LK forward + backward: rastreia projeções do modelo 3D e descarta pontos
#      cuja trajetória não é reversível (outliers por oclusão, ruído, rosto virado)
#   2. solvePnP (Gauss-Newton) com os pontos validados → nova pose 6-DoF
#   3. Re-inicializa com CNN apenas quando rastreamento genuinamente falha
# =============================================================================


class OneEuroFilter:
    """Filtro adaptativo: suave quando parado (min_cutoff baixo), responsivo em movimento."""
    def __init__(self, min_cutoff=0.15, beta=0.15, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = None
        self.dx_prev    = 0.0
        self.t_prev     = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t; self.x_prev = x
            return x
        dt      = max(1e-3, t - self.t_prev)
        dx      = (x - self.x_prev) / dt
        a_d     = self._alpha(self.d_cutoff, dt)
        dx_hat  = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff  = self.min_cutoff + self.beta * abs(dx_hat)
        a       = self._alpha(cutoff, dt)
        x_hat   = a * x + (1.0 - a) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat

    def reset(self):
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None


def precision_curve(norm, knee=0.45, knee_out=0.3):
    """Piecewise linear: centro mais rápido (navegação), bordas mais lentas (precisão)."""
    s = np.sign(norm)
    a = abs(norm)
    out = (a / knee) * knee_out if a <= knee else knee_out + ((a - knee) / (1.0 - knee)) * (1.0 - knee_out)
    return s * out


def soft_deadzone(val, inner_dz, outer_dz):
    """Zero dentro de inner_dz; ramp quadrática até outer_dz. Elimina snap binário."""
    s = np.sign(val)
    a = abs(val)
    if a <= inner_dz:
        return 0.0
    elif a >= outer_dz:
        return s * (a - inner_dz)
    t = (a - inner_dz) / (outer_dz - inner_dz)
    return s * (a - inner_dz) * (t * t)


def load_calibration():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_calibration(data):
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibração salva em {CALIB_FILE}")


def run():
    try:
        engine = VoiceCommandEngine(model_path=MODEL_PATH, comandos=COMANDOS)
        voz_thread = threading.Thread(target=engine.iniciar, daemon=True)
        voz_thread.start()
    except Exception as e:
        print(f"[!] Voz não iniciada: {e}")

    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector  = FaceDetector("assets/face_detector.onnx")
    mark_detector  = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    print(f"Face detector providers: {face_detector.session.get_providers()}")
    print(f"Mark detector providers: {mark_detector.model.get_providers()}")

    tm = cv2.TickMeter()

    # --- PARÂMETROS DE CONTROLE ---
    max_yaw_deg    = 14.0
    max_pitch_deg  = 8.0
    # centro → movimento normal/rápido (navegação); bordas → mais lento (preciso)
    # knee_out > knee = desacelera nas bordas sem limitar range máximo (cantos sempre alcançáveis)
    curve_knee     = 0.65
    curve_knee_out = 0.65

    deadzone_inner_yaw   = 0.0
    deadzone_outer_yaw   = 0.0
    deadzone_inner_pitch = 0.0
    deadzone_outer_pitch = 0.0

    saved = load_calibration()
    if saved:
        neutral_pitch = saved.get("neutral_pitch", 0.0)
        neutral_yaw   = saved.get("neutral_yaw",   0.0)
        max_yaw_deg   = saved.get("max_yaw_deg",   max_yaw_deg)
        max_pitch_deg = saved.get("max_pitch_deg", max_pitch_deg)
        is_calibrated = True
        print(f"Calibração carregada: yaw±{max_yaw_deg:.1f}° pitch±{max_pitch_deg:.1f}°")
    else:
        neutral_pitch = 0.0
        neutral_yaw   = 0.0
        is_calibrated = False

    filt_pitch  = OneEuroFilter(min_cutoff=0.08, beta=0.15)
    filt_yaw    = OneEuroFilter(min_cutoff=0.08, beta=0.15)
    have_sample = False

    curr_mouse_x, curr_mouse_y = screen_w / 2, screen_h / 2

    MAX_ANGLE_DELTA = 2.5
    prev_raw_pitch  = None
    prev_raw_yaw    = None
    WARMUP_FRAMES   = 6
    warmup_count    = WARMUP_FRAMES

    # --- PARÂMETROS DE RASTREAMENTO 3D ---
    REPROJ_THRESH = 8.0
    FB_THRESH     = 2.0
    LK_PARAMS = dict(
        winSize =(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    MIN_TRACKED = 20

    prev_gray       = None
    track_pts_2d    = None
    track_pts_3d    = None
    r_vec_track     = None
    t_vec_track     = None
    frames_tracked  = 0
    tracking_active = False

    print("=== CONTROLE INICIADO ===")
    print("'c' → calibrar centro  |  'r' → reiniciar  |  'q'/ESC → sair")

    while True:
        frame_got, frame = cap.read()
        if not frame_got:
            break
        if video_src == 0:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tm.start()

        need_landmarks = not tracking_active

        # ------------------------------------------------------------------ #
        #  PASSO 1 — Lucas-Kanade com verificação forward-backward            #
        # ------------------------------------------------------------------ #
        if tracking_active and not need_landmarks and prev_gray is not None:
            new_pts, status_fw, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts_2d, None, **LK_PARAMS)
            back_pts, status_bw, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, new_pts, None, **LK_PARAMS)

            fb_err    = np.linalg.norm(
                (track_pts_2d - back_pts).reshape(-1, 2), axis=1)
            good_mask = (
                (status_fw.ravel() == 1) &
                (status_bw.ravel() == 1) &
                (fb_err < FB_THRESH)
            )

            if good_mask.sum() >= MIN_TRACKED:
                obs_2d = new_pts[good_mask].reshape(-1, 2).astype(np.float64)
                mod_3d = track_pts_3d[good_mask].astype(np.float64)

                ok, r_new, t_new = cv2.solvePnP(
                    mod_3d, obs_2d,
                    pose_estimator.camera_matrix,
                    pose_estimator.dist_coeefs,
                    rvec=r_vec_track.copy(), tvec=t_vec_track.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)

                if ok:
                    proj_check, _ = cv2.projectPoints(
                        mod_3d, r_new, t_new,
                        pose_estimator.camera_matrix,
                        pose_estimator.dist_coeefs)
                    reproj_err = float(np.mean(
                        np.linalg.norm(proj_check.reshape(-1, 2) - obs_2d, axis=1)))

                    if reproj_err < REPROJ_THRESH:
                        r_vec_track  = r_new
                        t_vec_track  = t_new
                        proj_all, _ = cv2.projectPoints(
                            pose_estimator.model_points_68,
                            r_vec_track, t_vec_track,
                            pose_estimator.camera_matrix,
                            pose_estimator.dist_coeefs)
                        track_pts_2d = proj_all.astype(np.float32)
                        track_pts_3d = pose_estimator.model_points_68.copy()
                        frames_tracked += 1
                    else:
                        need_landmarks = True
                else:
                    need_landmarks = True
            else:
                need_landmarks = True

        # ------------------------------------------------------------------ #
        #  PASSO 2 — Re-inicialização por landmarks (só quando falha real)    #
        # ------------------------------------------------------------------ #
        if need_landmarks:
            faces, _ = face_detector.detect(frame, 0.7)
            if len(faces) > 0:
                face  = refine(faces, frame_width, frame_height, 0.15)[0]
                rx1, ry1, rx2, ry2 = face[:4].astype(int)
                rx1 = max(0, rx1); ry1 = max(0, ry1)
                rx2 = min(frame_width, rx2); ry2 = min(frame_height, ry2)
                patch = frame[ry1:ry2, rx1:rx2]
                if patch.size > 0:
                    marks = mark_detector.detect([patch])[0].reshape([68, 2])
                    marks *= (rx2 - rx1)
                    marks[:, 0] += rx1
                    marks[:, 1] += ry1

                    ok, r_init, t_init = cv2.solvePnP(
                        pose_estimator.model_points_68.astype(np.float64),
                        marks.astype(np.float64),
                        pose_estimator.camera_matrix,
                        pose_estimator.dist_coeefs,
                        flags=cv2.SOLVEPNP_EPNP)

                    if ok:
                        r_vec_track     = r_init
                        t_vec_track     = t_init
                        proj_all, _ = cv2.projectPoints(
                            pose_estimator.model_points_68,
                            r_vec_track, t_vec_track,
                            pose_estimator.camera_matrix,
                            pose_estimator.dist_coeefs)
                        track_pts_2d    = proj_all.astype(np.float32)
                        track_pts_3d    = pose_estimator.model_points_68.copy()
                        frames_tracked  = 0
                        tracking_active = True
                        need_landmarks  = False
                        warmup_count    = WARMUP_FRAMES
                        prev_raw_pitch  = None
                        prev_raw_yaw    = None
            else:
                tracking_active = False

        prev_gray = gray

        # ------------------------------------------------------------------ #
        #  PASSO 3 — Ângulos → mouse                                          #
        # ------------------------------------------------------------------ #
        raw_pitch = raw_yaw = 0.0
        delta_pitch = delta_yaw = 0.0

        if tracking_active and r_vec_track is not None:
            rmat, _ = cv2.Rodrigues(r_vec_track)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            t_now     = time.perf_counter()
            raw_pitch = float(filt_pitch(angles[0], t_now))
            raw_yaw   = float(filt_yaw(angles[1], t_now))

            if prev_raw_pitch is not None:
                raw_pitch = np.clip(raw_pitch,
                                    prev_raw_pitch - MAX_ANGLE_DELTA,
                                    prev_raw_pitch + MAX_ANGLE_DELTA)
                raw_yaw   = np.clip(raw_yaw,
                                    prev_raw_yaw - MAX_ANGLE_DELTA,
                                    prev_raw_yaw + MAX_ANGLE_DELTA)
            prev_raw_pitch = raw_pitch
            prev_raw_yaw   = raw_yaw
            have_sample = True

            if not is_calibrated:
                neutral_pitch = raw_pitch
                neutral_yaw   = raw_yaw
                is_calibrated = True
                print("Centro calibrado automaticamente!")

            delta_pitch = raw_pitch - neutral_pitch
            delta_yaw   = raw_yaw   - neutral_yaw

            delta_yaw   = soft_deadzone(delta_yaw,   deadzone_inner_yaw,   deadzone_outer_yaw)
            delta_pitch = soft_deadzone(delta_pitch, deadzone_inner_pitch, deadzone_outer_pitch)

            norm_x = float(np.clip(delta_yaw   / max_yaw_deg,   -1.0, 1.0))
            norm_y = float(np.clip(delta_pitch / max_pitch_deg, -1.0, 1.0))

            mapped_x = precision_curve(norm_x, curve_knee, curve_knee_out)
            mapped_y = precision_curve(norm_y, curve_knee, curve_knee_out)

            target_x = float(np.clip((screen_w / 2) + (mapped_x * (screen_w / 2)), CORNER_MARGIN, screen_w - 1 - CORNER_MARGIN))
            target_y = float(np.clip((screen_h / 2) - (mapped_y * (screen_h / 2)), CORNER_MARGIN, screen_h - 1 - CORNER_MARGIN))

            dist  = np.hypot(target_x - curr_mouse_x, target_y - curr_mouse_y)
            alpha = float(np.clip(dist / 100.0, 0.20, 0.55))
            curr_mouse_x += (target_x - curr_mouse_x) * alpha
            curr_mouse_y += (target_y - curr_mouse_y) * alpha
            if dist < 2.0:
                curr_mouse_x, curr_mouse_y = target_x, target_y
            curr_mouse_x = float(np.clip(curr_mouse_x, CORNER_MARGIN, screen_w - 1 - CORNER_MARGIN))
            curr_mouse_y = float(np.clip(curr_mouse_y, CORNER_MARGIN, screen_h - 1 - CORNER_MARGIN))

            if warmup_count > 0:
                warmup_count -= 1
            else:
                pyautogui.moveTo(int(curr_mouse_x), int(curr_mouse_y))
            tm.stop()

            pose_estimator.visualize(frame, (r_vec_track, t_vec_track), color=(0, 255, 0))
            if track_pts_2d is not None:
                for pt in track_pts_2d.reshape(-1, 2).astype(int):
                    if 0 <= pt[0] < frame_width and 0 <= pt[1] < frame_height:
                        cv2.circle(frame, tuple(pt), 2, (0, 220, 255), -1)

            mode_color = (0, 255, 0) if not need_landmarks else (0, 140, 255)
            mode_txt   = f"3D-TRACK {frames_tracked}" if not need_landmarks else "3D-REINIT"
            cv2.putText(frame, f"Pitch: {raw_pitch:+.1f} (d {delta_pitch:+.1f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Yaw:   {raw_yaw:+.1f} (d {delta_yaw:+.1f})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, mode_txt,
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            if delta_yaw == 0.0 and delta_pitch == 0.0:
                cv2.putText(frame, "DEADZONE", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        else:
            tm.stop()
            cv2.putText(frame, "Procurando rosto...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Head Tracker - HCI Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        if key == ord('c') and have_sample:
            neutral_pitch = float(filt_pitch.x_prev) if filt_pitch.x_prev is not None else raw_pitch
            neutral_yaw   = float(filt_yaw.x_prev)   if filt_yaw.x_prev   is not None else raw_yaw
            is_calibrated = True
            print("Centro recalibrado!")

        if key == ord('r'):
            tracking_active = False
            track_pts_2d    = None
            track_pts_3d    = None
            r_vec_track     = None
            t_vec_track     = None
            prev_raw_pitch  = None
            prev_raw_yaw    = None
            warmup_count    = WARMUP_FRAMES
            frames_tracked  = 0
            filt_pitch.reset()
            filt_yaw.reset()
            print("Rastreamento reinicializado!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
