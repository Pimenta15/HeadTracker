from argparse import ArgumentParser
import time
import threading
import json
import queue
import subprocess
import os
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

parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()


# =============================================================================
# COMANDOS DE VOZ — VOSK + PYAUTOGUI
# =============================================================================

MODEL_PATH  = r"C:\Users\bielr\Downloads\vosk-model-small-pt-0.3\vosk-model-small-pt-0.3"
SAMPLE_RATE = 16000
CHUNK_SIZE  = 2000  # blocos menores = menor latência (~125ms por bloco)

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
    "som":   volume_aumentar,
    "abaixa":   volume_diminuir,
    "março":              mutar,
    "copiar":            copiar,
    "colar":             colar,
    "desfazer":          desfazer,
    "rolar cima":        rolar_cima,
    "rolar baixo":       rolar_baixo,
    "minimizar":         minimizar,
    "abrir terminal":    abrir_terminal,
    "encerrar programa": encerrar_programa,
    "show": clique_mouse,
    "sou": clique_mouse,
    "aumenta": aumentar_zoom,
    "diminui": diminuir_zoom,
    "troca": alternar_windows,
    "quadro": teclado_windows,
    "fato": cliquedireito_mouse,
    "colo": segurar_mouse,
    "joia": soltar_mouse,
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
 
        # Vocabulário restrito: só as palavras dos comandos — reconhecimento 2-3x mais rápido
        palavras = []
        for frase in comandos.keys():
            palavras.extend(frase.split())
        vocab = json.dumps(list(set(palavras)) + ["[unk]"])
        self.recognizer.SetGrammar(vocab)
        print(f"[*] Vocabulário restrito a {len(set(palavras))} palavras dos comandos.")
 
        self.comandos        = comandos
        self.sample_rate     = sample_rate
        self.chunk_size      = chunk_size
        self._audio_q          = queue.Queue()
        self._running          = False
        self._ultimo_comando   = ""   # evita execução dupla parcial + final
        self._ultimo_exec_time = 0.0  # timestamp da última execução
        self._cooldown         = 1.2  # segundos mínimos entre dois disparos do mesmo comando
 
    def _audio_callback(self, in_data, frame_count, time_info, status):
        self._audio_q.put(in_data)
        return (None, pyaudio.paContinue)
 
    def _processar_texto(self, texto):
        texto = texto.lower().strip()
        if not texto:
            # fala terminou — libera o bloqueio de comando repetido
            self._ultimo_comando = ""
            return
        for frase, acao in self.comandos.items():
            if frase in texto:
                agora = time.time()
                mesmo_comando   = self._ultimo_comando == frase
                dentro_cooldown = (agora - self._ultimo_exec_time) < self._cooldown
                if mesmo_comando and dentro_cooldown:
                    return  # duplo disparo parcial+final, ignora
                print(f"[flash] Executando: {frase} -> {acao.__name__}")
                self._ultimo_comando   = frase
                self._ultimo_exec_time = agora
                try:
                    acao()
                except Exception as e:
                    print(f"[X] Erro ao executar '{frase}': {e}")
                return
        # nenhum comando encontrado — libera para próximo disparo
        self._ultimo_comando = ""
 
    def _recognition_loop(self):
        while self._running:
            try:
                data = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
 
            if self.recognizer.AcceptWaveform(data):
                # Resultado final (após silêncio detectado)
                resultado = json.loads(self.recognizer.Result())
                texto = resultado.get("text", "")
                if texto:
                    print(f"[mic] Final: \"{texto}\"")
                self._processar_texto(texto)
            else:
                # Resultado parcial — dispara comando sem esperar silêncio
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
# =============================================================================
 
class OneEuroFilter:
    def __init__(self, min_cutoff=0.2, beta=0.1, d_cutoff=1.0):
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
 
 
def precision_curve(norm, knee=0.45, knee_out=0.3):
    s = np.sign(norm)
    a = abs(norm)
    out = (a / knee) * knee_out if a <= knee else knee_out + ((a - knee) / (1.0 - knee)) * (1.0 - knee_out)
    return s * out
 
 
def run():
    # Inicia reconhecimento de voz em thread separada
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
 
    deadzone_yaw   = 1.0
    deadzone_pitch = 1.0
    max_yaw_deg    = 14.0
    max_pitch_deg  = 8.0
    curve_knee     = 0.45
    curve_knee_out = 0.3
 
    filt_pitch  = OneEuroFilter(min_cutoff=0.2, beta=0.1)
    filt_yaw    = OneEuroFilter(min_cutoff=0.2, beta=0.1)
    have_sample = False
 
    neutral_pitch = 0.0
    neutral_yaw   = 0.0
    is_calibrated = False
 
    curr_mouse_x, curr_mouse_y = screen_w / 2, screen_h / 2
 
    MAX_ANGLE_DELTA = 2.5
    prev_raw_pitch  = None
    prev_raw_yaw    = None
    WARMUP_FRAMES   = 6
    warmup_count    = WARMUP_FRAMES
 
    REPROJ_THRESH   = 8.0
    LK_PARAMS = dict(
        winSize =(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    MIN_TRACKED = 20
 
    prev_gray      = None
    track_pts_2d   = None
    track_pts_3d   = None
    r_vec_track    = None
    t_vec_track    = None
    frames_tracked  = 0
    tracking_active = False
 
    print("=== CONTROLE INICIADO (RASTREAMENTO 3D — MODELO DE NEWTON) ===")
    print("Pressione 'c' para CALIBRAR O CENTRO.")
    print("Pressione 'r' para forçar reinicialização do rastreamento.")
    print("Pressione 'q' ou 'ESC' para sair.")
 
    while True:
        frame_got, frame = cap.read()
        if not frame_got:
            break
        if video_src == 0:
            frame = cv2.flip(frame, 1)
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tm.start()
 
        need_landmarks = not tracking_active
 
        if tracking_active and not need_landmarks and prev_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, track_pts_2d, None, **LK_PARAMS)
 
            good_mask = status.ravel() == 1
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
                        r_vec_track = r_init
                        t_vec_track = t_init
                        proj_all, _ = cv2.projectPoints(
                            pose_estimator.model_points_68,
                            r_vec_track, t_vec_track,
                            pose_estimator.camera_matrix,
                            pose_estimator.dist_coeefs)
                        track_pts_2d = proj_all.astype(np.float32)
                        track_pts_3d = pose_estimator.model_points_68.copy()
                        frames_tracked = 0
                        tracking_active = True
                        need_landmarks  = False
                        warmup_count = WARMUP_FRAMES
                        prev_raw_pitch = None
                        prev_raw_yaw   = None
            else:
                tracking_active = False
 
        prev_gray = gray
 
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
 
            if abs(delta_yaw)   < deadzone_yaw:   delta_yaw   = 0.0
            if abs(delta_pitch) < deadzone_pitch:  delta_pitch = 0.0
 
            norm_x = max(-1.0, min(1.0, delta_yaw   / max_yaw_deg))
            norm_y = max(-1.0, min(1.0, delta_pitch / max_pitch_deg))
 
            mapped_x = precision_curve(norm_x, curve_knee, curve_knee_out)
            mapped_y = precision_curve(norm_y, curve_knee, curve_knee_out)
 
            target_x = (screen_w / 2) + (mapped_x * (screen_w / 2))
            target_y = (screen_h / 2) - (mapped_y * (screen_h / 2))
 
            dist = np.hypot(target_x - curr_mouse_x, target_y - curr_mouse_y)
            alpha = float(np.clip(dist / 250.0, 0.06, 0.5))
            curr_mouse_x += (target_x - curr_mouse_x) * alpha
            curr_mouse_y += (target_y - curr_mouse_y) * alpha
            if dist < 2.0:
                curr_mouse_x, curr_mouse_y = target_x, target_y
            if abs(mapped_x) >= 0.999: curr_mouse_x = target_x
            if abs(mapped_y) >= 0.999: curr_mouse_y = target_y
 
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
                cv2.putText(frame, "DEADZONE", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
            neutral_pitch = float(filt_pitch.x_prev)
            neutral_yaw   = float(filt_yaw.x_prev)
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
            print("Rastreamento reinicializado!")
 
 
if __name__ == '__main__':
    run()