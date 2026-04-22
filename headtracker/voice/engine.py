"""Voice recognition engine using Vosk."""
import json
import os
import queue
import threading
import time

import pyaudio
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
CHUNK_SIZE = 2000


class VoiceCommandEngine:
    """Listens on the microphone and dispatches voice commands in a background thread."""

    def __init__(self, model_path, comandos, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Pasta do modelo não encontrada: '{model_path}'\n"
                "Baixe em https://alphacephei.com/vosk/models e extraia como 'model/'"
            )
        print(f"[*] Carregando modelo Vosk de '{model_path}'...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)

        palavras = []
        for frase in comandos.keys():
            palavras.extend(frase.split())
        vocab = json.dumps(list(set(palavras)) + ["[unk]"])
        self.recognizer.SetGrammar(vocab)
        print(f"[*] Vocabulário restrito a {len(set(palavras))} palavras dos comandos.")

        self.comandos = comandos
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._audio_q = queue.Queue()
        self._running = False
        self._ultimo_comando = ""
        self._ultimo_exec_time = 0.0
        self._cooldown = 1.2

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
                if self._ultimo_comando == frase and (agora - self._ultimo_exec_time) < self._cooldown:
                    return
                print(f"[flash] Executando: {frase} -> {acao.__name__}")
                self._ultimo_comando = frase
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
            print(f'   * "{cmd}"')
