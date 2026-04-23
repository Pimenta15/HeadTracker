"""Voice command actions and vocabulary mapping (Portuguese)."""
import os
import sys
import subprocess
import time

import pyautogui


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
    caminho = os.path.join(
        os.path.expanduser("~"), "Desktop",
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
    print("\n[!] Encerrando programa...")
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
    "encerrar":          encerrar_programa,
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
