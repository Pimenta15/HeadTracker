"""Utility to enumerate and preview connected cameras."""
import cv2


def listar_cameras():
    print("Buscando câmeras conectadas...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCESSO] Câmera encontrada no Índice: {i}")
                cv2.imshow(f"Camera {i} - Pressione qualquer tecla para fechar", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cap.release()
        else:
            print(f"[FALHA] Nenhuma câmera no Índice: {i}")


if __name__ == "__main__":
    listar_cameras()
