import cv2
import threading

class Camara:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()  # Espera a que el hilo termine
        if self.cap is not None:
            self.cap.release()

    def camara_activa(self):
        return self.cap.isOpened() if self.cap else False

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    #Voltear horizontalmente 
                    self.frame = cv2.flip(frame, 1)

    def get_frame(self):
        with self.lock:
            return self.frame
