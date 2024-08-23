import cv2
import threading

class Camara:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self.thread = None

    def obtener_camaras(self, stop=True):
        if stop:
            if self.camara_activa():
                self.stop()
        camaras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camaras.append(i)
                cap.release()
        return camaras

    def start_aux(self, index):
        self.cap = cv2.VideoCapture(index)
        self.running = True
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.start()

    # Al iniciar la camara, se inicia en otro hilo para evitar lag al usar la app
    def start(self, index):
        threading.Thread(target=self.start_aux, args=(index,)).start()

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
            if self.running:
                #Reescalar a 640x480
                if self.frame is not None:
                    self.frame = cv2.resize(self.frame, (640, 480))
                return self.frame
            else:
                return None
