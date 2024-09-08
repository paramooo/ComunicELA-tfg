from cv2 import VideoCapture, flip, resize
from threading import Lock, Thread

class Camara:
    """Clase de la camara, esta maneja las camaras y la obtencion de los frames"""

    def __init__(self):
        """
        Inicializa la clase Camara con los atributos necesarios.
        """
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.cap = None
        self.thread = None

    def obtener_camaras(self, stop=True):
        """
        Obtiene una lista de índices de cámaras disponibles.
        
        Args:
            stop (bool): Si es True, detiene la cámara activa antes de buscar nuevas cámaras.
        
        Returns:
            list: Lista de índices de cámaras disponibles.
        """
        if stop:
            if self.camara_activa():
                self.stop()
        camaras = []
        for i in range(10):
            cap = VideoCapture(i)
            if cap.isOpened():
                camaras.append(i)
                cap.release()
        return camaras


    def start_aux(self, index):
        """
        Inicia la captura de video en un hilo separado.
        
        Args:
            index (int): Índice de la cámara a utilizar.
        """
        self.cap = VideoCapture(index)
        self.running = True
        self.thread = Thread(target=self.update_frame, args=())
        self.thread.start()


    def start(self, index):
        """
        Inicia la captura de video en un hilo separado.

        Args:
            index (int): Índice de la cámara a utilizar.
        """
        Thread(target=self.start_aux, args=(index,)).start()


    def stop(self):
        """
        Detiene la captura de video y libera los recursos.
        
        """
        self.running = False
        if self.thread is not None:
            self.thread.join()  
        if self.cap is not None:
            self.cap.release()


    def camara_activa(self):
        """
        Verifica si la cámara está activa.

        Returns:
            bool: True si la cámara está activa, False de lo contrario.
        """
        return self.cap.isOpened() if self.cap else False


    def update_frame(self):
        """
        Actualiza el frame de la cámara.
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = flip(frame, 1)


    def get_frame(self):
        """
        Obtiene el frame actual de la cámara.

        Returns:
            ndarray: Frame actual de la cámara.
        """
        with self.lock:
            if self.running:
                if self.frame is not None:
                    self.frame = resize(self.frame, (640, 480))
                return self.frame
            else:
                return None
