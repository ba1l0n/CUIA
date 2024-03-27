import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def popup(titulo, imagen):
    cv2.imshow(titulo, imagen)
    while True:
        if cv2.waitKey(10) > 0:
            cv2.destroyWindow(titulo)
            break
        elif cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
            break

def plot(image):
    if len(image.shape)==2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(h*px, w*px), layout='tight')
    plt.axis('off')
    if c==4:
        plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) )
    elif c==1:
        plt.imshow( image, cmap='gray' )
    else:
        plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )

def bestBackend(camid):
    backends = cv2.videoio_registry.getCameraBackends()
    bestCap = 0
    bestTime = 999
    for b in backends:
        start = time.time()
        cam = cv2.VideoCapture(camid, b)
        end = time.time()
        if cam.isOpened():
            if end-start < bestTime:
                bestTime = end-start
                bestCap = b
            cam.release()
    return bestCap

class myVideo:
    def __init__(self, source, backend=cv2.CAP_ANY):
        self.loop = False      #Para indicar si el video reiniciará al terminar
        self.process = None    #Para indicar la función opcional de procesado de frames
        if isinstance(source, str):
            if os.path.exists(source):
                self._cap = cv2.VideoCapture(source)
                self._camera = False
                self._nextFrame = 0
                self._startTime = time.time()
                self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._numFrames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self._currentFrame = None
            else:
                self._cap = cv2.VideoCapture(source)
                self._camera = True #IP Camera
        elif isinstance(source, int):
            self._cap = cv2.VideoCapture(source, backend)
            self._camera = True

    def __del__(self):
        self._cap.release()

    def release(self):
        self._cap.release()
        del self

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        if self._camera:
            ret, frame = self._cap.read()
            if ret and self.process != None:
                frame = self.process(frame)
            return(ret, frame)
        else:
            nextFrameStart = self._startTime + self._nextFrame / self._fps
            nextFrameEnd = self._startTime + (self._nextFrame + 1) / self._fps
            now = time.time()
            if now <= nextFrameStart:
                return (True, self._currentFrame)
            else:
                if now < nextFrameEnd:
                    correctFrame = self._nextFrame
                else:
                    correctFrame = int((now - self._startTime) * self._fps)

                if self.loop:
                    correctFrame = correctFrame % self._numFrames
                elif correctFrame >= self._numFrames:
                    return (False, None)

                if correctFrame != self._nextFrame:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, correctFrame)

                ret, frame = self._cap.read()
                if ret:
                    self._currentFrame = frame
                    self._nextFrame = correctFrame + 1
                    if self.loop:
                        self._nextFrame = self._nextFrame % self._numFrames

                    if self.process != None:
                        frame = self.process(frame)
                return (ret, frame)

    def get(self, prop):
        return(self._cap.get(prop))

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._nextFrame = value
        return(self._cap.set(prop, value))

    def play(self, titulo, key=27):
        cv2.namedWindow(titulo)
        if self._cap.isOpened():
            while True:
                ret, frame = self.read()
                if not ret or cv2.waitKey(20)==key:
                    break
                elif cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1: #Detenemos también si se cerró la ventana
                    break
                if frame is not None:
                    cv2.imshow(titulo, frame)
        cv2.destroyWindow(titulo)
