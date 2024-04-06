import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import os

def popup(titulo, imagen):
    cv2.imshow(titulo, imagen)
    while True:
        if cv2.waitKey(10) > 0:
            cv2.destroyWindow(titulo)
            break
        elif cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
            break

def plot(image, titulo=None, axis=False):
    dpi = mpl.rcParams['figure.dpi']
    if len(image.shape)==2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = w / float(dpi), h / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    if not axis:
        ax.axis('off')
    if isinstance(titulo, str):
        plt.title(titulo)
    
    # Display the image.
    if c==4:
        plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) )
    elif c==1:
        plt.imshow( image, cmap='gray' )
    else:
        plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGR2RGB) , aspect='equal')


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
                    self._currentFrame = framecv2.cvtColor(fg, cv2.COLOR_GRAY2BGRA)
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

def alphaBlending(fg, bg, x=0, y=0):
    sfg = fg.shape
    fgh = sfg[0]
    fgw = sfg[1]

    sbg = bg.shape
    bgh = sbg[0]
    bgw = sbg[1]

    h = max(bgh, y + fgh) - min(0, y)
    w = max(bgw, x + fgw) - min(0, x)

    CA = np.zeros(shape=(h, w, 3))
    aA = np.zeros(shape=(h, w))
    CB = np.zeros(shape=(h, w, 3))
    aB = np.zeros(shape=(h, w))

    bgx = max(0, -x)
    bgy = max(0, -y)

    if len(sbg) == 2 or sbg[2]==1:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = np.ones(shape=sbg)
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    elif sbg[2] == 3:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = np.ones(shape=sbg[0:2])
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = bg
    else:
        aB[bgy:bgy+bgh, bgx:bgx+bgw] = bg[:, :, 3] / 255.0
        CB[bgy:bgy+bgh, bgx:bgx+bgw, :] = bg[:, :, 0:3]

    fgx = max(0, x)
    fgy = max(0, y)

    if len(sfg) == 2 or sfg[2]==1:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = np.ones(shape=sfg)
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    elif sfg[2] == 3:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = np.ones(shape=sfg[0:2])
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = fg
    else:
        aA[fgy:fgy+fgh, fgx:fgx+fgw] = fg[:, :, 3] / 255.0
        CA[fgy:fgy+fgh, fgx:fgx+fgw, :] = fg[:, :, 0:3]

    aA = cv2.merge((aA, aA, aA))
    aB = cv2.merge((aB, aB, aB))
    a0 = aA + aB * (1 - aA)
    C0 = np.divide(((CA * aA) + (CB * aB)*(1.0 - aA)), a0, out=np.zeros_like(CA), where=(a0!=0))

    res = cv2.cvtColor(np.uint8(C0), cv2.COLOR_BGR2BGRA)
    res[:, :, 3] = np.uint8(a0[:, :, 0] * 255.0)

    return res

def proyeccion(puntos, rvec, tvec, cameraMatrix, distCoeffs):
    if isinstance(puntos, list):
        return(proyeccion(np.array(puntos, dtype=np.float32), rvec, tvec, cameraMatrix, distCoeffs))
    if isinstance(puntos, np.ndarray):
        if puntos.ndim == 1 and puntos.size == 3:
            res, _ = cv2.projectPoints(puntos.astype(np.float32), rvec, tvec, cameraMatrix, distCoeffs)
            return(res[0][0].astype(int))
        if puntos.ndim > 1:
            aux = proyeccion(puntos[0], rvec, tvec, cameraMatrix, distCoeffs)
            aux = np.expand_dims(aux, axis=0)
            for p in puntos[1:]:
                aux = np.append(aux, [proyeccion(p, rvec, tvec, cameraMatrix, distCoeffs)], axis=0)
            return(np.array(aux))

def histogramahsv(imagen, solotono=True):
    if solotono:
        hist, (ax1) = plt.subplots(1)
    else:
        hist, (ax1, ax2, ax3) = plt.subplots(1,3)
    framehsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(framehsv)
    histoh = cv2.calcHist( [framehsv], [0], None, [180], [0, 180])
    ax1.set_title("Hue")
    ax1.get_yaxis().set_visible(False)
    ax1.plot(histoh)
    if not solotono:
        histos = cv2.calcHist( [framehsv], [1], None, [256], [0, 256])
        ax2.set_title("Sat")
        ax2.get_yaxis().set_visible(False)
        ax2.plot(histos)
        histov = cv2.calcHist( [framehsv], [2], None, [256], [0, 256])
        ax3.set_title("Val")
        ax3.get_yaxis().set_visible(False)
        ax3.plot(histov)
    plt.show()