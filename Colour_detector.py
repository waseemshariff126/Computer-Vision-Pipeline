import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


colorRange = {
        'Black':  (np.array([  0,   0,  27]), np.array([131,  65,  65])),
        'Silver': (np.array([117,   0, 122]), np.array([255,  19, 255])),
        'Red':    (np.array([155,  56,  60]), np.array([190, 170, 197])),
        'White':  (np.array([ 73,   0, 178]), np.array([140,  21, 255])),
        'Blue':   (np.array([ 67,  23,  66]), np.array([116, 106, 255]))    
             }

def colorDetector(img):
    hsvImage = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    maxPixel = 0
    for color in colorRange:
        mask = cv2.inRange(hsvImage, colorRange[color][0], colorRange[color][1])
        pixelStrength = np.sum(mask == 255) / 255
        if pixelStrength > maxPixel:
            carColor = color
            maxPixel = pixelStrength
    return carColor
