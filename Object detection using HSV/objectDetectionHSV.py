import cv2 as cv
import numpy as np
from PIL import Image
from util import get_limits

orange = [0, 255, 255]

path='/Users/amallmuhammed/Documents/Week 20 Open CV/images/cars.mp4'
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=orange)
    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv.imshow('Object detection',frame)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()