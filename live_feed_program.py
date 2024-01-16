# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:35:57 2024

@author: Loges
"""

import cv2
import numpy as np
from keras.models import load_model

model = load_model()
with open('classes.txt', 'r') as file:
    classes = file.readlines()

vdo=cv2.VideoCapture(0)
while True:
    isTrue,Frame=vdo.read()
    
    preds=model.predict(Frame)
    preds = np.argmax(preds, axis=1)
    class_name = classes[preds]
    
    cv2.putText(Frame, class_name, (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, (0,100,0), 2)
    cv2.imshow('Output',Frame)
    if cv2.waitKey(27) & 0xFF==ord('d'):
        break

vdo.release()
cv2 .destroyAllWindows()