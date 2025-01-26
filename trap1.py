# https://learnopencv.com/moving-object-detection-with-opencv/


#! pip install opencv-python gradio


import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

#vid_path = "/Users/danielbraun/Documents/trackcam/20240925/03renard/short.mov"
vid_path = "/Users/danielbraun/Documents/trackcam/20241109/VD_00003.MP4"
#vid_path = "/Users/danielbraun/Documents/trackcam/20240323/04chevreuil//DSCF0013.AVI"
#vid_path = "/Users/danielbraun/Documents/trackcam/20240218/renard/DSCF0268.AVI"




cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("Error opening video file")
    exit(1)

   
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Apply background subtraction
        print(".", end='');
        fg_mask = backSub.apply(frame)

        retval, mask_thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        #contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 200  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        frame_ct = cv2.drawContours(frame, large_contours, -1, (255, 255, 0), 2)

        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
        cv2.imshow('Frame_final', frame_out)
        #cv2.imshow('Frame_final', frame_ct)
        #cv2.imshow('mask', fg_mask)
        #cv2.imshow('mask', mask_eroded)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
         break
