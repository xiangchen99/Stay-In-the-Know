# Project: Eating Utensil Detector Using TensorFlow and OpenCV
# Author: Addison Sears-Collins
# Date created: August 1, 2021
# Description: This program detects forks, spoons, and knives
 
import cv2 as cv # OpenCV computer vision library
import numpy as np # Scientific computing library \
from gtts import gTTS
import os
from playsound import playsound
import threading
import pygame
pygame.mixer.init()
import datetime
 
# Just use a subset of the classes
classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
 
# Colors we will use for the object labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))
 
# Open the webcam
cam = cv.VideoCapture(0)
 
pb  = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
 
# Read the neural network
cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)   

#text to speech function
voiceCounter = 0
def speak(text):
  global voiceCounter
  tts = gTTS(text=text, lang="en")
  filename = "voice" + str(voiceCounter) + ".mp3"
  tts.save(filename)
  pygame.mixer.music.load(filename)
  pygame.mixer.music.play()
    
  while pygame.mixer.music.get_busy():
    pass
  voiceCounter += 1
  if voiceCounter > 2:
    os.remove("voice" + str(voiceCounter-2) + ".mp3")
 
while True:
 
  # Read in the frame
  ret_val, img = cam.read()
  rows = img.shape[0]
  cols = img.shape[1]
  cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 

 
  # Run object detection
  cvOut = cvNet.forward()
  # Go through each object detected and label it
  # Define variables that will be used
  isPerson = False
  isWater = False
  isToothbrush = False
  isCellPhone = False
  
  for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
 
      idx = int(detection[1])   # prediction class index. 
 
      # If you want all classes to be labeled instead of just forks, spoons, and knives, 
      # remove this line below (i.e. remove line 65)
      if classes[idx] == 'person' or classes[idx] == 'cell phone' or classes[idx] == 'toothbrush' or classes[idx] == 'bottle':          
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        if classes[idx] == 'person':
          isPerson = True
        if classes[idx] == 'bottle':
          isWater = True
        if classes[idx] == 'toothbrush':
          isToothbrush = True
        if classes[idx] == 'cell phone':
          isCellPhone = True
            
        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(classes[idx],score * 100)
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
        if (isPerson and isWater):
          cv.putText(img, "Chug, Chug, Chug, Chug", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          speak("Chug, Chug, Chug, Chug")
        if (isPerson and isToothbrush):
          cv.putText(img, "Make sure to brush twice a day", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          speak("Make sure to brush twice a day")
        if (isPerson and isCellPhone):
          cv.putText(img, "Stop getting distracted by your phone", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          speak("Stop getting distracted by your phone")

  # Display the frame
  cv.imshow('my webcam', img)
 
  # Press ESC to quit
  if cv.waitKey(1) == 27: 
    break
 
# Stop filming
cam.release()
 
# Close down OpenCV
cv.destroyAllWindows()