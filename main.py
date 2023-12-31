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
import time
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['TWILIO_ACCOUNT_SID'] ='AC10b029395650ed8afd05108e47968825'
auth_token = os.environ['TWILIO_AUTH_TOKEN'] = '8f71c8e313d0fba164e4321e8ee67cdf'
client = Client(account_sid, auth_token)

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
isPerson = False
isWater = False
isBook = False
isCellPhone = False
drinking = False
reading = False
distracted = False
count = 0;

#text to speech function
voiceCounter = 0
def speak(text):
  global voiceCounter
  tts = gTTS(text=text, lang="en")
  filename = "voice" + str(voiceCounter) + ".mp3"
  tts.save(filename)
  pygame.mixer.music.load(filename)
  pygame.mixer.music.play()
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

  
  for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
 
      idx = int(detection[1])   # prediction class index. 
 
      # If you want all classes to be labeled instead of just forks, spoons, and knives, 
      # remove this line below (i.e. remove line 65)
      if classes[idx] == 'person' or classes[idx] == 'cell phone' or classes[idx] == 'book' or classes[idx] == 'bottle':          
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        if classes[idx] == 'person':
          isPerson = True
        if classes[idx] == 'bottle':
          isWater = True
        if classes[idx] == 'book':
          isBook = True
        if classes[idx] == 'cell phone':
          isCellPhone = True
            
        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(classes[idx],score * 100)
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
        if (isPerson and isWater):
          cv.putText(img, "Drink more water", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          if drinking == False and isWater:
            speak("Drink more water")
            
            message = client.messages.create(
                              from_='+18883015401',
                              body='Xiang is drinking water right now',
                              to='+16463227786'
                          )
            
            print(message.sid)



            drinking = True
          isWater = False

        if (isPerson and isBook):
          cv.putText(img, "Reading is the key to success", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          if reading == False and isBook:

            speak("Reading is the key to success")
            
            message = client.messages.create(
                              from_='+18883015401',
                              body='Xiang is reading right now',
                              to='+16463227786'
                          )

            print(message.sid)

            reading = True
          isBook = False
        if (isPerson and isCellPhone):
          cv.putText(img, "Stop getting distracted by your phone", (100,100),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
          if distracted == False and isCellPhone:

            speak("Stop getting distracted by your phone")

            message = client.messages.create(
                              from_='+18883015401',
                              body='Xiang is being distracted right now',
                              to='+16463227786'
                          )

            print(message.sid)

            distracted = True
          isCellPhone = False

      if (count == 150):
        drinking = False
        distracted = False
        reading = False
        count = 0
      count = count + 1
      print(count)


          

  # Display the frame
  cv.imshow('my webcam', img)
 
  # Press ESC to quit
  if cv.waitKey(1) == 27: 
    break
 
# Stop filming
cam.release()
 
# Close down OpenCV
cv.destroyAllWindows()
