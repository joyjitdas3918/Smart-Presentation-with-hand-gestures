import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)
# Load images for the presentation
pathImages = sorted(os.listdir('ppt'), key=len)
imgNumber = 0

# Variables for annotation and control
pressed = False
counter = 0
delay = 10
annotations = [[]]
annotationIndex = 0
annotationStart = False

# Hand Detector setup
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Function to handle image switching
def change_slide(next_slide=True):
    global imgNumber, annotations, annotationIndex, annotationStart
    annotationStart = False
    if next_slide:
        imgNumber = min(len(pathImages) - 1, imgNumber + 1)
    else:
        imgNumber = max(0, imgNumber - 1)
    annotations = [[]]
    annotationIndex = 0

# Function to draw annotations
def draw_annotations(img, annotation_list):
    for ann in annotation_list:
        for i in range(1, len(ann)):
            cv2.line(img, ann[i-1], ann[i], (0, 0, 255), 12)

# Main loop
while True:
    # Read current slide image
    pathFullImage = os.path.join('ppt', pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    
    # Read webcam image
    success, img = cap.read()
    
    img = cv2.flip(img, 1)
    # Detect hands
    hands, img = detector.findHands(img, draw=False)
    if hands and not pressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        lmList = hand['lmList']
        indexFinger = (lmList[8][0], lmList[8][1])  # Index finger tip
        
        # Slide forward
        if fingers == [0, 0, 0, 0, 1]:
            change_slide(next_slide=True)
            pressed = True
        
        # Slide backward
        elif fingers == [1, 0, 0, 0, 0]:
            change_slide(next_slide=False)
            pressed = True
        
        # Drawing a circle (pause on slide)
        elif fingers == [0, 1, 1, 0, 0]:
            annotationStart = False
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        
        # Start drawing annotations
        elif fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationIndex += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationIndex].append(indexFinger)
        
        # Remove the last annotation
        elif fingers == [0, 1, 1, 1, 0] and annotationIndex > 0:
            annotations.pop(-1)
            annotationIndex -= 1
            pressed = True
    else:
        annotationStart = False
    
    # Delay mechanism to avoid repeated presses
    if pressed:
        counter += 1
        if counter > delay:
            counter = 0
            pressed = False
    
    # Draw annotations
    draw_annotations(imgCurrent, annotations)
    
    # Display images
    cv2.imshow("Presentation", imgCurrent)
    cv2.imshow("Webcam",img)
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
