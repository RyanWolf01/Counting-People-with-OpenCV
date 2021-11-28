# Code adapted from GeeksforGeeks at:
# https://www.geeksforgeeks.org/pedestrian-detection-using-opencv-python/
# Changed the video input and modified program to keep an active count of the
# number of people in an area at one time.

# Scatterplot code adapted from W3Schools:
# https://www.w3schools.com/python/matplotlib_scatter.asp


import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
   
# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
   
cap = cv2.VideoCapture('videos/airport_walking.mp4')

# cap = cv2.VideoCapture('people_closer.mp4')
   
iterations = 0
frame_tracker = []
people_counter = []
FRAMES_PER_SEC = 30

while cap.isOpened():
    # Reading the video stream
    # Changed this to read every 10th frame
    i = 0
    SKIPPED_FRAMES = FRAMES_PER_SEC * 5

    while i < SKIPPED_FRAMES:
        ret, image = cap.read()
        i += 1

    if ret:
        image = imutils.resize(image, 
                               width=min(400, image.shape[1]))
   
        # Detecting all the regions 
        # in the Image that has a 
        # pedestrians inside it
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)
   
        # Drawing the regions in the 
        # Image

        frame_tracker.append((iterations * SKIPPED_FRAMES) / FRAMES_PER_SEC)
        count = 0
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h), 
                          (0, 0, 255), 2)
            count += 1
        people_counter.append(count)
   
        # Showing the output Image
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    iterations += 1
  
# print(people_counter)
cap.release()
cv2.destroyAllWindows()

plt.bar(frame_tracker, people_counter)
plt.show()

# print(f"number of frames: {seconds * 10}")
# 830 in 29 seconds

# ~ 30 frames/second