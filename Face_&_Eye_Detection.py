import numpy as np
import cv2

# Load Haar Cascades
face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"haarcascade_eye.xml")

# Load the image
img = cv2.imread(r"pexels-danxavier-1212984.jpg")

if img is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    # Slightly expand the rectangle inward by 1 pixel to avoid clipping
    cv2.rectangle(img, (x + 1, y + 1), (x + w - 2, y + h - 2), (0, 0, 255), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex + 1, ey + 1), (ex + ew - 2, ey + eh - 2), (0, 255, 0), 2)

# Save and reload image for clean rendering (avoids display bugs)
cv2.imwrite("output_fixed.jpg", img)
img_fixed = cv2.imread("output_fixed.jpg")

cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
cv2.imshow("Detected", img_fixed)
cv2.waitKey(0)
cv2.destroyAllWindows()
