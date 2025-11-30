# import numpy as np
# import cv2


# # Load the face and eye haar cascades classifiers
# face_classifier = cv2.CascadeClassifier(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\Haarcascades\Haarcascades\haarcascade_frontalface_default.xml")
# eye_classifier = cv2.CascadeClassifier(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\Haarcascades\Haarcascades\haarcascade_eye.xml")



# # Load the image
# # img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\pexels-danxavier-1212984.jpg")
# img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\AI\Deep Learning\Happy_Face\pexels-justin-shaifer-501272-1222271.jpg")
# # img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\AI\Deep Learning\Happy_Face\pexels-alex-green-6625763.jpg")

# # Check if the image is loaded correctly
# if img is None:
#     print("Error: Image not found or cannot be loaded!")
#     exit()
    

# # Convert image to grayscale for face detection
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # Detect faces in the image
# faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# # Check if faces are detected
# if len(faces) == 0:
#     print("No face found.")
    
# # Draw rectangles around detected faces and detected eyes within each face
# for (x, y, w, h) in faces:
#     # Draw a rectangle around the face
#     cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 0), 2)
    
    
#     # Region of interest (ROI) for face
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
    
#     # Detect eyes within the face region
#     eyes = eye_classifier.detectMultiScale(roi_gray)
    
#     for (ex, ey, ew, eh) in eyes:
#         # Draw rectangle around each detected eye
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (127, 0, 0), 2)
        
        
# # Display the output image with rectangles around faces and eyes
# # cv2.imshow('img', img)

# # Resize image to fit screen window (e.g., width=800)
# img_resized = cv2.resize(img, (800, 600))  # You can adjust size
# cv2.imshow("img", img_resized)


# # Wait for a key press before closing the window
# cv2.waitKey(0)

# # Destroy all OpenCV Windows
# cv2.destroyAllWindows() 





# CHATGPT



import numpy as np
import cv2

# Load Haar Cascades
face_classifier = cv2.CascadeClassifier(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\Haarcascades\Haarcascades\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\Haarcascades\Haarcascades\haarcascade_eye.xml")

# Load the image
# img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\AI\Deep Learning\Happy_Face\pexels-justin-shaifer-501272-1222271.jpg")
# img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\AI\Deep Learning\Happy_Face\pexels-shvetsa-4587993.jpg")
img = cv2.imread(r"C:\Users\piyus\Desktop\NareshIT\OpenCV\pexels-danxavier-1212984.jpg")

if img is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    # ✅ Slightly expand the rectangle inward by 1 pixel to avoid clipping
    cv2.rectangle(img, (x + 1, y + 1), (x + w - 2, y + h - 2), (0, 0, 255), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex + 1, ey + 1), (ex + ew - 2, ey + eh - 2), (0, 255, 0), 2)

# ✅ Save and reload image for clean rendering (avoids display bugs)
cv2.imwrite("output_fixed.jpg", img)
img_fixed = cv2.imread("output_fixed.jpg")

cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
cv2.imshow("Detected", img_fixed)
cv2.waitKey(0)
cv2.destroyAllWindows()

