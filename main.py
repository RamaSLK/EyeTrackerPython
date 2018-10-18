import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
glass_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
right_eye_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_righteye_2splits.xml')
left_eye_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    cv2.GaussianBlur(gray, (9, 9), 4, gray, 4)

    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    #
    #     glasses = glass_cascade.detectMultiScale(roi_gray)
    #     for (gx, gy, gw, gh) in glasses:
    #         cv2.rectangle(roi_color,(gx, gy),(gx + gw, gy + gh), (0, 0, 255), 2)

            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        glasses = glass_cascade.detectMultiScale(roi_gray, 1.3, 6)
        for (rx, ry, rw, rh) in glasses:
            cv2.rectangle(roi_color, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            roi_gray2 = roi_gray[ry:ry + rh, rx:rx + rw]
            roi_color2 = img[ry:ry + rh, rx:rx + rw]
            circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10, minRadius=0, maxRadius=0)

            try:
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(roi_color2, (i[0], i[1]), i[2], (255, 255, 255), 2)
                    print("drawing circle", circles)
                    # draw the center of the circle
                    cv2.circle(roi_color2, (i[0], i[1]), 2, (255, 255, 255), 3)

            except Exception as e:
                print("Exception : ", e)

    cv2.flip(img, 1, img)
    cv2.imshow('video feed', img)
    # cv2.flip(roi_color2, 1, roi_color2)
    # cv2.imshow('roi_color2', roi_color2)
    k = cv2.waitKey(30)
    if k == 27:  # press Esc key to kill the program
        break

cap.release()
cv2.destroyAllWindows()
