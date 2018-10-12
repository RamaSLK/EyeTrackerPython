import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
camera = cv2.VideoCapture(0)
numerator_l = 0
denominator_l = 0
numerator_r = 0
denominator_r = 0
while True:
    ret, frame = camera.read()
    roi = frame
    roi2 = frame
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # left eye
        # horizontal lower line
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 2.1)), (int(x + w / 2.5), int(y + h / 2.1)), (0, 255, 0), 2)
        # horizontal upper line
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 3.5)), (int(x + w / 2.5), int(y + h / 3.5)), (0, 255, 0), 2)
        # vertical left line
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 3.5)), (int(x + w / 4.2), int(y + h / 2.1)), (0, 255, 0), 2)
        # vertical right line
        cv2.line(frame, (int(x + w / 2.5), int(y + h / 3.5)), (int(x + w / 2.5), int(y + h / 2.1)), (0, 255, 0), 2)

        # right eye
        # horizontal lower line
        cv2.line(frame, (int(x + w / 1.6), int(y + h / 2.1)), (int(x + w / 1.3), int(y + h / 2.1)), (0, 255, 0), 2)
        # horizontal upper line
        cv2.line(frame, (int(x + w / 1.6), int(y + h / 3.5)), (int(x + w / 1.3), int(y + h / 3.5)), (0, 255, 0), 2)
        # vertical left line
        cv2.line(frame, (int(x + w / 1.6), int(y + h / 3.5)), (int(x + w / 1.6), int(y + h / 2.1)), (0, 255, 0), 2)
        # vertical right line
        cv2.line(frame, (int(x + w / 1.3), int(y + h / 3.5)), (int(x + w / 1.3), int(y + h / 2.1)), (0, 255, 0), 2)

        # coordinates of interest
        # left eye roi
        x1 = int(x + w / 4.2) + 1
        x2 = int(x + w / 2.5)
        y1 = int(y + h / 3.5) + 1
        y2 = int(y + h / 2.1)

        # right eye roi
        x3 = int(x + w / 1.6) + 1
        x4 = int(x + w / 1.3)
        y3 = int(y + h / 3.5) + 1
        y4 = int(y + h / 2.1)

        # for left eye
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        thres = cv2.inRange(equ, 0, 30)
        kernel = np.ones((3, 3), np.uint8)
        # removing small noise inside the white image
        dilation = cv2.dilate(thres, kernel, iterations=2)
        # decreasing the size of the white region
        erosion = cv2.erode(dilation, kernel, iterations=3)
        # finding the contours
        image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # checking for 2 contours found or not
        if len(contours) == 2:
            numerator_l += 1
            # img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            # finding the centroid of the contour
            M = cv2.moments(contours[1])
            print(M['m00'])
            print(M['m10'])
            print(M['m01'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)
        # print cx,cy
        # checking for one contours presence
        elif len(contours) == 1:
            numerator_l += 1
            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

            # finding centroid of the contours
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print cx,cy
                cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)
        else:
            denominator_l += 1
        # print "iris not detected"

        # for right eye
        roi2 = frame[y3:y4, x3:x4]
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        equ2 = cv2.equalizeHist(gray2)
        thres2 = cv2.inRange(equ2, 0, 20)
        kernel2 = np.ones((3, 3), np.uint8)
        # removing small noise inside the white image
        dilation2 = cv2.dilate(thres2, kernel2, iterations=2)
        # decreasing the size of the white region
        erosion2 = cv2.erode(dilation2, kernel2, iterations=3)
        # finding the contours
        image2, contours2, hierarchy2 = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # checking for 2 contours found or not
        if len(contours2) == 2:
            numerator_r += 1
            # img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            # finding the centroid of the contour
            M = cv2.moments(contours2[1])
            print(M['m00'])
            print(M['m10'])
            print(M['m01'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.line(roi2, (cx, cy), (cx, cy), (0, 0, 255), 3)
        # print cx,cy
        # checking for one contours presence
        elif len(contours2) == 1:
            numerator_r += 1
            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

            # finding centroid of the contours
            M = cv2.moments(contours2[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print cx,cy
                cv2.line(roi2, (cx, cy), (cx, cy), (0, 0, 255), 3)
        else:
            denominator_r += 1
        # print "iris not detected"

    cv2.imshow("frame", frame)
    # cv2.imshow("Left eye", roi)
    # cv2.imshow("Right eye", roi2)
    if cv2.waitKey(30) == 27 & 0xff:
        break
camera.release()
print("accurracy=", (float(numerator_l) / float(numerator_l + denominator_l)) * 100)
print("accurracy=", (float(numerator_r) / float(numerator_r + denominator_r)) * 100)
cv2.destroyAllWindows()
