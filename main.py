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

# import numpy as np
# import cv2
# import time
#
# cap = cv2.VideoCapture(0)  # 640,480
# # w = 640
# # h = 480
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#
#         # downsample
#         # frameD = cv2.pyrDown(cv2.pyrDown(frame))
#         # frameDBW = cv2.cvtColor(frameD,cv2.COLOR_RGB2GRAY)
#
#         # detect face
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         faces = cv2.CascadeClassifier('/Users/dhanukaramanayake/.virtualenvs/EyeTrackerPy/lib/python3.7/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
#         detected = faces.detectMultiScale(frame, 1.3, 5)
#
#         # faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#         # detected2 = faces.detectMultiScale(frameDBW, 1.3, 5)
#
#         pupilFrame = frame
#         pupilO = frame
#         windowClose = np.ones((5, 5), np.uint8)
#         windowOpen = np.ones((2, 2), np.uint8)
#         windowErode = np.ones((2, 2), np.uint8)
#
#         # draw square
#         for (x, y, w, h) in detected:
#             cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)
#             cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 1)
#             pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):(y + h), x:(x + w)])
#             pupilO = pupilFrame
#             ret, pupilFrame = cv2.threshold(pupilFrame, 55, 255, cv2.THRESH_BINARY)  # 50 ..nothin 70 is better
#             pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
#             pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
#             pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
#
#             # so above we do image processing to get the pupil..
#             # now we find the biggest blob and get the centriod
#
#             threshold = cv2.inRange(pupilFrame, 250, 255)  # get the blobs
#             buff, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#             # if there are 3 or more blobs, delete the biggest and delete the left most for the right eye
#             # if there are 2 blob, take the second largest
#             # if there are 1 or less blobs, do nothing
#
#             if len(contours) >= 2:
#                 # find biggest blob
#                 maxArea = 0
#                 MAindex = 0  # to get the unwanted frame
#                 distanceX = []  # delete the left most (for right eye)
#                 currentIndex = 0
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     center = cv2.moments(cnt)
#                     if center['m00'] != 0:
#                         cx = int(center["m10"] / center["m00"])
#                         cy = int(center["m01"] / center["m00"])
#                     else:
#                         cx, cy = 0, 0
#                     distanceX.append(cx)
#                     if area > maxArea:
#                         maxArea = area
#                         MAindex = currentIndex
#                     currentIndex = currentIndex + 1
#
#                 del contours[MAindex]  # remove the picture frame contour
#                 del distanceX[MAindex]
#
#             eye = 'right'
#
#             if len(contours) >= 2:  # delete the left most blob for right eye
#                 if eye == 'right':
#                     edgeOfEye = distanceX.index(min(distanceX))
#                 else:
#                     edgeOfEye = distanceX.index(max(distanceX))
#                 del contours[edgeOfEye]
#                 del distanceX[edgeOfEye]
#
#             if len(contours) >= 1:  # get largest blob
#                 maxArea = 0
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     if area > maxArea:
#                         maxArea = area
#                         largeBlob = cnt
#
#             if len(largeBlob) > 0:
#                 center = cv2.moments(largeBlob)
#                 cx, cy = int(center['m10'] / center['m00']), int(center['m01'] / center['m00'])
#                 cv2.circle(pupilO, (cx, cy), 5, 255, -1)
#
#     # show picture
#     cv2.flip(pupilO, 1, pupilO)
#     cv2.flip(pupilFrame, 1, pupilFrame)
#     cv2.imshow('frame', pupilO)
#     cv2.imshow('frame2', pupilFrame)
#     k = cv2.waitKey(30)
#     if k == 27:  # press Esc key to kill the program
#         break
#
# cap.release()
# cv2.destroyAllWindows()
