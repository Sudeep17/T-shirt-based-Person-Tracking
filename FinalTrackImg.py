import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys

# def draw_detections(img, rects, thickness = 5):



img1 = cv2.imread('LogoFinal.jpg')          # queryImage
# img2 = cv2.imread('box_in_scene.png',0) # trainImage

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
video = cv2.VideoCapture(0)
MIN_MATCH_COUNT = 15

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
b=0
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 # Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
while True and b!=27:

    # Read first frame.
    ok, img2 = video.read()
    # img2 = cv2.cvtColor()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    personHeight = []
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        w, h = img1.shape[:-1]
        # print("width and Height ->",w, h)
        pts = np.float32([[-w, -h], [-w, 7*h - 1], [2*w - 1, 7*h - 1], [2*w - 1, -h]]).reshape(-1, 1, 2)
        ptsLogo = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        if M is not None:
            dst = cv2.perspectiveTransform(pts, M)
            dstLogo = cv2.perspectiveTransform(ptsLogo, M)
            found, w = hog.detectMultiScale(img2, winStride=(8, 8), padding=(32, 32), scale=1.05)

            coords = [np.int32(dst)]

            x0 = coords[0][0][0][0]
            y0 = coords[0][0][0][1]
            x1 = coords[0][2][0][0]
            y1 = coords[0][2][0][1]
            x2 = coords[0][3][0][0]
            y2 = coords[0][3][0][1]

            coordslogo = [np.int32(dstLogo)]

            lx0 = coordslogo[0][0][0][0]
            ly0 = coordslogo[0][0][0][1]
            lx1 = coordslogo[0][2][0][0]
            ly1 = coordslogo[0][2][0][1]
            lx2 = coordslogo[0][3][0][0]

            #
            # print(coords, (x0,y0), (x1,y1))
            # cv2.circle(img2, (int(x0), int(y0)), 11, (255, 0, 255), 1)
            # cv2.circle(img2, (int(x1), int(y1)), 11, (255, 0, 255), 1)
            #cv2.imshow("test", rectCoords)
            #
            # print(np.int32(pts))
            # roi = img2[-h:3*h, 0:w]
            # cv2.imshow("roi",roi)
            #
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            roi_gray1 = gray[y0:y0+int(h/2), x0:x1]
            # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            img2 = cv2.polylines(img2, [np.int32(dstLogo)], True, (0, 255, 255), 3, cv2.LINE_AA)
            # print("--->",roi_gray1)
            # print(len(found))

            for xh, yh, wh, hh in found:
                # the HOG detector returns slightly larger rectangles than the real objects.
                # so we slightly shrink the rectangles to get a nicer output.
                # print("hog loc---",xh, yh, wh+xh, hh+yh)
                # print("logo loc---", lx0, ly0, lx1, ly1)
                # print("pixel height of logo", ly1-ly0)
                PtoHRatio= (ly1-ly0)/35

                if xh<lx0 and yh<ly0 and (wh+xh) > lx1 and (hh+yh)>lx2:
                    pad_w, pad_h = int(0.15 * wh), int(0.10 * hh)
                    cv2.rectangle(img2, (xh + pad_w, yh + pad_h), (xh + wh - pad_w, yh + hh - pad_h), (0, 0, 255), 4)
                    personHeighttemp = (hh - 2 * pad_h) / PtoHRatio
                    personHeight.append(personHeighttemp)
                    personHeight = sorted(personHeight)
                    medianHeight = personHeight[round(len(personHeight)/2)]
                    print("Average height of a person(in cm)", medianHeight)

                    print("Average height of a person(in inches)", medianHeight/2.54)
                    # cv2.putText(img2, 'height ->'+str(personHeight), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.rectangle(img, (x , y), (x + w, y + h), (0, 255, 0), thickness)


            # draw_detections(img2, found)
            # cv2.imshow("hogroi", img2)

            if roi_gray1.size != 0:
                # roi_gray = roi_gray1[yy:yy + hh, xx:xx + ww]


                # cv2.imshow("roi",roi_gray1)
                faces = face_cascade.detectMultiScale(roi_gray1)
                # print(len(faces))
                if len(faces):

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img2, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (255, 255, 255), 2)
                    roi_gray = roi_gray1[y:y + h, x:x + w]
                    # roi_color = img2[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img2, (x0 + x + ex, y0 + y + ey), (x0 + x + ex + ew, y0 + y + ey + eh), (0, 255, 0),
                                      2)


        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

            matchesMask = None
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

    # print(gray)

    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imshow("detected",img2)
    b = cv2.waitKey(10)


