import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_matches(im1,im2):
    # sift = cv2.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY),None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY),None)

    # matching features
    BF_Matcher = cv2.BFMatcher()
    initial_matches = BF_Matcher.knnMatch(des1,des2,k=2)

    # ratio test
    good_matches = []
    for m,n in initial_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    return good_matches,kp1,kp2

def find_homography(matches,kp1,kp2):
    if len(matches) < 4:
        exit(0)

    pts1 = []
    pts2 = []

    for match in matches:
        # get matched points from both images
        pts1.append(kp1[match[0].queryIdx].pt)
        pts2.append(kp2[match[0].trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    (H,status) = cv2.findHomography(pts1,pts2,cv2.RANSAC,4.0)

    return H,status

def update_frame_matrix(H,im1,im2):
    h,w = im2.shape[:2]

    # 1.corners of secondary image:
    # 12
    # 43
    # [[x1,x2,x3,x4],
    # [y1,y2,y3,y4],
    # [1,1,1,1]]


    init_matrix = np.array([[0,w-1,w-1,0],
                            [0,0,h-1,h-1],
                            [1,1,1,1]])

    # 2.finding resulting corners(after transformation)
    # transforming every column by H matrix final_matrix[:,i] = H*init_matrix[:,i]
    final_matrix = np.dot(H,init_matrix)

    [x,y,c] = final_matrix
    x = np.divide(x,c)
    y = np.divide(y,c)


    # find dimensions of transformed corners
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    new_w = max_x
    new_h = max_y
    correction = [0,0]
    if min_x < 0:
        new_w -= min_x
        correction[0] = abs(min_x)
    if min_y < 0:
        new_h -= min_y
        correction[1] = abs(min_y)

    # finding height and width  
    h_base,w_base = im2.shape[:2]
    if new_w < w_base + correction[0]:
        new_w = w_base + correction[0]
    if new_h < h_base + correction[1]:
        new_h = h_base + correction[1]


    x = np.add(x, correction[0])
    y = np.add(y, correction[1])

    old_initial_points = np.float32([[0,0],
                                     [w-1,0],
                                     [w-1,h-1],
                                     [0,h-1]])

    # final points after correction()
    new_final_points = np.float32(np.array([x,y]).transpose())

    homography_matrix = cv2.getPerspectiveTransform(old_initial_points,new_final_points)

    print(homography_matrix)






im1 = cv2.cvtColor(cv2.imread("img/1.jpg"),cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread("img/2.jpg"),cv2.COLOR_BGR2RGB)


good_matches, kp1, kp2 = find_matches(im1,im2)
print(f"n good matches:{len(good_matches)}")

H,status = find_homography(good_matches,kp1,kp2)



print(H)


update_frame_matrix(H,im1,im2)