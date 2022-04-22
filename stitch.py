from hashlib import algorithms_available
import cv2
import matplotlib.pyplot as plt

import numpy as np




def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # coordinates of left image corners
    list_of_points_1 = np.float32([[0,0],[0,rows1],[cols1,rows1],[cols1,0]]).reshape(-1,1,2)
    # coordinates of right image corners
    temp_points = np.float32([[0,0],[0,rows2],[cols2,rows2],[cols2,0]]).reshape(-1,1,2)

    # transform right image corners into left image with Homography
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    # combine left and right image corners
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2),axis=0)
    print(f"list_of_points : {list_of_points}")

    # find max and min coords and convert them to closest int
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel()-0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel()+0.5)

    print(f"x_min : {x_min} | y_min : {y_min}")
    print(f"x_max : {x_max} | y_max : {y_max}")

    translation_dist = [-x_min, -y_min]

    print(f"translation_dist : {translation_dist}")

    # additional translation as minimum translation in x and y axis
    H_translation = np.array([[1,0,translation_dist[0]],
                              [0,1,translation_dist[1]],
                              [0,0,1]])

    print(f"total H:{H_translation.dot(H)}")

    output_img = cv2.warpPerspective(img2, H_translation.dot(H),(x_max-x_min,y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1


    return output_img


# coordinates of end-effector when making images
n_images = 9
x_coords = [716.82] * n_images
y_coords = [114.94,
            124.54,
            135.5,
            151.07,
            162.03,
            177.32,
            182.86,
            189.68,
            197.22]
z_coords = [437.19] * n_images

angle_1 = -90.17
angle_2 = 0.37
angle_3 = -91.13


def homography_raw(path1,path2):
    # load
    im1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2RGB)

    # keypoints,descriptors
    descriptor = cv2.SIFT_create()

    (kps1, features1) = descriptor.detectAndCompute(im1, None)
    (kps2, features2) = descriptor.detectAndCompute(im2, None)

    # keypoint matching(FLANN - Fast library for approximate nearest neighbors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(features1,features2,k=2)

    # remove bad matches(mask might be necessary for matching visualization):
    matches_mask = [[0,0] for i in range(len(matches))]

    print(f"len(features1) : {len(features1)}")

    good_matches = []
    for i,(m,n) in matches:
        # if distance to feature 1 is much less than distance to feature 2( dist1 < 70% of dist)
        if m.distance < 0.7 * n.distance:
            matches_mask[i] = [1,0]

            good_matches.append(m)

    
    # draw_params = dict(matchColor = (0,255,0),
    #                     singlePointColor = (255,0,0),
    #                     matchesMask = matchesMask,
    #                     flags = cv2.DrawMatchesFlags_DEFAULT)

    # img3 = cv2.drawMatchesKnn(im1,kps1,im2,kps2,matches,None,**draw_params)

    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    # calculate homography(src -> dst transformation)
    (H,status) = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)

    return H





if __name__=="__main__":
    path1 = "./FREZ/1.png"
    path2 = "./FREZ/2.png"

    # load images
    im1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2RGB)

    # find keypoints and descriptors
    kps1, feats1 = detect_and_describe(im1)
    kps2, feats2 = detect_and_describe(im2)

    # keypoint matching(brute force - works bad, )
    # matcher = cv2.BFMatcher()
    # matches = matcher.match(feats1,feats2)


    # KNN like
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(feats1, feats2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    print(f"len(feats1): {len(feats1)}")
    print(f"len(feats2): {len(feats2)}")
    print(len(matches))
    print(matches[0][0].distance)
    print(matches[0][1].distance)

    
    # tuples of matched ids
    good_matches = []    
    
    
    for i,(m,n) in enumerate(matches):
        
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]

            good_matches.append(m)



    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    print(src_pts.shape)


    (H,status) = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)

    print(H)
    H = np.eye(3)
    H[1,2] = y_coords[1] - y_coords[0]

    warped_image = warpImages(im2,im1,H)

    plt.imshow(warped_image)
    plt.show()


    # warped_image = cv2.warpPerspective(im1,H,(im2.shape[1],im2.shape[0]))

    # plt.imshow(warped_image)
    # plt.show()



    # plt.imshow(img3)
    # plt.show()










