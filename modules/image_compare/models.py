import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from scipy.ndimage import binary_opening, binary_dilation


sift = cv2.SIFT_create()


def keep_largest_area_mask(im, connectivity=8):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity)

    try:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    except:
        return np.zeros_like(labels, dtype=np.uint8)

    largest_area_mask = np.zeros_like(labels, dtype=np.uint8)
    largest_area_mask[labels == largest_label] = 255

    return largest_area_mask


def depth_compare(im0, im1, thresh=10, open=5):
    diff = cv2.absdiff(im0, im1)

    _, binary_diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((open, open), dtype=np.uint8)

    morphed_mask = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel, 1)
    morphed_mask = cv2.dilate(morphed_mask, kernel, 1)
    
    return keep_largest_area_mask(morphed_mask)


def rgb_compare(im0, im1, open=5):
    (score, diff) = compare_ssim(im0, im1, full=True, gaussian_weights=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    kernel = np.ones((open, open), dtype=np.uint8)
    
    morphed_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 1)
    morphed_mask = cv2.dilate(morphed_mask, kernel, 1)

    return keep_largest_area_mask(morphed_mask)


def sift_compare(im0, im1):

    keypoints1, descriptors1 = sift.detectAndCompute(im0, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im1, None)

    flann = cv2.FlannBasedMatcher()

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, _ = cv2.findHomography(matched_keypoints1, matched_keypoints2, cv2.RANSAC, 5.0)
        transformed_image1 = cv2.warpPerspective(im0, H, (im1.shape[1], im1.shape[0]))
    except:
        return np.zeros_like(im0, dtype=np.uint8)

    _, binary_image1 = cv2.threshold(transformed_image1, 127, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(im1, 127, 255, cv2.THRESH_BINARY)

    diff_image = cv2.absdiff(binary_image1, binary_image2)

    return keep_largest_area_mask(diff_image)


def depth_compare_np(im0, im1, thresh=10, open=5): #without OpenCV
    diff = np.abs(im0.astype(np.int16) - im1.astype(np.int16))

    binary_diff = np.where(diff > thresh, 255, 0).astype(np.uint8)
    
    kernel = np.ones((open, open), dtype=np.uint8)

    #morphed_mask = binary_opening(binary_diff, structure=kernel).astype(np.uint8)
    #morphed_mask = binary_dilation(morphed_mask, structure=kernel).astype(np.uint8) * 255

    morphed_mask = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel, 1)
    morphed_mask = cv2.dilate(morphed_mask, kernel, 1)
    
    return keep_largest_area_mask(morphed_mask)
