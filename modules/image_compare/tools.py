import cv2
import numpy as np


def get_mask_bbox(diff):
    imh, imw = diff.shape[:2]
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x0, y0, w, h = cv2.boundingRect(largest_contour)

    x1, y1 = x0 + w, y0 + h

    bbox = [[x0, y0], [x1, y1]]

    points = largest_contour.flatten().tolist()
    #mask on original image
    points = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
    #mask in bbox
    #points_crop = [[i[0]-x0, i[1]-y0] for i in points]

    #binary_map = np.zeros((h, w), dtype=np.uint8)
    #cv2.fillPoly(binary_map, np.array([points_crop], dtype=np.int32), 1)

    original_map = np.zeros((imh, imw), dtype=np.uint8)
    cv2.fillPoly(original_map, np.array([points], dtype=np.int32), 1)
    
    #return bbox, points_crop, binary_map, points, original_map
    return bbox, points, original_map


def mask_iou(mask0, mask1):#mask iou base on binary map
    area0 = mask0.sum()
    area1 = mask1.sum()
    inter = ((mask0+mask1)==2).sum()
    iou = inter / (area0+area1-inter)

    return iou

