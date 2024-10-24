import os
import random
import numpy as np
import cv2
import json
import time


def get_png_filenames(directory):
    all_files = os.listdir(directory)
    png_files = [file for file in all_files if file.lower().endswith('.png')]
    return png_files


def get_random_images(image_list, mask_list, min_num=19, max_num=20):
    num_images = random.randint(min_num, max_num)
    chosen_indices = [random.randint(0, len(image_list) - 1) for _ in range(1, num_images)]
    random_image = [image_list[idx] for idx in chosen_indices]
    random_mask = [mask_list[idx] for idx in chosen_indices]
    return random_image, random_mask


def polygon_from_points(points):
    """transfer points to polygen"""
    return np.array(points, dtype=np.int32)


def check_overlap_and_adjust(poly1_points, poly2_points, w, h):
    """check overlap"""
    poly1 = polygon_from_points(poly1_points)
    poly2 = polygon_from_points(poly2_points)
    
    img1 = np.zeros((h, w), dtype=np.uint8)
    img2 = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(img1, [poly1], 255)
    cv2.fillPoly(img2, [poly2], 255)
    
    overlap = cv2.bitwise_and(img1, img2)
    
    if np.any(overlap):
        contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlap_contour = contours[0]
        
        poly2_adjusted = cv2.subtract(img2, overlap)
        
        contours, _ = cv2.findContours(poly2_adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            adjusted_points = [contour[:, 0, :].tolist() for contour in contours]
            return adjusted_points[0], 'partial'
        
    return poly2_points, 'complete'


def merge_image_from_file(in_img_root, out_img_root, db_size, bg_file):
    total_img_list = get_png_filenames(in_img_root)

    for i in range(db_size):

        bg = cv2.imread(bg_file)
        bgh, bgw = bg.shape[:2]
        random_imgs = get_random_images(total_img_list)

        mask_list = []
        shapes = []

        for img in random_imgs:
            label = 'complete'

            ob = cv2.imread(os.path.join(in_img_root, img))
            obh, obw = ob.shape[:2]

            x_offset = random.randint(0, bgw)
            y_offset = random.randint(0, bgh)

            if y_offset+obh > bgh or x_offset+obw > bgw:
                continue

            ann = os.path.join(in_img_root.replace('visible', 'anns'), img.replace('png', 'json'))

            d = json.load(open(ann, 'r'))
            mask_points = d['shapes'][0]['points']

            trans_mask_points = [[p[0]+x_offset, p[1]+y_offset] for p in mask_points]

            for m in mask_list:
                trans_mask_points, t_label = check_overlap_and_adjust(m, trans_mask_points, bgh, bgw)
                if t_label == 'partial':
                    label = t_label

            mask_list.append(trans_mask_points)

            mask_points = [[p[0]-x_offset, p[1]-y_offset] for p in trans_mask_points]

            mask = np.zeros((obh, obw), dtype=np.uint8)

            cv2.fillPoly(mask, [np.array(mask_points, dtype=np.int32)], 255)

            _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_binary)

            roi = bg[y_offset:y_offset+obh, x_offset:x_offset+obw]

            small_img_fg = cv2.bitwise_and(ob, ob, mask=mask_binary)

            large_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            result = cv2.add(large_img_bg, small_img_fg)

            bg[y_offset:y_offset+obh, x_offset:x_offset+obw] = result

            shapes.append({"flags": {}, "group_id": '', "shape_type": "polygon", "label": label, "points": trans_mask_points})
    
        if not mask_list:
            continue

        fname = str(int(time.time()*1000))

        cv2.imwrite(os.path.join(out_img_root, fname+'.png'), bg)

        annotation = {
            "version": '',
            "flags":{},
            "imagePath": fname+'.png',
            "imageHeight": bgh, 
            "imageWidth": bgw,
            "imageData": None,
            "shapes": shapes
        }

        with open(os.path.join(out_img_root.replace('visible', 'anns'), fname+'.json'), 'w') as f:
            json.dump(annotation, f)

    return 0


def merge_image_online(img_list, in_mask_list, out_img_root, bg_img, fname, db_size=10):

    for i in range(db_size):
        bg = bg_img.copy()
        bgh, bgw = bg.shape[:2]
        random_imgs, random_masks = get_random_images(img_list, in_mask_list, 3, 20)

        mask_list = []
        shapes = []

        for i in range(len(random_imgs)):
            label = 'complete'

            ob = random_imgs[i]
            obh, obw = ob.shape[:2]

            x_offset = random.randint(0, bgw)
            y_offset = random.randint(0, bgh)

            if y_offset+obh > bgh or x_offset+obw > bgw:
                continue

            mask_points = random_masks[i]

            trans_mask_points = [[p[0]+x_offset, p[1]+y_offset] for p in mask_points]

            for m in mask_list:
                trans_mask_points, t_label = check_overlap_and_adjust(m, trans_mask_points, bgh, bgw)
                if t_label == 'partial':
                    label = t_label

            mask_list.append(trans_mask_points)

            mask_points = [[p[0]-x_offset, p[1]-y_offset] for p in trans_mask_points]

            mask = np.zeros((obh, obw), dtype=np.uint8)

            cv2.fillPoly(mask, [np.array(mask_points, dtype=np.int32)], 255)

            _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask_binary)

            roi = bg[y_offset:y_offset+obh, x_offset:x_offset+obw]

            small_img_fg = cv2.bitwise_and(ob, ob, mask=mask_binary)

            large_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            result = cv2.add(large_img_bg, small_img_fg)

            bg[y_offset:y_offset+obh, x_offset:x_offset+obw] = result

            if type(trans_mask_points) != list:
                trans_mask_points = trans_mask_points.tolist()

            shapes.append({"flags": {}, "group_id": None, "shape_type": "polygon", "label": label, "points": trans_mask_points})
    
        if not mask_list:
            continue

        ft = str(int(time.time()*1000))

        cv2.imwrite(os.path.join(out_img_root, fname+'_'+ft+'_visible.png'), bg)

        annotation = {
            "version": '',
            "flags":{},
            "imagePath": fname+'_'+ft+'_visible.png',
            "imageHeight": bgh, 
            "imageWidth": bgw,
            "imageData": None,
            "shapes": shapes
        }

        with open(os.path.join(out_img_root.replace('visible', 'anns'), fname+'_'+ft+'_visible.json'), 'w') as f:
            json.dump(annotation, f)

    return 0

