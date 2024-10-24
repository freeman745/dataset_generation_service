import cv2
from modules.dataset.utils import crop_tote, save_data
from modules.image_compare.models import depth_compare, rgb_compare, depth_compare_np
from modules.image_compare.tools import  get_mask_bbox, mask_iou
from modules.dataset.merge import merge_image_online
import numpy as np
import json
import requests
import pickle
import blosc


def read_compressed_file(file_path):
    with open(file_path, "rb") as f:
        compressed_array = f.read()
    array = blosc.unpack_array(compressed_array)
    return array


def call_api(url, data):
    s_request = pickle.dumps(data)
    s_response = requests.post(url, data=s_request).content
    response = pickle.loads(s_response)

    return response


def call_ol_tm(template, whole, tm_url, ol_url, ol_thresh=0.0, tm_thresh=0.7):
    data = (
            "predict",
            {
                "arr": whole,
                "score_thr_multiplier": ol_thresh,
            },
            )
    response = call_api(ol_url, data)
    bboxes = [box.tolist() for box in response['bboxes']]
    masks =  response['segms']

    data = (
            {
                "image": whole,
                "template": template,
                "bboxes": bboxes,
                "masks": masks,
                "thresh": tm_thresh
            },
            )
    response = call_api(tm_url, data)

    return response['bboxes'], response['masks']


def run(date, order_id, rgb_list, depth_list, json_list, mask_list, config, save_flag):
    iou_list = []
    rgb_im = cv2.imread(rgb_list[0])
    im0 = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2GRAY)
    im0 = crop_tote(im0, json_list[0])
    rgb_im = crop_tote(rgb_im, json_list[0])
    ''' # using trimmed depth
    depth0 = np.load(depth_list[0])
    depth0 = cv2.normalize(depth0, None, 0, 255, cv2.NORM_MINMAX)
    depth0 = depth0.astype(np.uint8)
    '''
    depth0 = read_compressed_file(depth_list[0])
    depth0 = depth0[:, :, -1]
    depth0 = crop_tote(depth0, json_list[0])

    img4merge = []
    mask4merge = []

    bg = cv2.imread(config.bg)

    for i in range(1, len(rgb_list)):
        imh, imw = im0.shape[:2]
        ol_detail = json.load(open(json_list[i-1], 'r'))
        jan = ol_detail['jan']
        npz = mask_list[i-1]
        im1 = cv2.imread(rgb_list[i], 0)
        im1 = crop_tote(im1, json_list[i])
        ''' # using trimmed depth
        depth1 = np.load(depth_list[i])
        depth1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX)
        depth1 = depth1.astype(np.uint8)
        '''
        depth1 = read_compressed_file(depth_list[i])
        depth1 = depth1[:, :, -1]
        depth1 = crop_tote(depth1, json_list[i])

        rgb_box, ori_points_rgb, ori_map_rgb, depth_box, ori_points_depth, ori_map_depth = predict_one(im0, im1, depth0, depth1, config)

        masks = np.load(npz)
        bb_ordered = ol_detail['PickDetectorDepth']['bb_ordered'][0]

        t_iou = []

        for index, mask_key in enumerate(masks):
            try:
                ol_mask = masks[mask_key]
                ol_bbox = bb_ordered[index]['bb']
                scale = np.zeros((imh, imw), dtype=np.uint8)
                scale[ol_bbox[1]:ol_bbox[1]+ol_bbox[3],ol_bbox[0]:ol_bbox[0]+ol_bbox[2]] += ol_mask

                depth_iou = mask_iou(ori_map_depth, scale)
                rgb_iou = mask_iou(ori_map_rgb, scale)

                t_iou.append(max(depth_iou, rgb_iou))
            except:
                t_iou.append(1.0)
        
        iou_list.append(max(t_iou))

        if config.save != '0' and not save_flag:
            fname = str(date)+'_'+str(order_id)+'_'+str(jan)+'_'+str(i-1)
            template, mask = save_data(rgb_im, depth_box, ori_points_depth, config.image_path, config.anns_path, config.padding, fname)

            img4merge.append(template.copy())
            mask4merge.append(mask)

            if config.template_match:
                filted_bbox, filted_mask = call_ol_tm(template, rgb_im, config.tm, config.ol)

                for j in range(len(filted_bbox)):
                    bbox = filted_bbox[j]

                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                    crop = rgb_im[y1:y2, x1:x2]

                    img4merge.append(crop.copy())
                    mask4merge.append(filted_mask[j])

        im0 = im1
        depth0 =depth1
        rgb_im = cv2.imread(rgb_list[i])
        rgb_im = crop_tote(rgb_im, json_list[i])

    if config.save != '0':
        merge_image_online(img4merge, mask4merge, config.image_path, bg, fname, config.db_size)

    return iou_list


def predict_one(rgb0, rgb1, depth0, depth1, config):
    if depth0.shape != depth1.shape:
        #new_size = (max(depth0.shape[1], depth1.shape[1]), max(depth0.shape[0], depth1.shape[0]))
        new_size = (depth0.shape[1], depth0.shape[0])
        depth1 = cv2.resize(depth1, new_size)

    if rgb0.shape != rgb1.shape:
        #new_size = (max(im0.shape[1], im1.shape[1]), max(im0.shape[0], im1.shape[0]))
        new_size = (rgb0.shape[1], rgb0.shape[0])
        rgb1 = cv2.resize(rgb1, new_size)

    rgb_diff = rgb_compare(rgb0, rgb1, config.rgb_open_thresh)
    #depth_diff = depth_compare(r_depth0, r_depth1, config.binary_thresh, config.depth_open_thresh) # using trimmed depth
    depth_diff = depth_compare_np(depth0, depth1, config.binary_thresh, config.depth_open_thresh)

    rgb_box, ori_points_rgb, ori_map_rgb = get_mask_bbox(rgb_diff)
    depth_box, ori_points_depth, ori_map_depth = get_mask_bbox(depth_diff)

    return rgb_box, ori_points_rgb, ori_map_rgb, depth_box, ori_points_depth, ori_map_depth
