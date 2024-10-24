import cv2
import os
import os.path as osp
import json
import datetime
import collections
import labelme
import pycocotools.mask as mk
import glob
import uuid
import numpy as np
import re


def get_order_files(source, order_id):
    detect_extra = 0
    index = 0
    rgb_list = []
    depth_list = []
    json_list = []
    mask_list = []

    try:
        while not detect_extra:
            current_dir = os.path.join(source, order_id+'_'+str(index))
            #depth_dir = os.path.join(current_dir, 'PickDetectorDepth', '_detectOnCam', 'trimmed_depth') # using trimmed depth
            #depth_list.append(glob.glob(os.path.join(depth_dir, '*.np'))[0])
            depth_dir = os.path.join(current_dir, 'PickDetectorDepth', 'detect', 'depth') # using original sensor output
            depth_list.append(glob.glob(os.path.join(depth_dir, '*.bin'))[0])
            rgb_dir = os.path.join(current_dir, 'PickDetectorDepth', 'detect', 'visible')
            rgb_list.append(glob.glob(os.path.join(rgb_dir, '*.png'))[0])
            json_list.append(glob.glob(os.path.join(current_dir, '*.json'))[0])
            mask_dir = os.path.join(current_dir, 'PickDetectorDepth', '_log_valid', 'segms_valid')
            mask_list.append(glob.glob(os.path.join(mask_dir, '*.npz'))[0])
            extra_path = os.path.join(current_dir, 'PickDetectorDepth', 'extra')
            index += 1
            next_path = os.path.join(source, order_id+'_'+str(index))
            if os.path.exists(extra_path) and not os.path.exists(next_path):
                sub_dirs = os.listdir(extra_path)
                for subdir in sub_dirs:
                    if os.path.isdir(os.path.join(extra_path, subdir)):
                        if 'depth' in subdir:
                            extra_depth_dir = os.path.join(extra_path, subdir)
                        elif 'visible' in subdir:
                            extra_rgb_dir = os.path.join(extra_path, subdir)

                depth_list.append(glob.glob(os.path.join(extra_depth_dir, '*.bin'))[0])
                rgb_list.append(glob.glob(os.path.join(extra_rgb_dir, '*.png'))[0])
                json_list.append(glob.glob(os.path.join(current_dir, '*.json'))[0])
                detect_extra = 1
    except:
        pass

    return rgb_list, depth_list, json_list, mask_list


def crop_tote(image, json_file):
    crop_area_pattern = r"crop_area=\[(\d+), (\d+), (\d+), (\d+)\]"

    data = json.load(open(json_file, 'r'))
    for i in range(len(data['events'])):
        if 'crop_area' in data['events'][i]['brief']:
            crop_area_match = re.search(crop_area_pattern, data['events'][i]['brief'])
            crop_area = [int(x) for x in crop_area_match.groups()]
            break
    output = image[crop_area[1]:crop_area[1]+crop_area[3],crop_area[0]:crop_area[0]+crop_area[2]]

    return output


def save_data(image, bbox, points, image_path, anns_path, ratio, fname):
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(anns_path, exist_ok=True)

    height, width = image.shape[:2]

    x1 = bbox[0][0]
    y1 = bbox[0][1]
    x2 = bbox[1][0]
    y2 = bbox[1][1]

    tx = max(0, int(x1-ratio*(x2-x1)))
    ty = max(0, int(y1-ratio*(y2-y1)))
    bx = min(int(x2+ratio*(x2-x1)), width)
    by = min(int(y2+ratio*(y2-y1)), height)

    img_name = image_path+'/'+fname+'_visible.png'
    template = image[ty:by, tx:bx]
    cv2.imwrite(img_name, template)

    json_filename = anns_path+'/'+fname+'_visible.json'

    mask = [[i[0]-tx, i[1]-ty] for i in points]

    annotation = {
            "version": '',
            "flags":{},
            "imagePath": fname+'_visible.png',
            "imageHeight": int(by-ty), 
            "imageWidth": int(bx-tx),
            "imageData": None,
            "shapes": [{"flags": {}, "group_id": None, "shape_type": "polygon", "label": 'complete', "points": [[i[0]-tx, i[1]-ty] for i in points]}]
        }
    
    with open(json_filename, 'w') as f:
        json.dump(annotation, f)

    return template, mask


def generate_coco(input_dir, output_dir, labels, jan_code):
    now = datetime.datetime.now()
    out_ann_file = osp.join(output_dir, f'annotations_rgb.json')

    if os.path.exists(out_ann_file):
        with open(out_ann_file, 'r') as data_file:
            coco_data = json.load(data_file)
        jan_list = coco_data.get('jan_codes', [])
    else:
        jan_list = []

    jan_list.append(jan_code)
    jan_list = list(set(jan_list))
    
    data = dict(
        jan_codes=jan_list,
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    label_files = glob.glob(osp.join(input_dir, '*.json'))
    image_id = -1
    annotation_id = -1
    skipping_files = []

    for label_file in label_files:
        with open(label_file) as f:
            label_data = json.load(f)

        img_fname = label_data['imagePath']

        current_image_ann = []
        masks = {}
        segmentations = collections.defaultdict(list)
        skip_this_img = False

        if not os.path.exists(output_dir + '/visible/' + img_fname):
            continue

        image_id += 1
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = 'polygon'
            try:
                mask = labelme.utils.shape_to_mask((label_data['imageHeight'], label_data['imageWidth']), points, shape_type)
            except:
                skip_this_img = True
                break

            if not group_id or group_id == '':
                group_id = uuid.uuid1()

            instance = (label, group_id)
            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        if skip_this_img:
            image_id -= 1
            continue

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if 'complete' in cls_name: cls_name = 'complete'
            if 'partial' in cls_name: cls_name = 'partial'
            if cls_name not in class_name_to_id:
                skip_this_img = True
                break
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = mk.encode(mask)
            area = float(mk.area(mask))
            bbox = mk.toBbox(mask).flatten().tolist()
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
                skipping_files.append(label_file)
                skip_this_img = True
                break
            
            annotation_id += 1
            current_image_ann.append(dict(
                id=annotation_id,
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

        if not skip_this_img:
            data['annotations'].extend(current_image_ann)
            data['images'].append(dict(
                license=0,
                url=None,
                file_name=img_fname,
                height=label_data['imageHeight'],
                width=label_data['imageWidth'],
                date_captured=None,
                id=image_id,
                ))
        else:
            image_id -= 1

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)

    return out_ann_file

