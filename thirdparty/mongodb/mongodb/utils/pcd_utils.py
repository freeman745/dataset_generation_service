import numpy as np
import cv2


def get_pcd_of_obj_from_pmap(obj, pmap, area, area_num, down_sample=3.0, mask=None):
    """!
    @param[in] obj object dict returned by _find_object
    @param[in] depth depth image
    @param[in] area_num area number
    @param[in] down_smaple Down sample number for voxelize.
    @param[in] mask mask image
    @return Point cloud
    """
    # mask outer area of rectangle
    masked_depth = pmap.copy()
    masked_depth[:, :, 2] = add_rect_mask(
        pmap[:, :, 2], obj["rect"], pmap.shape[:2], area[area_num]
    )
    if mask is not None:
        masked_depth = add_bg_mask(masked_depth, area, area_num, mask)
        # plt.imshow(masked_depth[:,:,2].astype(int))
        # plt.show()
    # align bb to the depth image.
    bb = [
        obj["bb"][0] + area[area_num][0],
        obj["bb"][1] + area[area_num][1],
        obj["bb"][2],
        obj["bb"][3],
    ]
    return get_pcd_in_bb_from_pmap(bb, masked_depth, 1.0)


def add_bg_mask(img_in, area, area_num, bg_mask, too_deep_depth=3000):
    too_deep = np.full((bg_mask.shape[0], bg_mask.shape[1]), too_deep_depth)
    x_id = area[area_num][0]
    y_id = area[area_num][1]
    img_in[y_id : y_id + too_deep.shape[0], x_id : x_id + too_deep.shape[1], 2][
        bg_mask > 0
    ] = too_deep[bg_mask > 0]
    return img_in


def add_rect_mask(target, rect_param, shape2d, offset=(0, 0), too_deep_depth=3000):
    # mask outer area of rectangle
    mask = np.zeros(shape2d, dtype=np.uint8)
    too_deep = np.ones(shape2d, dtype=target.dtype) * too_deep_depth
    if rect_param[2] == 0:
        start_point = (
            int(rect_param[0][0] + offset[0]),
            int(rect_param[0][1] + offset[1]),
        )
        end_point = (
            int(start_point[0] + rect_param[1][0]),
            int(start_point[1] + rect_param[1][1]),
        )
        mask = cv2.rectangle(mask, start_point, end_point, 255, -1)
    else:
        rect = (
            (rect_param[0][0] + offset[0], rect_param[0][1] + offset[1]),
            rect_param[1],
            rect_param[2],
        )
        ## Note, this function seems to not work properly with non rotated rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    not_mask = cv2.bitwise_not(mask)
    too_deep = cv2.copyTo(too_deep, not_mask)
    masked_target = cv2.copyTo(target, mask)
    masked_target += too_deep
    return masked_target


def get_pcd_in_bb_from_pmap(bb, pmap, down_sample=1.0, depth_max=3000):
    """!
    get point cloud in boundary box from a point map
    @param[in] bb boundary box
    @param[in] pmap point map
    @param[in] down_sample pcd vox down sample size
    @return point_cloud
    """
    import open3d as o3d

    if bb != None and bb[-1] != -1:
        x = bb[0]
        y = bb[1]
        w = bb[2]
        h = bb[3]
        # trim bb area from depth image
        obj_pmap = pmap[y : y + h, x : x + w, :]
    else:
        obj_pmap = pmap
    obj_points_w_nan = obj_pmap.reshape(obj_pmap.shape[0] * obj_pmap.shape[1], 3)
    obj_points_w_fur = obj_points_w_nan[~np.isnan(obj_points_w_nan).any(axis=1)]
    obj_points = obj_points_w_fur[obj_points_w_fur[:, 2] < depth_max - 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points.astype(np.float64))
    pcd = pcd.voxel_down_sample(down_sample)
    return pcd
