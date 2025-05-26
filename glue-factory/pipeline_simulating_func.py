import numpy as np
import cv2
import torch
import json
from pathlib import Path

# original glue-factory tool
from gluefactory.utils.tensor import batch_to_device
from gluefactory.datasets.augmentations import IdentityAugmentation
from gluefactory.models import get_model
from omegaconf import OmegaConf

# in vessel_projection.py
from gluefactory.utils.vessel_projection import (cta_2_dsa_2d_wbex, metrics_mse,
                                                 metrics_l2_norm_angle_difference, optimization_by_matching_points,
                                                 bbox_center_normalize)
from skimage.morphology import skeletonize


def read_json(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def sample_points2(points, num, return_idx=False):
    interval = (len(points) - 1) / (num - 1)
    idx = [int(np.floor(interval * i)) for i in range(num)]
    idx[-1] = min(idx[-1], len(points) - 1)
    if return_idx:
        return np.array(points[idx]), idx
    return np.array(points[idx])


def binary_image_2_points(img: np.array, sample_num: int = None):
    shape = img.shape
    if len(shape) == 3:
        sk = skeletonize((img / 255).sum(axis=-1) > 0)
    else:
        sk = skeletonize(img > 0)
    y, x = np.where(sk > 0)  # h, w -> y, x
    # change iy to num x 2
    points_coords = np.concatenate([x[:, None], y[:, None]], axis=1)
    return sample_points2(points_coords, num=sample_num) if sample_num is not None else points_coords


def caculate_circle_parrallel(x, y, radius, half_box_size=None):
    """
    :param x: float
    :param y: float
    :param radius: float
    :return: circle index
    """
    if half_box_size is None:
        half_box_size = np.ceil(max(radius)).astype(int)
    x_center = x - np.floor(x)
    y_center = y - np.floor(y)
    xx = np.arange(-half_box_size, half_box_size + 1)
    yy = np.arange(-half_box_size, half_box_size + 1)
    xx, yy = np.meshgrid(xx, yy)
    x_index, y_index, l_index = (
        np.where(((xx[:, :, None] - x_center) ** 2 + (yy[:, :, None]-y_center) ** 2) < radius ** 2))  # H x W x m
    x_index_final = xx[x_index, y_index] + np.floor(x[l_index])
    y_index_final = yy[x_index, y_index] + np.floor(y[l_index])
    return x_index_final.astype(int), y_index_final.astype(int)


def generate_2d_image_parreral(points_position: np.array, radius: np.array, image_size: np.array):
    """
    :param points_position: m x 2 (x->w, y->h)
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    x_index, y_index = caculate_circle_parrallel(points_position[:, 0], points_position[:, 1], radius=radius,
                                                 half_box_size=np.ceil(max(radius)))
    filter = (x_index >= 0) & (x_index < image_size[1]) & (y_index >= 0) & (y_index < image_size[0])
    image[y_index[filter], x_index[filter]] = 1
    return image


def vessel_bound(points, h, w, side=2):
    min_h, max_h, min_w, max_w = (np.floor(min(0, points[:, 1].min() - side)).astype(np.int16),
                                  np.ceil(max(h, points[:, 1].max()) + side).astype(np.int16),
                                  np.floor(min(0, points[:, 0].min()) - side).astype(np.int16),
                                  np.ceil(max(w, points[:, 0].max()) + side).astype(np.int16))
    return min_h, max_h, min_w, max_w


def case_evaluation(cta_3d_points, radius, target_params, grid_params, cut_index0, cut_index1, lm_index,
                    model, cuda_id, dsa_sample_num=1024, cta_sample_num=1024, case_name="case"):
    """
    :param cta_3d_points:
    :param label:
    :param radius: in 512
    :param target_params:
    :param grid_params:
    :param cut_index0: for dsa
    :param cut_index1: for cta
    :param lm_index: lm_index in 3d points
    :param model: superglue+lightglue model
    :return:
    """

    target, target_radius = cta_2_dsa_2d_wbex(cta_3d_points, target_params, radius)
    grid, grid_radius = cta_2_dsa_2d_wbex(cta_3d_points, grid_params, radius)

    # lm located target and grid | for input
    h, w = 512, 512
    cl_keypoints0, delta_coords = bbox_center_normalize(target, (h, w))
    cl_keypoints1 = grid + cl_keypoints0[lm_index] - grid[lm_index]
    # TODO: cl_keypoints1 should be in the same bbox as cl_keypoints0, need changed
    min_h_0, max_h_0, min_w_0, max_w_0 = vessel_bound(points=cl_keypoints0, h=h, w=w, side=2)
    min_h_1, max_h_1, min_w_1, max_w_1 = vessel_bound(points=cl_keypoints1, h=h, w=w, side=2)
    min_h, max_h, min_w, max_w = min(min_h_0, min_h_1), max(max_h_0, max_h_1), min(min_w_0, min_w_1), max(max_w_0, max_w_1)
    after_padding_size = max_h - min_h, max_w - min_w

    """
    cta (grid): padding -> plot image -> resize_2_224
    dsa (target): padding -> plot image -> skeletonize and sample -> resize_2_224
    """

    input_size = (224, 224)

    cl_keypoints0, cl_keypoints1 = cl_keypoints0 - np.array([min_w, min_h]), cl_keypoints1 - np.array(
        [min_w, min_h])

    # TODO: Different Rigid Transformation should led to different projected radius
    target_radius = target_radius
    grid_radius = grid_radius

    cl_keypoints0, cl_keypoints1 = cl_keypoints0[~cut_index0], cl_keypoints1[~cut_index1]
    image0 = generate_2d_image_parreral(points_position=cl_keypoints0, radius=target_radius[~cut_index0],
                                        image_size=after_padding_size)
    image1 = generate_2d_image_parreral(points_position=cl_keypoints1, radius=grid_radius[~cut_index1],
                                        image_size=after_padding_size)
    # skeletonize and sample
    cl2d_keypoints0 = binary_image_2_points(image0, sample_num=dsa_sample_num)
    cl2d_keypoints0_512 = cl2d_keypoints0.copy() + np.array(
        [min_w, min_h]) - delta_coords  # 2d skeleton in original projection
    # resize image, points
    image0 = cv2.resize(image0, input_size)
    image1 = cv2.resize(image1, input_size)
    scale_factor = np.array(input_size) / np.array(after_padding_size)  # for h, w
    cl2d_keypoints0, cl_keypoints1 = cl2d_keypoints0 * scale_factor[::-1], cl_keypoints1 * scale_factor[::-1]

    # resample the cta points
    cl_keypoints1, sample_index = sample_points2(cl_keypoints1, num=cta_sample_num, return_idx=True)
    cl_keypoints1_3d = cta_3d_points[~cut_index1][sample_index]

    img_to_tensor = IdentityAugmentation()
    # predict
    inputs = {
        "view0": {"image": img_to_tensor(image0, return_tensor=True).unsqueeze(0),
                  "image_size": torch.tensor(input_size).unsqueeze(0)},
        "view1": {"image": img_to_tensor(image1, return_tensor=True).unsqueeze(0),
                  "image_size": torch.tensor(input_size).unsqueeze(0)},
        "keypoints0": torch.tensor(cl2d_keypoints0, dtype=torch.float32).unsqueeze(0),
        "keypoints1": torch.tensor(cl_keypoints1, dtype=torch.float32).unsqueeze(0),
        "keypoints1_3d": torch.tensor(cl_keypoints1_3d, dtype=torch.float32).unsqueeze(0),
    }
    with torch.no_grad():
        inputs = batch_to_device(inputs, f"cuda:{cuda_id}")
        pred = model(inputs)
        pred = rbd(pred)
        pred = {k: v.cpu().numpy() for k, v in pred.items()}

    pred_matches0, pred_matches1 = pred["matches0"], pred["matches1"]
    valid_index = pred_matches0 > -1

    ## 估计投影参数
    optimized_params = optimization_by_matching_points(cl_keypoints1_3d[pred_matches0[valid_index]],
                                                       dsa_2d_points=cl2d_keypoints0_512[valid_index],
                                                       initial_params=grid_params)
    # using lm_delta to enhance grid performance
    lm_delta = grid[lm_index] - target[lm_index]
    mse_grid = metrics_mse(target_params, grid_params, cl_keypoints1_3d, lm_delta=lm_delta)
    mse_cut_grid = metrics_mse(target_params, grid_params, cta_3d_points[cut_index0])
    angle_distance_grid = metrics_l2_norm_angle_difference(target_params, grid_params)
    mse_optimize = metrics_mse(target_params, optimized_params, cl_keypoints1_3d)
    mse_cut_optimize = metrics_mse(target_params, optimized_params, cta_3d_points[cut_index0])
    angle_distance_optimize = metrics_l2_norm_angle_difference(target_params, optimized_params)
    print(f"name: {case_name} "
          f"| mse (opt, grid, cut_opt, cut_grid): {mse_optimize, mse_grid, mse_cut_optimize, mse_cut_grid}"
          f"| angle (opt, grid): {angle_distance_optimize, angle_distance_grid}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run case evaluation")
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing the data files")
    parser.add_argument("--conf_path", type=str, default="", help="Path to the configuration file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint")
    args = parser.parse_args()
    data_dir = args.data_dir
    conf_path = args.conf_path
    checkpoint_path = args.checkpoint_path
    data_list = data_dir.glob("*/*.npz")
    cuda_id = 3
    conf = OmegaConf.load(conf_path)
    conf.model.matcher.filter_threshold = 0.1
    conf.model.matcher.width_confidence = -1
    conf.model.matcher.depth_confidence = -1
    conf.model.matcher.lax = True
    model = get_model(conf.model.name)(conf.model)
    cuda_id = 3
    model = model.to(f"cuda:{cuda_id}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"], strict=False)
    model.eval()

    for data_path in data_list:
        d = np.load(data_path, allow_pickle=True)
        params = read_json(str(data_path.parent / "params.json"))
        cta_3d_points = d["cta_3d_points"]
        label = d["label"]
        radius = d["radius"]
        lm_index = d["lm_index"]
        cut_index0 = d["cut_index0"]
        cut_index1 = d["cut_index1"]
        target_params = params["target_params"]
        grid_params = params["grid_params"]
        radius_3d = radius * (target_params["SOD"] / target_params["SID"]) * target_params["DSAImagerSpacing"][0]
        grid, grid_radius = cta_2_dsa_2d_wbex(cta_3d_points, grid_params, radius=radius_3d)
        target, target_radius = cta_2_dsa_2d_wbex(cta_3d_points, target_params, radius=radius_3d)
        case_evaluation(cta_3d_points=cta_3d_points, radius=radius_3d, lm_index=lm_index, cut_index0=cut_index0,
                        cut_index1=cut_index1, target_params=target_params, grid_params=grid_params,  model=model,
                        cuda_id=cuda_id, dsa_sample_num=1024, cta_sample_num=1024,
                        case_name=data_path.name.split(".")[0])