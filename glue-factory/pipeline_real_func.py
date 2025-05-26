import numpy as np
import cv2
import torch
import json
from pathlib import Path
from skimage.morphology import skeletonize
from gluefactory.utils.tensor import batch_to_device
from omegaconf import OmegaConf
from gluefactory.utils.vessel_projection import (cta_2_dsa_2d_wbex, metrics_mse,
                                                 metrics_l2_norm_angle_difference, optimization_by_matching_points,)
from gluefactory.datasets.augmentations import IdentityAugmentation
from gluefactory.models import get_model


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
        half_box_size = np.ceil(radius).astype(int)
    x_center = x - np.floor(x)
    y_center = y - np.floor(y)
    xx = np.arange(-half_box_size, half_box_size + 1)
    yy = np.arange(-half_box_size, half_box_size + 1)
    xx, yy = np.meshgrid(xx, yy, indexing="ij")
    x_index, y_index, l_index = (
        np.where(((xx[:, :, None] - x_center) ** 2 + (yy[:, :, None]-y_center) ** 2) < radius ** 2))  # H x W x m
    x_index_final = xx[x_index, y_index] + np.around(x[l_index])
    y_index_final = yy[x_index, y_index] + np.around(y[l_index])
    return x_index_final.astype(int), y_index_final.astype(int)


def generate_2d_image_parreral(points_position: np.array, radius: np.array, image_size: np.array):
    """
    :param points_position: m x 2 (x->w, y->h)
    :param radius: m x 1
    :param image_size: 2 x 1; case: (H, W)
    :return: image array
    """
    image = np.zeros(image_size)
    x_index, y_index = caculate_circle_parrallel(points_position[:, 1], points_position[:, 0], radius=radius,
                                                 half_box_size=np.ceil(max(radius)))
    filter = (x_index >= 0) & (x_index < image_size[0]) & (y_index >= 0) & (y_index < image_size[1])
    image[x_index[filter], y_index[filter]] = 1
    return image


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def read_json(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)


def vessel_bound(points, h, w, side=2):
    min_h, max_h, min_w, max_w = (np.floor(min(0, points[:, 1].min() - side)).astype(np.int16),
                                  np.ceil(max(h, points[:, 1].max()) + side).astype(np.int16),
                                  np.floor(min(0, points[:, 0].min()) - side).astype(np.int16),
                                  np.ceil(max(w, points[:, 0].max()) + side).astype(np.int16))
    return min_h, max_h, min_w, max_w


def case_evaluation(cta_3d_points, dsa, dsa_lm, lm_index, radius, target_params, grid_params, cta_sample_num,
                    dsa_sample_num, model, cuda_id,):
    """
    :param cta_3d_points:
    :param label:
    :param radius: in 3D
    :param target_params:
    :param grid_params:
    :param lm_index: lm_index in 3d points
    :param model: superglue+lightglue model
    :return:
    """

    grid, radius = cta_2_dsa_2d_wbex(cta_3d_points, grid_params, radius)

    # lm located target and grid | for input
    h, w = int(target_params["DSAImagerSize"][0]), int(target_params["DSAImagerSize"][0])
    cl_keypoints1 = grid + dsa_lm - grid[lm_index]
    min_h, max_h, min_w, max_w = vessel_bound(points=cl_keypoints1, h=h, w=w, side=2)
    after_padding_size = max_h - min_h, max_w - min_w

    """
    cta (grid): padding -> plot image -> resize_2_224
    dsa (target): padding -> plot image -> skeletonize and sample -> resize_2_224
    """

    input_size = (224, 224)

    cl_keypoints1 = cl_keypoints1 - np.array([min_w, min_h])
    cl_keypoints1_o = cl_keypoints1.copy()
    lm_position = dsa_lm

    # TODO: Different Rigid Transformation should led to different projected radius
    image0 = dsa
    image0 = cv2.copyMakeBorder(image0, -min_h, max_h - h, -min_w, max_w - w, cv2.BORDER_CONSTANT, value=0)
    image1 = generate_2d_image_parreral(points_position=cl_keypoints1, radius=radius,
                                        image_size=after_padding_size)
    # skeletonize and sample
    cl2d_keypoints0 = binary_image_2_points(image0, sample_num=dsa_sample_num)
    # TODO:pixel和物理距离的转换
    cl2d_keypoints0_512 = cl2d_keypoints0.copy() + np.array([min_w, min_h]) # 2d skeleton in original projection

    # resize image, points
    image0 = cv2.resize(image0, input_size)
    image1 = cv2.resize(image1, input_size)
    scale_factor = np.array(input_size) / np.array(after_padding_size)  # for h, w
    cl2d_keypoints0, cl_keypoints1 = cl2d_keypoints0 * scale_factor[::-1], cl_keypoints1 * scale_factor[::-1]
    cl_keypoints1_o = cl_keypoints1_o * scale_factor[::-1]

    # resample the cta points
    cl_keypoints1, sample_index = sample_points2(cl_keypoints1, num=cta_sample_num, return_idx=True)
    cl_keypoints1_3d = cta_3d_points[sample_index]

    img_to_tensor = IdentityAugmentation()
    # predict
    inputs = {
        # "view0": {"image": img_to_tensor(image0, return_tensor=True).unsqueeze(0),
        #           "image_size": torch.tensor(input_size).unsqueeze(0)},
        # "view1": {"image": img_to_tensor(image1, return_tensor=True).unsqueeze(0),
        #           "image_size": torch.tensor(input_size).unsqueeze(0)},
        "view0": {"image": torch.tensor(image0, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255,
                  "image_size": torch.tensor(input_size).unsqueeze(0)},
        "view1": {"image": torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
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
    mse_grid = metrics_mse(target_params, grid_params, cta_3d_points, lm_delta=dsa_lm - grid[lm_index])
    angle_distance_grid = metrics_l2_norm_angle_difference(target_params, grid_params)
    mse_optimize = metrics_mse(target_params, optimized_params, cta_3d_points)
    angle_distance_optimize = metrics_l2_norm_angle_difference(target_params, optimized_params)
    print(f"name: {case_name} "
          f"| mse (opt, grid): {mse_optimize, mse_grid}"
          f"| angle (opt, grid): {angle_distance_optimize, angle_distance_grid}"
    )
    pass

def get_mid_root_index(label, root_id=5):
    indices = np.where(label == root_id)[0]
    return indices[len(indices) // 2]


if __name__ == "__main__":
    """
    重写整个pipeline,主要是增加：
    （1）512->224, 224->512的转换
    （2）增加投影
    现在主要设计三个类，一个适用于在线训练（直接从3D数据开始），一个是用于离线生成（离线生成匹配数据），评估（仿真数据和真实数据）
    """

    import argparse
    parser = argparse.ArgumentParser(description="Run case evaluation")
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing the data files")
    parser.add_argument("--conf_path", type=str, default="", help="Path to the configuration file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the results")
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device ID to use")
    args = parser.parse_args()

    data_dir = args.data_dir
    conf_path = args.conf_path
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    data_dir = Path(data_dir)
    data_list = list(data_dir.glob("**/cta_info.npy"))

    conf = OmegaConf.load(conf_path)
    conf.model.matcher.filter_threshold = 0.0001  # 这里和仿真数据不太一样，取0.0001
    conf.model.matcher.width_confidence = -1
    conf.model.matcher.depth_confidence = -1
    conf.model.matcher.lax = True
    model = get_model(conf.model.name)(conf.model)
    cuda_id = args.cuda_id
    model = model.to(f"cuda:{cuda_id}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"], strict=False)
    model.eval()
    for i, cta_info_path in enumerate(data_list):
        # show the progress
        print(f"{i}/{len(data_list)}")
        # data_loading
        case_name = cta_info_path.parent.name
        cta_info = np.load(cta_info_path)
        dsa_img_path = cta_info_path.parent / "dsa_img.png"
        dsa_seg_path = cta_info_path.parent / "dsa_seg.png"
        target_path = cta_info_path.parent / "target_params.json"
        grid_path = cta_info_path.parent / "grid_params.json"
        dsa_img = read_image(dsa_img_path, grayscale=True)
        dsa_seg = read_image(dsa_seg_path, grayscale=True)
        cta_3d_points = cta_info['cta_3d_points']
        label = cta_info['cta_labels']
        radius = cta_info['cta_radius']  # in 512
        dsa_lm = cta_info['dsa_lm']

        target_params = read_json(str(target_path))
        grid_params = read_json(str(grid_path))
        radius_3d = radius * target_params["SOD"] / target_params["SID"] * target_params["DSAImagerSpacing"][0]
        dsa_sample_num = 1024
        cta_sample_num = 1024
        root_id = 5
        lm_index = get_mid_root_index(label, root_id=root_id)

        case_evaluation(cta_3d_points=cta_3d_points, dsa=dsa_seg, dsa_lm=dsa_lm, lm_index=lm_index,
                        radius=radius_3d, target_params=target_params, grid_params=grid_params, model=model,
                        cuda_id=cuda_id, dsa_sample_num=dsa_sample_num,
                        cta_sample_num=cta_sample_num)
