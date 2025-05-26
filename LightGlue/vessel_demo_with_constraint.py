# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path
import json

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor_type = "superpoint"
if extractor_type == "superpoint":
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
elif extractor_type == "sift":
    extractor = SIFT(max_num_keypoints=2048).eval().to(device)  # load the extractor
elif extractor_type == "disk":
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
else:
    raise ValueError("Unknown extractor type")

matcher = LightGlue(features=extractor_type).eval().to(device)

# paired images are in H:\CTA2DSA\LightGlue\paired_data\CCTA_segmentation_projection and H:\CTA2DSA\LightGlue\paired_data\DSA_alg_mask_png
# 这两个文件夹中每个文件前9位是一样的，所以可以直接对应
# 迭代读取两个文件夹中的文件，然后进行匹配

CCTA_projection_path = Path("H:/CTA2DSA/LightGlue/new_paired_data/cta_image")
DSA_mask_path = Path("H:/CTA2DSA/LightGlue/new_paired_data/dsa_image")
lm_position_path = Path("H:/CTA2DSA/LightGlue/new_paired_data/dsa_lm")
save_path = Path(f"H:/CTA2DSA/LightGlue/output/new_vessel_filtered_{extractor_type}/100_100")
save_path.mkdir(parents=True, exist_ok=True)


def distance_matrix(kp1, kp2):
    """
    Calculate the pairwise Euclidean distance between two sets of keypoints.

    Args:
    kp1: torch.Tensor of shape (B, M, 2) representing keypoints in one set
    kp2: torch.Tensor of shape (B, N, 2) representing keypoints in another set

    Returns:
    A dictionary containing a tensor of distances of shape (B, M, N)
    """
    # Ensure the input tensors are floats for precise distance calculation
    kp1 = kp1.float()
    kp2 = kp2.float()

    # Get the squared differences in dimensions between each pair of keypoints
    # We expand kp1 to (B, M, 1, 2) and kp2 to (B, 1, N, 2) to enable broadcasting
    diff = kp1.unsqueeze(2) - kp2.unsqueeze(1)

    # Compute the squared Euclidean distances (sum of squared differences along the last dimension)
    dist_squared = torch.sum(diff ** 2, dim=-1)

    # Compute the Euclidean distance by taking the square root of the sum of squared distances
    distance = torch.sqrt(dist_squared)

    # Return the result as a dictionary
    return distance


def distance_filter_index(distance_map, dt):
    """
    Args:
    kp1: torch.Tensor of shape (B, M, 2)
    kp2: torch.Tensor of shape (B, N, 2)
    distance_map: distances of shape (B, M, N)
    dt: distance_threshold
    """
    map_index = distance_map < dt
    index_1 = map_index.sum(dim=2) > 0
    index_2 = map_index.sum(dim=1) > 0

    return index_1, index_2


def multi_distance_matrix(kp1, kp2, o):
    """
    kp1: B x M x 2
    kp2: B x N x 2
    o: 2
    Output: {'disdance metric kp1_2_kp2': B x M x N, 'distance metric kp1_o': B x M, 'distance metric kp2_o': B x N
    'angle metric kp1_o_angle': B x M, 'angle metric kp1_o_angle': B x N}
    the angle is
    """
    pass


for file in CCTA_projection_path.iterdir():
    if file.suffix == ".png":
        img1 = load_image(file)
        img0 = load_image(DSA_mask_path / file.name)

        # load json file in lm position path
        # with open(lm_position_path / file.name.replace(".png", ".json"), "r") as f:
        #     lm_position = json.load(f)['dsa_lm'][:2]

        ## plot lm position in img1 using red color by matplotlib
        # x_numpy = img1.permute(1, 2, 0)
        # x, y = np.where((x_numpy[:, :, 1] > 0) * (x_numpy[:, :, 1] < 0.99))[0].mean(), \
        #        np.where((x_numpy[:, :, 1] > 0) * (x_numpy[:, :, 1] < 0.99))[1].mean()


        feats0 = extractor.extract(img0.to(device))
        feats1 = extractor.extract(img1.to(device))
        distance_map = distance_matrix(feats0['keypoints'], feats1['keypoints'])
        index0, index1 = distance_filter_index(distance_map, dt=img1.shape[-1]*1)
        feats0_filtered, feats1_filtered = feats0['keypoints'][~index0], feats1['keypoints'][~index1]
        feats0['keypoints'], feats1['keypoints'] = feats0['keypoints'][index0], feats1['keypoints'][index1]
        feats0['descriptors'], feats1['descriptors'] = feats0['descriptors'][index0], feats1['descriptors'][index1]

        viz2d.plot_images([img0, img1])
        viz2d.plot_keypoints([feats0['keypoints'], feats1['keypoints']], colors="blue", ps=10)
        viz2d.plot_keypoints([feats0_filtered, feats1_filtered], colors="red", ps=10)
        # plt.show()
        plt.gcf()
        plt.savefig(save_path / file.name.replace(".png", "initial_point.png"))

        feats0['keypoints'], feats1['keypoints'] = feats0['keypoints'].unsqueeze(0), feats1['keypoints'].unsqueeze(0)
        feats0['descriptors'], feats1['descriptors'] = feats0['descriptors'].unsqueeze(0), feats1['descriptors'].unsqueeze(0)

        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        # # distance_filter_2
        filter_index_2 = ((m_kpts0 - m_kpts1)**2).sum(axis=-1).sqrt() < img1.shape[-1]*1
        matches_remain = matches[filter_index_2]
        matches_filter = matches[~filter_index_2]
        m_kpts0_remain, m_kpts1_remain = kpts0[matches_remain[..., 0]], kpts1[matches_remain[..., 1]]
        m_kpts0_filter, m_kpts1_filter = kpts0[matches_filter[..., 0]], kpts1[matches_filter[..., 1]]

        # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        axes = viz2d.plot_images([img0, img1])
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2, dt=img1.shape[-1]*0.3)
        viz2d.plot_matches(m_kpts0_remain, m_kpts1_remain, color="lime", lw=0.2)
        viz2d.plot_matches(m_kpts0_filter, m_kpts1_filter, color="red", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers; \n patient_id: {file.name}', fs=20)
        # save the matched image
        # plt.show()
        plt.gcf()
        plt.savefig(save_path / file.name)

        # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        # viz2d.plot_images([img0, img1])
        # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        # plt.show()


# image0 = load_image(images / "DSC_0411.JPG")
# image1 = load_image(images / "DSC_0410.JPG")
#
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension
#
# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# plt.show()

