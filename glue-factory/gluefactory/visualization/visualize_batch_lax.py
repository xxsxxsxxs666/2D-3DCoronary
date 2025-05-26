import torch
import numpy as np

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}


def make_match_figures_dict(data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in data_.keys():
        data_ = data_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    m0 = data["matches0"]
    gtm0 = data["gt_matches0"].long()

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        # todo: 这里会取到gt, -1的点，这个点是不应该被取到的
        correct = (gtm0[i][valid] == m0[i][valid])
        gt_kpm1 = kp1[i][gtm0[i][valid]].numpy()

        gt_unmatched_index = gtm0[i][valid] == -1
        gt_kpm1[gt_unmatched_index] = np.inf
        # convert gtmo to torch.int64
        # calculate distance between kpm0 and gt_kpm1 using torch
        distance = np.linalg.norm(kpm1 - gt_kpm1, axis=-1)
        correct = correct.numpy()
        # from IPython import embed; embed(colors="linux")
        # if distance is less than 5, then it is half correct, so we set it to 0.5
        # convert correct to correct_lax to float
        correct_lax = correct.copy().astype(np.float32)
        correct_lax[(distance < 2) & (~correct)] = 0.5
        # from IPython import embed; embed(colors="linux")
        # print(((distance < 3) & (~correct)).sum(), valid.sum(), (gtm0[i][valid] == m0[i][valid]).sum())


        if "heatmap0" in data.keys():
            heatmaps.append(
                [
                    torch.sigmoid(data["heatmap0"][i, 0]),
                    torch.sigmoid(data["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct_lax).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}


def lax_color_generator(m0, gtm0, kp0, kp1):
    valid = (m0 > -1) & (gtm0 >= -1)
    kpm0, kpm1 = kp0[valid], kp1[m0[valid]]

    # todo: 这里会取到gt, -1的点，这个点是不应该被取到的
    correct = (gtm0[valid] == m0[valid])
    gt_kpm1 = kp1[gtm0[valid]]

    gt_unmatched_index = gtm0[valid] == -1
    gt_kpm1[gt_unmatched_index] = np.inf

    distance = np.linalg.norm(kpm1 - gt_kpm1, axis=-1)
    correct = correct

    correct_lax = correct.copy().astype(np.float32)
    correct_lax[(distance < 2) & (~correct)] = 0.5

    return cm_RdGn(correct_lax), correct_lax

