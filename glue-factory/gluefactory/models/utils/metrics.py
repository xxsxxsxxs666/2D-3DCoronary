import torch


@torch.no_grad()
def matcher_metrics(pred, data, prefix="", prefix_gt=None, lax=False, lax_distance_threshold=2.0):
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()
        sort_ind = torch.argsort(-scores)
        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)
        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, -1) / (
            1e-8 + torch.cumsum(sorted_p_mask, -1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, -1) / (
            1e-8 + sorted_r_mask.sum(-1)[:, None]
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]
        return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    def lax_metric(m0, gtm0, kp0, kp1, distance_threshold=lax_distance_threshold):
        n_pairs = m0.shape[0]
        correct_number = torch.zeros(n_pairs, dtype=torch.float32, device=m0.device)
        valid_batch = ((m0 > -1) & (gtm0 > -1))

        # todo: vectorize this part
        for i in range(n_pairs):
            valid = valid_batch[i]
            correct = (gtm0[i][valid] == m0[i][valid])
            kpm1 = kp1[i][m0[i][valid]]
            gt_kpm1 = kp1[i][gtm0[i][valid]]
            distance = torch.linalg.norm(kpm1 - gt_kpm1, axis=-1)
            correct_number[i] = ((distance < distance_threshold) & (~correct)).sum().item() + correct.sum().item()

        # calculate the recall_lax
        mask = (gtm0 > -1).float()
        recall_lax = correct_number / mask.sum(1).float()

        # calculate the precision_lax
        mask = ((m0 > -1) & (gtm0 >= -1)).float()
        precision_lax = (correct_number) / mask.sum(1).float()

        # calculate the accuracy_lax
        mask = (gtm0 >= -1).float()
        accuracy_lax = (correct_number + ((gtm0 == -1) & (m0 == -1)).sum(1)) / mask.sum(1).float()

        return recall_lax, precision_lax, accuracy_lax


    if prefix_gt is None:
        prefix_gt = prefix
    rec = recall(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    prec = precision(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    acc = accuracy(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    ap = ranking_ap(
        pred[f"{prefix}matches0"],
        data[f"gt_{prefix_gt}matches0"],
        pred[f"{prefix}matching_scores0"],
    )
    metrics = {
        f"{prefix}match_recall": rec,
        f"{prefix}match_precision": prec,
        f"{prefix}accuracy": acc,
        f"{prefix}average_precision": ap,
    }
    if lax:
        rec_lax, prec_lax, acc_lax = lax_metric(
            pred[f"{prefix}matches0"],
            data[f"gt_{prefix_gt}matches0"].long(),
            data[f"keypoints0"],
            data[f"keypoints1"],
        )
        metrics[f"{prefix}recall_lax"] = rec_lax
        metrics[f"{prefix}precision_lax"] = prec_lax
        metrics[f"{prefix}accuracy_lax"] = acc_lax
    return metrics
