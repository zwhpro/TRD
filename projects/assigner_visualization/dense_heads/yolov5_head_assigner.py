# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Union

import numpy as np
import torch
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.models import YOLOv5Head
from mmyolo.registry import MODELS

from mmyolo.models.losses import bbox_overlaps
torch.set_printoptions(profile="full")
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def bbox_overlaps(bboxes1, bboxes2, mode='smoothGD', eps=1e-6):
    assert mode in ['iou','ciou','giou','diou','wd', 'kl', 'exp_kl', 'kl_10','smoothGD'], f'Unsupported mode {mode}'
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    bbox1_x1, bbox1_y1 = bboxes1[..., 0], bboxes1[..., 1]
    bbox1_x2, bbox1_y2 = bboxes1[..., 2], bboxes1[..., 3]
    bbox2_x1, bbox2_y1 = bboxes2[..., 0], bboxes2[..., 1]
    bbox2_x2, bbox2_y2 = bboxes2[..., 2], bboxes2[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(bboxes1[..., :2], bboxes2[..., :2])
    enclose_x2y2 = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if mode == 'ciou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w ** 2 + enclose_h ** 2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2)) ** 2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2)) ** 2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))
    elif mode == 'giou':
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area
    elif mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw ** 2 + sigma_ch ** 2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w) ** 2  # ρ_x
        rho_y = (sigma_ch / enclose_h) ** 2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
                1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                               4) + torch.pow(
            1 - torch.exp(-1 * omiga_h), 4)

        ious = ious - ((distance_cost + shape_cost) * 0.5)
    elif mode == 'kl':
        center1 = (bboxes1[..., :2] + bboxes1[...,  2:]) / 2
        center2 = (bboxes2[...,:2] + bboxes2[...,  2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., 2] - bboxes1[...,  0] + eps
        h1 = bboxes1[...,  3] - bboxes1[..., 1] + eps
        w2 = bboxes2[..., 2] - bboxes2[...,  0] + eps
        h2 = bboxes2[...,  3] - bboxes2[..., 1] + eps

        kl = (w2 ** 2 / w1 ** 2 + h2 ** 2 / h1 ** 2 + 4 * whs[..., 0] ** 2 / w1 ** 2 + 4 * whs[
            ..., 1] ** 2 / h1 ** 2 + torch.log(w1 ** 2 / w2 ** 2) + torch.log(h1 ** 2 / h2 ** 2) - 2) / 2

        ious = 1 / (1 + kl)
    elif mode == 'kl_10':
        center1 = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
        center2 = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., 2] - bboxes1[..., 0] + eps
        h1 = bboxes1[..., 3] - bboxes1[..., 1] + eps
        w2 = bboxes2[..., 2] - bboxes2[..., 0] + eps
        h2 = bboxes2[..., 3] - bboxes2[..., 1] + eps

        kl = (w2 ** 2 / w1 ** 2 + h2 ** 2 / h1 ** 2 + 4 * whs[..., 0] ** 2 / w1 ** 2 + 4 * whs[
            ..., 1] ** 2 / h1 ** 2 + torch.log(w1 ** 2 / w2 ** 2) + torch.log(h1 ** 2 / h2 ** 2) - 2) / 2

        ious = 1 / (10 + kl)
    elif mode == 'exp_kl':
        center1 = (bboxes1[...,  :2] + bboxes1[..., 2:]) / 2
        center2 = (bboxes2[...,  :2] + bboxes2[..., 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[...,  2] - bboxes1[..., 0] + eps
        h1 = bboxes1[...,  3] - bboxes1[..., 1] + eps
        w2 = bboxes2[...,  2] - bboxes2[...,  0] + eps
        h2 = bboxes2[...,  3] - bboxes2[...,  1] + eps

        kl = (w2 ** 2 / w1 ** 2 + h2 ** 2 / h1 ** 2 + 4 * whs[..., 0] ** 2 / w1 ** 2 + 4 * whs[
            ..., 1] ** 2 / h1 ** 2 + torch.log(w1 ** 2 / w2 ** 2) + torch.log(h1 ** 2 / h2 ** 2) - 2) / 2

        ious = torch.exp(-kl / 10)
    elif mode == 'wd':
        center1 = (bboxes1[..., :2] + bboxes1[...,2:]) / 2
        center2 = (bboxes2[..., :2] + bboxes2[...,2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[...,  2] - bboxes1[..., 0] + eps
        h1 = bboxes1[..., 3] - bboxes1[..., 1] + eps
        w2 = bboxes2[..., 2] - bboxes2[..., 0] + eps
        h2 = bboxes2[..., 3] - bboxes2[..., 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        wasserstein = center_distance + wh_distance

        ious = torch.exp(-torch.sqrt(wasserstein)/12.8)
    elif mode == 'smoothGD':
        center1 = (bboxes1[..., :2] + bboxes1[..., 2:]) / 2
        center2 = (bboxes2[..., :2] + bboxes2[..., 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., 2] - bboxes1[..., 0] + eps
        h1 = bboxes1[..., 3] - bboxes1[..., 1] + eps
        w2 = bboxes2[..., 2] - bboxes2[..., 0] + eps
        h2 = bboxes2[..., 3] - bboxes2[..., 1] + eps

        kl = (w2 ** 2 / w1 ** 2 + h2 ** 2 / h1 ** 2 + 4 * whs[..., 0] ** 2 / w1 ** 2 + 4 * whs[
            ..., 1] ** 2 / h1 ** 2 + torch.log(w1 ** 2 / w2 ** 2) + torch.log(h1 ** 2 / h2 ** 2) - 2) / 2

        # 计算bbox1和bbox2的外接框的左上角和右下角坐标
        # 计算bbox1和bbox2的左上角和右下角坐标
        x1_min, y1_min = bboxes1[..., 0:1], bboxes1[..., 1:2]
        x1_max, y1_max = bboxes1[..., 2:3], bboxes1[..., 3:4]
        x2_min, y2_min = bboxes2[..., 0:1], bboxes2[..., 1:2]
        x2_max, y2_max = bboxes2[..., 2:3], bboxes2[..., 3:4]

        union_x1_min, union_y1_min = torch.min(x1_min, x2_min), torch.min(y1_min, y2_min)
        union_x2_max, union_y2_max = torch.max(x1_max, x2_max), torch.max(y1_max, y2_max)

        union_diagonal2 = (union_x2_max - union_x1_min) ** 2 + (union_y2_max - union_y1_min) ** 2
        # 计算bbox1和bbox2的中心点之间的距离
        center_distance2 = whs[..., 0] ** 2 + whs[..., 1] ** 2

        smoothGD = (1 / (1 + kl) - center_distance2 / union_diagonal2.squeeze(2) + 1) / 2
        # smoothGD = 1 / (1 + kl) - center_distance2 / union_diagonal2.squeeze(2)
        ious = smoothGD
    return ious
def G_MLA(t,anchors,mode="iou"):
    # -----------建模为高斯------------------------------
    # 将t_xywh转换为t_xyxy
    t_xyxy = xywh2xyxy(t[0, :, 2:6])
    t_xyxy = t_xyxy.repeat(len(anchors), 1, 1)
    t_center = (t_xyxy[..., :2] + t_xyxy[..., 2:]) / 2
    anchors_center = torch.floor(t_center)
    # anchors_center = t_center
    # 将anchors建模出中心位置anchors和t的中心坐标一样
    anchors_wh = []
    for i in range(len(anchors)):
        anchor = anchors[i].repeat(len(t_xyxy[1, :]), 1)
        anchors_wh.append(anchor)
    anchors_wh = torch.cat(anchors_wh).view(len(anchors_center), len(anchors_center[0, ...]), 2)
    anchors_xywh = torch.cat((anchors_center[..., :2], anchors_wh), dim=-1)
    # 将anchors_xywh转换为anchors_xyxy
    anchors_xyxy = []
    for anchor_xywh in anchors_xywh:
        anchor_xyxy = xywh2xyxy(anchor_xywh)
        anchors_xyxy.append(anchor_xyxy)

    # 将anchors_xyxy固定形状维3维4通道
    anchors_xyxy = torch.cat(anchors_xyxy).view(len(anchors_xywh), len(anchors_xywh[0, ...]), 4)
    kld = bbox_overlaps(t_xyxy, anchors_xyxy, mode=mode)
    # m每一个gt框匹配的三个锚的得分topk
    topk = 3
    kld_vals, kld_indices = kld.topk(k=topk, dim=0, largest=True, sorted=True)

    max_kld_flag = kld > kld_vals[2, :]
    threshold_kld_flag = kld > -1
    j = max_kld_flag & threshold_kld_flag

    return j
@MODELS.register_module()
class YOLOv5HeadAssigner(YOLOv5Head):

    def assign_by_gt_and_feat(
        self,
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        inputs_hw: Union[Tensor, tuple] = (640, 640)
    ) -> dict:
        """Calculate the assigning results based on the gt and features
        extracted by the detection head.

        Args:
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            inputs_hw (Union[Tensor, tuple]): Height and width of inputs size.
        Returns:
            dict[str, Tensor]: A dictionary of assigning results.
        """
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = batch_targets_normed.device
        scaled_factor = torch.ones(7, device=device)
        gt_inds = torch.arange(
            batch_targets_normed.shape[1],
            dtype=torch.long,
            device=device,
            requires_grad=False).unsqueeze(0).repeat((self.num_base_priors, 1))

        assign_results = []
        for i in range(self.num_levels):
            assign_results_feat = []
            h = inputs_hw[0] // self.featmap_strides[i]
            w = inputs_hw[1] // self.featmap_strides[i]

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                for k in range(self.num_base_priors):
                    assign_results_feat.append({
                        'stride':
                        self.featmap_strides[i],
                        'grid_x_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'grid_y_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'img_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'class_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'retained_gt_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'prior_ind':
                        k
                    })
                assign_results.append(assign_results_feat)
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor([w, h, w, h])
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            # wh_ratio = batch_targets_scaled[...,
            #                                 4:6] / priors_base_sizes_i[:, None]
            # match_inds = torch.max(
            #     wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            # 2. G-MLA
            match_inds = G_MLA(batch_targets_scaled,priors_base_sizes_i,'iou')

            batch_targets_scaled = batch_targets_scaled[match_inds]
            match_gt_inds = gt_inds[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                for k in range(self.num_base_priors):
                    assign_results_feat.append({
                        'stride':
                        self.featmap_strides[i],
                        'grid_x_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'grid_y_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'img_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'class_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'retained_gt_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'prior_ind':
                        k
                    })
                assign_results.append(assign_results_feat)
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            retained_gt_inds = match_gt_inds.repeat((5, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(
                -1), img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            for k in range(self.num_base_priors):
                retained_inds = priors_inds == k
                assign_results_prior = {
                    'stride': self.featmap_strides[i],
                    'grid_x_inds': grid_x_inds[retained_inds],
                    'grid_y_inds': grid_y_inds[retained_inds],
                    'img_inds': img_inds[retained_inds],
                    'class_inds': class_inds[retained_inds],
                    'retained_gt_inds': retained_gt_inds[retained_inds],
                    'prior_ind': k
                }
                assign_results_feat.append(assign_results_prior)
            assign_results.append(assign_results_feat)
        return assign_results

    def assign(self, batch_data_samples: Union[list, dict],
               inputs_hw: Union[tuple, torch.Size]) -> dict:
        """Calculate assigning results. This function is provided to the
        `assigner_visualization.py` script.

        Args:
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            inputs_hw: Height and width of inputs size

        Returns:
            dict: A dictionary of assigning components.
        """
        if isinstance(batch_data_samples, list):
            outputs = unpack_gt_instances(batch_data_samples)
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs

            assign_inputs = (batch_gt_instances, batch_img_metas,
                             batch_gt_instances_ignore, inputs_hw)
        else:
            # Fast version
            assign_inputs = (batch_data_samples['bboxes_labels'],
                             batch_data_samples['img_metas'], inputs_hw)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)

        return assign_results
