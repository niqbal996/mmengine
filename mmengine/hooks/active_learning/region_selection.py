import math
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore

def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):

    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                score, purity, entropy = floating_region_score(output)

                score[active] = -float('inf')

                active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)

                for pixel in range(active_regions):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    active_start_w = w - active_radius if w - active_radius >= 0 else 0
                    active_start_h = h - active_radius if h - active_radius >= 0 else 0
                    active_end_w = w + active_radius + 1
                    active_end_h = h + active_radius + 1

                    mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    mask_end_w = w + mask_radius + 1
                    mask_end_h = h + mask_radius + 1

                    # mask out
                    score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    # active sampling
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()