import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from tqdm import tqdm
import math
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import print_log
from glob import glob
from .active_learning import FloatingRegionScore

@HOOKS.register_module()
class RegionLabelHook(Hook):
    def __init__(self, label_budget_per_round=0.1, region_size=11, selection_mode='ratio', uncertainty_threshold=None):
        self.label_budget_per_round = label_budget_per_round  # e.g. 5 for 5%
        self.region_size = region_size
        self.selection_mode = selection_mode  # 'ratio' or 'threshold'
        self.uncertainty_threshold = uncertainty_threshold
        # TODO Update using the constructor arguments
        self.active_radius = 1
        self.mask_radius = self.active_radius * 2
        self.region_scorer = FloatingRegionScore(in_channels=3, size=2*self.active_radius+1).cuda()
        self.per_region_pixels = (2 * self.active_radius + 1) ** 2
        self.active_ratio = 0.022 / 1000

    def visualize_active_mask(self, active_mask, mask_path, n_labels=None):
        plt.ion()  # Enable interactive mode
        img_name = os.path.basename(mask_path)
        unique_labels = torch.unique(active_mask)
        if n_labels is None:
            n_labels = int(unique_labels.max().item()) + 1 if (unique_labels != 255).any() else 1
        # Use tab20 colormap for up to 20 classes, otherwise use hsv
        base_cmap = plt.get_cmap('tab20' if n_labels <= 20 else 'hsv', n_labels)
        color_list = [base_cmap(i) for i in range(n_labels)]
        # Set color for 255 (unlabeled) to grey
        color_list.append((0.5, 0.5, 0.5, 1.0))
        cmap = ListedColormap(color_list)
        bounds = list(range(n_labels)) + [256]
        norm = BoundaryNorm(bounds, cmap.N)
        # Prepare mask for display: set 255 to n_labels (last color)
        mask_np = active_mask.cpu().numpy() if hasattr(active_mask, 'cpu') else np.array(active_mask)
        mask_vis = np.where(mask_np == 255, n_labels, mask_np)
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_vis, cmap=cmap, norm=norm, interpolation='nearest')
        plt.title(f'Active Mask: {img_name}')
        plt.axis('off')
        plt.show()
        plt.pause(0.1)
        
    def get_all_active_regions_mask(self, scores, groundtruth_mask, active_mask, active_regions, active, selected):
        # scores: torch.Tensor (H, W)
        # groundtruth_mask: np.ndarray (H, W)
        # active_mask: np.ndarray (H, W)
        # active, selected: torch.BoolTensor (H, W)
        scores = scores.clone().detach()
        H, W = scores.shape
        region_radius = self.region_size // 2
        per_region_pixels = self.region_size ** 2

        scores[active] = -float('inf')
        for _ in range(active_regions):
            max_val = torch.max(scores)
            if max_val == -float('inf'):
                break
            h, w = (scores == max_val).nonzero(as_tuple=True)
            h, w = h[0].item(), w[0].item()
            h0 = max(h - region_radius, 0)
            h1 = min(h + region_radius + 1, H)
            w0 = max(w - region_radius, 0)
            w1 = min(w + region_radius + 1, W)
            # Label region
            active_mask[h0:h1, w0:w1] = groundtruth_mask[h0:h1, w0:w1].cpu()
            active[h0:h1, w0:w1] = True
            selected[h0:h1, w0:w1] = True
            scores[h0:h1, w0:w1] = -float('inf')
        return active_mask

    def get_regions_above_threshold(self, scores, groundtruth_mask, active_mask, threshold, active, selected, max_regions=None):
        scores = scores.clone().detach()
        H, W = scores.shape
        region_radius = self.region_size // 2
        count = 0
        while True:
            max_val = torch.max(scores)
            if max_val < threshold or max_val == -float('inf'):
                break
            h, w = (scores == max_val).nonzero(as_tuple=True)
            h, w = h[0].item(), w[0].item()
            h0 = max(h - region_radius, 0)
            h1 = min(h + region_radius + 1, H)
            w0 = max(w - region_radius, 0)
            w1 = min(w + region_radius + 1, W)
            active_mask[h0:h1, w0:w1] = groundtruth_mask[h0:h1, w0:w1]
            active[h0:h1, w0:w1] = True
            selected[h0:h1, w0:w1] = True
            scores[h0:h1, w0:w1] = -float('inf')
            count += 1
            if max_regions is not None and count >= max_regions:
                break
        return active_mask

    def init_masks(self, dataloader, mask_dir, indicator_dir):
        # Initialize masks for the dataset
        for data in tqdm(dataloader, total=len(dataloader), desc="Initializing active masks", unit='mask(s)'):
            img_name = os.path.basename(data['data_samples'][0].img_path)
            mask_path = os.path.join(mask_dir, img_name)
            indicator_path = os.path.join(indicator_dir, img_name)[:-4]+'.pth'
            h, w = data['data_samples'][0].img_shape
            active_mask = np.ones((h, w), dtype=np.uint8) * 255
            active_mask = Image.fromarray(active_mask)
            active_mask.save(mask_path)
            indicator = {
                'active': torch.tensor([0], dtype=torch.bool),
                'selected': torch.tensor([0], dtype=torch.bool),
            }
            torch.save(indicator, indicator_path)

    def region_selection(self,
                         score, 
                         gt_mask,
                         active_mask,
                         active_regions,
                         active,
                         selected):
        
        for pixel in range(active_regions):
            values, indices_h = torch.max(score, dim=0)
            _, indices_w = torch.max(values, dim=0)
            w = indices_w.item()
            h = indices_h[w].item()

            active_start_w = w - self.active_radius if w - self.active_radius >= 0 else 0
            active_start_h = h - self.active_radius if h - self.active_radius >= 0 else 0
            active_end_w = w + self.active_radius + 1
            active_end_h = h + self.active_radius + 1

            mask_start_w = w - self.mask_radius if w - self.mask_radius >= 0 else 0
            mask_start_h = h - self.mask_radius if h - self.mask_radius >= 0 else 0
            mask_end_w = w + self.mask_radius + 1
            mask_end_h = h + self.mask_radius + 1

            # mask out
            score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
            active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
            selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
            # active sampling
            active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                gt_mask[active_start_h:active_end_h, active_start_w:active_end_w]

        return score, active_mask, active, selected

    def after_region_label(self, runner):
        model = runner.model
        model.eval()
        target_loader = getattr(runner.train_loop, 'dataloader_active', None)
        dataset = target_loader.dataset
        if target_loader is None:
            raise AttributeError("train_loop does not have a 'target_dataloader_iterator'.")
        # Find the original label directory and create the new one
        orig_label_dir = dataset.data_prefix['seg_map_path']
        budget_int = int(runner.train_loop.label_budget * 100) if runner.train_loop.label_budget < 1.0 else int(runner.train_loop.label_budget)
        transforms = runner._train_loop.dataloader_active.dataset.pipeline.transforms
        idx = next(
            (i for i, t in enumerate(transforms) if t.__class__.__name__ == "LoadActiveMask"),
            None
        )
        new_label_dir = os.path.join(os.path.dirname(orig_label_dir), 
                                         runner._train_loop.dataloader_active.dataset.pipeline.transforms[idx].active_mask_path)
        new_indicator_dir = os.path.join(os.path.dirname(orig_label_dir), 
                                        runner._train_loop.dataloader_active.dataset.pipeline.transforms[idx].active_indicator_path)
        state_path = os.path.join(os.path.dirname(new_label_dir), 'active_state.txt')
        # Check if active_state.txt exists and read the first line
        skip_init_masks = False
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                first_line = f.readline().strip()
                masks_list = glob(os.path.join(new_label_dir, '*.png'))
                if first_line == '1' and len(masks_list) == len(target_loader):
                    skip_init_masks = True
        if runner.train_loop.active_round == 0 and not skip_init_masks:
            os.makedirs(new_label_dir, exist_ok=True)
            os.makedirs(new_indicator_dir, exist_ok=True)
            self.init_masks(target_loader, new_label_dir, new_indicator_dir)
        else:
            print_log(f"Masks have already been initiated from previous run with zeros. Skipping!")

        # Update active learning state
        runner.train_loop.active_round += 1
        runner.train_loop.label_budget += self.label_budget_per_round
        print_log(f"Active learning round {runner.train_loop.active_round}, budget: {runner.train_loop.label_budget}", logger='current')
        
        # Write active learning state to a small file for cross-process sharing
        with open(state_path, 'w') as f:
            f.write(f"{runner.train_loop.active_round}\n{runner.train_loop.label_budget}\n")
        print_log(f"Wrote active state to {state_path}: round={runner.train_loop.active_round}, budget={runner.train_loop.label_budget}", logger='current')
        region_ratio = runner.train_loop.label_budget / 100.0
        # Simple iteration through dataset indices
        processed_count = 0
        file_count = len(target_loader)
        print_log(f"Processing {file_count} files for active labeling from the target dataset", logger='current')
        for data in tqdm(target_loader, total=file_count, desc="Generating active masks", unit='image(s)'):  # Limit for testing
        # for idx in tqdm(range(min(file_count, 100))):  # Limit for testing
            # TODO this wont work with a multibatch scenario
            gt_mask = data['data_samples'][0]._gt_sem_seg.data[0]
            gt_mask_path = data['data_samples'][0].seg_map_path
            active_mask = data['data_samples'][0]._active_mask.data[0]
            active_mask_path = data['data_samples'][0].active_mask_path
            active_selected = data['data_samples'][0]._active_selected.data[0]
            active_indicator = data['data_samples'][0]._active_indicator.data[0]
            active_indicator_path = data['data_samples'][0].active_indicator_path

            with torch.no_grad():
                output = model.test_step(data)[0]
                if hasattr(output, 'seg_logits'):
                    seg_logits = output.seg_logits.data
                # pred = output.pred_sem_seg.data
                
                # scores = torch.softmax(seg_logits, dim=0).max(dim=0)[0]
                # uncertainty = 1 - scores

            H, W = 1024, 1024
            # TODO there is no caching of active_mask from previous runs. Should the mask from previous run be taken or 
            # should it be recalculated based on current uncertainties from the model.

            if self.selection_mode == 'ratio':
                per_region_pixels = self.region_size ** 2
                total_pixels = H * W
                active_regions = int(region_ratio * total_pixels // per_region_pixels)
                active_mask, active, active_selected = self.get_all_active_regions_mask(
                    uncertainty, 
                    gt_mask, 
                    active_mask, 
                    active_regions, 
                    active, 
                    active_selected)
            elif self.selection_mode == 'region_selection':
                score, purity, entropy = self.region_scorer(seg_logits)
                score[active_indicator] = -float('inf')
                active_regions = math.ceil(H*W * self.active_ratio / self.per_region_pixels)
                score, active_mask, active_indicator, active_selected = self.region_selection(
                    score, 
                    gt_mask, 
                    active_mask, 
                    active_regions,
                    active_indicator, 
                    active_selected)
            elif self.selection_mode == 'pixel_selection':
                score, purity, entropy = self.region_scorer(seg_logits)
                score[active] = -float('inf')
                active_regions = math.ceil(H*W * self.active_ratio / self.per_region_pixels)
                active_mask, active, active_selected = self.region_selection(
                    score, 
                    gt_mask, 
                    active_mask, 
                    active_regions, 
                    active, 
                    active_selected)
            else:
                active_mask, active, active_selected = self.get_regions_above_threshold(
                    uncertainty, 
                    gt_mask, 
                    active_mask, 
                    self.uncertainty_threshold, 
                    active, 
                    active_selected)

            # Save the new mask
            # img_name = os.path.basename(active_mask_path)
            # save_path = os.path.join(new_label_dir, img_name)
            Image.fromarray(active_mask.numpy()).save(active_mask_path)
            indicator = {
                'active': active_indicator,
                'selected': active_selected,
            }
            torch.save(indicator, active_indicator_path)

        # print_log(f"Finished processing {file_count} files, saved to {new_label_dir}", logger='current')