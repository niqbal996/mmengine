import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class RegionLabelHook(Hook):
    def __init__(self, label_budget=5, region_size=11, selection_mode='ratio', uncertainty_threshold=None):
        self.label_budget = label_budget  # e.g. 5 for 5%
        self.region_size = region_size
        self.selection_mode = selection_mode  # 'ratio' or 'threshold'
        self.uncertainty_threshold = uncertainty_threshold

    def visualize_active_mask(self, active_mask, mask_path, n_labels=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
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

    def after_region_label(self, runner):
        model = runner.model
        model.eval()
        device = next(model.parameters()).device
        target_loader = getattr(runner.train_loop, 'target_dataloader_iterator', None)
        dataset = getattr(runner.train_loop, 'dataloader_target', None).dataset
        if target_loader is None:
            raise AttributeError("train_loop does not have a 'target_dataloader_iterator'.")
        # Find the original label directory and create the new one
        orig_label_dir = dataset.data_prefix['seg_map_path']
        new_label_dir = os.path.join(os.path.dirname(orig_label_dir), f'semantics_{self.label_budget}')
        os.makedirs(new_label_dir, exist_ok=True)

        region_ratio = self.label_budget / 100.0

        for data in target_loader:
            # img = data['inputs'][0].to(device).unsqueeze(0)
            img_path = data['data_samples'][0].img_path
            mask_path = data['data_samples'][0].seg_map_path
            gt_mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            with torch.no_grad():
                output = model.test_step(data)[0]
                if hasattr(output, 'seg_logits'):
                    seg_logits = output.seg_logits.data
                pred = output.pred_sem_seg.data
                scores = torch.softmax(seg_logits, dim=0).max(dim=0)[0]  # uncertainty: 1 - max prob
                uncertainty = 1 - scores

            H, W = uncertainty.shape
            active_mask = torch.full((H, W), 255, dtype=torch.uint8)
            active = torch.zeros((H, W), dtype=torch.bool, device=device)
            selected = torch.zeros((H, W), dtype=torch.bool, device=device)

            if self.selection_mode == 'ratio':
                per_region_pixels = self.region_size ** 2
                total_pixels = H * W
                active_regions = int(region_ratio * total_pixels // per_region_pixels)
                active_mask = self.get_all_active_regions_mask(
                    uncertainty, gt_mask, active_mask, active_regions, active, selected)
            else:
                active_mask = self.get_regions_above_threshold(
                    uncertainty, gt_mask, active_mask, self.uncertainty_threshold, active, selected)

            # Save the new mask
            img_name = os.path.basename(mask_path)
            save_path = os.path.join(new_label_dir, img_name)
            # Visualization: show active_mask with unique color per label, grey for 255
            self.visualize_active_mask(active_mask, mask_path)
            Image.fromarray(active_mask.numpy()).save(save_path)