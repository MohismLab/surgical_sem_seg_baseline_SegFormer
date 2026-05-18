import json
import glob
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.runner import Hook

from .metrics import intersect_and_union



class ExtraMetricsHook(Hook):
    """Compute val_loss, train_mIoU, and val_HD95 every ``interval`` iters.

    Also records metric history and generates a 4-subplot training figure
    (matching SAM-LoRA style) at the end of training.

    Designed to be registered AFTER ``EvalHook`` with the same interval so that
    val_mIoU / val_mDice / val_loss / train_mIoU all land in the same JSON log
    entry.
    """

    def __init__(self,
                 val_loss_dataloader,
                 train_eval_dataloader,
                 interval,
                 n_val_samples=200,
                 n_train_samples=200,
                 num_classes=30,
                 ignore_index=255,
                 iters_per_epoch=8000):
        self.val_loss_dl = val_loss_dataloader
        self.train_eval_dl = train_eval_dataloader
        self.interval = interval
        self.n_val_samples = n_val_samples
        self.n_train_samples = n_train_samples
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.iters_per_epoch = iters_per_epoch

        # history for plotting (recorded at each eval interval)
        self.history = dict(
            iter=[], val_loss=[], train_mIoU=[], val_hd95=[])

    @staticmethod
    def _scalar_loss_sum(loss_dict):
        total = 0.0
        for k, v in loss_dict.items():
            if 'loss' not in k:
                continue
            if isinstance(v, (list, tuple)):
                total += sum(float(x) for x in v)
            elif torch.is_tensor(v):
                total += float(v.detach())
            else:
                total += float(v)
        return total

    def _compute_val_loss(self, runner):
        model = runner.model
        was_training = model.training
        model.train()
        total = 0.0
        n_seen = 0
        try:
            with torch.no_grad():
                for batch in self.val_loss_dl:
                    losses = model(return_loss=True, **batch)
                    bs = len(batch['img_metas'].data[0]) if hasattr(
                        batch['img_metas'], 'data') else len(batch['img_metas'])
                    total += self._scalar_loss_sum(losses) * bs
                    n_seen += bs
                    if n_seen >= self.n_val_samples:
                        break
        finally:
            if not was_training:
                model.eval()
        return total / max(n_seen, 1)

    def _compute_train_iou(self, runner):
        model = runner.model
        model.eval()
        total_int = np.zeros(self.num_classes, dtype=np.float64)
        total_uni = np.zeros(self.num_classes, dtype=np.float64)
        n_seen = 0
        ds = self.train_eval_dl.dataset
        with torch.no_grad():
            for batch in self.train_eval_dl:
                preds = model(return_loss=False, **batch)
                for pred in preds:
                    img_info = ds.img_infos[n_seen]
                    seg_map_path = osp.join(ds.ann_dir, img_info['ann']['seg_map'])
                    gt = mmcv.imread(seg_map_path, flag='unchanged', backend='pillow')
                    a_int, a_uni, _, _ = intersect_and_union(
                        pred, gt, self.num_classes, self.ignore_index)
                    total_int += a_int
                    total_uni += a_uni
                    n_seen += 1
                    if n_seen >= self.n_train_samples:
                        break
                if n_seen >= self.n_train_samples:
                    break
        iou = total_int / np.maximum(total_uni, 1e-12)
        return float(np.nanmean(iou))

    def _compute_val_hd95(self, runner):
        """Compute mean HD95 on a subset of val images."""
        model = runner.model
        model.eval()
        hd95_list = []
        n_seen = 0
        ds = self.train_eval_dl.dataset  # reuse train_eval for structure
        # Use val_loss_dl's dataset for val images
        val_ds = self.val_loss_dl.dataset
        with torch.no_grad():
            for batch in self.val_loss_dl:
                # Get predictions: need to run in test mode
                # We use the gt from batch directly
                img = batch['img'].data[0] if hasattr(batch['img'], 'data') else batch['img']
                img_meta = batch['img_metas'].data[0] if hasattr(
                    batch['img_metas'], 'data') else batch['img_metas']
                gt_seg = batch['gt_semantic_seg'].data[0] if hasattr(
                    batch['gt_semantic_seg'], 'data') else batch['gt_semantic_seg']

                # Forward pass to get prediction
                with torch.no_grad():
                    model.eval()
                    seg_logit = model.module.encode_decode(
                        img.cuda(), img_meta) if hasattr(model, 'module') \
                        else model.encode_decode(img.cuda(), img_meta)
                    pred = seg_logit.argmax(dim=1).cpu().numpy()

                for i in range(pred.shape[0]):
                    gt_i = gt_seg[i].squeeze().numpy()
                    pred_i = pred[i]
                    # Resize pred to gt size if needed
                    if pred_i.shape != gt_i.shape:
                        from PIL import Image
                        pred_i = np.array(Image.fromarray(pred_i.astype(np.uint8)).resize(
                            (gt_i.shape[1], gt_i.shape[0]), Image.NEAREST))

                    hd95_val = self._per_image_hd95(pred_i, gt_i)
                    if not np.isnan(hd95_val):
                        hd95_list.append(hd95_val)
                    n_seen += 1
                    if n_seen >= self.n_val_samples:
                        break
                if n_seen >= self.n_val_samples:
                    break

        return float(np.mean(hd95_list)) if hd95_list else float('nan')

    def _per_image_hd95(self, pred, gt):
        """Compute mean HD95 across all foreground classes in a single image."""
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion
        except ImportError:
            return float('nan')

        hd95_per_class = []
        for cls in range(1, self.num_classes):  # skip background
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
            if not pred_mask.any() and not gt_mask.any():
                continue  # class not present
            if not pred_mask.any() or not gt_mask.any():
                continue  # one side empty, skip (like SAM-LoRA behavior)

            dt_gt = distance_transform_edt(~gt_mask)
            dt_pred = distance_transform_edt(~pred_mask)

            struct = np.ones((3, 3), dtype=bool)
            surf_pred = pred_mask ^ binary_erosion(pred_mask, structure=struct, border_value=0)
            surf_gt = gt_mask ^ binary_erosion(gt_mask, structure=struct, border_value=0)
            if not surf_pred.any() or not surf_gt.any():
                continue

            d_pred2gt = dt_gt[surf_pred]
            d_gt2pred = dt_pred[surf_gt]
            all_d = np.concatenate([d_pred2gt, d_gt2pred])
            hd95_per_class.append(np.percentile(all_d, 95))

        return float(np.mean(hd95_per_class)) if hd95_per_class else float('nan')

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        cur_iter = runner.iter + 1
        runner.logger.info(f'[ExtraMetrics] Computing extra metrics @ iter {cur_iter} ...')

        val_loss = self._compute_val_loss(runner)
        train_iou = self._compute_train_iou(runner)
        val_hd95 = self._compute_val_hd95(runner)
        runner.model.train()

        # Record history
        self.history['iter'].append(cur_iter)
        self.history['val_loss'].append(val_loss)
        self.history['train_mIoU'].append(train_iou)
        self.history['val_hd95'].append(val_hd95)

        is_main = (not hasattr(runner, 'rank')) or runner.rank == 0
        if is_main:
            runner.log_buffer.output['val_loss'] = val_loss
            runner.log_buffer.output['train_mIoU'] = train_iou
            runner.log_buffer.output['val_hd95'] = val_hd95
            runner.log_buffer.ready = True
            runner.logger.info(
                f'[ExtraMetrics] iter {cur_iter}: '
                f'val_loss={val_loss:.4f} train_mIoU={train_iou:.4f} '
                f'val_hd95={val_hd95:.2f}')

    def after_run(self, runner):
        """Generate the 4-subplot training metrics figure after training."""
        try:
            self._plot_metrics(runner.work_dir)
        except Exception as e:
            runner.logger.warning(f'[ExtraMetrics] Failed to plot: {e}')

    def _iter_to_epoch(self, iters):
        """Convert iteration numbers to epoch numbers."""
        return [it / self.iters_per_epoch for it in iters]

    def _plot_metrics(self, work_dir):
        """Generate 4-subplot figure matching SAM-LoRA training_metrics style."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Parse the JSON log for standard metrics
        json_logs = sorted(glob.glob(osp.join(work_dir, '*.log.json')))
        if not json_logs:
            return
        log_path = json_logs[-1]

        train_iters, train_losses = [], []
        val_iters, val_mious, val_mdices = [], [], []

        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                mode = entry.get('mode', '')
                if mode == 'train':
                    it = entry.get('iter')
                    loss = entry.get('loss')
                    if it is not None and loss is not None:
                        train_iters.append(it)
                        train_losses.append(loss)
                elif mode == 'val':
                    it = entry.get('iter')
                    miou = entry.get('mIoU')
                    mdice = entry.get('mDice')
                    if it is not None and miou is not None:
                        val_iters.append(it)
                        val_mious.append(miou)  # keep 0~1 scale
                        val_mdices.append(mdice if mdice else 0)

        # Average train loss per epoch
        epoch_train_losses = []
        epoch_train_epochs = []
        if train_iters and train_losses:
            cur_epoch_losses = []
            cur_epoch = 0
            for it, loss in zip(train_iters, train_losses):
                ep = int(it / self.iters_per_epoch)
                if ep > cur_epoch and cur_epoch_losses:
                    epoch_train_losses.append(np.mean(cur_epoch_losses))
                    epoch_train_epochs.append(cur_epoch)
                    cur_epoch_losses = []
                    cur_epoch = ep
                cur_epoch_losses.append(loss)
            if cur_epoch_losses:
                epoch_train_losses.append(np.mean(cur_epoch_losses))
                epoch_train_epochs.append(cur_epoch)

        # Extra metrics from self.history
        extra_epochs = self._iter_to_epoch(self.history['iter'])
        val_losses = self.history['val_loss']
        train_mious = self.history['train_mIoU']  # keep 0~1
        val_hd95s = self.history['val_hd95']
        val_epochs = self._iter_to_epoch(val_iters)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Subplot 1: Training and Validation Mean Loss
        ax = axes[0, 0]
        if epoch_train_epochs:
            ax.plot(epoch_train_epochs, epoch_train_losses,
                    label='Training Mean Loss', marker='o', markersize=5,
                    color='tab:blue', linewidth=1.5)
        if extra_epochs and val_losses:
            ax.plot(extra_epochs, val_losses,
                    label='Validation Mean Loss', marker='s', markersize=5,
                    color='tab:orange', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Mean Loss')
        ax.grid(True)
        ax.legend()

        # Subplot 2: Training and Validation IoU
        ax = axes[0, 1]
        if extra_epochs and train_mious:
            ax.plot(extra_epochs, train_mious,
                    label='Training Mean IoU', color='orange',
                    marker='o', markersize=5, linewidth=1.5)
        if val_epochs and val_mious:
            ax.plot(val_epochs, val_mious,
                    label='Validation Mean IoU', color='red',
                    marker='s', markersize=5, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Training and Validation Mean IoU')
        ax.grid(True)
        ax.legend()

        # Subplot 3: Validation Dice and IoU
        ax = axes[1, 0]
        if val_epochs:
            ax.plot(val_epochs, val_mdices,
                    label='Validation Mean Dice', marker='s',
                    markersize=5, linewidth=1.5)
            ax.plot(val_epochs, val_mious,
                    label='Validation Mean IoU', marker='s',
                    markersize=5, color='tab:orange', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation Mean Dice and Mean IoU')
        ax.grid(True)
        ax.legend()

        # Subplot 4: Validation HD95
        ax = axes[1, 1]
        if extra_epochs and val_hd95s:
            valid = [(ep, h) for ep, h in zip(extra_epochs, val_hd95s)
                     if not np.isnan(h)]
            if valid:
                eps, hs = zip(*valid)
                ax.plot(eps, hs, label='Validation Mean HD95',
                        color='red', marker='s', markersize=5, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('HD95')
        ax.set_title('Validation Mean HD95')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        output_path = osp.join(work_dir, 'training_metrics.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f'[ExtraMetrics] Training metrics saved to {output_path}')

