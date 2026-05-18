import os
import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_best (str | None): If a metric name is provided, after each
            evaluation the model with the best score on this metric will be
            saved as ``best_<metric>_iter_<iter>.pth`` under ``runner.work_dir``.
            Only one such file is kept at any time (older one is removed when
            a better score is found). Default: None.
        rule (str | None): Comparison rule when ``save_best`` is set. Either
            'greater' (higher is better, e.g. mIoU/mDice/mAcc/aAcc) or 'less'
            (lower is better, e.g. loss/HD95). If None, inferred from the
            metric name. Default: None.
    """

    _greater_keys = ('mIoU', 'mDice', 'mAcc', 'aAcc', 'IoU', 'Dice', 'Acc')
    _less_keys = ('loss', 'hd95', 'HD95')
    _init_value_map = {'greater': -float('inf'), 'less': float('inf')}

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch

        self.save_best = eval_kwargs.pop('save_best', None)
        rule = eval_kwargs.pop('rule', None)
        self.eval_kwargs = eval_kwargs

        self.rule = None
        self._compare_func = None
        self.best_score = None
        self.best_ckpt_path = None
        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def _init_rule(self, rule, key_indicator):
        if rule is not None:
            assert rule in ('greater', 'less'), \
                f"rule must be 'greater' or 'less', got {rule}"
        else:
            if any(k in key_indicator for k in self._less_keys):
                rule = 'less'
            elif any(k in key_indicator for k in self._greater_keys):
                rule = 'greater'
            else:
                rule = 'greater'
        self.rule = rule
        if rule == 'greater':
            self._compare_func = lambda new, best: new > best
        else:
            self._compare_func = lambda new, best: new < best
        self.best_score = self._init_value_map[rule]

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            self._save_best_ckpt(runner, eval_res)

    def _save_best_ckpt(self, runner, eval_res):
        key = self.save_best
        if key not in eval_res:
            runner.logger.warning(
                f'[EvalHook] save_best key "{key}" not in eval results '
                f'{list(eval_res.keys())}; skip saving best ckpt.')
            return
        score = eval_res[key]
        if not self._compare_func(score, self.best_score):
            return

        cur_iter = runner.iter + 1
        new_path = osp.join(runner.work_dir,
                            f'best_{key}_iter_{cur_iter}.pth')

        if (self.best_ckpt_path is not None
                and self.best_ckpt_path != new_path
                and osp.isfile(self.best_ckpt_path)):
            try:
                os.remove(self.best_ckpt_path)
            except OSError as e:
                runner.logger.warning(
                    f'[EvalHook] failed to remove old best ckpt '
                    f'{self.best_ckpt_path}: {e}')

        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl=f'best_{key}_iter_{{}}.pth',
            save_optimizer=False,
            create_symlink=False)
        self.best_score = score
        self.best_ckpt_path = new_path
        runner.logger.info(
            f'[EvalHook] New best {key}: {score:.4f} at iter {cur_iter}, '
            f'saved to {new_path}')


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch

        self.save_best = eval_kwargs.pop('save_best', None)
        rule = eval_kwargs.pop('rule', None)
        self.eval_kwargs = eval_kwargs

        self.rule = None
        self._compare_func = None
        self.best_score = None
        self.best_ckpt_path = None
        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
