
from typing import Dict, List, Optional, Sequence, Union, Tuple

import torch
from mmengine.evaluator import BaseMetric


from mmseg.registry import METRICS

from torch import Tensor, tensor
import torch.nn.functional as F
import cv2
import pickle
import math
from mmengine.visualization import Visualizer
import torch.distributed as dist

# code modified from torchmetrics
def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """PyTorch currently does not support``torch.bincount`` for:

        - deterministic mode on GPU.
        - MPS devices

    This implementation fallback to a for-loop counting occurrences in that case.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x
    """
    if minlength is None:
        minlength = len(torch.unique(x))
    if torch.are_deterministic_algorithms_enabled() and x.is_mps:
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    return torch.bincount(x, minlength=minlength)


def _binary_precision_recall_curve_update(
        preds: Tensor,
        target: Tensor,
        thresholds: Optional[Tensor],
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.
    """
    if thresholds is None:
        return preds, target
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0)).long()  # num_samples x num_thresholds
    unique_mapping = preds_t + 2 * target.unsqueeze(-1) + 4 * torch.arange(len_t, device=target.device)
    bins = _bincount(unique_mapping.flatten(), minlength=4 * len_t)
    return bins.reshape(len_t, 2, 2)


def _binary_precision_recall_curve_format(
        preds: Tensor,
        target: Tensor,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor
    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if not torch.all((0 <= preds) * (preds <= 1)):
        preds = preds.sigmoid()

    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    return preds, target, thresholds


def _binary_precision_recall_curve_compute(
        state: Union[Tensor, Tuple[Tensor, Tensor]],
        thresholds: Optional[Tensor],
        pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve.
    """
    if isinstance(state, Tensor):
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, dtype=recall.dtype, device=recall.device)])
        return precision, recall, thresholds
    else:
        fps, tps, thresholds = _binary_clf_curve(state[0], state[1], pos_label=pos_label)
        precision = tps / (tps + fps)
        recall = tps / tps[-1]

        # stop when full recall attained and reverse the outputs so recall is decreasing
        last_ind = torch.where(tps == tps[-1])[0][0]
        sl = slice(0, last_ind.item() + 1)

        # need to call reversed explicitly, since including that to slice would
        # introduce negative strides that are not yet supported in pytorch
        precision = torch.cat([reversed(precision[sl]), torch.ones(1, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
        thresholds = reversed(thresholds[sl]).detach().clone()  # type: ignore

    return precision, recall, thresholds


def _binary_average_precision_compute(
        state: Union[Tensor, Tuple[Tensor, Tensor]],
        thresholds: Optional[Tensor],
) -> Tensor:
    precision, recall, _ = _binary_precision_recall_curve_compute(state, thresholds)
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])


def _binary_roc_compute(
        state: Union[Tensor, Tuple[Tensor, Tensor]],
        thresholds: Optional[Tensor],
        pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        tns = state[:, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0)
        fpr = _safe_divide(fps, fps + tns).flip(0)
        thresholds = thresholds.flip(0)
    else:
        fps, tps, thresholds = _binary_clf_curve(preds=state[0], target=state[1], pos_label=pos_label)
        # Add an extra threshold position to make sure that the curve starts at (0, 0)
        tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
        fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
        thresholds = torch.cat([torch.ones(1, dtype=thresholds.dtype, device=thresholds.device), thresholds])

        if fps[-1] <= 0:
            # rank_zero_warn(
            #     "No negative samples in targets, false positive value should be meaningless."
            #     " Returning zero tensor in false positive score",
            #     UserWarning,
            # )
            fpr = torch.zeros_like(thresholds)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            # rank_zero_warn(
            #     "No positive samples in targets, true positive value should be meaningless."
            #     " Returning zero tensor in true positive score",
            #     UserWarning,
            # )
            tpr = torch.zeros_like(thresholds)
        else:
            tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def _binary_auroc_compute(
        state: Union[Tensor, Tuple[Tensor, Tensor]],
        thresholds: Optional[Tensor],
        max_fpr: Optional[float] = None,
        pos_label: int = 1,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    fpr, tpr, _ = _binary_roc_compute(state, thresholds, pos_label)
    if max_fpr is None or max_fpr == 1:
        return _auc_compute_without_check(fpr, tpr, 1.0)

    _device = fpr.device if isinstance(fpr, Tensor) else fpr[0].device
    max_area: Tensor = tensor(max_fpr, device=_device)
    # Add a single point at max_fpr and interpolate its tpr value
    stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = torch.lerp(tpr[stop - 1], tpr[stop], weight)
    tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
    fpr = torch.cat([fpr[:stop], max_area.view(1)])

    # Compute partial AUC
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)

    # McClish correction: standardize result to be 0.5 if non-discriminant and 1 if maximal
    min_area: Tensor = 0.5 * max_area ** 2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float, axis: int = -1) -> Tensor:
    """Computes area under the curve using the trapezoidal rule.

    Assumes increasing or decreasing order of `x`.
    """
    with torch.no_grad():
        auc_: Tensor = torch.trapz(y, x, dim=axis) * direction
    return auc_


def _binary_clf_curve(
        preds: Tensor,
        target: Tensor,
        sample_weights: Optional[Sequence] = None,
        pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the tps and false positives for all unique thresholds in the preds tensor. Adapted from
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py.

    Args:
        preds: 1d tensor with predictions
        target: 1d tensor with true values
        sample_weights: a 1d tensor with a weight per sample
        pos_label: interger determining what the positive class in target tensor is

    Returns:
        fps: 1d tensor with false positives for different thresholds
        tps: 1d tensor with true positives for different thresholds
        thresholds: the unique thresholds use for calculating fps and tps
    """
    with torch.no_grad():
        if sample_weights is not None and not isinstance(sample_weights, Tensor):
            sample_weights = tensor(sample_weights, device=preds.device, dtype=torch.float)

        # remove class dimension if necessary
        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = torch.argsort(preds, descending=True)

        preds = preds[desc_score_indices]
        target = target[desc_score_indices]

        if sample_weights is not None:
            weight = sample_weights[desc_score_indices]
        else:
            weight = 1.0

        # pred typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
        target = (target == pos_label).to(torch.long)
        tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

        if sample_weights is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps

        return fps, tps, preds[threshold_idxs]


def _adjust_threshold_arg(
        thresholds: Optional[Union[int, List[float], Tensor]] = None, device: Optional[torch.device] = None
) -> Optional[Tensor]:
    """Utility function for converting the threshold arg for list and int to tensor format."""
    if isinstance(thresholds, int):
        thresholds = torch.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        thresholds = torch.tensor(thresholds, device=device)
    return thresholds


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom


@METRICS.register_module()
class OoDMetric(BaseMetric):
    def __init__(self,
                 mode='SML',
                 mean_var_path=None,
                 true_num_classes: int = 19,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if mode == 'SML':
            assert mean_var_path is not None
            with open(mean_var_path,"rb") as f:
                mean_var=pickle.load(f)
            self.mean=mean_var['mean']
            self.var=mean_var['var']
        self.mode = mode
        self.mean_var_path=mean_var_path
        self.true_num_classes = true_num_classes
        self.ignore_index = ignore_index
        self.inner_counter=0
        self.counter=0

    def _draw_activation_map(self,img_path,scores):
        if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
            self.inner_counter += 1
            if self.inner_counter%5==0:
                probs = scores.clone()
                probs = probs - probs.min()
                probs = probs / probs.max()
                vis = Visualizer.get_current_instance()
                img = cv2.imread(img_path)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                window_name = f'val_{self.counter}'

                drawn_img = vis.draw_featmap(
                    probs.unsqueeze(0),
                    img,
                    alpha=0.95
                )

                vis.add_image(window_name, drawn_img, step=self.inner_counter)


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            scores = data_sample['seg_logits']['data']
            target = data_sample['gt_sem_seg']['data'].to(
                scores)
            target[target == 2] = 1
            if self.mode=='SML':
                scores,prediction=scores.max(0)
                for c in torch.unique(prediction):
                    scores=torch.where(prediction==c,(scores-self.mean[c.item()])/math.sqrt(self.var[c.item()]),scores)
                scores=-scores
            elif self.mode=='ML':
                scores=-scores.amax(0)
            elif self.mode=='MSP':
                scores=-scores.softmax(0).amax(0)
            elif self.mode=='Entropy':
                scores=torch.softmax(scores,dim=0)
                scores=-torch.sum(scores*torch.log(scores),dim=0)
            elif self.mode=='Energy':
                scores=-torch.logsumexp(scores,dim=0)
            else:
                scores=scores[0]
                # raise NotImplementedError
                # scores = -torch.logsumexp(scores,dim=0)
                # scores=1-torch.sigmoid(scores)
            # self._draw_activation_map(data_sample['img_path'],scores)
            # probs=scores.clone()
            # probs = probs - probs.min()
            # probs = probs / probs.max()
            # probs = np.uint8(probs.cpu().numpy() * 255)
            # # probs[target.cpu().numpy()[0]==255]=0
            # img = cv2.applyColorMap(probs, cv2.COLORMAP_JET)
            # cv2.imwrite(f'ood_second_{self.counter}.png', img)
            # # scores=1-torch.max(scores[:19,...],dim=0)
            

            preds, target, _ = _binary_precision_recall_curve_format(scores, target, None, self.ignore_index)
            state = _binary_precision_recall_curve_update(preds, target, None)
            self.results.append((state[0], state[1]))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        # ap
        preds = [d[0] for d in results]
        targets = [d[1] for d in results]
        state = [dim_zero_cat(preds), dim_zero_cat(targets)]
        ap = _binary_average_precision_compute(state, None)
        fpr, tpr, thresholds = _binary_roc_compute(state, None)
        auroc = _binary_auroc_compute(state, None, None)
        self.counter+=1
        return {
            "AUROC": auroc,
            "AUPRC": ap,
            "FPR@95": fpr[tpr > 0.95][0]
        }
