import torch
import numpy as np
from typing import (
    Any,
    List,
    Dict,
)


__all__ = (
    'METRIC_NAMES',
    'calc_batch_rec_metrics_per_k',
)


METRIC_NAMES = ('HR', 'Recall', 'NDCG', 'APR')

def KLD(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    p = torch.clamp(p, min=epsilon, max=1-epsilon)
    q = torch.clamp(q, min=epsilon, max=1-epsilon)
    kl_div = p * torch.log(p / q)
    kl_div = torch.where(torch.isnan(kl_div), torch.zeros_like(kl_div), kl_div)
    kl_div = torch.clamp(kl_div, min=0)  # 将负值强制设置为0
    return kl_div.sum()

def calc_batch_rec_metrics_per_k(rankers: torch.LongTensor,
                                 labels: torch.LongTensor,
                                 ks: List[int]
                                 ) -> Dict[str, Any]:
    """
        Args:
            rankers: LongTensor, (b x C), pos per rank (0 to C-1)
            labels: LongTensor, (b x C), binary per pos (0 or 1)
            ks: list of top-k values

        Returns:
            a dict of various metrics.
            keys are 'count', 'mean', 'values'.
            names are 'HR', 'Recall', 'NDCG'.

        put'em all in the same device.
    """

    # prepare
    batch_size = rankers.size(0)
    metrics: Dict[str, Any] = {
        'count': batch_size,
        'values': {},
        'mean': {},
    }
    answer_count = labels.sum(1)
    device = labels.device
    ks = sorted(ks, reverse=True)

    overall_popularity = labels.sum(0).float() / batch_size  # 计算整体流行度
    # for each k
    for k in ks:
        rankers_at_k = rankers[:, :k]
        hit_per_pos = labels.gather(1, rankers_at_k)

        # hr
        hrs = hit_per_pos.sum(1).bool().float()
        hrs_list = list(hrs.detach().cpu().numpy())
        metrics['values'][f'HR@{k}'] = hrs_list
        metrics['mean'][f'HR@{k}'] = sum(hrs_list) / batch_size

        # recall
        divisor = torch.min(
            torch.Tensor([k]).to(device),
            answer_count,
        )
        recalls = (hit_per_pos.sum(1) / divisor.float())
        recalls_list = list(recalls.detach().cpu().numpy())
        metrics['values'][f'Recall@{k}'] = recalls_list
        metrics['mean'][f'Recall@{k}'] = sum(recalls_list) / batch_size

        # ndcg
        positions = torch.arange(1, k + 1).to(device).float()
        weights = 1 / (positions + 1).log2()
        dcg = (hit_per_pos * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(device)
        ndcgs = dcg / idcg
        ndcgs_list = list(ndcgs.detach().cpu().numpy())
        metrics['values'][f'NDCG@{k}'] = ndcgs_list
        metrics['mean'][f'NDCG@{k}'] = sum(ndcgs_list) / batch_size

        # APR: Average Popularity of Recommended Items
        recommended_popularity = labels.gather(1, rankers[:, :k]).sum(1).float() / k
        apr_list = list(recommended_popularity.detach().cpu().numpy())
        metrics['values'][f'APR@{k}'] = apr_list
        metrics['mean'][f'APR@{k}'] = sum(apr_list) / batch_size


    return metrics
