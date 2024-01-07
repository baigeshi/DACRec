import torch

from typing import (
    Any,
    List,
    Dict,
)


__all__ = (
    'METRIC_NAMES',
    'calc_batch_rec_metrics_per_k',
)


METRIC_NAMES = ('HR', 'Recall', 'NDCG', 'MinSkew', 'MaxSkew', 'NDKL')


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

        # MinSkew
        hit_ranks = torch.arange(1, k + 1, device=device).float() * hit_per_pos
        non_zero_hit_ranks = hit_ranks[hit_ranks != 0]
        mean_hr = non_zero_hit_ranks.mean()
        std_hr = non_zero_hit_ranks.std(unbiased=False)
        skewness = ((non_zero_hit_ranks - mean_hr) / std_hr).pow(3) if std_hr > 0 else torch.zeros_like(non_zero_hit_ranks)
        
        min_skew = skewness.min().item() if len(skewness) > 0 else 0
        metrics['values'][f'MinSkew@{k}'] = min_skew
        metrics['mean'][f'MinSkew@{k}'] = min_skew / batch_size
        
        # MaxSkew
        max_skew = skewness.max().item() if len(skewness) > 0 else 0
        metrics['values'][f'MaxSkew@{k}'] = max_skew
        metrics['mean'][f'MaxSkew@{k}'] = max_skew / batch_size

        # NDKL
        Z = sum([1 / np.log(i+1) for i in range(1, k+1)])
        ndkl_values = []
        for i in range(k):
            p = labels[:, i]  
            q = rankers[:, i]  
            kl_div = p * (torch.log(p) - torch.log(q))  
            ndkl_values.append(kl_div / np.log(i + 2))  
        ndkl = sum(ndkl_values) / Z
        ndkl_list = list(ndkl.detach().cpu().numpy())
        metrics['values'][f'NDKL@{k}'] = ndkl_list
        metrics['mean'][f'NDKL@{k}'] = sum(ndkl_list) / batch_size


    return metrics
