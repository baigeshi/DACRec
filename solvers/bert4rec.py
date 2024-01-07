import os
import pickle

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models import BERT4Rec
from datasets import (
    MLMTrainDataset,
    MLMEvalDataset,
)

from .base import BaseSolver


__all__ = (
    'BERT4RecSolver',
)


# Contrastive Learning
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # revise to translate contrastive loss, where the current sentence itself and its translation is 1.
            mask = torch.eye(int(batch_size / 2), dtype=torch.float32)
            mask = torch.cat([mask, mask], dim=1)
            mask = torch.cat([mask, mask], dim=0)
            mask = mask.to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BERT4RecSolver(BaseSolver):

    def __init__(self, config: dict) -> None:
        C = config

        # before super
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        self.num_items = len(self.iid2iindex)

        super().__init__(config)

    # override
    def init_model(self) -> None:
        C = self.config
        cm = C['model']
        self.model = BERT4Rec(
            num_items=self.num_items,
            sequence_len=C['dataloader']['sequence_len'],
            max_num_segments=C['dataloader']['max_num_segments'],
            use_session_token=C['dataloader']['use_session_token'],
            num_layers=cm['num_layers'],
            hidden_dim=cm['hidden_dim'],
            temporal_dim=cm['temporal_dim'],
            num_heads=cm['num_heads'],
            dropout_prob=cm['dropout_prob'],
            random_seed=cm['random_seed']
        ).to(self.device)

    # override
    def init_criterion(self) -> None:
        self.ce_losser = CrossEntropyLoss(ignore_index=0)

    # override
    def init_dataloader(self) -> None:
        C = self.config
        name = C['dataset']
        sequence_len = C['dataloader']['sequence_len']
        max_num_segments = C['dataloader']['max_num_segments']
        use_session_token = C['dataloader']['use_session_token']
        self.train_dataloader = DataLoader(
            MLMTrainDataset(
                name=name,
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                random_cut_prob=C['dataloader']['random_cut_prob'],
                mask_prob=C['dataloader']['mlm_mask_prob'],
                use_session_token=use_session_token,
                random_seed=C['dataloader']['random_seed']
            ),
            batch_size=C['train']['batch_size'],
            shuffle=True,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.valid_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='valid',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.test_ns_random_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_popular_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='popular',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_all_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='all',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )

    # override
    def calculate_loss(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        labels = batch['labels'].to(self.device)  # b x L

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # loss
        logits = logits.view(-1, logits.size(-1))  # bL x (V + 1)
        labels = labels.view(-1)  # bL
        loss = self.ce_losser(logits, labels)
        
        # Contrastive Learning
        sloss = SupConLoss(contrast_mode='all', temperature=0.9).to(device)
        sloss_logits = logits
        sloss_labels = labels
        sloss_logits = sloss_logits.unsqueeze(1)
        sloss1 = sloss(sloss_logits, sloss_labels)
        sloss1 = sloss1 / len(sloss_logits)
        r = 0.00001
        loss = sloss1 * r + loss * (1-r)

        # Adversary Learning Input Data Reshape
        tokens_adv = tokens  # torch.Size([128, 200])
        # print(logits_ad.size(),tokens.size())
        # ml1m=torch.Size([128, 200, 3328])  reshape([-1, 3328, 200])
        # steam2=torch.Size([1024, 15, 4332])  reshape([-1, 4332, 15])
        logits_adv = logits_ad.reshape([-1, 3328, 200])
        logits_adv = torch.max(logits_adv.data, 1)[1].cpu()
        
        return loss, tokens_adv, logits_adv

    # override
    def calculate_rankers(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        cands = batch['cands'].to(self.device)  # b x C

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # gather
        logits = logits[:, -1, :]  # b x (V + 1)
        scores = logits.gather(1, cands)  # b x C
        rankers = scores.argsort(dim=1, descending=True)

        return rankers
