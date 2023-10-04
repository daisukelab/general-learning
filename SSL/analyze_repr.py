import torch


def normalized_singular_value(representations: torch.Tensor) -> torch.Tensor:
    # https://arxiv.org/abs/2209.15007 Alexander C. Li, et al., "Understanding Collapse in Non-Contrastive Siamese Representation Learning," in ECCV, 2022
    # https://github.com/alexlioralexli/noncontrastive-ssl/blob/69e6ad106c588461b3c120153bd4737045c04330/test_collapse.py#L19-L30
    norms = torch.linalg.norm(representations, dim=1)
    normed_reprs = representations / (1e-6 + norms.unsqueeze(1))
    normed_reprs -= normed_reprs.mean(dim=0, keepdims=True)
    stds = torch.svd(normed_reprs).S
    return stds


def cumulative_explained_variance(nsv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # https://arxiv.org/abs/2209.15007 Alexander C. Li, et al., "Understanding Collapse in Non-Contrastive Siamese Representation Learning," in ECCV, 2022
    nsv_sum = nsv.sum()
    raw_cev = torch.tensor([nsv[:i].sum()/nsv_sum for i in range(1, len(nsv) + 1)])
    # AUC
    auc_cev = 1/len(nsv) * raw_cev.sum()
    return auc_cev, raw_cev


def get_cumulative_explained_variance_auc(representations: torch.Tensor) -> torch.Tensor:
    # https://arxiv.org/abs/2209.15007 Alexander C. Li, et al., "Understanding Collapse in Non-Contrastive Siamese Representation Learning," in ECCV, 2022
    nsv = normalized_singular_value(representations)
    assert representations.shape[-1] == nsv.shape[0], 'Ensure that the number of samples is greater than the number of feature dimensions' \
        f': representations.shape[-1]({representations.shape[-1]}) != nsv.shape[0]({nsv.shape[0]})'
    auc, cev = cumulative_explained_variance(nsv)
    return auc
