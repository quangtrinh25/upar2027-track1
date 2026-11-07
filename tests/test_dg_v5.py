import torch

from upar.dg import v5
from upar.dg.models import SPECS, build, param_groups


def test_group_ce_uses_only_single_answer_rows():
    targets = torch.zeros(3, 40)
    targets[0, 9] = 1                   # one upper colour -> counted
    targets[1, 9] = targets[1, 12] = 1  # two upper colours -> skipped for that group
    logits = torch.zeros(3, 40)
    logits[0, 9] = 10.
    loss = v5.group_ce(logits, targets)
    assert torch.isfinite(loss) and loss < .01  # only row 0 / upper colour enters, and it is confidently right
    assert v5.group_ce(torch.zeros(2, 40), torch.zeros(2, 40)) == 0


class Synthetic(torch.utils.data.Dataset):
    def __len__(self): return 4

    def __getitem__(self, i):
        y = torch.zeros(40); y[[1, 4, 9, 22, 33]] = 1
        return torch.randn(3, 252, 126), y


def test_v5_epoch_runs_and_erases():
    erased = v5.ErasingDataset(Synthetic(), p=1.)
    image, _ = erased[0]
    assert (image == 0).any()
    model = build('dinov2_b_v5', initialize=False)
    optimizer = torch.optim.AdamW(param_groups(model, 'dinov2_b_v5'))
    loader = torch.utils.data.DataLoader(erased, batch_size=2)
    stats = v5.train_epoch(model, loader, optimizer, device=torch.device('cpu'), scaler=None, smoothing=.02,
                           group_weight=SPECS['dinov2_b_v5']['group_ce'])
    assert stats['images'] == 4 and stats['optimizer_steps'] == 1 and stats['loss'] > 0
