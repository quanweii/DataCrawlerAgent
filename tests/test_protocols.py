import torch

from benchmark.data.protocols import group_inductive_split, temporal_split
from benchmark.data.splits import make_split_disjoint_view


def test_temporal_split_keeps_time_bucket_together():
    times = [2000, 2000, 2001, 2002, 2003, 2003, 2004, 2005, 2006, 2007]
    masks, _ = temporal_split(times)
    split = torch.full((len(times),), -1, dtype=torch.long)
    for i, mask in enumerate(masks):
        split[mask] = i
    assert split[0] == split[1]
    assert split[4] == split[5]
    assert int(split.min()) == 0 and int(split.max()) == 2


def test_group_inductive_has_zero_group_overlap():
    groups = [f"g{i // 2}" for i in range(30)]
    y = torch.tensor([i % 3 for i in range(30)])
    masks, assignment = group_inductive_split(groups, y, seed=7)
    for group in set(groups):
        splits = {i for i, mask in enumerate(masks) for j, value in enumerate(groups)
                  if value == group and bool(mask[j])}
        assert len(splits) == 1
        assert next(iter(splits)) == assignment[group]


def test_disjoint_view_has_no_cross_split_edges():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    masks = (
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        torch.tensor([0, 0, 1, 0], dtype=torch.bool),
        torch.tensor([0, 0, 0, 1], dtype=torch.bool),
    )
    view = make_split_disjoint_view(edge_index, masks)
    assert view.tolist() == [[0, 1], [1, 0]]
