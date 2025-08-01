from dataset.MBPDataset import MultithreadBatchParallelDataset
from dataset.MBPTransforms import MultithreadBatchParallelTransforms


def build_dataset(mode, args):
    transforms = MultithreadBatchParallelTransforms(mode, args)
    return MultithreadBatchParallelDataset(mode, transforms, args)


EMRS_category2name = {
    0: 'car',
    1: 'two-wheel',
    2: 'pedestrian',
    3: 'bus',
    4: 'truck',
}

EMRS_name2category = {
    'car': 0,
    'two-wheel': 1,
    'pedestrian': 2,
    'bus': 3,
    'truck': 4,
}

EMRS_category2label = {k: i for i, k in enumerate(EMRS_category2name.keys())}
EMRS_label2category = {v: k for k, v in EMRS_category2label.items()}
