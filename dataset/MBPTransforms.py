from dataset.transforms import *


def MultithreadBatchParallelTransforms(mode='train', args=None):
    if mode == 'train' and args.mosaic_transform:
        return Compose(
            [
                Mosaic(args.multimodal, imgsz=args.imgsz, rep=args.representation),
                RandomPhotometricDistort(p=0.5),
                RandomPerspective(translate=0.1, scale=0.5),
                StreamingHorizontalFlip(),
                StreamingVerticalFlip(),
                # RandomResize(mode),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
            ]
        )
    if mode == 'train' and not args.mosaic_transform:
        return Compose(
            [
                RandomPhotometricDistort(p=0.5),
                StreamingHorizontalFlip(),
                StreamingVerticalFlip(),
                # RandomResize(mode),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
            ]
        )
    if mode == 'val' or mode == 'test':
        return Compose(
            [
                # RandomResize(mode),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
             ]
        )
    raise ValueError(f'unknown {mode}')
