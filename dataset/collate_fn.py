import torch
import torch.nn.functional as F
import torchvision
from numbers import Number
torchvision.disable_beta_transforms_warning()
from collections import abc
from util.misc.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from itertools import repeat


def _ntuple(n):
    def parse(x):
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(self, stop_epoch=None, center=True, mosaic=False) -> None:
        super().__init__()
        self._decisions = {}
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.center = center
        self.mosaic = mosaic

    def __call__(self, items):
        H_max, W_max = self._max_hw(items)
        image_list, targets = [], []
        for x in items:
            image = x[0][None]
            target = x[1]
            H, W = image.shape[-2:]
            if H != W:
                H_max = W_max
                dw, dh = 0, W - H
            else:
                dw, dh = W_max - W, H_max - H
            if self.center:
                dw /= 2
                dh /= 2
            top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

            image = self.pad_image(image, top, bottom, left, right)

            if target["boxes"].numel() != 0:
                target["boxes"] = self._update_labels(target["boxes"], target["resized_shape"], left, top)
                target["boxes"] = target["boxes"] / torch.tensor((W_max, H_max)).tile(2)[None]
                target["boxes"] = box_xyxy_to_cxcywh(target["boxes"])

            if target.get("ratio_pad"):
                target["ratio_pad"] = (target["ratio_pad"], (left, top))
            target["resized_shape"] = [H, W]

            image_list.append(image)
            targets.append(target)

        images = torch.cat(image_list, dim=0)

        index = [x[2] for x in items]
        index = [[t[0] for t in index], [t[1] for t in index]]

        batch = {
            'img': images,
        }
        keys = targets[0].keys()
        values = list(zip(*[list(b.values()) for b in targets]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"boxes", "labels"}:
                if k == "labels":
                    value = [v.flatten() if v.ndim > 1 else v for v in value]
                elif k == "boxes":
                    for v in value:
                        if v.numel() % 4 != 0:
                            print(v)
                            print(f"Warning: Invalid boxes shape {v.shape}")

                    value = [v.reshape(-1, 4) if v.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32) for v in value]

                value = torch.cat(value, 0)
            batch[k] = value
        batch["labels"] = batch["labels"].view(-1, 1)
        batch["batch_idx"] = list(batch["batch_idx"])

        for i in range(len(batch["batch_idx"])):
            batch["batch_idx"][i] = batch["batch_idx"][i].view(-1)
            batch["batch_idx"][i] += i

        batch["batch_idx"] = torch.cat(batch["batch_idx"], 0)
        batch["data_idx"] = index

        return batch

    def add(self, boxes, offset):
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        boxes[:, 0] += offset[0]
        boxes[:, 1] += offset[1]
        boxes[:, 2] += offset[2]
        boxes[:, 3] += offset[3]
        return torch.Tensor(boxes)

    def _update_labels(self, boxes, orig_shape, padw, padh):
        boxes = box_cxcywh_to_xyxy(boxes).numpy()
        h, w = orig_shape
        boxes = boxes * (w, h, w, h)
        boxes = self.add(boxes, offset=(padw, padh, padw, padh))
        return boxes

    def _max_hw(self, items):
        max_h = 0
        max_w = 0
        for item in items:
            image = item[0]
            _, h, w = image.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        return max_h, max_w

    def pad_image(self, image, top, bottom, left, right):
        padding = (left, right, top, bottom)
        padded_image = F.pad(image, padding, mode='constant', value=114 / 255)
        return padded_image






