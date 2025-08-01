import PIL
import torch
import torch.nn as nn
import random
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import Any, Callable, Dict, List, Tuple, Type, Union, cast, Sequence
from dataset.function import convert_to_tv_tensor, _boxes_keys, BoundingBoxes, _parse_labels_getter
import torchvision
from torchvision import tv_tensors
import math
from PIL import Image
from torchvision.ops.boxes import box_iou
from dataset.representations import EventRepresentation
from dataset.representations import BidirecticalIntegralFusion as BIF
import numpy as np
import os
import cv2
import torch.nn.functional as F_v1
from dataset.function import convert_to_tv_tensor
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import _get_kernel
from torchvision.transforms.v2._utils import (query_chw,
                                              check_type,
                                              _check_sequence_input,
                                              _get_fill,
                                              _setup_fill_arg,
                                              has_all,
                                              query_size,
                                              get_bounding_boxes,
                                              has_any,
                                              is_pure_tensor
                                              )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, data):
        data_idx, data_structure, decision = data['data_idx'], data['data_structure'], data['decision']

        for t in self.transforms:
            image, target, data_idx, decision, data_structure = \
                t(image, target, data_idx, decision, data_structure)
        return image, target, data_idx


class Transform(nn.Module):
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor, PIL.Image.Image)

    def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
        needs_transform_list = []
        transform_pure_tensor = not has_any(flat_inputs, tv_tensors.Image, tv_tensors.Video, PIL.Image.Image)
        for inpt in flat_inputs:
            needs_transform = True
            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def remove_zero_area_boxes(self, boxes):
        widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
        boxes_areas = widths * heights
        good = boxes_areas > 0
        if not all(good):
            boxes = boxes[good]
        return boxes, good

    def box_clip(self, boxes, w, h):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    def box_scale(self, boxes, scale_w, scale_h):
        scale = (scale_w, scale_h, scale_w, scale_h)
        boxes[:, 0] *= scale[0]
        boxes[:, 1] *= scale[1]
        boxes[:, 2] *= scale[2]
        boxes[:, 3] *= scale[3]
        return boxes

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


class RandomPhotometricDistort(Transform):
    def __init__(self, brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def _generate_value(self, left: float, right: float) -> float:
        return torch.empty(1).uniform_(left, right).item()

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_channels, *_ = query_chw(flat_inputs)
        params: Dict[str, Any] = {
            key: self._generate_value(range[0], range[1]) if torch.rand(1) < self.p else None
            for key, range in [
                ("brightness_factor", self.brightness),
                ("contrast_factor", self.contrast),
                ("saturation_factor", self.saturation),
                ("hue_factor", self.hue),
            ]
        }
        params["contrast_before"] = bool(torch.rand(()) < 0.5)
        params["channel_permutation"] = torch.randperm(num_channels) if torch.rand(1) < self.p else None
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["brightness_factor"] is not None:
            inpt = self._call_kernel(F.adjust_brightness, inpt, brightness_factor=params["brightness_factor"])
        if params["contrast_factor"] is not None and params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["saturation_factor"] is not None:
            inpt = self._call_kernel(F.adjust_saturation, inpt, saturation_factor=params["saturation_factor"])
        if params["hue_factor"] is not None:
            inpt = self._call_kernel(F.adjust_hue, inpt, hue_factor=params["hue_factor"])
        if params["contrast_factor"] is not None and not params["contrast_before"]:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params["contrast_factor"])
        if params["channel_permutation"] is not None:
            inpt = self._call_kernel(F.permute_channels, inpt, permutation=params["channel_permutation"])
        return inpt

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )
        flat_outputs = [self._transform(inpt, params) if needs_transform else inpt
                        for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)]
        return tree_unflatten(flat_outputs, spec)


class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.horizontal_flip, inpt)

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class RandomVerticalFlip(Transform):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.vertical_flip, inpt)

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class StreamingHorizontalFlip(Transform):
    def __init__(self) -> None:
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.horizontal_flip, inpt)

    def forward(self, *inputs: Any) -> Any:
        _idx = inputs[2][0]
        _decision = inputs[3]["horizontal_flip"]
        should_flip = _decision[_idx]
        if not should_flip:
            return inputs

        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class StreamingVerticalFlip(Transform):
    def __init__(self) -> None:
        super().__init__()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.vertical_flip, inpt)

    def forward(self, *inputs: Any) -> Any:
        _idx = inputs[2][0]
        _decision = inputs[3]["vertical_flip"]
        should_flip = _decision[_idx]
        if not should_flip:
            return inputs

        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class RandomResize(Transform):
    def __init__(self, mode, center=True) -> None:
        super().__init__()
        self.mode = mode
        self.center = center

    def forward(self, *inputs: Any) -> Any:
        image, target, data_idx, decision, data_structure = inputs

        if self.mode == "train":
            _idx = data_idx[0]
            _decision = decision["random_resize"]
            resized_shape = _decision[_idx]
        elif self.mode in ["val", "test"]:
            resized_shape = 640
        else:
            raise ValueError(f'unknown {self.mode}')

        h0, w0 = target.get("resized_shape", target.get('orig_shape'))
        r = resized_shape / max(h0, w0)

        image = np.array(image)
        w, h = (min(math.ceil(w0 * r), resized_shape), min(math.ceil(h0 * r), resized_shape))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)

        resized_w, resized_h = image.size
        target['resized_shape'] = (resized_h, resized_w)

        if target['boxes'].numel() != 0:
            target['boxes'] = target['boxes'] / torch.Tensor([w0, h0, w0, h0]) *\
                              torch.Tensor([resized_w, resized_h, resized_w, resized_h])
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        target["ratio_pad"] = (
            resized_h / target["orig_shape"][0],
            resized_w / target["orig_shape"][1]
        )

        image = np.array(image)
        image = torch.Tensor(image).permute(2, 0, 1)
        H, W = image.shape[-2:]
        if H != W:
            dw, dh = 0, W - H
        if self.center:
            dw /= 2
            dh /= 2
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        image = self.pad_image(image, top, bottom, left, right)
        if target["boxes"].numel() != 0:
            target["boxes"] = self._update_labels(target["boxes"], left, top)
        if target.get("ratio_pad"):
            target["ratio_pad"] = (target["ratio_pad"], (left, top))
        image = image.permute(1, 2, 0)
        image = image.numpy().astype(np.uint8)
        image = Image.fromarray(image)

        return image, target, data_idx, decision, data_structure

    def pad_image(self, image, top, bottom, left, right):
        padding = (left, right, top, bottom)
        padded_image = F_v1.pad(image, padding, mode='constant', value=114)
        return padded_image

    def add(self, boxes, offset):
        boxes[:, 0] += offset[0]
        boxes[:, 1] += offset[1]
        boxes[:, 2] += offset[2]
        boxes[:, 3] += offset[3]
        return boxes

    def _update_labels(self, boxes, padw, padh):
        boxes = self.add(boxes, offset=(padw, padh, padw, padh))
        return boxes


class Mosaic(Transform):
    def __init__(self, multimodal, imgsz=640, n=4, rep="vg", uniform=False):
        super().__init__()
        assert n in {4, }, "grid must be equal to 4."
        self.multimodal = multimodal
        self.uniform = uniform
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        self.n = n

        self.representation = rep
        if self.multimodal == "event":
            self.rep = EventRepresentation(self.representation)
        elif self.multimodal == "multimodal":
            self.fusion = BIF()

    def forward(self, *inputs: Any) -> Any:
        image, target, data_idx, decision, data_structure = inputs
        mix = self.load_image(data_structure)
        image, target = self._mix_transform(image, target, mix)
        return image, target, data_idx, decision, data_structure

    def load_image(self, data_structure):
        from dataset.MBPDataset import MultithreadBatchParallelDataset as MBPDataset

        _idxs = random.choices(range(len(data_structure)), k=self.n - 1)
        mix = []
        for _idx in _idxs:
            idx = random.randint(0, data_structure[_idx]['num'] - 1)
            image_path, event_path, label_path = data_structure[_idx]['pairs'][idx]
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            if self.multimodal == 'event':
                event = np.load(event_path)
                image = self.rep.run(event, H, W)
            elif self.multimodal == 'multimodal':
                event = np.load(event_path)
                fusion = self.fusion.run(image.copy().convert("L"), event.copy())
                image_ary = np.array(image)
                fusion_ary = np.array(fusion)
                image_ = np.zeros_like(image_ary)
                image_[:, :, 0] = fusion_ary[:, :, 0]
                image_[:, :, 1] = image_ary[:, :, 1]
                image_[:, :, 2] = image_ary[:, :, 2]
                image = Image.fromarray(image_)
            else:
                assert self.multimodal in ['image', 'event', 'multimodal']

            if os.path.exists(label_path):
                anno = MBPDataset.get_json_boxes(label_path)
                target = {'image_id': idx, 'boxes': anno['boxes'], 'labels': anno['labels']}
                target = MBPDataset.prepare(image, target)
            else:
                target = {
                    'orig_shape': [H, W],
                    'boxes': torch.Tensor([]),
                    'labels': torch.Tensor([]),
                    'image_id': idx,
                    'batch_idx': torch.zeros(len(torch.Tensor([])))
                }

            image, target = MBPDataset.scale_transform(image, target, self.imgsz)
            mix.append({
                'image': image,
                'target': target
            })
        return mix

    def _mix_transform(self, image, target, mix):
        return self._mosaic4_uniform(image, target, mix) if self.uniform else self._mosaic4(image, target, mix)

    def _mosaic4_uniform(self, image, target, mix):
        mosaic_targets = []
        s = self.imgsz
        h, w = target.get('resized_shape', target['orig_shape'])
        for i in range(4):
            img = np.array(image) if i == 0 else np.array(mix[i - 1]['image'])
            tgt = target if i == 0 else mix[i - 1]['target']

            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(s - w, 0), max(s - h, 0), s, s
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = s, max(s - h, 0), min(s + w, s * 2), s
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(s - w, 0), s, s, min(s + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, w, h
            elif i == 3:
                x1a, y1a, x2a, y2a = s, s, min(s + w, s * 2), min(s + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, w, h

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if tgt['boxes'].numel() != 0:
                tgt = self._update_labels(tgt, padw, padh)
            mosaic_targets.append(tgt)
        final_target = self._cat_targets(mosaic_targets)
        image = Image.fromarray(img4)
        return image, final_target

    def _mosaic4(self, image, target, mix):
        mosaic_targets = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        h, w = target.get('resized_shape', target['orig_shape'])
        for i in range(4):
            img = np.array(image) if i == 0 else np.array(mix[i - 1]['image'])
            tgt = target if i == 0 else mix[i - 1]['target']

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if tgt['boxes'].numel() != 0:
                tgt = self._update_labels(tgt, padw, padh)
            mosaic_targets.append(tgt)
        final_target = self._cat_targets(mosaic_targets)
        image = Image.fromarray(img4)
        return image, final_target

    def _update_labels(self, tgt, padw, padh):
        offset = (padw, padh, padw, padh)
        tgt['boxes'][:, 0] += offset[0]
        tgt['boxes'][:, 1] += offset[1]
        tgt['boxes'][:, 2] += offset[2]
        tgt['boxes'][:, 3] += offset[3]
        return tgt

    def _cat_targets(self, mosaic_targets):
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for target in mosaic_targets:
            if target['boxes'].numel() != 0:
                cls.append(target["labels"])
                instances.append(target["boxes"])

        final_targets = {
            "orig_shape": mosaic_targets[0]["orig_shape"],
            "resized_shape": [imgsz, imgsz],
            "labels": torch.concatenate(cls, 0) if len(cls) != 0 else torch.Tensor([]),
            "boxes": torch.concatenate(instances, 0) if len(instances) != 0 else torch.Tensor([]),
            "mosaic_border": self.border,
        }

        final_targets['boxes'] = self.box_clip(final_targets['boxes'], imgsz, imgsz) \
            if final_targets['boxes'].numel() != 0 else final_targets['boxes']
        final_targets['boxes'], good = self.remove_zero_area_boxes(final_targets["boxes"]) \
            if final_targets['boxes'].numel() != 0 else (final_targets['boxes'], None)
        final_targets["labels"] = final_targets["labels"][good] \
            if good is not None else final_targets["labels"]

        final_targets['batch_idx'] = torch.zeros(len(final_targets['labels']))  # debug了 没问题

        if final_targets['boxes'].numel() != 0:
            if not isinstance(final_targets['boxes'], BoundingBoxes):
                final_targets['boxes'] = convert_to_tv_tensor(
                    final_targets['boxes'],
                    key='boxes',
                    spatial_size=final_targets['resized_shape']
                )
        return final_targets


class RandomPerspective(Transform):
    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border

    def affine_transform(self, img, border):
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def forward(self, *inputs: Any) -> Any:
        image, target, data_idx, decision, data_structure = inputs
        cls = target['labels']
        border = target.pop("mosaic_border", self.border)
        image = np.array(image)
        self.size = image.shape[1] + border[1] * 2, image.shape[0] + border[0] * 2
        image, M, scale = self.affine_transform(image, border)
        target['resized_shape'] = image.shape[:2]
        image = Image.fromarray(image)

        if target['boxes'].numel() != 0:
            boxes_orig = target.pop("boxes").numpy()
            boxes = self.apply_bboxes(boxes_orig, M)
            boxes_new = self.box_clip(boxes.copy(), *self.size)
            boxes_orig = self.box_scale(boxes_orig, scale_w=scale, scale_h=scale)
            i = self.box_candidates(box1=boxes_orig.T, box2=boxes_new.T)

            target['labels'] = cls[i]
            target['batch_idx'] = torch.zeros(len(target['labels']))  # debug了 没问题
            target['boxes'] = torch.Tensor(boxes_new[i])

        if target['boxes'].numel() != 0:
            if not isinstance(target['boxes'], BoundingBoxes):
                target['boxes'] = convert_to_tv_tensor(
                    target['boxes'],
                    key='boxes',
                    spatial_size=target['resized_shape']
                )
        else:
            if not isinstance(target['boxes'], torch.Tensor):
                target['boxes'] = torch.Tensor(target['boxes'])

        return image, target, data_idx, decision, data_structure


class ConvertPILImage(Transform):
    _transformed_types = (PIL.Image.Image,)
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.
        inpt = tv_tensors.Image(inpt)
        return inpt

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)


class ConvertBoxes(Transform):
    _transformed_types = (BoundingBoxes,)
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
        needs_transform_list = self._needs_transform_list(flat_inputs)
        flat_outputs = [
            self._transform(inpt, {}) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]
        return tree_unflatten(flat_outputs, spec)




