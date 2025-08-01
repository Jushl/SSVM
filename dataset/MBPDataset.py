from torch.utils.data import Dataset, DataLoader
import random
import os
from glob import glob
from PIL import Image
import numpy as np
from dataset.representations import EventRepresentation
from dataset.representations import BidirecticalIntegralFusion as BIF
from dataset.function import BoundingBoxes
import json
import torch
from dataset.function import convert_to_tv_tensor
import dataset
import cv2
import math
from dataset.MBPTransforms import MultithreadBatchParallelTransforms


class MultithreadBatchParallelDataset(Dataset):
    def __init__(self, mode, transforms, args):
        self.dataset_path = args.dataset_path
        self.representation = args.representation
        if mode == "train":
            self.base_path = os.path.join(args.dataset_path, "train")
            self.batch_size = args.batch_size_train
        elif mode == "val":
            self.base_path = os.path.join(args.dataset_path, "val")
            self.batch_size = args.batch_size_val
        else:
            assert mode in ["train", "val"]

        self.args = args
        self.imgsz = args.imgsz
        self.multimodal = args.multimodal

        if self.multimodal == "event":
            self.rep = EventRepresentation(self.representation)
        elif self.multimodal == "multimodal":
            self.fusion = BIF()

        self._transforms = transforms
        self.data_structure = []
        self._build_data_structure()
        self._build_random_decision(args.streaming_scales, args.mosaic_transform)

    def _build_data_structure(self):
        images_base_path = os.path.join(self.base_path, "images")
        for scene in sorted(os.listdir(images_base_path)):
            scene_path = os.path.join(images_base_path, scene)
            for data in sorted(os.listdir(scene_path)):
                data_path = os.path.join(scene_path, data)
                if os.path.isdir(data_path):
                    images_files = sorted(glob(os.path.join(data_path, "*.png")))
                    data_structure = []
                    for image_path in images_files:
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        event_path = os.path.join(data_path.replace('images', 'events'), f"{base_name}.npy")
                        label_path = os.path.join(data_path.replace('images', 'labels'), f"{base_name}.json")
                        data_structure.append((image_path, event_path, label_path))
                    if data_structure:
                        self.data_structure.append({
                            'pairs': data_structure,
                            'num': len(data_structure)
                        })
        random.shuffle(self.data_structure)
        if self.batch_size != 1:
            for _ in range(self.batch_size - len(self.data_structure) % self.batch_size):
                self.data_structure.append(random.choice(self.data_structure))

    def _build_random_decision(self, scales, mosaic=False):
        num_decision = len(self.data_structure)
        if mosaic:
            self._decision = {
                "vertical_flip": {i: random.random() < 0.5 for i in range(num_decision)},
                "horizontal_flip": {i: random.random() < 0.5 for i in range(num_decision)},
            }
        else:
            self._decision = {
                "vertical_flip": {i: random.random() < 0.5 for i in range(num_decision)},
                "horizontal_flip": {i: random.random() < 0.5 for i in range(num_decision)},
                "random_resize": {i: random.choice(scales) for i in range(num_decision)}
            }

    @staticmethod
    def get_json_boxes(label_filename):
        with open(label_filename, 'r') as json_file:
            data = json.load(json_file)
            objects = data['shapes']
            class_indexes = []
            bounding_boxes = []
            for i in range(len(objects)):
                bounding_boxes_points = objects[i]['points']
                if 'label' in objects[i]:
                    bounding_boxes_class = objects[i]['label']
                else:
                    bounding_boxes_class = objects[i]['label']

                class_index = int(dataset.EMRS_name2category[bounding_boxes_class])
                bounding_box = [int(bounding_boxes_points[0][0]), int(bounding_boxes_points[0][1]),
                                int(bounding_boxes_points[2][0]), int(bounding_boxes_points[2][1])]

                class_indexes.append(class_index)
                bounding_boxes.append(bounding_box)

        return {'labels': class_indexes, 'boxes': bounding_boxes}

    @staticmethod
    def prepare(image, target):
        w, h = image.size
        gt = {}
        gt["orig_shape"] = [int(h), int(w)]
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        boxes = target['boxes']
        classes = target['labels']
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        gt["boxes"] = boxes
        gt["labels"] = classes
        gt["image_id"] = image_id
        gt["batch_idx"] = torch.zeros(len(classes))
        if 'boxes' in gt:
            gt['boxes'] = convert_to_tv_tensor(gt['boxes'], key='boxes', spatial_size=image.size[::-1])
        return gt

    @staticmethod
    def scale_transform(image, target, imgsz, rect_mode=True):
        h0, w0 = target['orig_shape']
        im = np.array(image)
        if rect_mode:
            r = imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == imgsz):
            im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        resized_h, resized_w = im.shape[:2]
        target['resized_shape'] = (resized_h, resized_w)

        target["ratio_pad"] = (
            resized_h / target["orig_shape"][0],
            resized_w / target["orig_shape"][1]
        )

        image = Image.fromarray(im)
        if target['boxes'].numel() != 0:
            target['boxes'] = target['boxes'] / torch.Tensor([w0, h0, w0, h0]) \
                              * torch.Tensor([resized_w, resized_h, resized_w, resized_h])
            if not isinstance(target['boxes'], BoundingBoxes):
                target['boxes'] = convert_to_tv_tensor(
                    target['boxes'],
                    key='boxes',
                    spatial_size=target['resized_shape']
                )

        return image, target

    def close_mosaic(self):
        self.args.mosaic_transform = False
        self._transforms = MultithreadBatchParallelTransforms(args=self.args)
        print('Close Streaming Mosaic Transform !!!')

    def __len__(self):
        return sum(data['num'] for data in self.data_structure)

    def __getitem__(self, idx):
        _idx = 0
        while idx >= self.data_structure[_idx]['num']:
            idx -= self.data_structure[_idx]['num']
            _idx += 1
        data_idx = [_idx, idx]
        image_path, event_path, label_path = self.data_structure[_idx]['pairs'][idx]
        data = {
            'data_idx': data_idx,
            'data_structure': self.data_structure,
            'decision': self._decision
        }
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
            anno = self.get_json_boxes(label_path)
            target = {'image_id': idx, 'boxes': anno['boxes'], 'labels': anno['labels']}
            target = self.prepare(image, target)
        else:
            target = {
                'orig_shape': [H, W],
                'boxes': torch.Tensor([]),
                'labels': torch.Tensor([]),
                'image_id': idx,
                'batch_idx': torch.zeros(len(torch.Tensor([])))
            }

        image, target = self.scale_transform(image, target, self.imgsz)
        image, target, data_idx = self._transforms(image, target, data)

        return image, target, data_idx
