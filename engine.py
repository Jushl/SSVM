import warnings
import numpy as np
from util.misc.torch_utils import autocast, convert_optimizer_state_dict_to_fp16
import torch
from models.SMVM.modules.head import Detect
import gc
from copy import copy, deepcopy
import os
from pathlib import Path
from torch import nn
from torch import optim
import math
from util.misc.val import DetectionValidator
import time
from util.optim.ema import ModelEMA
from models.SMVM import unfreeze_streaming_layers


def save_batch_vis(batch, save_dir="dataset/input_batch"):
    import cv2

    def cxcywh_to_xyxy(bboxes, h, w):
        x1 = bboxes[:, 0] - bboxes[:, 2] / 2
        y1 = bboxes[:, 1] - bboxes[:, 3] / 2
        x2 = bboxes[:, 0] + bboxes[:, 2] / 2
        y2 = bboxes[:, 1] + bboxes[:, 3] / 2
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

    images = batch["img"].cpu()  # [8,3,H,W]
    boxes = batch["boxes"].cpu()  # [n,4]
    mask = batch["batch_idx"].cpu() if isinstance(batch["batch_idx"], torch.Tensor) else torch.tensor(batch["batch_idx"])
    data_idx = batch["data_idx"]

    num_images, _, H, W = images.shape
    for i in range(num_images):
        img = images[i].to("cpu").permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        img_with_boxes = img.copy()
        box_indices = torch.where(mask == i)[0]
        for idx in box_indices:
            x1, y1, x2, y2 = cxcywh_to_xyxy(boxes[idx][None], H, W)  # 转换为整数坐标
            color = (0, 255, 0)  # 绿色 (BGR格式)
            thickness = 2
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        folder_idx, img_idx = data_idx[0][i], data_idx[1][i]
        folder_path = f"check_input_bathch_train_sm/folder{folder_idx}"
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"image_{folder_idx}_{img_idx}.png")

        # 保存图片
        cv2.imwrite(save_path, img_with_boxes)
        print(f"Saved: {save_path}")


def check_batch_type(batch):
    import cv2
    def cxcywh_to_xyxy(bboxes, h, w):
        x1 = bboxes[:, 0] - bboxes[:, 2] / 2
        y1 = bboxes[:, 1] - bboxes[:, 3] / 2
        x2 = bboxes[:, 0] + bboxes[:, 2] / 2
        y2 = bboxes[:, 1] + bboxes[:, 3] / 2
        return torch.stack([x1 * w, y1 * h, x2 * w, y2 * h], dim=1)

    mask = (batch['batch_idx'] == 0).squeeze()
    bboxes = batch['boxes'][mask].to("cpu")
    img = batch['img'][0].to("cpu")
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    h, w, _ = img_np.shape

    bboxes_xyxy = cxcywh_to_xyxy(bboxes, h, w)

    img_with_boxes = img_np.copy()
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox.int().tolist()  # 转换为整数坐标
        color = (0, 255, 0)  # 绿色 (BGR格式)
        thickness = 2
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('Image with Bounding Boxes', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Detection(object):
    def __init__(
            self,
            model,
            scaler,
            ema,
            data_loader_train,
            data_loader_val,
            device,
            args
                 ):

        self.device = device
        self.epochs = args.epochs

        self.model = model
        self.scaler = scaler
        self.ema = ema
        self.amp = args.amp
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.args = args

        self.best_fitness = None
        self.fitness = None

        self.save_dir = Path(
            os.path.join(
                args.output_dir,
                model.model_name +
                '_epochs[' +
                str(args.epochs) +
                ']_batch[' +
                str(args.batch_size_train) +
                ']_numwork[' +
                str(args.num_workers) +
                ']_rep[' +
                args.representation +
                ']_multimodal[' +
                args.multimodal +
                ']'
            )
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.save_dir / "results.csv"
        self.wdir = self.save_dir / "weights"
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths

        # Validator
        self.validator = self.get_validator()
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.args.batch_size_train), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.args.batch_size_train * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(data_loader_train) / max(self.args.batch_size_train, self.args.nbs)) * self.args.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        if self.args.resume:
            self.load_resume_state()
        if self.args.finetune:
            self.load_finetune_state()

        # Scheduler
        self._setup_scheduler()
        self.scheduler.last_epoch = self.args.start_epoch - 1

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def train(self):
        nb = len(self.data_loader_train)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.train_time_start = time.time()
        self.epoch_time_start = time.time()

        epoch = self.epoch if self.args.resume else 0
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            keep_train = epoch < self.epochs
            if self.epoch == self.args.stop_mosaic:
                unfreeze_streaming_layers(self.model)
                self.data_loader_train.dataset.close_mosaic()

            if not keep_train:
                print("train done...")
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            _decisions = {}
            self.tloss = None
            self.data_loader_train.dataset._build_random_decision(self.args.streaming_scales, self.args.mosaic_transform)
            print("start training...")
            for i, batch in enumerate(self.data_loader_train):
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.args.batch_size_train]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # forward
                with autocast(self.args.amp):
                    batch = self.preprocess_batch(batch)
                    _idx = tuple(batch["data_idx"][0])
                    if _idx not in _decisions:
                        _decisions[_idx] = True
                        state = None
                    assert _decisions[_idx] is True

                    (self.loss, self.loss_items), state = self.model(
                        batch,
                        (
                            (state[0].detach(), state[1].detach())
                            if isinstance(state, tuple)
                            else state.detach()
                            if state is not None
                            else state
                        )
                    )

                    if batch["boxes"].numel() != 0:
                        self.tloss = (
                            (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                        )
                    else:
                        continue

                # backward
                self.scaler.scale(self.loss).backward()
                # optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                if i % 10 == 0:
                    print(f"Epoch:{epoch} {i + 10}/{nb} loss:{self.loss / self.args.batch_size_train}")

            # validation
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            if keep_train:
                self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.save_model()

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            self._clear_memory()
            epoch += 1

    def val(self):
        model, weight = self.torch_safe_load(self.args.resume)
        metrics = self.validator(model=model)

    def torch_safe_load(self, weight):
        ckpt = torch.load(weight, map_location="cpu")
        model = (ckpt.get("ema") or ckpt["model"]).to(self.device).float()
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])
        for m in model.modules():
            if hasattr(m, "inplace"):
                m.inplace = True
            elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None
            if isinstance(m, Detect):
                m.anchors = m.anchors.float().to(self.device)
                m.strides = m.strides.float().to(self.device)
        return model, weight

    def load_resume_state(self, inplace=True):
        ckpt = torch.load(self.args.resume, map_location='cpu')
        weights = (ckpt.get("ema") or ckpt["model"]).to(self.device).float()
        weights = weights.eval()
        for m in weights.modules():
            if hasattr(m, "inplace"):
                m.inplace = inplace
            elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        self.model.load(weights)
        self.ema = ModelEMA(self.model)

        self.accumulate = max(round(self.args.nbs / self.args.batch_size_train), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.args.batch_size_train * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.data_loader_train) / max(self.args.batch_size_train, self.args.nbs)) * self.args.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            self.best_fitness = ckpt["best_fitness"]

        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]

        self.epoch = ckpt.get("epoch")

    def load_finetune_state(self):
        weights = torch.load(self.args.finetune, map_location='cpu')
        self.model.finetune(weights)
        self.ema = ModelEMA(self.model)

        self.accumulate = max(round(self.args.nbs / self.args.batch_size_train), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.args.batch_size_train * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.data_loader_train) / max(self.args.batch_size_train, self.args.nbs)) * self.args.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        return batch

    def validate(self):
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        return optimizer

    def _setup_scheduler(self):
        if self.args.cos_lr:
            def one_cycle(y1=0.0, y2=1.0, steps=100):
                return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(self.data_loader_val, save_dir=self.save_dir, args=copy(self.args))

    def read_results_csv(self):
        import pandas as pd  # scope for faster 'import ultralytics'
        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def save_model(self):
        import io
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)

    def _clear_memory(self):
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()
