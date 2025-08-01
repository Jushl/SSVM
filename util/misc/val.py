from util.misc.metrics import DetMetrics
from util.misc.box_ops import xywh2xyxy, scale_boxes, box_iou
import numpy as np
from util.misc.checks import check_version
import torch
from util.misc.metrics import ConfusionMatrix

TORCH_1_9 = check_version(torch.__version__, "1.9.0")

def smart_inference_mode():
    def decorate(fn):
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)
    return decorate


class DetectionValidator:
    def __init__(self, dataloader=None, save_dir=None, args=None):
        self.args = args
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.confusion_matrix = None
        self.iouv = None
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            model.eval()
        else:
            self.device = self.args.device
            model = model.half() if self.args.half else model.float()
            model = model.to(self.device)
            self.stride = model.stride
            model.eval()

        self.init_metrics(model)
        self.jdict = []
        _decisions = {}


        for batch_i, batch in enumerate(self.dataloader):
            self.batch_i = batch_i
            # Preprocess
            batch = self.preprocess(batch)

            _idx = tuple(batch["data_idx"][0])
            if _idx not in _decisions:
                _decisions[_idx] = True
                state = None
            assert _decisions[_idx] is True

            preds, state = model(
                batch["img"],
                (
                    (state[0].detach(), state[1].detach())
                    if isinstance(state, tuple)
                    else state.detach()
                    if state is not None
                    else state
                )
            )

            # Loss
            if self.training:
                self.loss += model.loss(batch, state, preds)[0][1]

            # Postprocess
            preds = self.postprocess(preds)

            self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.confusion_matrix.plot(names=self.names.values(), save_dir=self.save_dir)
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        header = ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
        print("%22s%11s%11s%11s%11s%11s%11s" % header)
        print((pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())))
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            return stats

    def init_metrics(self, model):
        self.class_map = list(range(1, len(model.names) + 1))
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)


    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float())
        for k in ["batch_idx", "labels", "boxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def postprocess(self, preds):
        return self.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            max_det=self.args.max_det,
        )

    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=0,  # number of classes (optional)
            max_nms=30000,
            max_wh=7680,
            in_place=True,
    ):
        import torchvision

        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        prediction = prediction.transpose(-1, -2)

        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            if not x.shape[0]:
                continue

            box, cls, mask = x.split((4, nc, nm), 1)
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)

            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * max_wh
            scores = x[:, 4]
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]
            output[xi] = x[i]

        return output

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue

            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            self.confusion_matrix.process_batch(predn, bbox, cls)
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["labels"][idx].squeeze(-1)
        bbox = batch["boxes"][idx]
        ori_shape = batch["orig_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = xywh2xyxy(bbox) * torch.tensor(imgsz, device=batch["img"].device)[[1, 0, 1, 0]]
            scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_stats(self):
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict