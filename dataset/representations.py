import numpy as np
from PIL import Image
import cv2


class BidirecticalIntegralFusion:
    TIME_SCALE = 1e6
    TIME_MOD = 1e8

    def norm_image(self, image):
        min_img = np.min(image)
        max_img = np.max(image)
        if max_img != min_img:
            image = (image - min_img) / (max_img - min_img)
        return image

    def load_image(self, image):
        image = cv2.imread(image, 0)
        return image.astype(np.float32)

    def _run(self, image, event, delta):
        x = event[:, 0].astype(np.int32)
        y = event[:, 1].astype(np.int32)
        p = event[:, 2]
        p[p == 0] = -1
        timestamp = event[:, 3]
        t = np.mod(timestamp, self.TIME_MOD)
        if t[0] > t[-1]:
            t = timestamp - 1e7
            t = np.mod(t, self.TIME_MOD)
        t /= self.TIME_SCALE
        t_num = len(t)
        t_0_index = int(t_num / 2)
        t_0 = t[t_0_index]
        _t3 = t[0]
        t3_ = t[-1]
        thr = delta / 255
        img_h, img_w = image.shape

        e_t = np.zeros((img_h * img_w), dtype=np.float32)
        integral_time = np.full(img_h * img_w, t_0, dtype=np.float32)
        E_t = np.zeros((img_h * img_w), dtype=np.float32)

        for i in range(t_0_index, t_num):
            pix_index = y[i] * img_w + x[i]

            E_t[pix_index] = E_t[pix_index] + np.exp(e_t[pix_index]) * (t[i] - integral_time[pix_index])
            e_t[pix_index] = e_t[pix_index] + p[i] * thr
            integral_time[pix_index] = t[i]

        E_t3_ = E_t
        e_t = np.clip(e_t, -20, 20)
        E_t3_ = E_t3_ + np.exp(e_t) * (t3_ - integral_time)

        e_t.fill(0)
        integral_time.fill(t_0)
        E_t.fill(0)
        for i in range(t_0_index, -1, -1):
            pix_index = y[i] * img_w + x[i]

            E_t[pix_index] = E_t[pix_index] + np.exp(e_t[pix_index]) * np.abs(t[i] - integral_time[pix_index])
            e_t[pix_index] = e_t[pix_index] - p[i] * thr
            integral_time[pix_index] = t[i]

        _E_t3 = E_t
        e_t = np.clip(e_t, -20, 20)
        _E_t3 = _E_t3 + np.exp(e_t) * np.abs(_t3 - integral_time)

        E_t3 = (_E_t3 + E_t3_) / (t3_ - _t3)
        I_t3 = image / E_t3.reshape(img_h, img_w)
        I_t3[I_t3 > 1.0] = 1.0
        I = self.norm_image(I_t3) * 255
        return Image.fromarray(I.astype(np.uint8)).convert("RGB")

    def run(self, image, event):
        return self._run(self.norm_image(image), event, delta=110)


class EventRepresentation:
    def __init__(self, r):
        self.r = r
        self.num_bins = 1

        if self.r == "vg":
            self.run = self.VoxelGrid
        elif self.r == "bi":
            self.run = self.BinaryImage
        elif self.r == "eh":
            self.run = self.EventHistogram
        elif self.r == "ts":
            self.run = self.TimeSurface
            self.tau = 30000
            self.decay = 'exp'
            self.surface_dimensions = None
        elif self.r == "ev":
            self.TIME_SCALE = 1e6
            self.TIME_MOD = 1e8
            self.run = self.EdgeVoxel
        else:
            self.run = self.VoxelGrid
            assert self.r in ['vg', 'bi', 'eh', 'ts', 'ev']

    def VoxelGrid(self, events, height, width):
        def mapping(voxel_grid, height, width):
            map_img = np.ones((height, width)) * 128
            for i in range(len(voxel_grid)):
                img = voxel_grid[i][0]
                mean_pos = np.mean(img[img > 0])
                mean_neg = np.mean(img[img < 0])
                img = np.clip(img, a_min=2 * mean_neg, a_max=2 * mean_pos)
                mean_pos = np.mean(img[img > 0])
                mean_neg = np.mean(img[img < 0])
                var_pos = np.var(img[img > 0])
                var_neg = np.var(img[img < 0])
                img = np.clip(img, a_min=mean_neg - 3 * var_neg, a_max=mean_pos + 3 * var_pos)
                max = np.max(img)
                min = np.min(img)
                img[img > 0] /= max
                img[img < 0] /= abs(min)
                map_img[img < 0] = img[img < 0] * 128 + 128
                map_img[img >= 0] = img[img >= 0] * 127 + 128
            return map_img

        voxel_grid = np.zeros((self.num_bins, height, width), np.float32).ravel()
        last_stamp = events[-1][3]
        first_stamp = events[0][3]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0
        events[:, 3] = (self.num_bins - 1) * (events[:, 3] - first_stamp) / deltaT
        ts = events[:, 3]
        xs = events[:, 0].astype(int)
        ys = events[:, 1].astype(int)
        pols = events[:, 2]
        pols[pols == 0] = -1

        tis = ts.astype(int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, 1, height, width))
        map_img = mapping(voxel_grid, height, width)
        return Image.fromarray(map_img.astype(np.uint8)).convert("RGB")

    def BinaryImage(self, events, height, width):
        image = np.full((height, width), 0, dtype=np.uint8)
        for row in events:
            x, y, p = int(row[0]), int(row[1]), row[2]
            if p == 0:
                image[y, x] = (255.0)
            elif p == 1:
                image[y, x] = (255.0)
        return Image.fromarray(image).convert("RGB")

    def EventHistogram(self, events, height, width):
        per_seg = len(events) // self.num_bins
        events_slices = []
        for i in range(self.num_bins):
            start = i * per_seg
            end = (i + 1) * per_seg if i < self.num_bins - 1 else len(events)
            seg_data = events[start:end]
            events_slices.append(seg_data)
        frames = np.zeros((self.num_bins, 2, height, width), dtype=np.int16)  # T,P,H,W
        for i, event_slice in enumerate(events_slices):
            y, x, t, p = event_slice[:, 0].astype(int), event_slice[:, 1].astype(int), event_slice[:, 3], event_slice[:, 2].astype(int)
            for j in range(len(event_slice)):
                np.add.at(frames, (i, p[j], x[j], y[j]), 1)

        image = np.zeros((260, 346, 3))
        for i in range(len(frames)):
            frame = frames[i].astype(float)
            frame_pos = frame[0]
            frame_pos = np.clip(frame_pos, a_min=0, a_max=10)
            max_pos = np.max(frame_pos)
            frame_pos = (frame_pos / max_pos)
            frame_neg = frame[1]
            frame_neg = np.clip(frame_neg, a_min=0, a_max=10)
            max_neg = np.max(frame_neg)
            frame_neg = (frame_neg / max_neg)
            image[:, :, 0] = frame_pos * 255
            image[:, :, 1] = frame_neg * 255
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    def TimeSurface(self, events, height, width):
        def to_timesurface_numpy(x, y, t, p, indices, timestamp_memory, all_surfaces, tau=5e3):
            current_index_pos = 0
            for index in range(len(x)):
                timestamp_memory[p[index], y[index], x[index]] = t[index]

                if index == indices[current_index_pos]:
                    timestamp_context = timestamp_memory - t[index]
                    all_surfaces[current_index_pos, :, :, :] = np.exp(timestamp_context / tau)
                    current_index_pos += 1
                    if current_index_pos > len(indices) - 1:
                        break

        def transform(events, idx):
            timestamp_memory = np.zeros((2, height, width))
            timestamp_memory -= self.tau * 3 + 1
            all_surfaces = np.zeros(
                (
                    len(idx),
                    2,
                    height,
                    width,
                )
            )

            to_timesurface_numpy(
                events[:, 0].astype(int),
                events[:, 1].astype(int),
                events[:, 3],
                events[:, 2].astype(int),
                idx,
                timestamp_memory,
                all_surfaces,
                tau=self.tau,
            )
            return all_surfaces

        TimeSteps = 3
        t = events[:, 3]
        t_norm = (t - t[0]) / (t[-1] - t[0]) * TimeSteps
        idx = np.searchsorted(t_norm, np.arange(TimeSteps) + 1)
        img = transform(events, idx)

        frame_poss = []
        frame_negs = []
        for i in range(len(img)):
            frame = img[i].astype(float)
            frame_pos = frame[0]
            frame_poss.append(frame_pos)
            frame_neg = frame[1]
            frame_negs.append(frame_neg)
        image = np.zeros((260, 346, 3))
        for i in range(TimeSteps):
            image[:, :, i] = frame_poss[i] * 128 + frame_negs[i] * 128
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    def EdgeVoxel(self, events, height, width, delta=110):
        def mapping(img):
            map_img = np.ones_like(img) * 128
            mean_pos = np.mean(img[img > 0])
            mean_neg = np.mean(img[img < 0])
            img = np.clip(img, a_min=3 * mean_neg, a_max=3 * mean_pos)
            mean_pos = np.mean(img[img > 0])
            mean_neg = np.mean(img[img < 0])
            var_pos = np.var(img[img > 0])
            var_neg = np.var(img[img < 0])
            img = np.clip(img, a_min=mean_neg - 5 * var_neg, a_max=mean_pos + 5 * var_pos)
            max = np.max(img)
            min = np.min(img)
            img[img > 0] /= max
            img[img < 0] /= abs(min)
            map_img[img < 0] = img[img < 0] * 128 + 128
            map_img[img >= 0] = img[img >= 0] * 127 + 128
            return map_img

        thr = delta / 255
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        p = events[:, 2]
        p[p == 0] = -1
        timestamp = events[:, 3]
        t = np.mod(timestamp, self.TIME_MOD)
        if t[0] > t[-1]:
            t = timestamp - 1e7
            t = np.mod(t, self.TIME_MOD)
        t /= self.TIME_SCALE
        t_num = len(t)
        t_0_index = int(t_num / 2)
        t_0 = t[t_0_index]

        edgevoxel = np.zeros((height * width))
        integral_time = np.ones((height * width)) * t_0
        for i in range(t_0_index, t_num):
            pix_index = y[i] * width + x[i]
            pk = p[i]
            edgevoxel[pix_index] = edgevoxel[pix_index] + pk * thr * np.exp(-(t[i] - integral_time[pix_index]))
            integral_time[pix_index] = t[i]

        integral_time = np.ones((height * width)) * t_0
        for i in range(t_0_index, 0, -1):
            pix_index = y[i] * width + x[i]
            pk = -p[i]
            edgevoxel[pix_index] = edgevoxel[pix_index] + pk * thr * np.exp((t[i] - integral_time[pix_index]))
            integral_time[pix_index] = t[i]
        edgevoxel = edgevoxel.reshape((height, width))
        edgevoxel = mapping(edgevoxel)
        return Image.fromarray(edgevoxel.astype(np.uint8)).convert("RGB")
