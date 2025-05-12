"""
This is a file script used for loading the dataset
"""
from copy_test import parse_evt3_to_numpy
import pathlib
from typing import List, Callable, Optional
import os
import torch
import pdb
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import tonic
from tonic.io import make_structured_array

from data.datasets.ini_30.ini_30_aeadat_processor import read_csv, AedatProcessorLinear

load_dotenv() 


class Ini30Dataset:
    def __init__(
        self,
        training_params,
        dataset_params, 
        list_experiments: list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        raw_evt3_path: Optional[str] = None,
        raw_label_csv_path: Optional[str] = None
    ):  
        # Transforms
        self.transform = transform
        self.target_transform = target_transform

        # Parameters
        self.num_bins = dataset_params["num_bins"]
        self.events_per_frame = dataset_params["events_per_frame"] 
        self.fixed_window = dataset_params["fixed_window"]
        self.fixed_window_dt = dataset_params["fixed_window_dt"]

        self.data_dir = os.getenv("INI30_DATA_PATH")
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # augmentations
        self.event_drop = dataset_params["event_drop"]
        self.uniform_noise = dataset_params["uniform_noise"]
        self.time_jitter = dataset_params["time_jitter"]

        self.y = pd.read_csv(os.path.join(self.data_dir, "silver.csv"), delimiter='\t') 
        self.experiments = np.unique(self.y["exp_name"]).tolist()

        filter_values = [self.experiments[item] for item in list_experiments]
        self.y = self.y[self.y["exp_name"].isin(filter_values)]

        # correct cropped
        self.min_x, self.max_x = 96, 608
        self.y = self.y[(self.y.x_coord > self.min_x) & (self.y.x_coord < self.max_x)]
        self.y.x_coord -= self.min_x
        self.y.y_coord += 16

        self.avg_dt = 0
        self.items = 0
        self.raw_evt3_path = raw_evt3_path
        self.raw_label_csv_path = raw_label_csv_path

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return self.__class__.__name__

    def load_labels(self, index):
        # collect labels
        # —— 新增 raw case —— 
        if self.raw_label_csv_path is not None:
            df = pd.read_csv(self.raw_label_csv_path)
            raw_ts = df["timestamp"].to_numpy()
            raw_x  = df["center_x"].to_numpy()
            raw_y  = df["center_y"].to_numpy()
            
            mask = (raw_x > self.min_x) & (raw_x < self.max_x)
            x_crop = raw_x[mask] - self.min_x
            y_crop = raw_y[mask] + 16
            ts     = raw_ts[mask]
            
            # 4. 构建标准 DataFrame：保持 timestamp、center_x/center_y 这几列名
            df2 = pd.DataFrame({
                "timestamp": ts,
                "center_x":  x_crop,
                "center_y":  y_crop,
            })
            # 5. 排序并重索引
            df2 = df2.sort_values("timestamp").reset_index(drop=True)

            return df2
        item = self.y.iloc[index]
        path_to_exp = os.path.join(self.data_dir, item["exp_name"])

        tab = read_csv(
            pathlib.Path(os.path.join(path_to_exp, "annotations.csv")), False, True
        )
        tab = tab.sort_values(by="timestamp")
        tab = tab[tab["timestamp"] <= item["t_end"]]
        if self.fixed_window:
            tab = tab[
                tab["timestamp"]
                >= item["t_end"] - self.fixed_window_dt * (self.num_bins + 1)
            ]

        # center crop labels : 640x480 -> 512x512
        tab.center_x = 512 - (tab.center_x - self.min_x)
        tab.center_y = 512 - (tab.center_y + 16)

        return tab

    def load_events(self, index):
        if self.raw_evt3_path is not None:
            # 1) 读 raw EVT3
            data = parse_evt3_to_numpy(self.raw_evt3_path)
            # 2) 可以做和原来一样的 center-crop + 下采样
            xy  = data["xy"]
            t   = data["t"]
            p   = data["p"]
            # center crop: 640x480 -> 512x512
            mask = (xy[:,0] > self.min_x) & (xy[:,0] < self.max_x)
            xy  = xy[mask];  t = t[mask]; p = p[mask]
            xy[:,0] -= self.min_x;  xy[:,1] += 16
            # downsample: 512 -> img_width
            xy //= (512 // self.img_width)
            sort_idx = np.argsort(t)
            xy = xy[sort_idx]
            t = t[sort_idx]
            p = p[sort_idx]
            # df = pd.DataFrame({"xy": xy, "t": t, "p": p})
            # df.to_csv("rawdata.csv", index=False)
            return {"xy": xy, "t": t, "p": p}
        item = self.y.iloc[index]
        path_to_exp = os.path.join(self.data_dir, item["exp_name"])
        aedat_path = pathlib.Path(os.path.join(path_to_exp, "events.aedat4"))
        aedat_processor = AedatProcessorLinear(aedat_path, 0.25, 1e-7, 0.5)  # 1e-7
        events = aedat_processor.collect_events(0, item["t_end"])
        evs_coord = events.coordinates()
        evs_timestamp = events.timestamps()
        evs_features = events.polarities().astype(np.byte)

        # center crop events : 640x480 -> 512x512
        evs_idx = (evs_coord[:, 0] > self.min_x) & (evs_coord[:, 0] < self.max_x)
        evs_timestamp = evs_timestamp[evs_idx]
        evs_features = evs_features[evs_idx]
        evs_coord = evs_coord[evs_idx, :]
        evs_coord[:, 0] -= self.min_x
        evs_coord[:, 1] += 16

        # down sample : 512x512 -> img_width x img_height
        evs_coord //= 512 // self.img_width

        return {"t": evs_timestamp, "p": evs_features, "xy": evs_coord}

    def load_static_window(self, data, labels):
        
        # collect labels
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        start_label = (int(tab_start.center_x.item()), int(tab_start.center_y.item()))
        end_label = (int(tab_last.center_x.item()), int(tab_last.center_y.item()))

        start_time = tab_last["timestamp"] - self.fixed_window_dt * (self.num_bins + 1)
        evs_t = data["t"][data["t"] >= start_time]
        evs_p, evs_xy = data["p"][-evs_t.shape[0] :], data["xy"][-evs_t.shape[0] :, :]

        # frame
        data = np.zeros((self.num_bins, self.input_channel, self.img_width, self.img_height))

        # indexes
        start_idx = 0

        # get intermediary labels based on num of bins
        fixed_timestamps = np.linspace(start_time, tab_last["timestamp"], self.num_bins)
        x_axis, y_axis = [], []

        for i, fixed_tmp in enumerate(fixed_timestamps):
            # label
            idx = np.searchsorted(labels["timestamp"], fixed_tmp, side="left")
            if idx == 0:
                x_axis.append(start_label[0])
                y_axis.append(start_label[1])
            elif idx == len(labels["timestamp"]):
                x_axis.append(end_label[0])
                y_axis.append(end_label[1])
            else:  # Weighted interpolation
                t0 = labels["timestamp"].iloc[idx - 1]
                t1 = labels["timestamp"].iloc[idx]

                weight0 = (t1 - fixed_tmp) / (t1 - t0)
                weight1 = (fixed_tmp - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_x"] * weight0
                        + labels.iloc[idx]["center_x"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_y"] * weight0
                        + labels.iloc[idx]["center_y"] * weight1
                    )
                )

            # slice
            t = evs_t[start_idx:][evs_t[start_idx:] <= fixed_tmp]
            if t.shape[0] == 0:
                continue
            
            xy = evs_xy[start_idx : start_idx + t.shape[0], :]
            p = evs_p[start_idx : start_idx + t.shape[0]]

            np.add.at(data[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
            
            if self.input_channel > 1:
                np.add.at(
                    data[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                )
                data[i, 0, :, :][
                    data[i, 1, :, :] >= data[i, 0, :, :]
                ] = 0  # if ch 1 has more evs than 0
                data[i, 1, :, :][
                    data[i, 1, :, :] < data[i, 0, :, :]
                ] = 0  # if ch 0 has more evs than 1

            data[i] = data[i].clip(0, 1)  # no double events

            # move pointers
            start_idx += t.shape[0]

        frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        frames = frames.permute(0, 1, 3, 2) 
        labels = self.target_transform(np.vstack([x_axis, y_axis]))

        self.avg_dt += (evs_t[-1] - evs_t[0]) / self.num_bins
        self.items += 1

        return frames, labels

    def find_first_n_unique_pairs(self, events, N):
        seen_pairs = set()
        result = []
        seen = 0

        for i in reversed(range(len(events))):
            event_tuple = tuple(events[i])

            if event_tuple not in seen_pairs:
                seen_pairs.add(event_tuple)
                result.append(events[i])
                seen += 1
                if seen == N:
                    break
            else:
                result.append(events[i])

        return np.array(result)

    def load_dynamic_window(self, data, labels):
        tab_start, tab_last = labels.iloc[0], labels.iloc[-1]
        start_label = (int(tab_start.center_x.item()), int(tab_start.center_y.item()))
        end_label = (int(tab_last.center_x.item()), int(tab_last.center_y.item()))

        evs_t, evs_p, evs_xy = data["t"], data["p"], data["xy"]

        # frame
        data = np.zeros(
            (self.num_bins, self.input_channel, self.img_width, self.img_height)
        )

        # label
        x_axis, y_axis = [], []

        # indexes
        start_idx, end_idx = 0, len(evs_p) - 1
        for i in reversed(range(self.num_bins)):
            xy = self.find_first_n_unique_pairs(
                evs_xy[:end_idx, :], self.events_per_frame
            )
            start_idx = end_idx - len(xy)
            p = evs_p[start_idx:end_idx]

            np.add.at(data[i, 0], (xy[p == 0, 0], xy[p == 0, 1]), 1)
            if self.input_channel > 1:
                np.add.at(
                    data[i, self.input_channel - 1], (xy[p == 1, 0], xy[p == 1, 1]), 1
                )
                data[i, 0, :, :][
                    data[i, 1, :, :] >= data[i, 0, :, :]
                ] = 0  # if ch 1 has more evs than 0
                data[i, 1, :, :][
                    data[i, 1, :, :] < data[i, 0, :, :]
                ] = 0  # if ch 0 has more evs than 1

            data[i] = data[i].clip(0, 1)

            # label time
            label_time = (evs_t[start_idx] + evs_t[end_idx]) / 2

            # move pointers
            end_idx = start_idx

            # label
            idx = np.searchsorted(labels["timestamp"], label_time, side="left")
            if idx == 0:
                x_axis.append(start_label[0])
                y_axis.append(start_label[1])
            elif idx == len(labels["timestamp"]):
                x_axis.append(end_label[0])
                y_axis.append(end_label[1])
            else:  # Weighted interpolation
                t0 = labels["timestamp"].iloc[idx - 1]
                t1 = labels["timestamp"].iloc[idx]

                weight0 = (t1 - label_time) / (t1 - t0)
                weight1 = (label_time - t0) / (t1 - t0)

                x_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_x"] * weight0
                        + labels.iloc[idx]["center_x"] * weight1
                    )
                )
                y_axis.append(
                    int(
                        labels.iloc[idx - 1]["center_y"] * weight0
                        + labels.iloc[idx]["center_y"] * weight1
                    )
                )
        frames = torch.rot90(torch.tensor(data), k=2, dims=(2, 3))
        frames = frames.permute(0, 1, 3, 2)
        x_axis.reverse()
        y_axis.reverse()
        labels = self.target_transform(np.vstack([x_axis, y_axis]))
        avg_dt = (evs_t[-1] - evs_t[start_idx + len(xy)]) / self.num_bins

        return frames, labels, avg_dt

    def __getitem__(self, index):
        data = self.load_events(index)
        ts, xy, p = data["t"], data["xy"], data["p"]

        if self.raw_evt3_path is not None:
            # 1) 初始化一个全 0 的事件帧数组 (T, C, W, H)
            frames_np = np.zeros(
                (self.num_bins, self.input_channel, self.img_width, self.img_height),
                dtype=np.float32,
            )
            # 2) 定义时间切分边界 [t0, t1, …, tN]
            t_start, t_end = ts.min(), ts.max()
            cuts = np.linspace(t_start, t_end, self.num_bins + 1)

            # 3) 遍历每个 bin，累加事件
            for i in range(self.num_bins):
                mask = (ts >= cuts[i]) & (ts < cuts[i + 1])
                this_xy = xy[mask].copy()
                this_p = p[mask]

                # 中心裁剪 + Y 平移
                valid = (this_xy[:, 0] > self.min_x) & (this_xy[:, 0] < self.max_x)
                this_xy = this_xy[valid]
                this_p = this_p[valid]
                this_xy[:, 0] -= self.min_x
                this_xy[:, 1] += 16

                # 下采样 512->img_width
                this_xy //= (512 // self.img_width)

                # 累加不同极性
                for polarity in [0, 1]:
                    coords = this_xy[this_p == polarity]
                    if coords.size > 0:
                        np.add.at(
                            frames_np[i, polarity],
                            (coords[:, 0], coords[:, 1]),
                            1,
                        )

                # 极性冲突处理：同一像素若两个通道都有事件，保留事件数更多的那个
                if self.input_channel > 1:
                    ch0 = frames_np[i, 0]
                    ch1 = frames_np[i, 1]
                    # mask0: 0 通道事件 >= 1 通道
                    keep0 = ch0 >= ch1
                    # mask1: 1 通道事件 > 0 通道
                    keep1 = ch1 > ch0
                    ch0[~keep0] = 0
                    ch1[~keep1] = 0

                # 二值化（任何大于 1 的计数都变成 1）
                frames_np[i] = np.clip(frames_np[i], 0, 1)

            # 4) 转为 tensor 并对齐方向（与 AEDAT 分支保持一致）
            frames = torch.from_numpy(frames_np)
            frames = torch.rot90(frames, k=2, dims=(2, 3))
            frames = frames.permute(0, 1, 3, 2)

            # 5) 构造 dummy labels（推理时可随意）
            dummy_labels = torch.zeros(self.num_bins, 2, dtype=torch.float32)
            avg_dt = (t_end - t_start) / self.num_bins

            return frames, dummy_labels, avg_dt
        labels = self.load_labels(index)
        events = self.load_events(index)
        tmp_struct = make_structured_array(
            events["xy"][:, 0], events["xy"][:, 1], events["t"], events["p"]
        )

        if self.time_jitter:
            tj_fn = tonic.transforms.TimeJitter(std=100, clip_negative=True)
            tmp_struct = tj_fn(tmp_struct)

        if self.uniform_noise:
            un_fn = tonic.transforms.UniformNoise(
                sensor_size=(self.img_width, self.img_height, self.input_channel),
                n=1000,
            )
            tmp_struct = un_fn(tmp_struct)

        if self.event_drop:
            tj_fn = tonic.transforms.DropEvent(p=1 / 100)
            tmp_struct = tj_fn(tmp_struct)

        events = {
            "xy": np.hstack(
                [tmp_struct["x"].reshape(-1, 1), tmp_struct["y"].reshape(-1, 1)]
            ),
            "p": tmp_struct["p"] * 1,
            "t": tmp_struct["t"],
        }

        # interpolate labels and events
        if self.fixed_window:
            events, labels = self.load_static_window(events, labels)
            avg_dt = self.fixed_window_dt
        else:
            events, labels, avg_dt = self.load_dynamic_window(events, labels)

        event_tensor = events.float()
        labels_tensor = labels.float() 
        
        if event_tensor.shape[1] == 1:
            event_tensor = 1 - event_tensor

        return event_tensor, labels_tensor, avg_dt
