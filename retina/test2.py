import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.ops import nms

from sinabs.from_torch import from_model
from data.utils import load_yaml_config
from training.models.retina.retina import Retina
from training.module import EyeTrackingModelModule

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import glob
from PIL import Image

def make_gif(image_folder, output_path, duration=200):
    """
    Args:
        image_folder: 可视化图像所在文件夹（frame_00.png 等）
        output_path: 生成 GIF 的保存路径
        duration: 每帧显示时间，单位毫秒
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, "frame_*.png")))
    images = [Image.open(img) for img in image_files]
    if not images:
        print("No images found for GIF.")
        return
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)
    print(f"[✓] GIF saved to {output_path}")

def make_video(image_folder, output_path, fps=10):
    """
    Args:
        image_folder: 图像所在目录
        output_path: 输出 mp4 路径
        fps: 帧率
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, "frame_*.png")))
    if not image_files:
        print("No images found for video.")
        return

    # 读取第一张图确定大小
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_files:
        frame = cv2.imread(img_path)
        writer.write(frame)

    writer.release()
    print(f"[✓] MP4 video saved to {output_path}")


def visualize_detections(frames, detections, save_path="F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth"):
    """
    Args:
        frames: Tensor [T, C, H, W]，每帧事件图（0/1 二值）
        detections: List[List[x1, y1, x2, y2, score]]，YOLO每帧检测结果
        save_path: 若指定路径则保存图像
    """
    T, C, H, W = frames.shape
    for t in range(T):
        fig, ax = plt.subplots(1, figsize=(6,6))
        # 可视化事件图（取 ch0）
        img = frames[t, 0].cpu().numpy()
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(f"Frame {t} with Detections")
        ax.axis("off")

        # 画检测框
        for det in detections[t]:
            x1, y1, x2, y2, score = det
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-4, f"{score:.2f}", color="yellow", fontsize=8)

        # 保存或显示
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"frame_{t:02d}.png"), bbox_inches="tight")
        else:
            plt.show()

        plt.close()

def slice_static(raw, num_bins, img_w, img_h, min_x=96, max_x=608):
    t, xy, p = raw['t'], raw['xy'], raw['p']
    # 裁剪 & 坐标变换
    mask = (xy[:,0] > min_x) & (xy[:,0] < max_x)
    xy, p, t = xy[mask], p[mask], t[mask]
    xy[:,0] -= min_x;  xy[:,1] += 16
    xy //= (512 // img_w)
    # 安全限制 xy 范围在 [0, img_w-1] 和 [0, img_h-1]
    xy[:, 0] = np.clip(xy[:, 0], 0, img_w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, img_h - 1)

    t0, t1 = t.min(), t.max()
    cuts = np.linspace(t0, t1, num_bins+1)

    frames = np.zeros((num_bins, 2, img_w, img_h), dtype=np.float32)
    for i in range(num_bins):
        m = (t >= cuts[i]) & (t < cuts[i+1])
        this_xy, this_p = xy[m], p[m]
        # 累加
        for pol in [0,1]:
            sel = this_p == pol
            if sel.any():
                np.add.at(frames[i, pol],
                          (this_xy[sel,0], this_xy[sel,1]), 1)
        # 冲突 & 二值化
        ch0, ch1 = frames[i,0], frames[i,1]
        keep0 = ch0 >= ch1; keep1 = ch1 > ch0
        ch0[~keep0] = 0; ch1[~keep1] = 0
        frames[i] = np.clip(frames[i], 0, 1)

    # 转 Tensor 并对齐
    tensor = torch.from_numpy(frames)
    tensor = torch.rot90(tensor, k=2, dims=(2,3)).permute(0,1,3,2)
    return tensor


def slice_dynamic(raw, num_bins, events_per_bin, img_w, img_h, min_x=96, max_x=608):
    t, xy, p = raw['t'], raw['xy'], raw['p']
    # 裁剪 & 坐标变换
    mask = (xy[:,0] > min_x) & (xy[:,0] < max_x)
    xy, p, t = xy[mask], p[mask], t[mask]
    xy[:,0] -= min_x;  xy[:,1] += 16
    xy //= (512 // img_w)
    # 安全限制 xy 范围在 [0, img_w-1] 和 [0, img_h-1]
    xy[:, 0] = np.clip(xy[:, 0], 0, img_w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, img_h - 1)


    frames = np.zeros((num_bins, 2, img_w, img_h), dtype=np.float32)
    end_idx = len(t)
    for i in reversed(range(num_bins)):
        start_idx = max(0, end_idx - events_per_bin)
        this_xy = xy[start_idx:end_idx]
        this_p  = p[start_idx:end_idx]
        # 累加
        for pol in [0,1]:
            sel = this_p == pol
            if sel.any():
                np.add.at(frames[i, pol],
                          (this_xy[sel,0], this_xy[sel,1]), 1)
        # 冲突 & 二值化
        ch0, ch1 = frames[i,0], frames[i,1]
        keep0 = ch0 >= ch1; keep1 = ch1 > ch0
        ch0[~keep0] = 0; ch1[~keep1] = 0
        frames[i] = np.clip(frames[i], 0, 1)
        end_idx = start_idx

    tensor = torch.from_numpy(frames)
    tensor = torch.rot90(tensor, k=2, dims=(2,3)).permute(0,1,3,2)
    return tensor

class RawPredictDataset(Dataset):
    def __init__(self, raw_evt3_path, dataset_params,
                 slice_type="static", events_per_bin=None):
        """
        slice_type: "static" 或 "dynamic"
        events_per_bin: 仅在 dynamic 模式下使用
        """
        from copy_test import parse_evt3_to_numpy
        data = parse_evt3_to_numpy(raw_evt3_path)
        self.raw = {'t': data['t'], 'xy': data['xy'], 'p': data['p']}

        self.num_bins = dataset_params["num_bins"]
        self.img_w = dataset_params["img_width"]
        self.img_h = dataset_params["img_height"]
        self.slice_type = slice_type
        self.events_per_bin = events_per_bin or dataset_params.get("events_per_frame", 200)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.slice_type == "static":
            frames = slice_static(
                self.raw, self.num_bins, self.img_w, self.img_h
            )
        else:
            frames = slice_dynamic(
                self.raw, self.num_bins, self.events_per_bin,
                self.img_w, self.img_h
            )
        dummy_labels = torch.zeros(self.num_bins, 2)
        avg_dt = (self.raw['t'].max() - self.raw['t'].min()) / self.num_bins

        return frames, dummy_labels, avg_dt
    
def decode_yolo(outputs, S, B, img_w, img_h, conf_th=0.3, iou_th=0.5, topk=1):
    """
    outputs: Tensor [N, output_dim]  N = batch_size * num_bins
    返回: List[List[x1,y1,x2,y2,score]]，每个 sub-list 是一个 sample 的检测框（像素坐标）
    """
    device = outputs.device
    N = outputs.shape[0]
    preds = outputs.reshape(N, S, S, B*5)
    all_boxes = []

    for n in range(N):
        boxes = []
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    off = b*5
                    cx, cy, w, h, conf = preds[n,i,j,off:off+5].tolist()
                    if conf < conf_th:
                        continue
                    # 恢复到像素坐标
                    x1 = max(0, cx - w/2) * img_w
                    y1 = max(0, cy - h/2) * img_h
                    x2 = min(1, cx + w/2) * img_w
                    y2 = min(1, cy + h/2) * img_h
                    boxes.append([x1,y1,x2,y2,conf])
        if not boxes:
            all_boxes.append([])
            continue
        tb = torch.tensor(boxes, device=device)
        if topk == 1:
            idx = tb[:,4].argmax()
            best = tb[idx].cpu().tolist()
            # 恢复像素坐标
            x1,y1,x2,y2,conf = best
            return [[x1*img_w, y1*img_h, x2*img_w, y2*img_h, conf]]

    # 否则，执行标准 NMS
    keep = nms(tb[:,:4], tb[:,4], iou_th)
    keep = keep[:topk]  # 只保留 topk
    final = tb[keep].cpu().tolist()
    # pixel-recover
    return [[x1*img_w, y1*img_h, x2*img_w, y2*img_h, conf] 
            for x1,y1,x2,y2,conf in final]
    #     tb = torch.tensor(boxes, device=device)
    #     keep = nms(tb[:,:4], tb[:,4], iou_th)
    #     final = tb[keep].cpu().tolist()
    #     all_boxes.append(final)

    # return all_boxes


if __name__ == "__main__":
    raw_file    = "F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth\\output3.raw"

    # ---- 载入参数 ----
    path_to_run='F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v2\\output\\retina-ann-v2'
    training_params = load_yaml_config(os.path.join(path_to_run, "training_params.yaml"))
    dataset_params = load_yaml_config(os.path.join(path_to_run, "dataset_params.yaml"))
    if training_params["arch_name"][:6] =="retina":
        layers_config = load_yaml_config(os.path.join(path_to_run, "layer_configs.yaml"))
    S = training_params["SxS_Grid"]
    B = training_params["num_boxes"]

    # ---- 数据 & DataLoader ----
    ds = RawPredictDataset(
        raw_evt3_path=raw_file,
        dataset_params=dataset_params,
        slice_type="dynamic",  # 或 "dynamic"
        events_per_bin=dataset_params["events_per_frame"]
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

    # ---- 构建模型 & 加载权重 ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Retina(dataset_params, training_params, layers_config)  # 或者 Baseline_3ET
    checkpoint = torch.load("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v2\\output\\retina-ann-v2\\event_eye_tracking\\cwt3w1ml\\checkpoints\\epoch=0-step=395.ckpt",map_location='cuda:0')
    state_dt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    state_dict = {k.replace("seq_", "seq_model."): v for k, v in state_dt.items()}
    model.load_state_dict(state_dict)
    lit_model = EyeTrackingModelModule(model, dataset_params, training_params)

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, enable_progress_bar=False)
    preds = trainer.predict(lit_model, dataloaders=loader)

    for out in preds:
        out = out.squeeze(0)  # [num_bins, output_dim]
        img_w, img_h = dataset_params["img_width"], dataset_params["img_height"]
        boxes = decode_yolo(out, S, B, img_w, img_h, conf_th=0.0, iou_th=0.5, topk=1)
        print("Detection per-bin:", boxes)
        visualize_detections(ds[0][0], boxes, save_path="vis_output")  # ds[0][0] 是 [T,C,H,W]
        make_gif("vis_output", "output_detection.gif", duration=150)
        make_video("vis_output", "output_detection.mp4", fps=10)