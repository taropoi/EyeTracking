import torch
import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

# 1) 导入你的 Dataset helper
from data.datasets.ini_30.helper import get_ini_30_dataset, get_indexes
from data.utils import load_yaml_config
from data.transforms.helper import get_transforms

# 2) 导入你的 LightningModule
from training.module import EyeTrackingModelModule
from training.models.retina.retina import Retina
from sinabs.from_torch import from_model

# （如有必要，还要 import convert_to_n6、convert_to_dynap 等）


def main():
    # --- 2.1 读取配置 ---
    path_to_run='F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v2\\output\\retina-ann-v2'
    training_params = load_yaml_config(os.path.join(path_to_run, "training_params.yaml"))
    dataset_params = load_yaml_config(os.path.join(path_to_run, "dataset_params.yaml"))
    if training_params["arch_name"][:6] =="retina":
        layers_config = load_yaml_config(os.path.join(path_to_run, "layer_configs.yaml"))
    # --- 2.2 构造只包含一条 raw 数据的 Dataset ---
    # 这里 name="val" 或 "train" 都行，只要你 helper 里把 raw_evt3_path 和 raw_label_csv_path 传进去即可
    dataset = get_ini_30_dataset(
        name="val",
        training_params=training_params,
        dataset_params=dataset_params,
    )
    # 注意 helper 会把 raw_evt3_path、raw_label_csv_path 硬编码进去，
    # 或者你可以改 helper，让它接收额外的参数再传进来

    # DataLoader：batch_size=1，只跑这一条
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True, 
        drop_last=True
    )

    # --- 2.3 实例化模型并加载 checkpoint（如果有的话） ---
    model = Retina(dataset_params, training_params, layers_config)  # 或者 Baseline_3ET
    checkpoint = torch.load("F:\\1\\EyeTracking\\stage6_retina_gaze\\retina_v2\\output\\retina-ann-v2\\event_eye_tracking\\cwt3w1ml\\checkpoints\\epoch=0-step=395.ckpt",map_location='cuda:0')
    state_dt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    state_dict = {k.replace("seq_", "seq_model."): v for k, v in state_dt.items()}
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 把它封到 LightningModule 里
    lit_module = EyeTrackingModelModule(model, dataset_params, training_params)
    lit_module.eval()

    # --- 2.4 用 Trainer.predict ---
    trainer = pl.Trainer(
        accelerator="gpu", devices=[0],  
        enable_progress_bar=False,
    )
    outputs = trainer.predict(
        lit_module,
        dataloaders=loader,
    )
    print("type(outputs):", type(outputs))
    print("len(outputs):", len(outputs))
    all_norm=[]
    for out in outputs:
        pts_norm = out[:,:2]
        pts = pts_norm.squeeze(0)
        all_norm.append(pts.cpu().numpy())
    all_norm = np.stack(all_norm,axis=0)

    # ——— 5. 把归一化坐标还原到 1280×720 ———
    img_w, img_h = dataset_params["img_width"], dataset_params["img_height"]  # e.g. 64,64
    min_x, min_y_off = 96, 16

    all_raw = np.zeros_like(all_norm)
    for i in range(all_norm.shape[0]):
        pts = all_norm[i]
        # 1. 先乘回网络输入像素
        x_net = pts[:,0] * img_w
        y_net = pts[:,1] * img_h
        # 2. 上采样回 512×512
        x_512 = x_net * (512/img_w)
        y_512 = y_net * (512/img_h)
        # 3. 恢复到原始框架 1280×720
        all_raw[i,:,0] = x_512 + min_x
        all_raw[i,:,1] = y_512 - min_y_off

    # ——— 6. 加上每个 bin 对应的时间戳 ———
    bins_df = dataset.load_labels(0)  # DataFrame with 'timestamp' 对应每帧
    rows = []
    for sample_idx in range(all_raw.shape[0]):
        for bin_idx in range(all_raw.shape[1]):
            rows.append({
                "sample":    sample_idx,
                "bin_idx":   bin_idx,
                # "timestamp": time_bins[bin_idx],
                "x_pred":    all_raw[sample_idx, bin_idx, 0],
                "y_pred":    all_raw[sample_idx, bin_idx, 1],
            })
    df = pd.DataFrame(rows)
    df.to_csv("raw_predictions_with_time.csv", index=False)
    print("Saved", len(df), "rows to all_raw_predictions.csv")
    print("Saved predictions:", df.head())

if __name__ == "__main__":
    main()