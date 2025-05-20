import os
import pandas as pd
# 该脚本用于生成silver索引csv
INI30_DATA_PATH = "F:\\1\\EyeTracking\\stage6_retina_gaze\\rawdata"  # 绝对路径
EXPERIMENT_NAMES = ["ID_000","ID_001","ID_002","ID_003","ID_004","ID_005"]  # 可以是多个

num_bins = 10  # 10 bins
fixed_window_dt = 2500

silver_data = []

for exp in EXPERIMENT_NAMES:
    anno_path = os.path.join(INI30_DATA_PATH, exp, "annotations.csv")
    if not os.path.exists(anno_path):
        print(f"annotations.csv not found for {exp}")
        continue

    df = pd.read_csv(anno_path)
    df = df.sort_values("timestamp")

    # 遍历所有标签（或抽取一部分）
    for _, row in df.iterrows():
        silver_data.append({
            "exp_name": exp,
            "t_start": (int(row["timestamp"]) - fixed_window_dt * (num_bins + 1)) * 1000, 
            "t_end": int(row["timestamp"]),  # 注意单位是否为微秒
            "x_coord": int(row["center_x"]),
            "y_coord": int(row["center_y"]),
        })

# 写入 TSV 文件
silver_df = pd.DataFrame(silver_data)
# 假设 df 是你的 DataFrame，n 是要删除的行数
n = 2
silver_df = silver_df.iloc[n:].reset_index(drop=True)

silver_df.to_csv(os.path.join(INI30_DATA_PATH, "silver.csv"), sep='\t', index=True)
