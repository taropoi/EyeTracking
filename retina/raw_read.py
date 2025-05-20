import numpy as np
from metavision_core.event_io.raw_reader import RawReader

# Step 1: 设置输入 RAW 文件路径
# raw_file_path = "F:\\1\\EyeTracking\\stage6_retina_gaze\\rawdata\\ID_001\\rawdata1.raw"  # 修改为你的实际路径
raw_file_path = "test.raw"  # 修改为你的实际路径
# Step 2: 初始化 RawReader
reader = RawReader(record_base=raw_file_path)

# Step 3: 设置每次读取的事件数（或使用 load_delta_t 来按时间加载）
num_events_to_read = 1000000  # 一次读取的事件数

all_events = []
all_events = reader.load_n_events(num_events_to_read)
p_values = all_events['p']
unique_p = np.unique(p_values)
print("极性值：", unique_p)

# Step 4: 循环读取直到读完
# while not reader.is_done():
#     events = reader.load_n_events(num_events_to_read)

#     if events.size == 0:
#         break

#     # 将结构化数组转换为简单 ndarray 格式: (x, y, p, t)
#     x = events['x']
#     y = events['y']
#     p = events['p']
#     t = events['t']
#     stacked = np.stack([x, y, p, t], axis=-1)  # shape: (N, 4)

#     all_events.append(stacked)

# # Step 5: 合并所有事件
# all_events_np = np.concatenate(all_events, axis=0)  # shape: (Total_N, 4)
# p_values = all_events_np[:, 2]
# unique_p = np.unique(p_values)
# print("极性值：", unique_p)

# 现在 all_events_np 是一个 (N, 4) 的 numpy 数组，列分别是 x, y, p, t
# print("总共读取事件数：", all_events_np.shape[0])
# print("示例事件：", all_events_np[:10])
