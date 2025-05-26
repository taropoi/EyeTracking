import numpy as np
# 可在 get_ini_30_dataloader() 里读取事件流后立即调用。
def apply_sae_filter(events, width, height, decay_time_us=50000):
    sae = np.full((height, width), -np.inf)
    current_time = events[-1][2]  # 最新事件时间

    mask = []
    for x, y, t, p in events:
        dt = current_time - sae[int(y), int(x)]
        if dt > decay_time_us:
            mask.append(True)
            sae[int(y), int(x)] = t
        else:
            mask.append(False)
    return events[mask]

# 插入点建议：在 ini_30_dataset.py 中将事件切为 voxel grid 前调用
def normalize_event_density(events, num_bins, width, height):
    voxel_grid = np.zeros((num_bins, height, width))
    timestamps = events[:, 2]
    t_start, t_end = timestamps[0], timestamps[-1]
    delta_t = (t_end - t_start) / num_bins

    for x, y, t, p in events:
        bin_idx = int((t - t_start) // delta_t)
        bin_idx = min(max(bin_idx, 0), num_bins - 1)
        voxel_grid[bin_idx, int(y), int(x)] += 1

    # 归一化密度
    voxel_grid /= voxel_grid.max() + 1e-6
    return voxel_grid

# STC Filtering（Space-Time Contrast Filtering） 可以借用 Metavision 风格滤波，仅保留高空间对比事件，去除噪声背景
def stc_filter(events, spatial_radius=1, temporal_threshold=5000):
    filtered = []
    event_map = {}

    for x, y, t, p in events:
        key = (int(x), int(y))
        neighbors = [
            (x + dx, y + dy)
            for dx in range(-spatial_radius, spatial_radius + 1)
            for dy in range(-spatial_radius, spatial_radius + 1)
            if dx != 0 or dy != 0
        ]

        contrast = False
        for nx, ny in neighbors:
            if (nx, ny) in event_map and abs(t - event_map[(nx, ny)]) < temporal_threshold:
                contrast = True
                break

        if contrast:
            filtered.append([x, y, t, p])
            event_map[(int(x), int(y))] = t
    return np.array(filtered)
