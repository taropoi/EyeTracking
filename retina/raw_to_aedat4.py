import dv_processing as dv
import pandas as pd
import numpy as np
from metavision_core.event_io.raw_reader import RawReader

raw_file_path = "F:\\1\\EyeTracking\\stage6_retina_gaze\\rawdata\\ID_000\\rawdata3.raw"  # 修改为你的实际路径
reader = RawReader(record_base=raw_file_path)
num_events_to_read = 1000000  # 一次读取的事件数

resolution = (1280, 720)
config = dv.io.MonoCameraWriter.Config("DVXplorer_sample")
config.addEventStream(resolution)
writer = dv.io.MonoCameraWriter("F:\\1\\EyeTracking\\stage6_retina_gaze\\rawdata\\ID_000\\events.aedat4", config)
print(f"Is event stream available? {str(writer.isEventStreamConfigured())}")

store = dv.EventStore()
all_events = []
while not reader.is_done():
    events = reader.load_n_events(num_events_to_read)

    if events.size == 0:
        break

    # 将结构化数组转换为简单 ndarray 格式: (x, y, p, t)
    x = events['x']
    y = events['y']
    p = events['p']
    t = events['t'].astype(np.int64)*1000
    stacked = np.stack([x, y, p, t], axis=-1)  # shape: (N, 4)

    all_events.append(stacked)

# Step 5: 合并所有事件
all_events_np = np.concatenate(all_events, axis=0)  # shape: (Total_N, 4)

for i in range(all_events_np.shape[0]):
    x = all_events_np[i, 0]
    y = all_events_np[i, 1]
    p = all_events_np[i, 2]
    t = all_events_np[i, 3]
    store.push_back(t, x, y, p)

# Write 100 packet of event data
# for ev in store:
    # EventStore requires strictly monotonically increasing data, generate
    # a timestamp from the iteration counter value
    # timestamp = store[i].timestamp()

    # Empty event store
    # events = dv.data.generate.dvLogoAsEvents(timestamp, resolution)

    # Write the packet using the writer, the data is not going be written at the exact
    # time of the call to this function, it is only guaranteed to be written after
    # the writer instance is destroyed (destructor has completed)
writer.writeEvents(store)
print("events written")
