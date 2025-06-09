import numpy as np
import cv2
import matplotlib.pyplot as plt
import dv
import itertools
import pandas as pd
from scipy.ndimage import gaussian_filter
import struct
index = 1
def kde_center(mask, bandwidth=5):
    density = gaussian_filter(mask.astype(np.float32), sigma=bandwidth)
    center = np.unravel_index(np.argmax(density), density.shape)
    return center[::-1]  # return (x, y) 确定瞳孔位置

default_config = {
    "img_shape": (481, 641),  # 默认图像形状
    "step": 2000,  # 每次处理的事件数量
    "kernel_size_noise": 3,  # 形态学操作的核大小
    "threshold_noise": 0.001,  # 噪声过滤的阈值
    "kernel_size_eyelid": 5,  # 眼睑和glint掩码的核大小
    "kernel_size_eyelash_close_h": (20, 5), 
    "kernel_size_eyelash_disk": (10, 10), 
    "threshold_all_img": 0.1
}

def accumulate_events(events, pos_img, neg_img, all_img, img_shape, state):
    if state == 0:
        pos_img = np.zeros(img_shape, dtype=np.uint8)
        neg_img = np.zeros(img_shape, dtype=np.uint8)
        all_img = np.zeros(img_shape, dtype=np.uint8)

    for x, y, p in zip(events["x"], events["y"], events["p"]):
        all_img[y, x] += 2
        if all_img[y, x] > 255:
            all_img[y, x] = 255
        if p > 0:
            pos_img[y, x] += 2
            if pos_img[y, x] > 255:
                pos_img[y, x] = 255
        else:
            neg_img[y, x] += 2
            if neg_img[y, x] > 255:
                neg_img[y, x] = 255

    return pos_img, neg_img, all_img

def filter_noise(img, config): #单通道噪声过滤
    box = cv2.boxFilter(img.astype(np.float32), -1, (config["kernel_size_noise_single"], config["kernel_size_noise_single"]), normalize=0)
    _, noise_mask = cv2.threshold(box, config["threshold_noise_single"], 255, cv2.THRESH_BINARY_INV)
    noise_mask = noise_mask.astype(np.uint8)
    return noise_mask

def filter_noise_all(img, config): #双通道噪声过滤
    box = cv2.boxFilter(img.astype(np.float32), -1, (config["kernel_size_noise_all"], config["kernel_size_noise_all"]), normalize=0)
    _, noise_mask = cv2.threshold(box, config["threshold_noise_all"], 255, cv2.THRESH_BINARY_INV)
    noise_mask = noise_mask.astype(np.uint8)
    return noise_mask

def extract_eyelid_glint_mask(pos_img, neg_img, config): # 提取眼睑和glint掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["kernel_size_eyelid1"], config["kernel_size_eyelid1"]))
    pos_dilated = cv2.dilate(pos_img, kernel, iterations=2)
    neg_dilated = cv2.dilate(neg_img, kernel, iterations=2)
    intersection = cv2.bitwise_and(pos_dilated, neg_dilated)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["kernel_size_eyelid2"], config["kernel_size_eyelid2"]))
    mask = cv2.dilate(intersection, kernel_h, iterations=1)

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask, pos_dilated, neg_dilated, intersection, mask

def extract_eyelash_mask(all_img, eyelid_mask, config): # 提取睫毛掩码
    blur = cv2.medianBlur(all_img, config['blur_kernel_size'])
    _, blur_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

    _, all_img = cv2.threshold(all_img, 0, 255, cv2.THRESH_BINARY)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, config['kernel_size_eyelash_close_h'])
    morph = cv2.morphologyEx(all_img, cv2.MORPH_CLOSE, kernel_h)
    kernel_close_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['kernel_size_eyelash_close_disk'])
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close_disk)
    kernel_open_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['kernel_size_eyelash_open_disk'])
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open_disk)
    print(f"morph dtype: {morph.dtype}, eyelid_mask dtype: {eyelid_mask.dtype}")
    union = cv2.bitwise_or(morph, eyelid_mask)
    combined = cv2.bitwise_or(blur_mask, union)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    eyelash_mask = np.zeros_like(all_img)
    cv2.drawContours(eyelash_mask, [largest], -1, 255, -1)

    return eyelash_mask, blur, blur_mask, morph, union, combined

def extract_pupil_iris_mask(noise_mask, eyelid_mask, eyelash_mask, all_img): # 提取瞳孔和虹膜掩码
    total_mask = cv2.bitwise_or(noise_mask, eyelid_mask)
    total_mask = cv2.bitwise_or(total_mask, eyelash_mask)
    inverse_mask = cv2.bitwise_not(total_mask)

    # pupil_mask = cv2.bitwise_and(all_img, all_img, mask=inverse_mask)
    # pupil_mask = cv2.blur(pupil_mask.astype(np.float32), (3, 3))
    _, pupil_mask = cv2.threshold(inverse_mask, 0, 255, cv2.THRESH_BINARY)
    return pupil_mask

def try_segment_pupil(frame, config, state):
    """
    尝试从事件帧中提取面积最大的两个连通区域。
    返回 mask 以及是否成功分割（至少一个区域面积大于 min_area）。
    """
    mask = np.zeros_like(frame)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, state+1

    # # 计算所有轮廓的面积并排序
    # contour_areas = [(cv2.contourArea(cnt), cnt) for cnt in contours]
    # contour_areas.sort(key=lambda x: x[0], reverse=True)

    # success = False
    # for i, (area, cnt) in enumerate(contour_areas[:2]):  # 最多两个区域
    #     if area >= min_area:
    #         cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    #         success = True
    success = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        print(f"Contour area: {area}")
        if area >= config['min_area']:
            cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
            success = True

    if success == False:
        state += 1
        if state == 5:
            print("尝试分割瞳孔失败，连续失败5次。")
            state = 0
    else:
        state = 0

    return mask, state

def process_event_set(events, config, pos_img, neg_img, all_img, state): 
    pos_img, neg_img, all_img = accumulate_events(events, pos_img, neg_img, all_img, config["img_shape"], state)

    # 计算 噪声过滤与噪声mask
    pos_noise = filter_noise(pos_img, config)
    neg_noise = filter_noise(neg_img, config)
    all_noise = filter_noise_all(all_img, config)

    pos_rn = pos_img.astype(np.uint8) * (1 - pos_noise / 255.0) # 仅保留滤波后的事件
    neg_rn = neg_img.astype(np.uint8) * (1 - neg_noise / 255.0)
    all_rn = all_img.astype(np.uint8) * (1 - all_noise / 255.0)

    _, pos_rn_img = cv2.threshold(pos_rn, 0, 255, cv2.THRESH_BINARY)
    _, neg_rn_img = cv2.threshold(neg_rn, 0, 255, cv2.THRESH_BINARY)
    pos_rn_img = pos_rn_img.astype(np.uint8)
    neg_rn_img = neg_rn_img.astype(np.uint8)
    all_img = all_img.astype(np.uint8)

    eyelid_mask , pos_dilated, neg_dilated, inter, mask= extract_eyelid_glint_mask(pos_rn_img, neg_rn_img, config)
    eyelash_mask, blur, blur_mask, morph, union, combined = extract_eyelash_mask(all_img, eyelid_mask, config)

    # noise_mask = cv2.bitwise_and(pos_noise, neg_noise) 
    noise_mask = all_noise
    pupil_mask = extract_pupil_iris_mask(noise_mask, eyelid_mask, eyelash_mask, all_img)
    min_area = 5  # 最小面积阈值
    pupil_mask, state = try_segment_pupil(pupil_mask, config, state)
    print(f"Segment pupil success: {state}")

    pupil_events = {
        "x": events["x"][pupil_mask[events["y"], events["x"]] > 0],
        "y": events["y"][pupil_mask[events["y"], events["x"]] > 0]
    }

    center = kde_center(pupil_mask)
    return {
        "center": center,
        "masks": {
            "eyelid": eyelid_mask,
            "eyelash": eyelash_mask,
            "pupil": pupil_mask,
        }
    }, pos_rn, neg_rn, all_rn, state

def visualize_event_masks(pos_img, neg_img, all_img, masks, center):
    df = pd.read_csv(f"F:\\1\\EyeTracking\\stage6_retina_gaze\\evs_ini30\\evs_ini30\\ID_00{index}\\annotations.csv")
    x = df.loc[0, 'center_x']
    y = df.loc[0, 'center_y']
    timestamp = df.loc[0, 'timestamp']
    print(f"Timestamp: {timestamp}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # 1. 原始正极性图像
    axes[0].imshow(pos_img, cmap='Reds')
    axes[0].set_title('Positive Events')

    # 2. 原始负极性图像
    axes[1].imshow(neg_img, cmap='Greens')
    axes[1].set_title('Negative Events')

    # 3. 所有事件叠加图像
    axes[2].imshow(all_img, cmap='Greys')
    axes[2].set_title('All Events')

    # 4. 睫毛掩码
    axes[3].imshow(masks["eyelash"], cmap='Greys')
    # axes[3].plot(x, y, 'ro', markersize=5)
    axes[3].set_title('Eyelash Mask')

    # 5. 眼睑+glint掩码
    axes[4].imshow(masks["eyelid"], cmap='Greys')
    # axes[4].plot(x, y, 'ro', markersize=5)
    axes[4].set_title('Eyelid + Glint Mask')

    # 6. 瞳孔+虹膜掩码（以及拟合椭圆和中心点）
    # axes[5].imshow(masks["pupil"], cmap='Greys')
    axes[5].set_title('Pupil + Iris Mask')
    # axes[5].plot(x, y, 'ro', markersize=5)
    # cv2.ellipse(masks["pupil"], masks["ellipse"], (0,255,0), 2)  # 绘制椭圆
    axes[5].imshow(masks["pupil"], cmap='Reds')

    
    if center is not None:
        # axes[5].plot(center[0], center[1], 'bo', markersize=10, label="KDE Center")
        pass
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def load_aedat4_events(aedat_path, start_idx=0, step=200):
    file = dv.AedatFile(aedat_path)
    events = file['events']
    
    # 使用迭代器跳过前 start_idx 个事件，并获取 step 个事件
    sliced_events = list(itertools.islice(events, start_idx, start_idx + step))
    if not sliced_events:
        return None  # 没数据返回

    timestamp = [e.timestamp for e in sliced_events]
    x = [e.x for e in sliced_events]
    y = [e.y for e in sliced_events]
    polarity = [e.polarity for e in sliced_events]

    return {
        "x": np.array(x, dtype=np.int32),
        "y": np.array(y, dtype=np.int32),
        "t": np.array(timestamp),
        "p": np.array(polarity)
    }

def load_aerdat_events(aedat_path, start_idx=0, step=200):
    with open(aedat_path, mode='rb') as file:
        file_content = file.read()

    ''' Packet format'''
    packet_format = 'BHHI'                              # pol = uchar, (x,y) = ushort, t = uint32
    packet_size = struct.calcsize('='+packet_format)    # 16 + 16 + 8 + 32 bits => 2 + 2 + 1 + 4 bytes => 9 bytes
    num_events = len(file_content)//packet_size
    extra_bits = len(file_content)%packet_size

    '''Remove Extra Bits'''
    if extra_bits:
        file_content = file_content[0:-extra_bits]

    ''' Unpacking'''
    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    event_list.reverse()

    return event_list[start_idx * 4:(start_idx + step) * 4]  # 返回指定范围的事件数据

def run_visualization_on_file(file_path, config, file_type="aedat4"):
    print(f"Loading events from: {file_path}")
    start_idx = 0
    max_events = 1e7  # 假设最大限制，防止无限循环
    state = 0 
    while True:
        if file_type == "aedat4":
            events = load_aedat4_events(file_path, start_idx=start_idx, step=config["step"])
            print("First event timestamp:", events['t'][0])
            print("Last event timestamp:", events['t'][config["step"] - 1])
        elif file_type == "aerdat":
            events = load_aerdat_events(file_path, start_idx=start_idx, step=config["step"])
            events = {
                "x": np.array(events[1::4]),
                "y": np.array(events[2::4]),
                "t": np.array(events[0::4]),
                "p": np.array(events[3::4])
            }
        else:
            raise ValueError("Unsupported file type")

        if events is None or len(events["x"]) == 0:
            print("No more events to load.")
            break

        print(f"[{start_idx} - {start_idx + config['step']}) Events Loaded: {len(events['x'])}")
        if state == 0 : #判断state
            pos_img = []
            neg_img = []
            all_img = []
        result, pos_img, neg_img, all_img, state = process_event_set(events, config, pos_img, neg_img, all_img, state)
        visualize_event_masks(pos_img, neg_img, all_img, result["masks"], result["center"])

        start_idx += config["step"]
        user_input = input("按 Enter 查看下一段，输入 p 然后 Enter 可退出：")
        if user_input.lower() == "p":
            print("手动退出循环。")
            break

if __name__ == "__main__":
    config = default_config.copy()
    config["step"] = 2000  # 每次处理的事件数量
    config["img_shape"] = (261, 347)  # 图像形状
    # config["img_shape"] = (481, 641)  # 图像形状

    config["kernel_size_noise_single"] = 5  # 单通道噪声过滤的核大小
    config["threshold_noise_single"] = 3  # 单通道噪声过滤的阈值
    config["kernel_size_noise_all"] = 5  # 双通道噪声过滤的核大小
    config["threshold_noise_all"] = 10  # 双通道噪声过滤的阈值

    config["kernel_size_eyelid1"] = 10  # 眼睑和glint掩码, 正负图像分别膨胀的核大小
    config["kernel_size_eyelid2"] = 10  # 眼睑和glint掩码，相交图像膨胀的核大小

    config['blur_kernel_size'] = 3 # 睫毛模糊核大小
    config["kernel_size_eyelash_close_h"] = (15, 5)  # 睫毛闭操作的核大小
    config["kernel_size_eyelash_close_disk"] = (10, 10)  # 睫毛闭操作的椭圆核大小
    config["kernel_size_eyelash_open_disk"] = (25, 25)  # 睫毛开操作的椭圆核大小

    config["min_area"] = 10  # 判断瞳孔是否存在的最小面积阈值

    # path_to_data = f"F:\\1\\EyeTracking\\stage6_retina_gaze\\evs_ini30\\evs_ini30\\ID_00{index}\\events.aedat4"  
    path_to_data = "events.aerdat"
    # run_visualization_on_file(path_to_data, config=config, file_type="aedat4")
    run_visualization_on_file(path_to_data, config=config, file_type="aerdat")