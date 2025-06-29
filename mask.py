import numpy as np
import cv2
import matplotlib.pyplot as plt
import dv
import itertools
import pandas as pd
from scipy.ndimage import gaussian_filter
import struct

def kde_center(mask, bandwidth=5): # 使用高斯核密度估计来确定瞳孔位置
    density = gaussian_filter(mask.astype(np.float32), sigma=bandwidth)
    center = np.unravel_index(np.argmax(density), density.shape)
    return center[::-1]  # return (x, y) 确定瞳孔位置

def get_ellipse_from_mask(mask): # 椭圆拟合
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # 没找到轮廓

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None  # 不足以拟合椭圆

    ellipse = cv2.fitEllipse(largest)
    # 返回 (center(x, y), (major_axis, minor_axis), angle)
    return ellipse

def crop_rotated_roi(image, ellipse, output_size=(64, 64)): # 裁剪旋转的感兴趣区域，用于实际裁剪
    center, size, angle = ellipse
    # 构建仿射矩阵并旋转图像
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 仿射变换整个图像（对掩码图也适用）
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    # 从旋转后的图像中裁剪 64x64 框
    x, y = int(center[0]), int(center[1])
    w, h = output_size
    x1 = max(x - w // 2, 0)
    y1 = max(y - h // 2, 0)
    x2 = x1 + w
    y2 = y1 + h

    # 防止越界
    if x2 > rotated.shape[1] or y2 > rotated.shape[0]:
        pad_x = max(0, x2 - rotated.shape[1])
        pad_y = max(0, y2 - rotated.shape[0])
        rotated = cv2.copyMakeBorder(rotated, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)

    roi = rotated[y1:y2, x1:x2]
    return roi

def draw_rotated_crop_box(image, ellipse, box_size=(64, 64), color=(0, 0, 255), thickness=2): #裁剪旋转的感兴趣区域，用于可视化
    center, _, angle = ellipse
    w, h = box_size

    # 构造框的四个角点（相对于中心点）
    half_w, half_h = w / 2, h / 2
    box_pts = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ], dtype=np.float32)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 旋转加平移四个角点
    box_pts_rotated = np.dot(box_pts, M[:, :2].T)# + M[:, 2]
    box_pts_rotated = box_pts_rotated + np.array(center, dtype=np.float32)
    # print(f"center: {center}, angle: {angle}, box_pts_rotated: {box_pts_rotated}")
    # 转换为 int 并绘制
    box_pts_rotated = np.int32(box_pts_rotated).reshape((-1, 1, 2))
    image_inv = 255 - image if len(image.shape) == 2 else image.copy() # 反转方便RGB显示
    # image_inv = image.copy()
    image_with_box = cv2.cvtColor(image_inv, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv2.polylines(image_with_box, [box_pts_rotated], isClosed=True, color=color, thickness=thickness)

    return image_with_box

def process_and_draw_crop_on_mask(image, mask, box_size=(64, 64), color=(0, 0, 255), label='eyelid'):
    """
    对指定掩码进行椭圆拟合并裁剪，同时在原图上绘制方框可视化
    """
    ellipse = get_ellipse_from_mask(mask)
    if ellipse is None:
        print(f"[{label}] No ellipse found.")
        return None, image

    # 裁剪旋转后的感兴趣区域
    patch = crop_rotated_roi(mask, ellipse, output_size=box_size)

    # 可视化绘制红框
    image_with_box = draw_rotated_crop_box(image, ellipse, box_size=box_size, color=color, thickness=2)

    return patch, image_with_box

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

def accumulate_events(events, pos_img, neg_img, all_img, img_shape):
    # 创建空白图像
    pos_img = np.zeros(img_shape, dtype=np.uint8)
    neg_img = np.zeros(img_shape, dtype=np.uint8)
    all_img = np.zeros(img_shape, dtype=np.uint8)

    for x, y, p in zip(events["x"], events["y"], events["p"]):
        all_img[y, x] += 1
        if all_img[y, x] > 255:
            all_img[y, x] = 255
        if p > 0:
            pos_img[y, x] += 1
            if pos_img[y, x] > 255:
                pos_img[y, x] = 255
        else:
            neg_img[y, x] += 1
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
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["kernel_size_eyelid1"], config["kernel_size_eyelid1"]))
    pos_dilated = cv2.dilate(pos_img, kernel1, iterations=2)
    neg_dilated = cv2.dilate(neg_img, kernel1, iterations=2)
    intersection = cv2.bitwise_and(pos_dilated, neg_dilated)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["kernel_size_eyelid2"], config["kernel_size_eyelid2"]))
    mask = cv2.dilate(intersection, kernel2, iterations=1)

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask, pos_dilated, neg_dilated, intersection, mask

def extract_eyelash_mask(all_img, eyelid_mask, config): # 提取睫毛掩码
    # blur = cv2.medianBlur(all_img, config['blur_kernel_size'])
    # _, blur_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

    _, all_img = cv2.threshold(all_img, 0, 255, cv2.THRESH_BINARY)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, config['kernel_size_eyelash_close_h'])
    morph = cv2.morphologyEx(all_img, cv2.MORPH_CLOSE, kernel_h)
    kernel_close_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['kernel_size_eyelash_close_disk'])
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close_disk)
    kernel_open_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['kernel_size_eyelash_open_disk'])
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open_disk)
    union = cv2.bitwise_or(morph, eyelid_mask)
    # combined = cv2.bitwise_or(blur_mask, union)
    combined = union.copy()
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    eyelash_mask = np.zeros_like(all_img)
    cv2.drawContours(eyelash_mask, [largest], -1, 255, -1)

    # return eyelash_mask, blur, blur_mask, morph, union, combined
    return eyelash_mask, morph, union, combined

def extract_pupil_iris_mask(noise_mask, eyelid_mask, eyelash_mask, all_img): # 提取瞳孔和虹膜掩码
    total_mask = cv2.bitwise_or(noise_mask, eyelid_mask)
    total_mask = cv2.bitwise_or(total_mask, eyelash_mask)
    inverse_mask = cv2.bitwise_not(total_mask)

    # pupil_mask = cv2.bitwise_and(all_img, all_img, mask=inverse_mask)
    # pupil_mask = cv2.blur(pupil_mask.astype(np.float32), (3, 3))
    _, pupil_mask = cv2.threshold(inverse_mask, 0, 255, cv2.THRESH_BINARY)
    return pupil_mask

def try_segment_pupil(frame, config, state, count, sum, min_area=100): # 分割瞳孔
    mask = np.zeros_like(frame)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sum += 1
    if not contours:
        return mask, 0, count, sum  # 没有轮廓，直接返回

    # 计算所有轮廓的面积并排序
    contour_areas = [(cv2.contourArea(cnt), cnt) for cnt in contours]
    contour_areas.sort(key=lambda x: x[0], reverse=True)

    success = False
    for i, (area, cnt) in enumerate(contour_areas[:3]):  # 最多三个区域
        if area >= min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            print(f"Contour {i+1} area: {area}")
            success = True
    # success = False
    # if contours:
    #     largest = max(contours, key=cv2.contourArea)
    #     area = cv2.contourArea(largest)
    #     # print(f"Contour area: {area}")
    #     if area >= config['min_area']:
    #         cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    #         success = True
    #         count += 1

    if success == False:
        state = 0
    else:
        state = 1
        count += 1

    return mask, state, count, sum

def process_event_set(events, config, pos_img, neg_img, all_img, state, count=0, sum=0): 
    pos_img, neg_img, all_img = accumulate_events(events, pos_img, neg_img, all_img, config["img_shape"])

    # 计算 噪声过滤与噪声mask
    pos_noise = filter_noise(pos_img, config)
    neg_noise = filter_noise(neg_img, config)
    all_noise = filter_noise_all(all_img, config)

    # 仅保留噪声过滤后的像素
    pos_rn = pos_img.astype(np.uint8) * (1 - pos_noise / 255.0) 
    neg_rn = neg_img.astype(np.uint8) * (1 - neg_noise / 255.0)
    all_rn = all_img.astype(np.uint8) * (1 - all_noise / 255.0)

    # 将正负极性图像转换为二值图像
    _, pos_rn_img = cv2.threshold(pos_rn, 0, 255, cv2.THRESH_BINARY)
    _, neg_rn_img = cv2.threshold(neg_rn, 0, 255, cv2.THRESH_BINARY)
    _, all_rn_img = cv2.threshold(all_rn, 0, 255, cv2.THRESH_BINARY)
    pos_rn_img = pos_rn_img.astype(np.uint8)
    neg_rn_img = neg_rn_img.astype(np.uint8)
    all_rn_img = all_rn_img.astype(np.uint8)

    # 提取眼睑mask和睫毛mask
    eyelid_mask , pos_dilated, neg_dilated, inter, mask= extract_eyelid_glint_mask(pos_rn_img, neg_rn_img, config)
    eyelash_mask, morph, union, combined = extract_eyelash_mask(all_img, eyelid_mask, config)

    # 提取瞳孔mask
    # noise_mask = cv2.bitwise_and(pos_noise, neg_noise) 
    noise_mask = all_noise
    pupil_mask = extract_pupil_iris_mask(noise_mask, eyelid_mask, eyelash_mask, all_img)
    # 提取瞳孔位置（面积最大）
    min_area = config["min_area"]
    raw = pupil_mask.copy()
    pupil_mask, state, count, sum= try_segment_pupil(pupil_mask, config, state, count, sum, min_area)
    # print(f"Segment pupil success: {state}")

    pupil_events = {
        "x": events["x"][pupil_mask[events["y"], events["x"]] > 0],
        "y": events["y"][pupil_mask[events["y"], events["x"]] > 0]
    }
    # 提取密度中心作为预测瞳孔中心坐标
    center = kde_center(pupil_mask)

    # print(f"state:{state}")
    return {
        "center": center,
        "masks": {
            "eyelid": eyelid_mask,
            "eyelash": eyelash_mask,
            "pupil": pupil_mask,
        }
    }, pos_rn, neg_rn, all_rn, state, count, sum

def visualize_event_masks(pos_img, neg_img, all_img, index, masks, center):
    df = pd.read_csv(f"F:\\1\\EyeTracking\\stage6_retina_gaze\\evs_ini30\\evs_ini30\\ID_00{index}\\annotations.csv")
    x = df.loc[0, 'center_x']
    y = df.loc[0, 'center_y']
    timestamp = df.loc[0, 'timestamp']
    # print(f"Timestamp: {timestamp}")

    eyelid_patch, eyelid_boxed_img = process_and_draw_crop_on_mask(masks['eyelid'], masks['eyelid'], box_size=(64, 64), color=(255, 0, 0), label='eyelid')
    eyelash_patch, eyelash_boxed_img = process_and_draw_crop_on_mask(masks['eyelash'], masks['eyelash'], box_size=(64, 64), color=(0, 255, 0), label='eyelash')
    pupil_patch, pupil_boxed_img = process_and_draw_crop_on_mask(masks['pupil'], masks['pupil'], box_size=(64, 64), color=(0, 0, 255), label='pupil')

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
    # axes[3].imshow(masks["eyelash"], cmap='Greys')
    axes[3].imshow(eyelash_boxed_img, cmap='Greys')
    # axes[3].plot(x, y, 'ro', markersize=5)
    axes[3].set_title('Eyelash Mask')

    # 5. 眼睑+glint掩码
    # axes[4].imshow(masks["eyelid"], cmap='Greys')
    axes[4].imshow(eyelid_boxed_img, cmap='Greys')
    # axes[4].plot(x, y, 'ro', markersize=5)
    axes[4].set_title('Eyelid + Glint Mask')

    # 6. 瞳孔+虹膜掩码（以及拟合椭圆和中心点）
    # axes[5].imshow(masks["pupil"], cmap='Reds')
    # ellipse = get_ellipse_from_mask(masks["pupil"])
    axes[5].imshow(pupil_boxed_img, cmap='Reds')
    # if ellipse is not None:
    #     cropped_img = draw_rotated_crop_box(masks["pupil"], ellipse, box_size=(64, 64), color=(0, 255, 0), thickness=2)
    #     axes[5].imshow(cropped_img, cmap='Greys')
    #     axes[5].plot(x, y, 'ro', markersize=5)
    #     axes[5].set_title('Pupil + Iris Mask with Ellipse')
    # else:
    #     axes[5].set_title('Pupil + Iris Mask')
    #     axes[5].plot(x, y, 'ro', markersize=5)
    #     # cv2.ellipse(masks["pupil"], masks["ellipse"], (0,255,0), 2)  # 绘制椭圆
    #     axes[5].imshow(masks["pupil"], cmap='Reds')

    if center is not None:
        axes[5].plot(center[0], center[1], 'bo', markersize=10, label="KDE Center")
        # pass
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

def run_visualization_on_file(file_path, config, index, file_type="aedat4"):
    print(f"Loading events from: {file_path}")
    start_idx = 0 # 初始索引
    state = 0 # 初始状态，用于指示是否检查到瞳孔
    count = 0 # 成功检查到瞳孔的帧数
    sum = 0 # 读取总帧数
    while True:
        if file_type == "aedat4":
            events = load_aedat4_events(file_path, start_idx=start_idx, step=config["step"])
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
        pos_img = [] 
        neg_img = []
        all_img = []
        # print(f"[{start_idx} - {start_idx + config['step']}) Events Loaded: {len(events['x'])}")
        result, pos_img, neg_img, all_img, state, count, sum = process_event_set(events, config, pos_img, neg_img, all_img, state, count, sum)
        visualize_event_masks(pos_img, neg_img, all_img, index, result["masks"], result["center"])
        
        start_idx += config["step"]
        print(f"state: {state}, count: {count}, sum: {sum}")
        user_input = input("按 Enter 查看下一段，输入 p 然后 Enter 可退出：")
        if user_input.lower() == "p":
            print("手动退出循环。")
            break
    print(f"total count: {sum}, success: {count}")

if __name__ == "__main__":
    index = 1
    config = default_config.copy()
    config["step"] = 4000  # 每次处理的事件数量
    # config["img_shape"] = (261, 347)  # 图像形状
    config["img_shape"] = (481, 641)  # 图像形状

    config["kernel_size_noise_single"] = 5  # 单通道噪声过滤的核大小
    config["threshold_noise_single"] = 3  # 单通道噪声过滤的阈值
    config["kernel_size_noise_all"] = 5  # 双通道噪声过滤的核大小
    config["threshold_noise_all"] = 5  # 双通道噪声过滤的阈值

    config["kernel_size_eyelid1"] = 10  # 眼睑和glint掩码, 正负图像分别膨胀的核大小
    config["kernel_size_eyelid2"] = 10  # 眼睑和glint掩码，相交图像膨胀的核大小

    config['blur_kernel_size'] = 3 # 睫毛模糊核大小
    config["kernel_size_eyelash_close_h"] = (15, 5)  # 睫毛闭操作的核大小
    config["kernel_size_eyelash_close_disk"] = (10, 10)  # 睫毛闭操作的椭圆核大小
    config["kernel_size_eyelash_open_disk"] = (20, 20)  # 睫毛开操作的椭圆核大小

    config["min_area"] = 5  # 判断瞳孔是否存在的最小面积阈值

    path_to_data = f"F:\\1\\EyeTracking\\stage6_retina_gaze\\evs_ini30\\evs_ini30\\ID_00{index}\\events.aedat4"  
    # path_to_data = "events.aerdat"
    run_visualization_on_file(path_to_data, config, index, file_type="aedat4")
    # run_visualization_on_file(path_to_data, config=config, file_type="aerdat")