import numpy as np
import os
import csv
import cv2
import h5py
import math
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, SparseOpticalFlowConfigPreset, SparseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
from skvideo.io import FFmpegWriter
import matplotlib.pyplot as plt
import time

fig_per_frame = 0
begin_frame = 50
end_frame = 100

ours = 0
ours_dt_step = 200000
ours_delta = 100000
event_group_num = 350

raw_name = "output1"
ori_raw_path = "F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth\\" + raw_name + ".raw"  # car-300000/10   ped-7000/50
# event_path = "E:\EC\Share\event100000.npy"


ours_max_link_time = 7000
ours_min_cluster_size = 50
ours_distance_gain = 0.05000000074505806 #
ours_omega_cutoff = 7.0
ours_dampping = 0.7070000171661377
ours_match_polarity = 0

overlap_ratio = 1
disable_filter = 1  # 1: 关闭过滤器
use_monitor = 1
flow_box_extract_speed = 1
use_frame_or_shape = 0   # 1: 红框检测  0: 点蔟标蓝
jud_shape = 0    # 1: 根据聚类判断不同点蔟形状
detect_eys = 1   # 1: 追踪眼球圆形形状  *传统边缘检测算法检测眼球

# if use shape
cut_off = 0
cut_scale = 1
cut_swift = 200
cut_abs_swift = (0, 0)

save_csv = "F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth\\" if 0 else ""
save_avi = ("F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth\\" + raw_name) if 1 else ""

# 宽度：int，高度：int，distance_gain：float = 0.05000000074505806，
# 阻尼：float = 0.7070000171661377，omega_cutoff：float = 7.0，min_cluster_size：int = 7，
# max_link_time：int = 30000，match_polarity：bool = True，use_simple_match：bool = True，full_square：bool = True，
# last_event_only：bool = False，size_threshold：int = 100000000) -> None
from functools import cmp_to_key


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Sparse Optical Flow sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group(
        "Input", "Arguments related to input sequence.")
    input_group.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default=ori_raw_path,
        # '-i', '--input-event-file', dest='event_file_path', default="E:\EC\Share\event.hdf5",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    input_group.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    if ours:
        input_group.add_argument(
            '--dt-step', type=int, default=ours_dt_step, dest="dt_step",
            help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")
    else:
        input_group.add_argument(
            '--dt-step', type=int, default=33333, dest="dt_step",
            help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")

    noise_filtering_group = parser.add_argument_group(
        "Noise Filtering", "Arguments related to STC noise filtering.")
    noise_filtering_group.add_argument(
        "--disable-stc", dest="disable_stc", default=disable_filter, action="store_true",
        help="Disable STC noise filtering. All other options related to noise filtering are discarded.")
    noise_filtering_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int,  default=50000,
                                       help="Length of the time window for filtering (in us).")
    noise_filtering_group.add_argument(
        "--disable-stc-cut-trail", dest="stc_cut_trail", default=True, action="store_false",
        help="When stc cut trail is enabled, after an event goes through, it removes all events until change of polarity.")

    output_flow_group = parser.add_argument_group(
        "Output flow", "Arguments related to output optical flow.")
    output_flow_group.add_argument(
        "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
        default=save_csv,
        help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
        "format, the flow vx and vy are expressed in pixels per second.")
    output_flow_group.add_argument(
        "--output-dense-h5-filename", dest="output_dense_h5_filename",
        help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
        "averaged pixelwise over timeslices of --dt-step. The dense flow is expressed in terms of "
        "pixels per timeslice (of duration dt-step), not in pixels per second.")
    output_flow_group.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default=save_avi,
        help="Path to an output AVI file to save the resulting video.")
    output_flow_group.add_argument(
        '--fps', dest='fps', type=int, default=30,
        help="replay fps of output video")

    args = parser.parse_args()

    if args.output_sparse_npy_filename:
        assert not os.path.exists(args.output_sparse_npy_filename)
    if args.output_dense_h5_filename:
        assert not os.path.exists(args.output_dense_h5_filename)

    return args


def main():
    """ Main """
    args = parse_args()

    # 初始化瞳孔位置记录列表 kk_added
    pupil_positions = []  # 存储格式 (timestamp, x, y)

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(
        input_path=args.event_file_path, delta_t=args.dt_step)

    # Set ERC to 20Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(20000000)
            erc_module.enable(True)

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)

    if ours:
        width = 640
        height = 480
    else:
        height, width = mv_iterator.get_size()  # Camera Geometry
    # Event Frame Generator
    event_frame_gen = OnDemandFrameGenerationAlgorithm(
        width, height, args.dt_step)
    # Sparse Optical Flow Algorithm

    flow_algo = SparseOpticalFlowAlgorithm(
        width, height, max_link_time=ours_max_link_time, min_cluster_size=ours_min_cluster_size,
        distance_gain=ours_distance_gain, omega_cutoff=ours_omega_cutoff, damping=ours_dampping, match_polarity = ours_match_polarity)
    # else:
    #     flow_algo = SparseOpticalFlowAlgorithm(
    #         width, height, SparseOpticalFlowConfigPreset.FastObjects)


    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()

    # Flow Frame Generator
    flow_frame_gen = SparseFlowFrameGeneratorAlgorithm()

    # STC filter
    stc_filter = SpatioTemporalContrastAlgorithm(
        width, height, args.stc_filter_thr, args.stc_cut_trail)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    all_flow_events = []
    all_dense_flows = []
    all_dense_flows_start_ts = []
    all_dense_flows_end_ts = []

    # Window - Graphical User Interface
    with Window(title="Metavision Sparse Optical Flow", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        if args.out_video:
            video_name = args.out_video + ".avi"
            writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-r': str(args.fps)
            })

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        output_img = np.zeros((height, width, 3), np.uint8)
        processing_ts = mv_iterator.start_ts

        # Process events
        if ours:
            delta = ours_delta
        else:
            delta = mv_iterator.delta_t

        count = 0
        # eventgroup = np.load(event_path)

        # print(eventgroup.shape)

        temp = 0
        # 如果使用rawdata数据
        for evs in mv_iterator:
            start = time.perf_counter()
            # 画图
            if fig_per_frame:
                count = count + 1
                if count >= begin_frame and count <= end_frame:
                    k = np.zeros((720, 1280))
                    for temp in range(evs.shape[0]):
                        k[evs[temp]['y']][evs[temp]['x']] = 1
                    mage = plt.imshow(k, cmap='gray')
                    plt.show()
                    continue
                elif count >= end_frame:
                    return
                else:
                    continue

            processing_ts += delta
            print("processing_ts:", processing_ts)
            if evs.shape[0] > 0:
                print("all_event_count:", evs.shape)

            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # 过滤
            if args.disable_stc:
                events_buf = evs
            else:
                # Filter Events using STC
                stc_filter.process_events(evs, events_buf)

            # 事件帧生成
            event_frame_gen.process_events(events_buf)
            event_frame_gen.generate(processing_ts, output_img)

            # 评估光流
            flow_algo.process_events(events_buf, flow_buffer)
            # 保存
            if args.output_sparse_npy_filename:
                all_flow_events.append(flow_buffer.numpy().copy())
            if args.output_dense_h5_filename:
                all_dense_flows_start_ts.append(
                    processing_ts - args.dt_step)
                all_dense_flows_end_ts.append(processing_ts)
                flow_np = flow_buffer.numpy()
                if flow_np.size == 0:
                    all_dense_flows.append(
                        np.zeros((2, height, width), dtype=np.float32))
                else:
                    xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
                    coords = np.stack((ys, xs))
                    abs_coords = np.ravel_multi_index(coords, (height, width))
                    counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
                                            minlength=height*width).reshape(height, width)
                    flow_x = np.bincount(
                        abs_coords, weights=vx, minlength=height*width).reshape(height, width)
                    flow_y = np.bincount(
                        abs_coords, weights=vy, minlength=height*width).reshape(height, width)
                    mask_multiple_events = counts > 1
                    flow_x[mask_multiple_events] /= counts[mask_multiple_events]
                    flow_y[mask_multiple_events] /= counts[mask_multiple_events]

                    # flow expressed in pixels per delta_t
                    flow_x *= args.dt_step * 1e-6
                    flow_y *= args.dt_step * 1e-6
                    flow = np.stack((flow_x, flow_y)).astype(np.float32)
                    all_dense_flows.append(flow)

            # Draw the flow events on top of the events
            flow_frame_gen.add_flow_for_frame_update(flow_buffer)
            flow_frame_gen.clear_ids()
            flow_frame_gen.update_frame_with_flow(output_img)


            if use_monitor:
                # 获得flow点蔟信息
                flow_np = flow_buffer.numpy()

                red_frame = np.array([])
                if flow_np.size != 0:
                    xs, ys, _, _, id = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"], flow_np["id"]

                    # 1, 提取box位置
                    trial = np.transpose(np.stack((xs, ys, id)))
                    # 使用np.unique找出所有唯一的第三列值(点蔟id)
                    unique_values, inverse = np.unique(trial[:, 2], return_inverse=True)
                    # 对inverse进行排序，以便分组唯一值
                    sorted_inverse = np.argsort(inverse)
                    # 分割索引以形成组
                    unique_indices = np.split(sorted_inverse, np.cumsum(np.unique(inverse, return_counts=True)[1]))[:-1]
                    # 对于每个唯一值的组，找到第一列、第二列的最大最小值
                    box = np.transpose(np.array((
                                    [np.min(trial[group][:, 0]) for group in unique_indices],
                                    [np.min(trial[group][:, 1]) for group in unique_indices],
                                    [np.max(trial[group][:, 0]) for group in unique_indices],
                                    [np.max(trial[group][:, 1]) for group in unique_indices])))
                    print("event_group_count:", unique_values.shape[0])
                    # 创建映射字典
                    group_id_dict = {}
                    for i in range(unique_values.shape[0]):
                        group_id_dict[np.str(unique_values[i])] = i

                    if use_frame_or_shape:
                        # 2, 提取红框位置
                        for i in range(unique_values.shape[0]):
                            if box[i][0] != 100000:
                                # print(box[i][0], box[i][1], box[i][2], box[i][3], i)
                                red_frame = np.append(red_frame, (box[i][0], box[i][1], box[i][2], box[i][3]))
                                # print(red_frame)
                        # 3, 方框重叠度判断
                        for i in range(np.int32(red_frame.shape[0] / 4)):
                            for j in range(np.int32(red_frame.shape[0] / 4)):
                                if i == j:
                                    continue
                                x1, y1, x2, y2 = red_frame[i * 4 + 0], red_frame[i * 4 + 1], red_frame[i * 4 + 2], red_frame[i * 4 + 3]
                                xx1, yy1, xx2, yy2 = red_frame[j * 4 + 0], red_frame[j * 4 + 1], red_frame[j * 4 + 2], red_frame[j * 4 + 3]

                                # 根据重叠度合并矩形
                                if ( yy2 < y1 or xx2 < xx1 or yy1 > y2 or xx1 > xx2) == False: # 若重叠
                                    x_overlap1 = max(x1, xx1)
                                    y_overlap1 = max(y1, yy1)
                                    x_overlap2 = min(x2, xx2)
                                    y_overlap2 = min(y2, yy2)
                                    overlap_square = (y_overlap2 - y_overlap1) * (x_overlap2 - x_overlap1)
                                    a_square = (y2 - y1) * (x2 - x1)
                                    b_square = (yy2 - yy1) * (xx2 - xx1)
                                    if ((overlap_square / a_square > overlap_ratio) or (overlap_square / b_square > overlap_ratio)):
                                        if a_square > b_square:
                                            red_frame[j * 4: j * 4 + 4] = red_frame[i * 4 : i * 4 + 4]
                                        else:
                                            red_frame[i * 4: i * 4 + 4] = red_frame[j * 4 : j * 4 + 4]
                        # 4, 编辑视频
                        for i in range(np.int32(red_frame.shape[0] / 4)):
                            y1, x1, y2, x2 = np.int32(red_frame[i * 4 + 0]), np.int32(red_frame[i * 4 + 1]), \
                                np.int32(red_frame[i * 4 + 2]), np.int32(red_frame[i * 4 + 3])
                            # print(width, height)  # 1280 720
                            output_img[x1 : x2, y1, :] = (0, 0, 255)
                            output_img[x1 : x2, y2, :] = (0, 0, 255)
                            output_img[x1, y1 : y2, :] = (0, 0, 255)
                            output_img[x2, y1: y2, :] = (0, 0, 255)

                    else:
                        # 5, 所有点蔟标蓝
                        output_img[ys, xs, :] = (255, 0, 0)

                    # 裁剪中央区域
                    if cut_off:
                        cut_ymin = np.int32(width / (2.0 * cut_scale) - cut_swift) + cut_abs_swift[0]
                        cut_ymax = np.int32(width / (2.0 * cut_scale) + cut_swift) + cut_abs_swift[0]
                        for i in range(cut_ymax, width):
                            output_img[0:height, i, :] = (0, 0, 0)
                        for i in range(0, cut_ymin):
                            output_img[0:height, i, :] = (0, 0, 0)

                    if jud_shape:
                    # cv判断形状
                        b_img = np.zeros((height, width, 3), dtype=np.ubyte)
                        b_img[ys, xs, :] = (255, 0, 0)
                        if cut_off:
                            cut_ymin = np.int32(width / (2.0 * cut_scale) - cut_swift) + cut_abs_swift[0]
                            cut_ymax = np.int32(width / (2.0 * cut_scale) + cut_swift) + cut_abs_swift[0]
                            for i in range(cut_ymax, width):
                                b_img[0:height, i, :] = (0, 0, 0)
                            for i in range(0, cut_ymin):
                                b_img[0:height, i, :] = (0, 0, 0)
                        output_img = b_img

                        image = output_img
                        img2 = image.copy()
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # 使用高斯滤波去除噪声
                        gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)

                        # 应用Canny边缘检测
                        edges = cv2.Canny(gaussian_blur, 25, 75)

                        # 膨胀
                        kernel = np.ones((3, 3), np.uint8)
                        dilate = cv2.dilate(edges, kernel)

                        # 找到所有轮廓，记录轮廓的每一个点
                        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        options = []
                        for ci in contours:
                            area = cv2.contourArea(ci)
                            # print("找到轮廓面积:",area)
                            if area > 1000:
                                peri = 0.02 * cv2.arcLength(ci, True)
                                approx = cv2.approxPolyDP(ci, peri, True)
                                boxRect = cv2.boundingRect(approx)
                                print("顶点个数:", len(approx), boxRect)
                                if len(approx) == 3:
                                    objType = 'Triangle'
                                elif len(approx) == 4:
                                    aspRatio = 1.0 * boxRect[2] / boxRect[3]
                                    if aspRatio < 1.03 and aspRatio > 0.97:
                                        objType = 'Square'
                                    else:
                                        dist1 = distance(approx[0][0][0], approx[0][0][1], approx[1][0][0], approx[1][0][1])
                                        dist2 = distance(approx[0][0][0], approx[0][0][1], approx[3][0][0], approx[3][0][1])
                                        result = math.fabs(dist1 - dist2)
                                        if result <= 20:
                                            objType = "rhombus"
                                        else:
                                            objType = 'Rectangle'
                                elif len(approx) == 5:
                                    objType = 'Pentagon'
                                elif len(approx) == 7:
                                    objType = 'Arrow'
                                elif len(approx) > 8:
                                    objType = 'Circle'
                                else:
                                    continue


                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img2, objType, (boxRect[0], boxRect[1] - 5), font, 0.5, (0, 69, 255), 1)
                                cv2.rectangle(img2, (boxRect[0], boxRect[1]),
                                                (boxRect[0] + boxRect[2], boxRect[1] + boxRect[3]), (0, 0, 0), 3)# 绘制框
                                options.append(ci)
                                # dst=cv2.drawContours(img,ci,-1,(255,0,255),2)

                        # 此时得到了很多选项的轮廓，在 options 中是无规则存放的
                        print("共找到选项", len(options), "个")

                        # 绘制轮廓
                        dst = cv2.drawContours(image=img2, contours=options, contourIdx=-1, color=(0, 0, 0),
                                                thickness=2)

                        # cv2.imshow('dst', dst)
                        # cv2.waitKey()
                        #
                        # cv2.destroyAllWindows()
                        output_img = dst

                    if detect_eys:
                        # cv判断形状
                        b_img = np.zeros((height, width, 3), dtype=np.ubyte)
                        b_img[220:500, 420:880, :] = (255, 0, 0)
                        b_img[ys, xs, :] = (0, 0, 0)

                        output_img = b_img

                        image = output_img
                        img2 = image.copy()
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # 预处理：去噪并增强边缘
                        blurred = cv2.GaussianBlur(gray, (3, 3), 2)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        # 形态学操作消除内部小点
                        kernel = np.ones((5, 5), np.uint8)
                        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                        closed = 255 - closed

                        # 查找所有轮廓
                        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # 定义阈值参数
                        min_area = 500  # 最小面积阈值（根据实际调整）
                        min_circularity = 0.6  # 最小圆形度（0.7~1.0）

                        filtered_contours = []
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area < min_area:  # 跳过面积过小的轮廓（如小圆点）
                                continue
                            # 计算圆形度
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter == 0:
                                continue
                            circularity = 4 * np.pi * area / (perimeter ** 2)

                            if circularity >= min_circularity:
                                filtered_contours.append(cnt)
                        # 选择符合条件的最大的轮廓
                        if filtered_contours:
                            main_contour = max(filtered_contours, key=cv2.contourArea)
                            # 计算凸包填补缺口
                            hull = cv2.convexHull(main_contour)
                            # 椭圆拟合（需至少5个点）
                            if len(hull) >= 5:
                                ellipse = cv2.fitEllipse(hull)
                                cv2.ellipse(image, ellipse, (0, 255, 0), 2)

                                # 记录瞳孔中心坐标和时间戳 kk_added
                                center_x = int(ellipse[0][0])
                                center_y = int(ellipse[0][1])
                                pupil_positions.append( (processing_ts, center_x, center_y) )  # 添加记录

                            # 可选：绘制原轮廓对比
                            cv2.drawContours(image, [main_contour], -1, (0, 0, 255), 1)
                        # cv2.imshow('Original', image)
                        # cv2.imshow('Gray', gray)
                        # cv2.imshow('Thresh', thresh)
                        # cv2.imshow('closed', closed)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        output_img = image

            # Update the display
            window.show(output_img)
            # print(output_img.shape) # (720, 1280, 3)

            if args.out_video:
                writer.writeFrame(output_img.astype(np.uint8)[..., ::-1])

            if window.should_close():
                break

            print("------------")
            # temp = temp + 1
            # if temp == 1:
            #     break

            end = time.perf_counter()
            print(f"耗时: {end - start:.6f} 秒")
            # 处理单帧需要0.18s
            # 总共22s视频：127.189419 114.940271  5.19fps-> 5.74fps 0.56fps

    if args.out_video:
        writer.close()

    if args.output_sparse_npy_filename:
        print("Writing output file: ", args.output_sparse_npy_filename)
        all_flow_events = np.concatenate(all_flow_events)
        # np.save(args.output_sparse_npy_filename, all_flow_events)
        with open(args.output_sparse_npy_filename + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(all_flow_events)
    if args.output_dense_h5_filename:
        print("Writing output file: ", args.output_dense_h5_filename)
        flow_start_ts = np.array(all_dense_flows_start_ts)
        flow_end_ts = np.array(all_dense_flows_end_ts)
        flows = np.stack(all_dense_flows)
        N = flow_start_ts.size
        assert flow_end_ts.size == N
        assert flows.shape == (N, 2, height, width)
        dirname = os.path.dirname(args.output_dense_h5_filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        flow_h5 = h5py.File(args.output_dense_h5_filename, "w")
        flow_h5.create_dataset(
            "flow_start_ts", data=flow_start_ts, compression="gzip")
        flow_h5.create_dataset(
            "flow_end_ts", data=flow_end_ts, compression="gzip")
        flow_h5.create_dataset("flow", data=flows.astype(
            np.float32), compression="gzip")
        flow_h5["flow"].attrs["input_file_name"] = os.path.basename(
            args.event_file_path)
        flow_h5["flow"].attrs["checkpoint_path"] = "metavision_sparse_optical_flow"
        flow_h5["flow"].attrs["event_input_height"] = height
        flow_h5["flow"].attrs["event_input_width"] = width
        flow_h5["flow"].attrs["delta_t"] = args.dt_step
        flow_h5.close()

    # 程序结束前保存瞳孔位置数据 kk_added
    if pupil_positions:
        csv_path = os.path.join("F:\\1\\EyeTracking\\stage6_retina_gaze\\groundtruth", f"pupil_positions_{raw_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp(us)', 'x', 'y'])
            writer.writerows(pupil_positions)
        print(f"瞳孔位置已保存至: {csv_path}")
    else:
        print("未检测到有效瞳孔位置")


if __name__ == "__main__":
    main()
