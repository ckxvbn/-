import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from src.body import Body
from src.hand import Hand
from src import util
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from performance_monitor import PerformanceMonitor
from visualization import VisualizationManager

# 计算角度的辅助函数
def calculate_angle(point1, point2, point3):
    """计算三点之间的角度（point2为顶点）"""
    # 将点转换为numpy数组
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # 计算向量
    ba = a - b
    bc = c - b
    
    # 计算角度
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # 确保值在[-1, 1]范围内
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    # 转换为角度制
    return np.degrees(angle)

# 检查姿势是否标准
def check_standard_posture(candidate, subset):
    """检查姿势是否标准
    
    参数:
        candidate: 关键点数组，每个元素为[x, y, score, id]
        subset: 人体候选数组，每个元素为[索引0-17对应身体各部分，18为总分，19为总部件数]
    
    返回:
        bool: 是否为标准姿势
    """
    # 调试信息
    print(f"Detected {len(candidate)} key points, {len(subset)} persons")
    
    # 标准姿势定义：身体挺直，双手自然下垂或放在身体两侧，没有抬手动作
    # 基于关键点数量和特定关节位置判断
    has_enough_points = len(candidate) >= 8
    
    is_standard = False
    if has_enough_points and len(subset) > 0:
        # 检查是否有明显的抬手动作（基于肩、肘、腕关节的位置关系）
        # 关键点索引：5-左肩, 6-右肩, 7-左肘, 8-右肘, 9-左腕, 10-右腕
        has_raised_hand = False
        
        # 遍历每个人体候选
        for person in subset:
            # 检查左手臂关键点
            if person[5] >= 0 and person[7] >= 0 and person[9] >= 0:
                # 左肩、左肘、左腕都被检测到
                left_shoulder = candidate[int(person[5])][:2]
                left_wrist = candidate[int(person[9])][:2]
                
                # 左腕位置高于左肩，判断为抬手
                if left_wrist[1] < left_shoulder[1] - 20:
                    has_raised_hand = True
                    break
            
            # 检查右手臂关键点
            if person[6] >= 0 and person[8] >= 0 and person[10] >= 0:
                # 右肩、右肘、右腕都被检测到
                right_shoulder = candidate[int(person[6])][:2]
                right_wrist = candidate[int(person[10])][:2]
                
                # 右腕位置高于右肩，判断为抬手
                if right_wrist[1] < right_shoulder[1] - 20:
                    has_raised_hand = True
                    break
        
        # 标准姿势：关键点数量足够，且没有抬手动作
        is_standard = not has_raised_hand
        print(f"Has raised hand: {has_raised_hand}, Posture result: {is_standard}")
    
    return is_standard

# 绘制姿势结果
def draw_posture_result(canvas, is_standard):
    """在画面上绘制姿势标准度结果
    
    参数:
        canvas: 要绘制的图像
        is_standard: 是否为标准姿势
    """
    h, w, _ = canvas.shape
    
    # 简化绘制逻辑，确保文字和符号清晰可见
    
    # 绘制文字 - 只使用英文，确保字体兼容
    text = "STANDARD" if is_standard else "NOT STANDARD"
    color = (0, 255, 0) if is_standard else (0, 0, 255)
    
    # 文字位置：左上角，使用中号字体
    font_scale = 1.5
    font_thickness = 3
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
    
    # 绘制符号（使用简单的几何图形，确保显示正确）
    symbol_size = 50
    
    if is_standard:
        # 绘制绿色对号（使用线条绘制）
        cv2.line(canvas, (w - 100, 50), (w - 80, 70), (0, 255, 0), 10)
        cv2.line(canvas, (w - 80, 70), (w - 50, 30), (0, 255, 0), 10)
    else:
        # 绘制红色叉号（使用线条绘制）
        cv2.line(canvas, (w - 100, 30), (w - 50, 70), (0, 0, 255), 10)
        cv2.line(canvas, (w - 50, 30), (w - 100, 70), (0, 0, 255), 10)
    
    return canvas

# 模型路径
body_model_path = 'openpose/body_pose_model.pth'
hand_model_path = 'openpose/hand_pose_model.pth'
yolo_model_path = 'yolo_lidar_train/models/best.pt'

# 图像预处理函数
def preprocess_image(frame):
    """
    图像预处理，提高YOLO检测准确度
    
    参数:
        frame: 原始图像
        
    返回:
        预处理后的图像
    """
    # 1. 调整亮度和对比度
    alpha = 1.5  # 对比度增益
    beta = 50    # 亮度偏移
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # 2. 直方图均衡化（仅对灰度图）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    # 将均衡化后的灰度图转换回彩色
    frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    # 3. 高斯模糊降噪
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame

# 初始化模型
body_estimation = Body(body_model_path)
hand_estimation = Hand(hand_model_path)

# 尝试加载YOLO模型
yolo_model = None
try:
    if YOLO is not None and os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        print(f"YOLO模型加载成功: {yolo_model_path}")
        print(f"YOLO模型类别: {yolo_model.names}")
    else:
        print(f"YOLO模型未加载: YOLO={YOLO}, 模型文件存在={os.path.exists(yolo_model_path)}")
except Exception as e:
    print(f"YOLO模型加载失败: {e}")
    print("将使用--no-yolo模式运行")
    yolo_model = None

def process_image(image_path):
    """
    处理单张图像，识别人体姿态
    
    参数:
        image_path: 图像文件路径
    
    返回:
        处理后的图像，包含姿态关键点
    """
    # 加载图像
    oriImg = cv2.imread(image_path)  # B,G,R order
    if oriImg is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # 身体姿态估计
    candidate, subset = body_estimation(oriImg)
    
    # 绘制身体姿态
    canvas = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    # 手部姿态估计
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # 裁剪手部区域
        hand_img = oriImg[y:y+w, x:x+w, :]
        if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
            continue
        # 手部姿态估计
        peaks = hand_estimation(hand_img)
        # 调整关键点坐标到原图
        peaks[:, 0] = np.where(peaks[:, 0] == 0, 0, peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, 0, peaks[:, 1] + y)
        all_hand_peaks.append(peaks)
    
    canvas = util.draw_handpose(canvas, all_hand_peaks)
    eval_result = util.evaluate_posture(candidate, subset)
    canvas = util.annotate_posture(canvas, eval_result)
    return canvas

def process_video(video_path=None, camera_index=0, fast=False, no_hand=False, no_yolo=False):
    """
    处理视频或摄像头，实时识别人体姿态
    
    参数:
        video_path: 视频文件路径，如果为None则使用摄像头
        camera_index: 摄像头索引，默认为0
    """
    # 打开视频或摄像头
    cap = cv2.VideoCapture(video_path if video_path else camera_index)
    
    # 降低分辨率，提高帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 初始化性能监控器
    perf_monitor = PerformanceMonitor(fps_threshold=20, log_file="logs/performance_log.json")
    
    # 初始化可视化管理器
    vis_manager = VisualizationManager(window_title="AI Vision Interface", width=1280, height=720)
    
    # 初始化帧计数器和边界框缓存
    frame_count = 0
    last_bbox = None  # 缓存上一次的边界框，减少API调用频率
    api_status = "OK"  # API状态
    last_detections = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            perf_monitor.update_fps()
            low_fps = (perf_monitor.fps > 0 and perf_monitor.fps < perf_monitor.fps_threshold) or fast
            small_size = (160, 120) if low_fps else (240, 180)
            small_frame = cv2.resize(frame, small_size)
            
            # 身体姿态估计（在小分辨率图像上）
            candidate, subset = body_estimation(small_frame)
            
            # 将关键点坐标缩放回原图
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            if len(candidate) > 0:
                candidate[:, 0] *= scale_x
                candidate[:, 1] *= scale_y
            
            # 绘制身体姿态（在原图上）
            canvas = frame.copy()
            canvas = util.draw_bodypose(canvas, candidate, subset)
            
            use_hand = (not low_fps) and (not no_hand)
            hands_list = util.handDetect(candidate, subset, frame) if use_hand else []
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                # 裁剪手部区域
                hand_img = frame[y:y+w, x:x+w, :]
                if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                    continue
                # 降低手部图像分辨率
                hand_img_small = cv2.resize(hand_img, (128, 128))
                # 手部姿态估计（在小分辨率图像上）
                peaks = hand_estimation(hand_img_small)
                # 将关键点坐标缩放回原图
                peaks[:, 0] = np.where(peaks[:, 0] == 0, 0, peaks[:, 0] * (hand_img.shape[1] / hand_img_small.shape[1]) + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, 0, peaks[:, 1] * (hand_img.shape[0] / hand_img_small.shape[0]) + y)
                all_hand_peaks.append(peaks)
            
            # 绘制手部姿态
            canvas = util.draw_handpose_by_opencv(canvas, all_hand_peaks)
            
            eval_result = util.evaluate_posture(candidate, subset)
            canvas = util.annotate_posture(canvas, eval_result)
            
            # 使用YOLO模型检测激光雷达和扳手
            lidar_detected = False
            wrench_detected = False
            
            if (not no_yolo) and yolo_model is not None:
                yolo_interval = 12 if low_fps else 4
                yolo_imgsz = 320 if low_fps else 640
                if frame_count % yolo_interval == 0:
                    # 图像预处理，提高YOLO检测准确度
                    processed_frame = preprocess_image(canvas)
                    # 使用优化的YOLO检测参数
                    results = yolo_model(processed_frame, conf=0.2, imgsz=yolo_imgsz, augment=True, device=0 if torch.cuda.is_available() else 'cpu')
                    last_detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            class_name = yolo_model.names[cls]
                            last_detections.append((x1, y1, x2, y2, conf, class_name))
                for (x1, y1, x2, y2, conf, class_name) in last_detections:
                    if class_name == 'lidar':
                        lidar_detected = True
                    elif class_name == 'wrench':
                        wrench_detected = True
                    color = (0, 0, 255) if class_name == 'lidar' else (0, 255, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    label_y = y1 - 10 if y1 > 20 else y2 + label_size[1] + 10
                    cv2.rectangle(canvas, (x1, label_y - label_size[1] - 5), (x1 + label_size[0] + 10, label_y + 5), color, -1)
                    cv2.putText(canvas, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            info_data = {
                "is_standard": eval_result["is_standard"],
                "fps": perf_monitor.fps,
                "key_points": len(candidate),
                "persons": len(subset),
                "lidar_detected": lidar_detected,
                "wrench_detected": wrench_detected
            }
            
            # 创建完整的可视化界面
            interface = vis_manager.create_interface(canvas, info_data)
            
            vis_manager.show(interface)
            frame_count += 1
            if cv2.waitKey(1) == 27:
                break
    finally:
        # 停止性能监控，保存日志
        perf_monitor.stop()
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    import argparse
    import sys
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Human Pose Estimation')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', type=int, default=0, help='Use camera with specified index (default: 0)')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--no-hand', action='store_true')
    parser.add_argument('--no-yolo', action='store_true')
    parser.add_argument('--batch', type=str, help='Directory to process all images in batch')
    
    args = parser.parse_args()
    
    if args.image:
        # 处理单张图像
        result = process_image(args.image)
        plt.imshow(result)
        plt.axis('off')
        plt.show()
    elif args.video:
        process_video(video_path=args.video, camera_index=0, fast=args.fast, no_hand=args.no_hand, no_yolo=args.no_yolo)
    elif args.batch:
        # 批量处理目录中的所有图像
        import os
        from src import util
        
        # 检查目录是否存在
        if not os.path.exists(args.batch):
            print(f"Error: Directory {args.batch} not found")
            sys.exit(1)
        
        # 获取目录中的所有图像文件
        image_files = [f for f in os.listdir(args.batch) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) == 0:
            print(f"Error: No image files found in {args.batch}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files in {args.batch}")
        
        # 存储所有图像的姿态数据
        all_posture_data = {}
        
        for image_file in image_files:
            image_path = os.path.join(args.batch, image_file)
            print(f"Processing image: {image_file}")
            
            # 加载图像
            oriImg = cv2.imread(image_path)
            if oriImg is None:
                print(f"Error: Cannot read image {image_path}")
                continue
            
            # 身体姿态估计
            candidate, subset = body_estimation(oriImg)
            
            # 提取关键点坐标
            key_points = {}
            if len(subset) > 0:
                person = subset[0]
                # 关键点名称映射（根据OpenPose关键点顺序）
                keypoint_names = [
                    'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                    'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
                    'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                    'right_eye', 'left_eye', 'right_ear', 'left_ear'
                ]
                
                for i, name in enumerate(keypoint_names):
                    if person[i] >= 0:
                        x, y = candidate[int(person[i])][:2]
                        key_points[name] = {
                            'x': float(x),
                            'y': float(y),
                            'score': float(candidate[int(person[i])][2])
                        }
                
                # 计算关键关节角度
                angles = {}
                
                # 计算颈部倾斜角度
                if 'neck' in key_points and 'right_hip' in key_points and 'left_hip' in key_points:
                    neck = [key_points['neck']['x'], key_points['neck']['y']]
                    right_hip = [key_points['right_hip']['x'], key_points['right_hip']['y']]
                    left_hip = [key_points['left_hip']['x'], key_points['left_hip']['y']]
                    hip = np.mean(np.array([right_hip, left_hip]), axis=0)
                    vertical = [neck[0], neck[1] - 100]  # 虚拟垂直点
                    angles['torso_tilt'] = float(calculate_angle(hip, neck, vertical))
                
                # 计算右膝关节角度
                if 'right_hip' in key_points and 'right_knee' in key_points and 'right_ankle' in key_points:
                    right_hip = [key_points['right_hip']['x'], key_points['right_hip']['y']]
                    right_knee = [key_points['right_knee']['x'], key_points['right_knee']['y']]
                    right_ankle = [key_points['right_ankle']['x'], key_points['right_ankle']['y']]
                    angles['right_knee'] = float(calculate_angle(right_hip, right_knee, right_ankle))
                
                # 计算左膝关节角度
                if 'left_hip' in key_points and 'left_knee' in key_points and 'left_ankle' in key_points:
                    left_hip = [key_points['left_hip']['x'], key_points['left_hip']['y']]
                    left_knee = [key_points['left_knee']['x'], key_points['left_knee']['y']]
                    left_ankle = [key_points['left_ankle']['x'], key_points['left_ankle']['y']]
                    angles['left_knee'] = float(calculate_angle(left_hip, left_knee, left_ankle))
                
                # 计算右肘关节角度
                if 'right_shoulder' in key_points and 'right_elbow' in key_points and 'right_wrist' in key_points:
                    right_shoulder = [key_points['right_shoulder']['x'], key_points['right_shoulder']['y']]
                    right_elbow = [key_points['right_elbow']['x'], key_points['right_elbow']['y']]
                    right_wrist = [key_points['right_wrist']['x'], key_points['right_wrist']['y']]
                    angles['right_elbow'] = float(calculate_angle(right_shoulder, right_elbow, right_wrist))
                
                # 计算左肘关节角度
                if 'left_shoulder' in key_points and 'left_elbow' in key_points and 'left_wrist' in key_points:
                    left_shoulder = [key_points['left_shoulder']['x'], key_points['left_shoulder']['y']]
                    left_elbow = [key_points['left_elbow']['x'], key_points['left_elbow']['y']]
                    left_wrist = [key_points['left_wrist']['x'], key_points['left_wrist']['y']]
                    angles['left_elbow'] = float(calculate_angle(left_shoulder, left_elbow, left_wrist))
                
                # 计算左右手臂是否抬起
                arm_raised = {}
                if 'right_shoulder' in key_points and 'right_wrist' in key_points:
                    arm_raised['right_arm_raised'] = key_points['right_wrist']['y'] < key_points['right_shoulder']['y'] - 20
                if 'left_shoulder' in key_points and 'left_wrist' in key_points:
                    arm_raised['left_arm_raised'] = key_points['left_wrist']['y'] < key_points['left_shoulder']['y'] - 20
                
                # 计算重心位置
                center_of_gravity = {}
                if 'right_hip' in key_points and 'left_hip' in key_points:
                    right_hip = [key_points['right_hip']['x'], key_points['right_hip']['y']]
                    left_hip = [key_points['left_hip']['x'], key_points['left_hip']['y']]
                    hip_center = np.mean(np.array([right_hip, left_hip]), axis=0)
                    center_of_gravity['x'] = float(hip_center[0])
                    center_of_gravity['y'] = float(hip_center[1])
                
                # 姿态分类
                posture_category = 'stand'
                if arm_raised.get('right_arm_raised', False) or arm_raised.get('left_arm_raised', False):
                    posture_category = 'overhead'
                elif 'right_knee' in angles and 'left_knee' in angles:
                    if min(angles['right_knee'], angles['left_knee']) <= 120:
                        posture_category = 'squat'
                elif 'torso_tilt' in angles and angles['torso_tilt'] >= 25:
                    posture_category = 'bend'
                
                # 姿态评估
                eval_result = util.evaluate_posture(candidate, subset)
                
                # 保存姿态数据
                posture_data = {
                    'key_points': key_points,
                    'angles': angles,
                    'arm_raised': arm_raised,
                    'center_of_gravity': center_of_gravity,
                    'posture_category': posture_category,
                    'evaluation': eval_result,
                    'image_name': image_file
                }
                
                all_posture_data[image_file] = posture_data
        
        # 保存所有姿态数据到JSON文件
        output_file = 'posture_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_posture_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nBatch processing completed!")
        print(f"Posture data saved to: {output_file}")
        print(f"Images processed: {len(all_posture_data)}/{len(image_files)}")
    else:
        # 使用摄像头，支持指定摄像头索引
        process_video(video_path=None, camera_index=args.camera, fast=args.fast, no_hand=args.no_hand, no_yolo=args.no_yolo)
