import cv2
import numpy as np
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from video_auth import VideoAuthenticator

# 创建变换后的视频
def create_transformed_video(authenticator, original_path, transformed_path, 
                           brightness_change=0.0, rotation_angle=0.0, crop_ratio=0.0):
    """
    创建变换后的视频
    
    参数:
        authenticator: VideoAuthenticator实例
        original_path: 原始视频路径
        transformed_path: 输出变换视频路径
        brightness_change: 亮度变化
        rotation_angle: 旋转角度
        crop_ratio: 裁剪比例
    """
    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {original_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(transformed_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 应用变换
        transformed_frame = authenticator.apply_transformations(
            frame, brightness_change, rotation_angle, crop_ratio
        )
        
        out.write(transformed_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧...")
    
    cap.release()
    out.release()
    print(f"变换视频已创建: {transformed_path} (共 {frame_count} 帧)")
    return True

# 使用示例
def demo(video_path=None, signature_path=None, threshold=0.6,
         brightness_change=0.1, rotation_angle=5.0, crop_ratio=0.05):
    # 创建认证器
    authenticator = VideoAuthenticator(key="my_secret_key_123")
    
    print("=" * 50)
    print("视频真实性验证系统演示")
    print("=" * 50)
    
    # 设置输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理签名文件路径
    if signature_path is None or not os.path.isabs(signature_path):
        if signature_path is None:
            signature_path = "original_signature.pkl"
        signature_path = os.path.join(output_dir, signature_path)
    
    if video_path:
        print(f"\n使用自定义视频: {video_path}")
        original_video = video_path
    else:
        # 创建测试视频
        print("\n创建测试视频...")
        original_video = create_test_video()
    
    # 步骤1: 为原始视频生成签名
    print("\n1. 为原始视频生成签名...")
    original_signature = authenticator.generate_video_signature(original_video, sampling_rate=30)
    authenticator.save_signature(original_signature, signature_path)
    
    # 步骤2: 创建变换后的视频
    print("\n2. 创建变换后的视频...")
    transformed_video = os.path.join(output_dir, "transformed_video.mp4")
    success = create_transformed_video(
        authenticator, 
        original_video, 
        transformed_video,
        brightness_change=brightness_change,
        rotation_angle=rotation_angle,
        crop_ratio=crop_ratio
    )
    
    if not success:
        print("创建变换视频失败")
        return
    
    # 步骤3: 验证变换后视频的真实性
    print("\n3. 验证变换后视频的真实性...")
    is_authentic, similarity_score, details = authenticator.verify_video_authenticity(
        transformed_video, signature_path, threshold=threshold
    )
    
    print(f"\n验证结果:")
    print(f"是否真实: {'是' if is_authentic else '否'}")
    print(f"相似度分数: {similarity_score:.4f}")
    print(f"阈值: {details['threshold']}")
    print(f"帧数匹配: {'是' if details['frame_count_match'] else '否'}")
    
    return authenticator


# 创建更详细的测试示例
def create_test_video():
    """创建测试视频（用于演示）"""
    # 创建一个简单的测试视频
    width, height = 640, 480
    fps = 30
    duration = 5  # 5秒
    total_frames = fps * duration
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'test_video.mp4')
    
    # 使用OpenCV创建视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # 创建渐变帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加一些特征（矩形、圆形）
        color = (i % 255, (i * 2) % 255, (i * 3) % 255)
        cv2.rectangle(frame, (50, 50), (200, 200), color, 3)
        cv2.circle(frame, (400, 200), 80, (255, 0, 0), -1)
        cv2.putText(frame, f"Frame {i}", (50, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频已创建: {video_path}")
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="视频真实性验证系统 - 支持亮度、旋转、裁剪变换后的验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                                    # 使用测试视频
  python main.py --video my_video.mp4               # 使用自定义视频
  python main.py --video my_video.mp4 --threshold 0.7 --brightness 0.2
        """
    )
    parser.add_argument('--video', type=str, help='原始视频文件路径（可选，默认使用测试视频）')
    parser.add_argument('--signature', type=str, default=None, help='签名文件路径（默认为output/original_signature.pkl）')
    parser.add_argument('--threshold', type=float, default=0.6, help='验证阈值')
    parser.add_argument('--brightness', type=float, default=0.1, help='亮度变化（-1.0到1.0）')
    parser.add_argument('--rotation', type=float, default=5.0, help='旋转角度（度）')
    parser.add_argument('--crop', type=float, default=0.05, help='裁剪比例（0.0到0.5）')
    
    args = parser.parse_args()
    
    print("视频真实性验证系统")
    print("支持亮度、旋转、裁剪变换后的验证")
    
    # 运行演示
    auth_system = demo(
        video_path=args.video,
        signature_path=args.signature,
        threshold=args.threshold,
        brightness_change=args.brightness,
        rotation_angle=args.rotation,
        crop_ratio=args.crop
    )
    
    print("\n" + "=" * 50)
    print("算法特点总结:")
    print("1. 使用SIFT特征: 对旋转、缩放部分不变")
    print("2. 使用HSV直方图: 对亮度变化鲁棒")
    print("3. 使用图像矩: 对几何变换具有不变性")
    print("4. 使用频域特征: 对空间变换鲁棒")
    print("5. 多特征聚合: 提高鲁棒性和准确性")
    print("=" * 50)
