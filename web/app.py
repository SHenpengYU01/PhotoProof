from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from video_auth import VideoAuthenticator
import threading

# 获取当前文件的绝对路径
current_dir = Path(__file__).parent.absolute()

app = Flask(__name__)
app.secret_key = 'video_auth_secret_key'

# 使用绝对路径
UPLOAD_FOLDER = current_dir / 'uploads'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

print(f"上传文件夹路径: {app.config['UPLOAD_FOLDER']}")

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局认证器实例
authenticator = VideoAuthenticator(key="web_demo_secret_key")

# 存储处理状态
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv', 'webm'}

from werkzeug.utils import secure_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取变换参数
        brightness_change = float(request.form.get('brightness', 0.1))
        rotation_angle = float(request.form.get('rotation', 5.0))
        crop_ratio = float(request.form.get('crop', 0.05))
        
        # 异步处理视频签名
        thread = threading.Thread(target=process_video_signature, args=(filepath, brightness_change, rotation_angle, crop_ratio))
        thread.start()
        
        return render_template('processing.html', filename=filename)
    
    flash('文件类型不支持')
    return redirect(url_for('index'))

def process_video_signature(filepath, brightness_change=0.1, rotation_angle=5.0, crop_ratio=0.05):
    """异步处理视频签名和验证"""
    filename = os.path.basename(filepath)
    processing_status[filename] = {'status': 'processing', 'progress': 0}
    
    try:
        # 生成签名
        signature = authenticator.generate_video_signature(filepath, sampling_rate=30)
        
        # 保存签名
        sig_filename = filename.rsplit('.', 1)[0] + '_signature.pkl'
        sig_path = os.path.join(app.config['UPLOAD_FOLDER'], sig_filename)
        authenticator.save_signature(signature, sig_path)
        
        # 生成变换视频
        transformed_filename = filename.rsplit('.', 1)[0] + '_transformed.mp4'
        transformed_path = os.path.join(app.config['UPLOAD_FOLDER'], transformed_filename)
        create_transformed_video_demo(authenticator, filepath, transformed_path, brightness_change, rotation_angle, crop_ratio)
        
        # 验证
        is_authentic, similarity_score, details = authenticator.verify_video_authenticity(
            transformed_path, sig_path, threshold=0.6
        )
        
        processing_status[filename] = {
            'status': 'completed', 
            'signature': sig_filename,
            'transformed': transformed_filename,
            'transform_params': {
                'brightness_change': brightness_change,
                'rotation_angle': rotation_angle,
                'crop_ratio': crop_ratio
            },
            'result': {
                'is_authentic': is_authentic,
                'similarity_score': similarity_score,
                'threshold': details['threshold'],
                'frame_count_match': details['frame_count_match']
            }
        }
        
    except Exception as e:
        processing_status[filename] = {'status': 'error', 'error': str(e)}

@app.route('/status/<filename>')
def get_status(filename):
    """获取处理状态"""
    status = processing_status.get(filename, {'status': 'not_found'})
    return jsonify(status)

@app.route('/result/<filename>')
def show_result(filename):
    """显示验证结果"""
    status = processing_status.get(filename)
    if not status or status['status'] != 'completed':
        flash('结果不可用或仍在处理中')
        return redirect(url_for('index'))
    
    # 获取变换后的视频文件名
    transformed_filename = status.get('transformed', '')
    if not transformed_filename:
        transformed_filename = filename.rsplit('.', 1)[0] + '_transformed.mp4'
    
    return render_template('result.html', 
                         result=status['result'], 
                         filename=filename,
                         transformed_filename=transformed_filename,
                         transform_params=status.get('transform_params', {}))

@app.route('/verify', methods=['POST'])
def verify_video():
    original_file = request.form.get('original_video')
    test_file = request.form.get('test_video')
    
    if not original_file or not test_file:
        flash('请选择原始视频和测试视频')
        return redirect(url_for('index'))
    
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_file)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_file)
    
    # 获取签名文件
    sig_file = original_file.rsplit('.', 1)[0] + '_signature.pkl'
    sig_path = os.path.join(app.config['UPLOAD_FOLDER'], sig_file)
    
    if not os.path.exists(sig_path):
        flash('未找到原始视频的签名文件，请先上传并处理原始视频')
        return redirect(url_for('index'))
    
    try:
        # 验证视频
        is_authentic, similarity_score, details = authenticator.verify_video_authenticity(
            test_path, sig_path, threshold=0.6
        )
        
        return render_template('result.html', 
                             result={
                                 'is_authentic': is_authentic,
                                 'similarity_score': similarity_score,
                                 'threshold': details['threshold'],
                                 'frame_count_match': details['frame_count_match']
                             },
                             filename=original_file,
                             transformed_filename=test_file)
        
    except Exception as e:
        flash(f'验证失败: {str(e)}')
        return redirect(url_for('index'))

@app.route('/demo')
def demo():
    """自动演示功能"""
    demo_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo.mp4')
    if not os.path.exists(demo_video_path):
        create_demo_video(demo_video_path)
    
    # 异步处理
    thread = threading.Thread(target=process_demo, args=(demo_video_path,))
    thread.start()
    
    return render_template('processing.html', filename='demo.mp4')

def process_demo(filepath):
    """异步处理演示"""
    filename = 'demo.mp4'
    processing_status[filename] = {'status': 'processing'}
    
    try:
        # 生成签名
        signature = authenticator.generate_video_signature(filepath, sampling_rate=30)
        sig_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_signature.pkl')
        authenticator.save_signature(signature, sig_path)
        
        # 创建变换视频
        transformed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_transformed.mp4')
        create_transformed_video_demo(authenticator, filepath, transformed_path)
        
        # 验证
        is_authentic, similarity_score, details = authenticator.verify_video_authenticity(
            transformed_path, sig_path, threshold=0.6
        )
        
        processing_status[filename] = {
            'status': 'completed',
            'transformed': 'demo_transformed.mp4',
            'result': {
                'is_authentic': is_authentic,
                'similarity_score': similarity_score,
                'threshold': details['threshold'],
                'frame_count_match': details['frame_count_match']
            }
        }
        
    except Exception as e:
        processing_status[filename] = {'status': 'error', 'error': str(e)}

def create_demo_video(filepath):
    """创建演示视频 - 使用更兼容的编码和格式"""
    width, height = 640, 480
    fps = 30
    duration = 3  # 3秒
    total_frames = fps * duration
    
    # 优先尝试使用H.264编码，这是浏览器最支持的格式
    fourcc_options = ['avc1', 'X264', 'mp4v']
    writer = None
    
    for codec in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"使用编码器: {codec}")
            break
        else:
            writer = None
    
    if writer is None:
        print("无法创建视频写入器，尝试直接使用FFmpeg")
        return False
    
    # 创建渐变色背景
    for i in range(total_frames):
        # 创建彩色背景，而不是黑色
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加渐变色背景
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gradient, (height, 1))
        
        # 创建彩色帧
        r_channel = ((i * 2) % 255 + gradient // 2) % 255
        g_channel = ((i * 3) % 255 + gradient // 3) % 255
        b_channel = ((i * 5) % 255 + gradient // 4) % 255
        
        frame[:, :, 0] = b_channel  # OpenCV使用BGR格式
        frame[:, :, 1] = g_channel
        frame[:, :, 2] = r_channel
        
        # 添加一些图形
        center_x, center_y = width // 2, height // 2
        radius = 100
        
        # 画一个会移动的圆形
        circle_x = center_x + int(100 * np.sin(2 * np.pi * i / total_frames))
        circle_y = center_y + int(100 * np.cos(2 * np.pi * i / total_frames))
        cv2.circle(frame, (circle_x, circle_y), radius, (0, 255, 0), -1)
        
        # 画一个矩形
        rect_x1 = 100 + i % 100
        rect_y1 = 100
        rect_x2 = rect_x1 + 150
        rect_y2 = 250
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 3)
        
        # 添加文字
        text = f"Demo Video Frame: {i}"
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # 写入帧
        writer.write(frame)
    
    writer.release()
    print(f"演示视频已创建: {filepath}")
    
    # 验证视频
    cap = cv2.VideoCapture(filepath)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"视频验证: 可读取，帧数={frame_count}, 时长={duration:.2f}秒")
        cap.release()
        
        # 确保文件大小合理
        file_size = os.path.getsize(filepath)
        print(f"视频文件大小: {file_size / 1024:.2f} KB")
        
        return True
    else:
        print("视频验证失败: 无法打开")
        return False

def create_transformed_video_demo(authenticator, original_path, transformed_path, brightness_change=0.1, rotation_angle=5.0, crop_ratio=0.05):
    """创建变换演示视频"""
    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        print(f"无法打开原始视频: {original_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # 默认帧率
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 使用更兼容的编码器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(transformed_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("avc1编码器不可用，尝试使用X264...")
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(transformed_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("X264编码器不可用，尝试使用mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(transformed_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("无法创建视频写入器")
        cap.release()
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 应用变换
        transformed = authenticator.apply_transformations(
            frame, brightness_change=brightness_change, rotation_angle=rotation_angle, crop_ratio=crop_ratio
        )
        out.write(transformed)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"变换视频已创建: {transformed_path}, 帧数: {frame_count}")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """提供上传文件的访问"""
    try:
        print(f"请求文件: {filename}")
        print(f"上传文件夹路径: {app.config['UPLOAD_FOLDER']}")
        print(f"文件完整路径: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
        print(f"文件是否存在: {os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename))}")
        
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            print("文件不存在！")
            return "文件不存在", 404
        
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"发送文件时出错: {str(e)}")
        return str(e), 500

@app.route('/list_uploads')
def list_uploads():
    """列出上传文件夹中的文件（用于调试）"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        file_info = []
        for f in files:
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            file_info.append({
                'name': f,
                'exists': os.path.exists(full_path),
                'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
            })
        return jsonify(file_info)
    except Exception as e:
        return jsonify({'error': str(e)})

# 添加调试中间件
@app.after_request
def after_request(response):
    """添加调试头信息"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    print("=== Flask应用启动 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"应用目录: {current_dir}")
    print(f"上传文件夹: {app.config['UPLOAD_FOLDER']}")
    
    # 检查上传文件夹
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        print(f"上传文件夹内容: {os.listdir(app.config['UPLOAD_FOLDER'])}")
    else:
        print("上传文件夹不存在，正在创建...")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)