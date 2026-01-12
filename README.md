# 视频真实性验证系统

这是一个简单完整的视频真实性验证系统，支持视频经过亮度、旋转、裁剪变换后仍然能够验证其真实性。

## 项目结构

```txt
video-auth-system/
├── video_auth/          # 核心算法包
│   ├── __init__.py
│   └── authenticator.py # VideoAuthenticator类
├── web/                 # Web演示应用
│   ├── app.py          # Flask应用
│   ├── templates/      # HTML模板
│   ├── static/         # CSS/JS资源
│   └── uploads/        # 上传文件目录
├── scripts/            # 命令行工具
│   ├── main.py         # 命令行演示脚本
│   └── output/         # 测试输出文件目录
├── requirements.txt    # Python依赖
└── README.md          # 项目文档
```

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 命令行使用

- 使用自动生成的测试视频演示

```bash
python scripts/main.py
```

- 使用自定义视频和验证阈值

```bash
python scripts/main.py --video path/to/your/video.mp4 --threshold 0.6
```

- 使用自定义变换参数

```bash
python scripts/main.py --video path/to/your/video.mp4 --threshold 0.6 --brightness 0.1 --rotation 5 --crop 0.05
```

### 3. Web演示

```bash
# 启动Web应用
python web/app.py

# 访问 http://localhost:5000
```

## 算法特点

- **多特征融合**：结合SIFT、HSV直方图、Hu矩、频域特征
- **变换鲁棒性**：支持亮度±30%、旋转±15°、裁剪10%以内的变换
- **相似度计算**：加权特征相似度，确保关键特征贡献更大

## 支持的变换

- 亮度调整
- 图像旋转
- 边界裁剪

## API使用

```python
from video_auth import VideoAuthenticator

# 创建认证器
auth = VideoAuthenticator(key="your_secret_key")

# 生成视频签名
signature = auth.generate_video_signature("video.mp4")
auth.save_signature(signature, "signature.pkl")

# 验证视频真实性
is_authentic, score, details = auth.verify_video_authenticity(
    "transformed_video.mp4", "signature.pkl", threshold=0.7
)
```

## 技术栈

- **算法**：OpenCV, NumPy, SciPy
- **Web**：Flask, Bootstrap 5
