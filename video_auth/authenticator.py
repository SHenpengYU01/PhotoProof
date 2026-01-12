import cv2
import numpy as np
import hashlib
import json
from scipy import fftpack
import pickle
import base64
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class VideoAuthenticator:
    def __init__(self, key: str = None):
        """
        初始化视频认证器
        
        参数:
            key: 认证密钥（可选），用于增强安全性
        """
        self.key = key or "default_secret_key"
        self.sift = cv2.SIFT_create()
        
    def extract_robust_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        提取对亮度、旋转、裁剪鲁棒的特征
        
        参数:
            frame: 视频帧
            
        返回:
            包含鲁棒特征的字典
        """
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 1. 使用SIFT提取关键点和描述子（对旋转、缩放、亮度变化部分不变）
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # 如果没有检测到特征，使用备用方法
        if descriptors is None:
            descriptors = np.array([])
            
        # 2. 提取对亮度变化鲁棒的特征（使用归一化直方图）
        hist_features = self._extract_illumination_invariant_features(frame)
        
        # 3. 提取对裁剪鲁棒的特征（使用图像的统计特征）
        statistical_features = self._extract_crop_invariant_features(frame)
        
        # 4. 提取频域特征（对空间变换鲁棒）
        frequency_features = self._extract_frequency_features(gray)
        
        return {
            'sift_keypoints': len(keypoints),
            'sift_descriptors': descriptors.tolist() if len(descriptors) > 0 else [],
            'hist_features': hist_features,
            'statistical_features': statistical_features,
            'frequency_features': frequency_features,
            'frame_shape': frame.shape
        }
    
    def _extract_illumination_invariant_features(self, frame: np.ndarray) -> Dict[str, float]:
        """提取对亮度变化鲁棒的特征"""
        if len(frame.shape) == 3:
            # 使用HSV颜色空间的H和S通道（对亮度变化相对不敏感）
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            
            # 计算直方图并归一化
            h_hist = cv2.calcHist([h_channel], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([s_channel], [0], None, [16], [0, 256])
            
            h_hist = h_hist / (h_hist.sum() + 1e-6)
            s_hist = s_hist / (s_hist.sum() + 1e-6)
        else:
            # 对于灰度图，使用梯度方向直方图
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
            magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
            angle_hist = cv2.calcHist([angle.astype(np.uint8)], [0], None, [16], [0, 360])
            angle_hist = angle_hist / (angle_hist.sum() + 1e-6)
            h_hist, s_hist = angle_hist, angle_hist
        
        return {
            'h_hist': h_hist.flatten().tolist(),
            's_hist': s_hist.flatten().tolist()
        }
    
    def _extract_crop_invariant_features(self, frame: np.ndarray) -> Dict[str, float]:
        """提取对裁剪鲁棒的特征"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 图像矩（对平移、旋转、缩放具有不变性）
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # 对Hu矩取对数以增强数值稳定性
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 图像的统计特征（对部分裁剪相对鲁棒）
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        entropy = self._calculate_entropy(gray)
        
        # 边缘密度（对裁剪有一定鲁棒性）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.size + 1e-6)
        
        return {
            'hu_moments': hu_moments.tolist(),
            'mean': float(mean_val),
            'std': float(std_val),
            'entropy': float(entropy),
            'edge_density': float(edge_density)
        }
    
    def _extract_frequency_features(self, gray_frame: np.ndarray) -> Dict[str, Any]:
        """提取频域特征（对空间变换鲁棒）"""
        # 傅里叶变换
        f_transform = fftpack.fft2(gray_frame)
        f_transform_shifted = fftpack.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        
        # 提取低频和高频能量特征
        h, w = gray_frame.shape
        center_h, center_w = h // 2, w // 2
        
        # 低频区域（中心区域）
        low_freq_radius = min(h, w) // 4
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_w, center_h), low_freq_radius, 1, -1)
        mask = mask.astype(bool)
        low_freq_energy = np.sum(magnitude_spectrum[mask])
        
        # 高频区域
        high_freq_energy = np.sum(magnitude_spectrum[~mask])
        
        # 总能量
        total_energy = np.sum(magnitude_spectrum)
        
        return {
            'low_freq_ratio': float(low_freq_energy / (total_energy + 1e-6)),
            'high_freq_ratio': float(high_freq_energy / (total_energy + 1e-6)),
            'total_energy': float(total_energy)
        }
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """计算图像熵"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def generate_video_signature(self, video_path: str, sampling_rate: int = 10) -> Dict[str, Any]:
        """
        为视频生成鲁棒签名
        
        参数:
            video_path: 视频文件路径
            sampling_rate: 采样率（每N帧采样一帧）
            
        返回:
            视频签名
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        features_list = []
        frame_indices = []
        
        print(f"处理视频: {video_path}")
        print(f"总帧数: {frame_count}")
        
        for i in range(0, frame_count, sampling_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            features = self.extract_robust_features(frame)
            features_list.append(features)
            frame_indices.append(i)
            
            if len(features_list) % 10 == 0:
                print(f"已处理 {len(features_list)} 个关键帧...")
        
        cap.release()
        
        # 计算整个视频的聚合特征
        aggregated_features = self._aggregate_features(features_list)
        
        # 生成签名
        signature = {
            'video_path': video_path,
            'frame_count': frame_count,
            'sampling_rate': sampling_rate,
            'sampled_frames': len(features_list),
            'features': aggregated_features,
            'frame_features': features_list,
            'timestamp': np.datetime64('now').astype(str)
        }
        
        return signature
    
    def _aggregate_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个帧的特征"""
        aggregated = {}
        
        # 对数值特征取平均
        numerical_features = ['mean', 'std', 'entropy', 'edge_density']
        for feat in numerical_features:
            values = [f['statistical_features'][feat] for f in features_list]
            aggregated[f'avg_{feat}'] = float(np.mean(values))
            aggregated[f'std_{feat}'] = float(np.std(values))
        
        # 对Hu矩取平均
        hu_moments_list = [np.array(f['statistical_features']['hu_moments']) 
                          for f in features_list]
        avg_hu_moments = np.mean(hu_moments_list, axis=0)
        aggregated['avg_hu_moments'] = avg_hu_moments.tolist()
        
        # 对频域特征取平均
        freq_features = ['low_freq_ratio', 'high_freq_ratio', 'total_energy']
        for feat in freq_features:
            values = [f['frequency_features'][feat] for f in features_list]
            aggregated[f'avg_{feat}'] = float(np.mean(values))
        
        # 对直方图特征取平均
        h_hists = [np.array(f['hist_features']['h_hist']) for f in features_list]
        s_hists = [np.array(f['hist_features']['s_hist']) for f in features_list]
        
        aggregated['avg_h_hist'] = np.mean(h_hists, axis=0).tolist()
        return aggregated
    
    def save_signature(self, signature: Dict[str, Any], output_path: str):
        """保存签名到文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(signature, f)
        print(f"签名已保存到: {output_path}")
    
    def load_signature(self, signature_path: str) -> Dict[str, Any]:
        """从文件加载签名"""
        with open(signature_path, 'rb') as f:
            signature = pickle.load(f)
        return signature
    
    def apply_transformations(self, frame: np.ndarray, 
                            brightness_change: float = 0.0,
                            rotation_angle: float = 0.0,
                            crop_ratio: float = 0.0) -> np.ndarray:
        """
        应用变换到帧
        
        参数:
            frame: 输入帧
            brightness_change: 亮度变化（-1.0到1.0）
            rotation_angle: 旋转角度（度）
            crop_ratio: 裁剪比例（0.0到0.5）
            
        返回:
            变换后的帧
        """
        transformed = frame.copy()
        
        # 1. 亮度调整
        if brightness_change != 0:
            transformed = cv2.convertScaleAbs(transformed, alpha=1, beta=brightness_change * 255)
        
        # 2. 旋转
        if rotation_angle != 0:
            h, w = transformed.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            transformed = cv2.warpAffine(transformed, rotation_matrix, (w, h))
        
        # 3. 裁剪
        if crop_ratio > 0:
            h, w = transformed.shape[:2]
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)
            transformed = transformed[crop_h:h-crop_h, crop_w:w-crop_w]
            # 调整大小到原始尺寸
            transformed = cv2.resize(transformed, (w, h))
        
        return transformed
    
    def verify_video_authenticity(self, video_path: str, 
                                original_signature_path: str,
                                threshold: float = 0.6) -> Tuple[bool, float, Dict[str, Any]]:
        """
        验证视频的真实性
        
        参数:
            video_path: 待验证视频路径
            original_signature_path: 原始签名文件路径
            threshold: 相似度阈值
            
        返回:
            (是否真实, 相似度分数, 详细结果)
        """
        # 加载原始签名
        original_signature = self.load_signature(original_signature_path)
        
        # 为待验证视频生成签名
        test_signature = self.generate_video_signature(video_path, 
                                                     sampling_rate=original_signature['sampling_rate'])
        
        # 计算特征相似度
        similarity_score = self._compute_similarity(original_signature, test_signature)
        
        # 综合判断
        is_authentic = similarity_score >= threshold
        
        result_details = {
            'similarity_score': similarity_score,
            'threshold': threshold,
            'frame_count_match': (test_signature['frame_count'] == 
                                original_signature['frame_count'])
        }
        
        return is_authentic, similarity_score, result_details
    
    def _compute_similarity(self, sig1: Dict[str, Any], 
                          sig2: Dict[str, Any]) -> float:
        """计算两个签名的相似度"""
        similarities = []
        
        # 1. 比较统计特征
        stat_feats1 = sig1['features']
        stat_feats2 = sig2['features']
        
        # 比较Hu矩（余弦相似度）
        hu1 = np.array(stat_feats1['avg_hu_moments'])
        hu2 = np.array(stat_feats2['avg_hu_moments'])
        hu_sim = np.dot(hu1, hu2) / (np.linalg.norm(hu1) * np.linalg.norm(hu2) + 1e-10)
        similarities.append(max(0, hu_sim))
        
        # 比较直方图特征
        h_hist1 = np.array(stat_feats1['avg_h_hist'])
        h_hist2 = np.array(stat_feats2['avg_h_hist'])
        h_hist_sim = (cv2.compareHist(h_hist1.astype(np.float32), 
                                     h_hist2.astype(np.float32), 
                                     cv2.HISTCMP_CORREL) + 1) / 2
        similarities.append(max(0, h_hist_sim))
        
        # 2. 比较频域特征
        freq_keys = ['avg_low_freq_ratio', 'avg_high_freq_ratio']
        for key in freq_keys:
            val1 = stat_feats1[key]
            val2 = stat_feats2[key]
            freq_sim = 1 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
            similarities.append(max(0, freq_sim))
        
        # 3. 比较数值特征
        num_keys = ['avg_mean', 'avg_std', 'avg_entropy', 'avg_edge_density']
        for key in num_keys:
            val1 = stat_feats1[key]
            val2 = stat_feats2[key]
            num_sim = 1 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
            similarities.append(max(0, num_sim))
        
        # 返回加权平均相似度
        weights = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Hu矩2倍，直方图1.5，其他1倍
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)
        return float(weighted_sum / total_weight)


