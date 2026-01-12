"""
视频真实性验证算法包

提供对视频变换鲁棒的真实性验证功能。
支持亮度、旋转、裁剪等变换后的验证。
"""

from .authenticator import VideoAuthenticator

__version__ = "1.0.0"
__all__ = ["VideoAuthenticator"]