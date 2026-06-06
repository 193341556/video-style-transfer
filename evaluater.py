import cv2
import numpy as np
import os
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, Optional


torch.autograd.set_grad_enabled(False)
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def video_read_check(video_path: str) -> cv2.VideoCapture:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            "视频解码失败！解决方案：\n"
            "1. 使用FFmpeg转换格式：\n"
            "   ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4\n"
            "2. 确保视频分辨率为偶数尺寸"
        )

    if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
        cap.release()
        raise ValueError("视频不包含有效帧")

    return cap


def compute_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:

    # 尺寸一致性检查
    if prev_frame.shape != next_frame.shape:
        next_frame = cv2.resize(next_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )


def compute_metrics(
        style_img_path: str,
        content_video_path: str,
        output_video_path: str,
        sample_interval: int = 30,
        max_frames: Optional[int] = None
) -> Dict[str, float]:


    loss_fn = lpips.LPIPS(net='vgg')

    cap_cont = video_read_check(content_video_path)
    ret, base_frame = cap_cont.read()
    if not ret:
        cap_cont.release()
        raise ValueError("无法读取内容视频首帧")
    target_size = (base_frame.shape[1], base_frame.shape[0])  # (width, height)
    cap_cont.release()

    if not os.path.exists(style_img_path):
        raise FileNotFoundError(f"风格图像不存在: {style_img_path}")

    style_img = cv2.imread(style_img_path)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    style_img = cv2.resize(style_img, target_size)  # 统一到内容视频尺寸
    img0 = lpips.im2tensor(style_img)  # shape: [1,3,H,W]

    cap_cont = video_read_check(content_video_path)
    cap_out = video_read_check(output_video_path)

    try:
        total_frames = min(int(cap_cont.get(cv2.CAP_PROP_FRAME_COUNT)),
                           int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT)))
        import random
        frame_indices = list(range(total_frames))
        if len(frame_indices) < 8:
            sampled_indices = frame_indices
        else:
            sampled_indices = sorted(random.sample(frame_indices, 8))

        ssim_values, psnr_values = [], []
        lpips_values, flow_diffs = [], []
        prev_cont, prev_out = None, None

        for idx in sampled_indices:
            cap_cont.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_out.set(cv2.CAP_PROP_POS_FRAMES, idx)

            frame_cont, frame_out = None, None
            for _ in range(3):
                ret_cont, frame_cont = cap_cont.read()
                ret_out, frame_out = cap_out.read()
                if ret_cont and ret_out:
                    break

            if not (ret_cont and ret_out):
                print(f"警告：跳过损坏帧 {idx}")
                continue


            frame_out = cv2.resize(frame_out, target_size)

            # 结构相似性 (SSIM)
            gray_cont = cv2.cvtColor(frame_cont, cv2.COLOR_BGR2GRAY)
            gray_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2GRAY)
            ssim_val = ssim(gray_cont, gray_out, data_range=255)
            ssim_values.append(ssim_val)

            # 峰值信噪比 (PSNR)
            psnr_val = psnr(frame_cont, frame_out, data_range=255)
            psnr_values.append(psnr_val)

            # 风格相似性 (LPIPS)
            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            img1 = lpips.im2tensor(frame_rgb)
            lpips_val = loss_fn(img0, img1).item()
            lpips_values.append(lpips_val)

            # 光流一致性
            if prev_cont is not None and prev_out is not None:
                try:
                    flow_cont = compute_optical_flow(prev_cont, frame_cont)
                    flow_out = compute_optical_flow(prev_out, frame_out)
                    flow_diff = np.mean(np.abs(flow_cont - flow_out))
                    flow_diffs.append(flow_diff)
                except cv2.error as e:
                    print(f"光流计算跳过帧 {idx}: {str(e)}")

            prev_cont, prev_out = frame_cont.copy(), frame_out.copy()

        metrics = {
            "Sampled_Frames": len(sampled_indices),
            "Valid_Frames": len(ssim_values),
            "Content_SSIM": np.mean(ssim_values) if ssim_values else 0,
            "PSNR": np.mean(psnr_values) if psnr_values else 0,
            "Style_LPIPS": np.mean(lpips_values) if lpips_values else 0,
            "Flow_Consistency": np.mean(flow_diffs) if flow_diffs else 0,
        }

        return metrics

    finally:
        cap_cont.release()
        cap_out.release()


if __name__ == "__main__":
    try:
        metrics = compute_metrics(
            style_img_path="",
            content_video_path="",
            output_video_path=r"",
            sample_interval=10,
            max_frames=100
        )

        print("\n视频质量评估报告：")
        print(f"采样帧数: {metrics['Sampled_Frames']} (有效 {metrics['Valid_Frames']})")
        print("=" * 40)
        print(f"{'内容保真度(SSIM)':<20}: {metrics['Content_SSIM']:.4f}")
        print(f"{'图像质量(PSNR)':<20}: {metrics['PSNR']:.2f} dB")
        print(f"{'风格相似性(LPIPS)':<20}: {metrics['Style_LPIPS']:.4f}")
        print(f"{'运动一致性(Flow)':<20}: {metrics['Flow_Consistency']:.4f}")

    except Exception as e:
        print(f"\n评估错误: {str(e)}")
        if "video" in str(e).lower():
            print("推荐预处理命令：")
            print(
                "ffmpeg -i input.mp4 -vf 'scale=iw-mod(iw\,2):ih-mod(ih\,2)' -c:v libx264 -pix_fmt yuv420p output.mp4")
