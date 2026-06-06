
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import vgg
import transformer_any as transformer
import utils

import random
from torchvision import transforms
import time

# 配置参数
TRAIN_IMAGE_SIZE = 256
VIDEO_DIR = "database_video"
STYLE_IMAGES_DIR = "style_images"
BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
TEMPORAL_WEIGHT = 0
ADAM_LR = 1e-3
NUM_EPOCHS = 10
SAVE_MODEL_PATH = "models/"
PRINT_EVERY = 5
SAVE_EVERY = 5
GENERATED_IMAGE_DIR="outputs/"
CHECKPOINT_TEMPLATE = "epoch_{}_batch_{}.pth"


style_image_paths = [
    str(p) for p in Path(STYLE_IMAGES_DIR).glob("*")
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
]
assert len(style_image_paths) > 0, f"未找到风格图像：{STYLE_IMAGES_DIR}"

class VideoDataset(Dataset):
    def __init__(self, video_dir):

        self.video_paths = [
            str(p) for p in Path(video_dir).glob("*")
            if p.suffix.lower() in ('.mp4', '.avi', '.mov')
        ]
        assert len(self.video_paths) > 0, f"未找到视频文件：{video_dir}"


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])


        self.samples = []
        for vpath in self.video_paths:
            cap = cv2.VideoCapture(vpath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total_frames > 1:
                self.samples.extend([(vpath, i) for i in range(total_frames - 1)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx = self.samples[idx]


        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        _, prev_frame = cap.read()
        _, next_frame = cap.read()
        cap.release()

        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        prev_frame = cv2.resize(prev_frame, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
        next_frame = cv2.resize(next_frame, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )

        return {
            "current": self.transform(prev_frame),
            "next": self.transform(next_frame),
            "flow": torch.from_numpy(flow.transpose(2, 0, 1)).float()
        }



def train():
    content_losses = []
    style_losses = []
    temp_losses = []
    total_losses = []
    # 初始化设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # 数据加载
    dataset = VideoDataset(VIDEO_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0
    )

    # 网络初始化
    STYLE_FEATURE_DIM_FOR_ADAIN = 512 
    transformer_net = transformer.TransformerNetwork(
        style_feature_dim=STYLE_FEATURE_DIM_FOR_ADAIN, 
        norm_type="adain"
    ).to(device)
    vgg_net = vgg.VGG16().to(device)

    # 优化器
    optimizer = optim.Adam(transformer_net.parameters(), lr=ADAM_LR)

    # 风格特征提取
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68],
                                     dtype=torch.float32).view(1, 3, 1, 1).to(device)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}',
                            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        total_loss_sum = 0.0
        start_time = time.time()

        for batch_idx, batch in progress_bar:
            current = batch["current"].to(device)
            next_frame = batch["next"].to(device)
            flow = batch["flow"].to(device)

            # 每个 epoch 都随机采样风格，并始终计算时间损失
            style_path = random.choice(style_image_paths)
            calc_temp_loss = True

            style_image = utils.load_image(style_path)
            style_tensor = utils.itot(style_image, max_size=TRAIN_IMAGE_SIZE).to(device)
            current_batch_size = current.size(0)
            if current_batch_size > 1:
                repeated_style_tensor_for_vgg = style_tensor.repeat(current_batch_size, 1, 1, 1)
            else:
                repeated_style_tensor_for_vgg = style_tensor
            full_style_features_from_vgg = vgg_net(repeated_style_tensor_for_vgg.add(imagenet_neg_mean))
            style_grams = {layer: utils.gram(feat) for layer, feat in full_style_features_from_vgg.items()}
            style_features_for_adain_map = full_style_features_from_vgg['relu4_3']

            # 前向传播（动态风格特征输入，支持任意风格迁移）
            gen_current = transformer_net(current, style_feat=style_features_for_adain_map)
            gen_next = transformer_net(next_frame, style_feat=style_features_for_adain_map)

            # 内容损失
            content_features = vgg_net(current.add(imagenet_neg_mean))
            gen_features = vgg_net(gen_current.add(imagenet_neg_mean))
            content_loss = CONTENT_WEIGHT * nn.MSELoss()(gen_features['relu2_2'], content_features['relu2_2'])

            # 风格损失
            style_loss = 0
            for layer, feat in gen_features.items():
                style_loss += STYLE_WEIGHT * nn.MSELoss()(utils.gram(feat), style_grams[layer])


            temp_loss = torch.tensor(0.0, device=device)

            # 总损失
            total_loss = content_loss + style_loss + temp_loss
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            temp_losses.append(temp_loss.item())
            total_losses.append(total_loss.item())

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 更新进度条
            total_loss_sum += total_loss.item()
            if (batch_idx + 1) % PRINT_EVERY == 0:
                avg_loss = total_loss_sum / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.2f}',
                    'Content': f'{content_loss.item():.2f}',
                    'Style': f'{style_loss.item():.2f}',
                    'Speed': f'{(time.time() - start_time) / (batch_idx + 1):.2f}s/batch'
                })


            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

            if (batch_idx + 1) % SAVE_EVERY == 0:
                checkpoint_path = os.path.join(
                    SAVE_MODEL_PATH,
                    CHECKPOINT_TEMPLATE.format(epoch + 1, batch_idx + 1)
                )
                torch.save(transformer_net.state_dict(), checkpoint_path)
                print(f"\n{checkpoint_path}")
                utils.plot_loss_hist(
                    content_losses,
                    style_losses,
                    temp_losses,
                    total_losses,
                    title=f"Training Loss (Epochs={NUM_EPOCHS}, LR={ADAM_LR})"
                )
            if (batch_idx + 1) % SAVE_EVERY == 0:
                with torch.no_grad():

                    sample_idx = 0

                    gen_sample = gen_current[sample_idx].detach().cpu()
                    gen_np = gen_sample.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
                    gen_np = np.clip(gen_np, 0, 255).astype(np.uint8)
                    gen_bgr = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
                    gen_filename = os.path.join(
                        GENERATED_IMAGE_DIR,
                        f"epoch{epoch + 1}_batch{batch_idx + 1}_gen.png"
                    )
                    cv2.imwrite(gen_filename, gen_bgr)


        checkpoint_path = f"{SAVE_MODEL_PATH}epoch_{epoch + 1}.pth"
        torch.save(transformer_net.state_dict(), checkpoint_path)
        print(
            f"Epoch {epoch + 1} | 平均损失: {total_loss_sum / len(train_loader):.2f} | 耗时: {time.time() - start_time:.1f}s")

    final_path = f"{SAVE_MODEL_PATH}final_transformer.pth"
    torch.save(transformer_net.state_dict(), final_path)
    print(f"\n{final_path}")

if __name__ == "__main__":
    train()
