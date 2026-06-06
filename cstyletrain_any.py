import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import time
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms

import vgg
import transformer_any as transformer
import utils
from train import VideoDataset, TemporalConsistencyLoss

PTH="any3.pth"

TRAIN_IMAGE_SIZE = 256
VIDEO_DIR = "database_video"
STYLE_IMAGE_PATH = "images/night.jpg"

BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
TEMPORAL_WEIGHT = 30
ADAM_LR = 1e-3
NUM_EPOCHS = 1
SAVE_MODEL_PATH = "models/"
PRINT_EVERY = 50
SAVE_EVERY = 50
GENERATED_IMAGE_DIR = "outputs/"
CHECKPOINT_TEMPLATE = "epoch_{}_batch_{}.pth"



def continue_train(checkpoint_path=None):
    content_losses = []
    style_losses = []
    temp_losses = []
    total_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    dataset = VideoDataset(VIDEO_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0
    )

    STYLE_FEATURE_DIM_FOR_ADAIN = 512
    transformer_net = transformer.TransformerNetwork(
        style_feature_dim=STYLE_FEATURE_DIM_FOR_ADAIN,
        norm_type="adain"
    ).to(device)
    vgg_net = vgg.VGG16().to(device)

    optimizer = optim.Adam(transformer_net.parameters(), lr=ADAM_LR)


    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"加载模型参数: {checkpoint_path}")
        transformer_net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68],
                                     dtype=torch.float32).view(1, 3, 1, 1).to(device)

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

            style_path = STYLE_IMAGE_PATH
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

            gen_current = transformer_net(current, style_feat=style_features_for_adain_map)
            gen_next = transformer_net(next_frame, style_feat=style_features_for_adain_map)

            content_features = vgg_net(current.add(imagenet_neg_mean))
            gen_features = vgg_net(gen_current.add(imagenet_neg_mean))
            content_loss = CONTENT_WEIGHT * nn.MSELoss()(gen_features['relu2_2'], content_features['relu2_2'])

            style_loss = 0
            for layer, feat in gen_features.items():
                style_loss += STYLE_WEIGHT * nn.MSELoss()(utils.gram(feat), style_grams[layer])


            temp_loss = TemporalConsistencyLoss()(gen_current, gen_next, flow)

            total_loss = content_loss + style_loss + temp_loss
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            temp_losses.append(temp_loss.item())
            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            if (batch_idx + 1) % PRINT_EVERY == 0:
                avg_loss = total_loss_sum / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.2f}',
                    'Content': f'{content_loss.item():.2f}',
                    'Style': f'{style_loss.item():.2f}',
                    'Speed': f'{(time.time() - start_time) / (batch_idx + 1):.2f}s/batch'
                })
                print(
                    f"[Batch {batch_idx + 1}] Content Loss: {content_loss.item():.4f} | "
                    f"Style Loss: {style_loss.item():.4f} | "
                    f"Temporal Loss: {temp_loss.item():.4f} | "
                    f"Total Loss: {total_loss.item():.4f}"
                )

            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
            if (batch_idx + 1) % SAVE_EVERY == 0:
                checkpoint_path_save = os.path.join(
                    SAVE_MODEL_PATH,
                    CHECKPOINT_TEMPLATE.format(epoch + 1, batch_idx + 1)
                )
                torch.save(transformer_net.state_dict(), checkpoint_path_save)
                print(f"\n{checkpoint_path_save}")
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
                    gen_np = gen_sample.permute(1, 2, 0).numpy()
                    gen_np = np.clip(gen_np, 0, 255).astype(np.uint8)
                    gen_bgr = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
                    gen_filename = os.path.join(
                        GENERATED_IMAGE_DIR,
                        f"epoch{epoch + 1}_batch{batch_idx + 1}_gen.png"
                    )
                    cv2.imwrite(gen_filename, gen_bgr)

        checkpoint_path_epoch = f"{SAVE_MODEL_PATH}epoch_{epoch + 1}.pth"
        torch.save(transformer_net.state_dict(), checkpoint_path_epoch)
        print(
            f"✅ Epoch {epoch + 1} | 平均损失: {total_loss_sum / len(train_loader):.2f} | 耗时: {time.time() - start_time:.1f}s")

    final_path = f"{SAVE_MODEL_PATH}final_transformer.pth"
    torch.save(transformer_net.state_dict(), final_path)
    print(f"\n{final_path}")

if __name__ == "__main__":
    continue_train(PTH)
