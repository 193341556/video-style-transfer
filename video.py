import torch
import utils
import transformer
import cv2
import os
from stylize import stylize_folder_single
import numpy as np
from flowiz import read_flow
import pwc
import time

VIDEO_NAME = "street.mp4"
FRAME_SAVE_PATH = "frames/"
FRAME_CONTENT_FOLDER = "content_folder/"
FRAME_BASE_FILE_NAME = "frame"
FRAME_BASE_FILE_TYPE = ".jpg"
STYLE_FRAME_SAVE_PATH = "style_frames/"
STYLE_VIDEO_NAME = "adain_3.mp4"
STYLE_PATH = "transforms/adain_3.pth"
FLOW_PATH="flow/"
BATCH_SIZE = 1




def calculate_optical_flow(frames_path, flow_save_path):
    base_name_len = len(FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    images = [img for img in sorted(os.listdir(frames_path),
                                    key=lambda x: int(''.join(filter(str.isdigit, x[base_name_len:-filetype_len])))) if
              img.endswith(FRAME_BASE_FILE_TYPE)]
    count=0
    for i in range(len(images)-1):

        flow_path = os.path.join(flow_save_path, f"flow_{(count+1):05d}.flo")
        previmage=os.path.join(frames_path, images[count])
        nextimage = os.path.join(frames_path, images[count-1])
        pwc.pwcflow(previmage, nextimage, flow_path)
        count += 1
    print("Optical flow calculation complete.")


def video_transfer(video_path, style_path):
    print('Deleting')

    utils.delete_file(FRAME_SAVE_PATH)
    utils.delete_file(STYLE_FRAME_SAVE_PATH)
    utils.delete_file(FLOW_PATH)

    print("OpenCV {}".format(cv2.__version__))
    starttime = time.time()
    H, W, fps = getInfo(video_path)
    print("Height: {} Width: {} FPS: {}".format(H, W, fps))

    os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
    os.makedirs(FRAME_SAVE_PATH+FRAME_CONTENT_FOLDER, exist_ok=True)
    os.makedirs(STYLE_FRAME_SAVE_PATH, exist_ok=True)
    os.makedirs(FLOW_PATH, exist_ok=True)

    print("Extracting video frames")
    getFrames(video_path)

    print("Calculating optical flow")
    calculate_optical_flow(FRAME_SAVE_PATH+FRAME_CONTENT_FOLDER, FLOW_PATH)

    print("Performing style transfer on frames")
    stylize_folder_single(style_path, "frames/content_folder/", STYLE_FRAME_SAVE_PATH)

    print("Combining style frames into one video")
    makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, fps, int(H), int(W))
    print("Elapsed Time: {}".format(time.time() - starttime))


def getInfo(video_path):

    vidcap = cv2.VideoCapture(video_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    return height, width, fps


def getFrames(video_path):

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite("{}{}_{}{}".format(FRAME_SAVE_PATH + FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME, count,
                                      FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
        count += 1
    print("Done extracting all frames")


def makeVideo(frames_path, save_name, fps, height, width):

    base_name_len = len(FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    images = [img for img in sorted(os.listdir(frames_path), key=lambda x: int(''.join(filter(str.isdigit, x[base_name_len:-filetype_len])))) if img.endswith(FRAME_BASE_FILE_TYPE)]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    vout = cv2.VideoWriter(save_name, fourcc, fps, (width, height))

    for image_name in images:
        vout.write(cv2.imread(os.path.join(frames_path, image_name)))

    print("Done writing video")


video_transfer(VIDEO_NAME, STYLE_PATH)