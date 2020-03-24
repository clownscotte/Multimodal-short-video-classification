import csv
import os
import sys
from moviepy.editor import *
import cv2
import numpy
import torch





def video_to_audio(file_name):
  video = VideoFileClip(file_name)
  audio = video.audio
  return audio

"""
def extract_video_Name():
    # os.listdir(file)会历遍文件夹内的文件并返回一个列表
    path_list = os.listdir(file_path)
    # print(path_list)
    video_name = []
    # 利用循环历遍path_list列表并且利用split去掉后缀名
    for i in path_list:
      video_name.append(i.split(".")[0])
    return video_name
"""

def extract_video_Name (filename):
  video_Name=filename.split(".")[0]
  return video_Name



def extract_video_Frames (filename, frame_interval=500, max_num_frames=128):
  """
  提取视频的帧
  """
  video_capture = cv2.VideoCapture(filename)

  frames=[]

  last_ts = 99999
  num_retrieved = 0

  while num_retrieved < max_num_frames:

    while video_capture.get(CAP_PROP_POS_MSEC) < frame_interval + last_ts:#如果当前帧所在秒数在范围内
      if not video_capture.read()[0]:
        return

    last_ts = video_capture.get(CAP_PROP_POS_MSEC)
    Is_has_frames, frame = video_capture.read()
    if not Is_has_frames:
      break
    frames.append(frame)
    num_retrieved += 1

  return frames


"""
with open (File_name) as f:
 def extract_video_Frame_Feature(unused_argv):
   extractor = InceptionResNetV2()
   for video_file in csv.reader(f):
     rgb_features = []
     sum_rgb_features = None
     for rgb in frame_iterator(video_file):
       features = extractor(rgb[:, :, ::-1])
       if sum_rgb_features is None:
         sum_rgb_features = features
       else:
         sum_rgb_features += features
       rgb_features.append(_bytes_feature(quantize(features)))

   video_feature=numpy.array(rgb_features).reshape(([1538,8],128))
"""
