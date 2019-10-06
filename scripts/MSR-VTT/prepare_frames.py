# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import argparse as ap
import subprocess as sp
import glob


# 提取视频帧，放缩到相应大小
def allocate_videos(dir_name, output_path):
    def extract_frames(vid_path):
        vidname = os.path.basename(vid_path).strip().split('.')[0]
        save_path = os.path.join(output_path, vidname)
        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass

        video_to_frames_command = ['ffmpeg', '-y', '-i', vid_path, '-vf', 
        'scale=%d:%d'%(340, 256), '-qscale:v', '2', '-r', '8',
        os.path.join(save_path, '%04d.jpg')]
        sp.run(video_to_frames_command, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        print(vidname, 'has been extracted to', save_path)

    vids = glob.glob(os.path.join(dir_name, '*'))
    for vid in vids:
        extract_frames(vid) 


def main():
    parser = ap.ArgumentParser(description="Get video directory path")
    parser.add_argument('--videopath', type=str)
    parser.add_argument('--outputpath', type=str)
    args = parser.parse_args()

    allocate_videos(args.videopath, args.outputpath)


if __name__ == "__main__":
    main()
