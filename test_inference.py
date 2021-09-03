import numpy as np
import sys
sys.path.append('../app/src/main')
from inference import inference


def inference_video():
    video_dir = './ped2/testing/frames/01'
    anomaly_score = inference(video_dir)
    print(anomaly_score)

if __name__ == '__main__':
    inference_video()