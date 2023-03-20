import cv2
import os
import argparse
from tqdm import tqdm


"""
When training, on the time of loading data
If we getting this ERROR
" ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor "

This is due to Frames missing from some of the video data
"""


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data_dir", type=str, required=True,
                help="path to data dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")

args = vars(ap.parse_args())
path_to_dir = args["data_dir"]
path_to_save = args['save']

for i in os.listdir(path_to_dir):
    print(f'{i}:')
    for j in os.listdir(f'{path_to_dir}/{i}'):
        path_to_vids = f'{path_to_dir}/{i}/{j}'
        vids = os.listdir(path_to_vids)
        os.makedirs(f'{path_to_save}/{i}/{j}', exist_ok=True)
        print(f'{j}:')
        for vid in tqdm(vids):
            path_to_vid = f'{path_to_dir}/{i}/{j}/{vid}'
            cap = cv2.VideoCapture(path_to_vid)
            original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Write Video
            out_vid = cv2.VideoWriter(f'{path_to_save}/{i}/{j}/{vid}', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (original_video_width, original_video_height))
            while True:
                success, img = cap.read()
                if not success:
                    # print(f'[INFO] Failed to read {vid}')
                    break
                out_vid.write(img)

            cap.release()
            out_vid.release()
        print(f'[INFO] Completed {j}..')
