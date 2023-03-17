import cv2
import os


path_to_dir = 'data'
for i in os.listdir(path_to_dir):
    for j in os.listdir(f'{path_to_dir}/{i}'):
        path_to_vids = f'{path_to_dir}/{i}/{j}'
        vids = os.listdir(path_to_vids)
        os.makedirs(f'dataset/{i}/{j}', exist_ok=True)
        for vid in vids:
            print(f'[INFO] Loading {vid}')
            path_to_vid = f'{path_to_dir}/{i}/{j}/{vid}'
            cap = cv2.VideoCapture(path_to_vid)
            original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Write Video
            out_vid = cv2.VideoWriter(f'dataset/{i}/{j}/{vid}', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (original_video_width, original_video_height))
            while True:
                success, img = cap.read()
                if not success:
                    print(f'[INFO] Failed to read {vid}')
                    break
                out_vid.write(img)

            cap.release()
            out_vid.release()
            print(f'[INFO] Completed {vid}..')
