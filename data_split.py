from sklearn.model_selection import train_test_split
import glob
import os
import shutil
import subprocess
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data_dir", type=str, required=True,
                help="path to data dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")
ap.add_argument("-r", "--ratio", type=float, default=0.1,
                help="test ratio 0<ratio<1")

args = vars(ap.parse_args())
path_data_dir = args["data_dir"]
path_save_dir = args['save']
ratio = args['ratio']


class_names = sorted(os.listdir(path_data_dir))
print('class_names:\n', class_names)

for class_i in class_names:
    print(f'[INFO] Started {class_i}..')
    os.makedirs(f"{path_save_dir}/train/{class_i}", exist_ok=True)
    os.makedirs(f"{path_save_dir}/test/{class_i}", exist_ok=True)

    # Data Format if NOT AVI - Convert to -> *.avi
    vids_list_mp4 = glob.glob(f'{path_data_dir}/{class_i}/*.mp4')
    vids_list_avi = glob.glob(f'{path_data_dir}/{class_i}/*.avi')
    vids_list = vids_list_mp4 + vids_list_avi
    if len(vids_list):
        train, test = train_test_split(vids_list, test_size=ratio)
        for i in train:
            filename = os.path.split(i)[1]
            # Check the format its avi or NOT
            if os.path.splitext(filename)[1] != '.avi':
                save_path = f"{path_save_dir}/train/{class_i}/{os.path.splitext(filename)[0]}.avi"
                cmd = f'ffmpeg -i \"{i}\" -vcodec copy -acodec copy \"{save_path}\"'
                subprocess.call(cmd, shell=True)
            else:
                save_path = f"{path_save_dir}/train/{class_i}/{filename}"
                shutil.copy(i, save_path)
        print(f'[INFO] Completed {class_i} train Dir')
        
        for j in test:
            filename = os.path.split(j)[1]
            # Check the format its avi or NOT
            if os.path.splitext(filename)[1] != '.avi':
                save_path = f"{path_save_dir}/test/{class_i}/{os.path.splitext(filename)[0]}.avi"
                cmd = f'ffmpeg -i \"{i}\" -vcodec copy -acodec copy \"{save_path}\"'
                subprocess.call(cmd, shell=True)
            else:
                save_path = f"{path_save_dir}/test/{class_i}/{filename}"
                shutil.copy(j, save_path)
        print(f'[INFO] Completed {class_i} test Dir')
        print(f'[INFO] Completed {class_i}')
    else:
        print('[ERROR] Video data NOT found!!..')
