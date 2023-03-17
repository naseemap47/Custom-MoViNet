import tqdm
import random
import pathlib
import collections
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf


def list_files_per_class(zip_url):
    """
        List the files in each class of the dataset given the zip URL.

        Args:
        zip_url: URL from which the files can be unzipped. 

        Return:
        files: List of files in each of the classes.
    """
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_class(fname):
    """
        Retrieve the name of the class given a filename.

        Args:
        fname: Name of the file in the UCF101 dataset.

        Return:
        Class that the file belongs to.
    """
    return fname.split('_')[-3]

def get_files_per_class(files):
    """
        Retrieve the files that belong to each class. 

        Args:
        files: List of files in the dataset.

        Return:
        Dictionary of class names (key) and files (values).
    """
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
    """
        Download the contents of the zip file from the zip URL.

        Args:
        zip_url: Zip URL containing data.
        to_dir: Directory to download data to.
        file_names: Names of files to download.
    """
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn

            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file,)

def split_class_lists(files_for_class, count):
    """
        Returns the list of files belonging to a subset of data as well as the remainder of
        files that need to be downloaded.

        Args:
        files_for_class: Files belonging to a particular class of data.
        count: Number of files to download.

        Return:
        split_files: Files belonging to the subset of data.
        remainder: Dictionary of the remainder of files that need to be downloaded.
    """
    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return split_files, remainder

def download_ufc_101_subset(zip_url, num_classes, splits, download_dir):
    """
        Download a subset of the UFC101 dataset and split them into various parts, such as
        training, validation, and test. 

        Args:
        zip_url: Zip URL containing data.
        num_classes: Number of labels.
        splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
                (value is number of files per split).
        download_dir: Directory to download data to.

        Return:
        dir: Posix path of the resulting directories containing the splits of data.
    """
    files = list_files_per_class(zip_url)
    # print(files)
    for f in files:
        tokens = f.split('/')
        if len(tokens) <= 2:
            files.remove(f) # Remove that item from the list if it does not have a filename

    files_for_class = get_files_per_class(files)
    print(files_for_class)

    classes = list(files_for_class.keys())[:num_classes]

    for cls in classes:
        new_files_for_class = files_for_class[cls]
        random.shuffle(new_files_for_class)
        files_for_class[cls] = new_files_for_class

    # Only use the number of classes you want in the dictionary
    files_for_class = {x: files_for_class[x] for x in list(files_for_class)[:num_classes]}

    dirs = {}
    for split_name, split_count in splits.items():
        print(split_name, ":")
        split_dir = download_dir / split_name
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        # download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir

    return dirs

def format_frames(frame, output_size):
    """
        Pad and resize an image from a video.

        Args:
        frame: Image that needs to resized and padded. 
        output_size: Pixel size of the output frame image.

        Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
    """
        Creates frames from each video file present for each category.

        Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

        Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        """ Returns a set of frames with their associated label. 

        Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label