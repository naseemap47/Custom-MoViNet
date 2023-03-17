import os

import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import SparseCategoricalCrossentropy

import tensorflow_datasets as tfds
from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model
from collections import deque
import cv2


model_id = 'a1'

# Create the interpreter and signature runner
interpreter = tf.lite.Interpreter(model_path=f'movinet_{model_id}_stream.tflite')
runner = interpreter.get_signature_runner()

init_states = {
    name: tf.zeros(x['shape'], dtype=x['dtype'])
    for name, x in runner.get_input_details().items()
}
del init_states['image']


#################### Video Stream ###############################
cap = cv2.VideoCapture('archery.mp4')

original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Write Video
save = True
if save:
    out_vid = cv2.VideoWriter(f'output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (original_video_width, original_video_height))

image_size = (172, 172)

frames_queue = deque(maxlen=8)

with tf.io.gfile.GFile("ufc101_label_map.txt") as f:
    lines = f.readlines()
    KINETICS_600_LABELS_LIST = [line.strip() for line in lines]
    KINETICS_600_LABELS = tf.constant(KINETICS_600_LABELS_LIST)

def load_gif(file_path, image_size=(172, 172)):
    """Loads a gif file into a TF tensor."""
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        video = tf.io.decode_gif(f.read())
    video = tf.image.resize(video, image_size)
    video = tf.cast(video, tf.float32) / 255.
    return video

def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
    """Outputs the top k model labels and probabilities on the given video."""
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed Read Video..')
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames_queue.append(img_rgb)
    if len(frames_queue) == 8:
        # img_resize = cv2.resize(img, image_size)
        img_resize = tf.image.resize(frames_queue, image_size)
        img_norm = tf.cast(img_resize, tf.float32) / 255.
            
        clips = tf.split(img_norm[tf.newaxis], img_norm.shape[0], axis=1)

        # To run on a video, pass in one frame at a time
        states = init_states
        for clip in clips:
            # Input shape: [1, 1, 172, 172, 3] ---> 224, 224 for a2 version
            outputs = runner(**states, image=clip)
            logits = outputs.pop('logits')[0]
            states = outputs

        probs = tf.nn.softmax(logits)
        top_k = get_top_k(probs, k=1)
        print(top_k[0])
        # for label, prob in top_k:
        #     print(label, prob)
        cv2.putText(img, f'{top_k[0][0]} {top_k[0][1]:.3}', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # Write Video
    if save:
        out_vid.write(img)

    # cv2.imshow('img', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
if save:
    out_vid.release()
# cv2.destroyAllWindows()



# # Insert your video clip here
# video = load_gif('jumpingjack.gif', image_size=(172, 172))
# clips = tf.split(video[tf.newaxis], video.shape[0], axis=1)

# # To run on a video, pass in one frame at a time
# states = init_states
# for clip in clips:
#     # Input shape: [1, 1, 172, 172, 3] ---> 224, 224 for a2 version
#     outputs = runner(**states, image=clip)
#     logits = outputs.pop('logits')[0]
#     states = outputs

# probs = tf.nn.softmax(logits)
# top_k = get_top_k(probs)
# print()
# for label, prob in top_k:
#     print(label, prob)

