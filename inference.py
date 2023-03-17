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


model_id = 'a1'

# Create the interpreter and signature runner
interpreter = tf.lite.Interpreter(model_path=f'movinet_{model_id}_stream.tflite')
runner = interpreter.get_signature_runner()

init_states = {
    name: tf.zeros(x['shape'], dtype=x['dtype'])
    for name, x in runner.get_input_details().items()
}
del init_states['image']


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

# Insert your video clip here
video = load_gif('jumpingjack.gif', image_size=(172, 172))
clips = tf.split(video[tf.newaxis], video.shape[0], axis=1)

# To run on a video, pass in one frame at a time
states = init_states
for clip in clips:
    # Input shape: [1, 1, 172, 172, 3] ---> 224, 224 for a2 version
    outputs = runner(**states, image=clip)
    logits = outputs.pop('logits')[0]
    states = outputs

probs = tf.nn.softmax(logits)
top_k = get_top_k(probs)
print()
for label, prob in top_k:
    print(label, prob)

