import os
import tensorflow as tf
import tensorflow_datasets as tfds
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model


# Download and Load UCF-101 Data
dataset_name = 'ucf101'
builder = tfds.builder(dataset_name)
config = tfds.download.DownloadConfig(verify_ssl=False)
builder.download_and_prepare(download_config=config)

num_classes = builder.info.features['label'].num_classes
num_examples = {
    name: split.num_examples
    for name, split in builder.info.splits.items()
}

print('Number of classes:', num_classes)
print('Number of examples for train:', num_examples['train'])
print('Number of examples for test:', num_examples['test'])
print(builder.info)


batch_size = 8
num_frames = 8
frame_stride = 10
resolution = 172
model_id = 'a1' #---> You can change this for a0 (light), or a2 (robust)
resolution = 172
num_epochs = 2
num_classes=101


def format_features(features):
    video = features['video']
    video = video[:, ::frame_stride]
    video = video[:, :num_frames]

    video = tf.reshape(video, [-1, video.shape[2], video.shape[3], 3])
    video = tf.image.resize(video, (resolution, resolution))
    video = tf.reshape(video, (-1, num_frames, resolution, resolution, 3))
    video = tf.cast(video, tf.float32) / 255.

    label = tf.one_hot(features['label'], num_classes)
    return (video, label)
 
train_dataset = builder.as_dataset(
    split='train',
    batch_size=batch_size,
    shuffle_files=True)
train_dataset = train_dataset.map(
    format_features,
    num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(2)

test_dataset = builder.as_dataset(
    split='test',
    batch_size=batch_size)
test_dataset = test_dataset.map(
    format_features,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True)
test_dataset = test_dataset.prefetch(2)


tf.keras.backend.clear_session()

backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='swish',
    gating_activation='sigmoid'
)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(
    backbone, num_classes=600)
model.build([1, 1, 1, 1, 3])

# Load pre-trained weights, be sure you change de model id
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a1_stream.tar.gz -O movinet_a1_stream.tar.gz -q
# !tar -xvf movinet_a1_stream.tar.gz

checkpoint_dir = f'movinet_{model_id}_stream'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes, freeze_backbone=False):
    """Builds a classifier on top of a backbone model."""
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])

    return model

model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes)

train_steps = num_examples['train'] // batch_size
total_train_steps = train_steps * num_epochs
test_steps = num_examples['test'] // batch_size


loss_obj = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1)

metrics = [
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=1, name='top_1', dtype=tf.float32),
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name='top_5', dtype=tf.float32),
]

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=total_train_steps,
)
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=metrics)


checkpoint_path = f"movinet_{model_id}_stream_checkpoint/cptk-1"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,)

results = model.fit(train_dataset,
                    validation_data=test_dataset,
                    epochs=num_epochs,
                    steps_per_epoch=train_steps,
                    validation_steps=test_steps,
                    validation_freq=1,
                    callbacks=[cp_callback],
                    verbose=1)
results.history

weights=model.get_weights()

input_shape = [1, 1, 172, 172, 3]
batch_size, num_frames, image_size, = input_shape[:3]

tf.keras.backend.clear_session()
# Create the model
input_specs = tf.keras.layers.InputSpec(shape=input_shape)
stream_backbone = movinet.Movinet(
    model_id='a1',
    causal=True,
    input_specs=input_specs,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='swish',
    gating_activation='sigmoid',
    use_external_states=True)
stream_backbone.trainable=False
stream_model = movinet_model.MovinetClassifier(
    backbone=stream_backbone,
    num_classes=num_classes,
    output_states=True)
stream_model.build([1, 1, 172, 172, 3])
stream_model.set_weights(weights)
stream_model.get_weights()[0] 
model.get_weights()[0]

saved_model_dir=f"my_model/movinet_{model_id}_stream_UCF101"
export_saved_model.export_saved_model(
    model=stream_model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=True,
    bundle_input_init_states_fn=False)

model_id = 'a1'
saved_model_dir=f"my_model/movinet_{model_id}_stream_UCF101"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open(f'movinet_{model_id}_stream.tflite', 'wb') as f:
    f.write(tflite_model)
