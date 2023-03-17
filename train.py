import os
import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model
import pathlib
from utils import FrameGenerator


# Load Data
path_dir = pathlib.Path('dataset')
subset_paths = {}
for split_name in os.listdir('dataset'):
    split_dir = path_dir / split_name
    subset_paths[split_name] = split_dir
print(subset_paths)


batch_size = 8
num_frames = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)

for frames, labels in train_ds.take(10):
    print(labels)

print(f"Shape: {frames.shape}")
print(f"Label: {labels.shape}")



batch_size = 8
num_frames = 8
frame_stride = 10
resolution = 172
model_id = 'a1' #---> You can change this for a0 (light), or a2 (robust)
resolution = 172
num_epochs = 2
num_classes=10


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

num_epochs = 2

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

checkpoint_path = f"movinet_{model_id}_stream_checkpoint2/cptk-1"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,)
results = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    callbacks=[cp_callback],
                    verbose=1)

print(results.history)

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

saved_model_dir=f"my_model2/movinet_{model_id}_stream_UCF101"
export_saved_model.export_saved_model(
    model=stream_model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=True,
    bundle_input_init_states_fn=False)

model_id = 'a1'
saved_model_dir=f"my_model2/movinet_{model_id}_stream_UCF101"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open(f'movinet_{model_id}_stream2.tflite', 'wb') as f:
    f.write(tflite_model)
