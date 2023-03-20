import os
import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model
import pathlib
from utils import FrameGenerator
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to data dir")
ap.add_argument("-b", "--batch_size", type=int, default=8,
                help="batch_size")
ap.add_argument("-n", "--num_frames", type=int, default=8,
                help="num_frames")
ap.add_argument("-s", "--resolution", type=int, default=172,
                help="Video resolution")
ap.add_argument("-e", "--num_epochs", type=int, default=5,
                help="number of training epochs")
ap.add_argument("--pre_ckpt", type=str, required=True,
                help="path to pre-trained checkpoint dir")
ap.add_argument("--save_ckpt", type=str, required=True,
                help="path to save trained checkpoint eg: checkpoints/ckpt-1")
ap.add_argument("--export", type=str, required=True,
                help="path to export model")
ap.add_argument("-id", "--model_id", type=str, default='a1',
                help="model type, eg: a2")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to export tflite model")
ap.add_argument("-f", "--float", type=int, default=32,
                choices=[32, 16],
                help="path to export tflite model")
args = vars(ap.parse_args())


# Load Data
path_dir = pathlib.Path(args["data"])
subset_paths = {}
for split_name in os.listdir(args["data"]):
    split_dir = path_dir / split_name
    subset_paths[split_name] = split_dir
print('Data Dict:', subset_paths)

batch_size = args['batch_size']
num_frames = args['num_frames']

resolution = args['resolution']
# model_id = 'a1' #---> You can change this for a0 (light), or a2 (robust)
model_id = args['model_id']
num_epochs = args['num_epochs']
num_classes = len(os.listdir(os.path.join(args["data"], 'test')))

# checkpoint_dir = f'movinet_{model_id}_stream'
pre_ckpt_dir = args['pre_ckpt']
# checkpoint_path = f"movinet_{model_id}_stream_checkpoint1/ckpt-1"
save_ckpt_dir = args['save_ckpt']
# saved_model_dir=f"my_model/movinet_{model_id}_stream_violance"
saved_model_dir = args['export']
# path_save_tflite = 'model.tflite'
path_save_tflite = args['save']

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

checkpoint_path = tf.train.latest_checkpoint(pre_ckpt_dir)
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

# Build Model
model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

# Callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_ckpt_dir,
                                                 save_weights_only=True,
                                                 verbose=1,)

print('Number of Classes: ', num_classes)
print('Total Number of Epochs: ', num_epochs)
print('Batch Size: ', batch_size)

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

# Export Model
export_saved_model.export_saved_model(
    model=stream_model,
    input_shape=input_shape,
    export_path=saved_model_dir,
    causal=True,
    bundle_input_init_states_fn=False)
print(f'[INFO] Exported model: {saved_model_dir}')

# To TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
if args['float'] == 16:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(path_save_tflite, 'wb') as f:
    f.write(tflite_model)
print(f'[INFO] Saved TFLite model to : {path_save_tflite}')
