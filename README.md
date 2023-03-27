# Custom-MoViNet
Create your own custom **MoViNet model** using your custom data. You can train and inference from "a single of code"🥳.<br>
Currently only added **MoViNet Stream A1**, moving forward we will add all other types of **MoViNet models**.<br>

### Available MoViNet Models:
- [ ] MoViNet Base A0
- [ ] MoViNet Base A1
- [ ] MoViNet Base A2
- [ ] MoViNet Base A3
- [ ] MoViNet Base A4
- [ ] MoViNet Base A5
- [ ] MoViNet Base A6
- [ ] MoViNet Stream A0
- [x] MoViNet Stream A1
- [ ] MoViNet Stream A2
- [ ] MoViNet Stream A3
- [ ] MoViNet Stream A4
- [ ] MoViNet Stream A5


## Table of Contents:
- About MoViNet
- Prepare Dataset
- Train MoViNet Model
- Inference MoViNet Model

## 📖 About MoViNet
The **MoViNet** model family has been published in March 2021 by Google research. It tries to solve the recurring problem with video classification models, which is how resource hungry the models are. Figure below shows how **MoViNet** compares in resource usage to other **SOTA** video classification models such as **X3D**.

![ezgif-2-166afa34a2](https://user-images.githubusercontent.com/88816150/227913487-4ed5b612-304b-4dd1-acf7-cddfb4f195ca.jpg)

## 🎒 Prepare Dataset
Your custom dataset should be in Kinetics data format.<br>
**Kinetics Data Format**:
```
├── Dataset
│   ├── Classname1
│   │   ├── 1.mp4
│   │   ├── abc.avi
│   │   ├── ...
│   ├── Classname2
│   │   ├── example.mp4
│   │   ├── 2.avi
│   │   ├── ...
.   .
.   .
```

But to train MoViNet Model need a slightly different dataset format.But its need all videos in **AVI** format<br>
**MoViNet Training Data Format**:
```
├── Dataset
│   ├── train
│   │   ├── classname1
│   │   │   ├── 1.avi
│   │   │   ├── abc.avi
│   │   ├── classname2
│   │   │   ├── example.avi
│   │   │   ├── 2.avi
│   │   ├── ...
│   ├── test
│   │   ├── classname1
│   │   │   ├── 1.avi
│   │   │   ├── abc.avi
│   │   ├── classname2
│   │   │   ├── example.avi
│   │   │   ├── 2.avi
│   │   ├── ...
.   .
.   .
```
### 💼 Convert Dataset into MoViNet Training Data Format
But you don't worry you can convert your dataset into this format by runing single of code 🥳. If data is not in **AVI** format, it will convert **MP4** into **AVI** format 😎.

<details>
  <summary>Args</summary>
  
  `-i`, `--data_dir`: path to data dir <br>
  `-o`, `--save`: path to save dir <br>
  `-r`, `--ratio`: test ratio 0<ratio<1

</details>

**Example**
```
python3 data_split.py --data_dir data/ --save train_data_dir --ratio 0.2
```

## 🤖 Train

<details>
  <summary>Args</summary>
  
  `-i`, `--data`: path to data dir <br>
  `-b`, `--batch_size`: Training batch size <br>
  `-n`, `--num_frames`: Number of frame need to take to train the model from each video.<br>
  `-s`, `--resolution`: Video resolution to train the model.<br>
  `-e`, `--num_epochs`: number of training epochs.<br>
  `--pre_ckpt`: path to pre-trained checkpoint dir.<br>
  `--save_ckpt`: path to save trained checkpoint eg: checkpoints/ckpt-1.<br>
  `--export`: path to export model.<br>
  `-id`, `--model_id`: model type, eg: a2 <br>
  `-o`, `--save`: path to export tflite model.<br>
  `-f`, `--float`: model quantization, choices: 32 & 16

</details>

**Example:**
```
python3 train.py --data train_data_dir --batch_size 8 --num_frames 32 --resolution 172 --num_epochs 100 \
                 --pre_ckpt movinet_a1_stream/ --save_ckpt checkpoints/ckpt-1 --export my_model/custom_model1 \
                 --model_id a1 --save my_model.tflite --float 16
```

### ⚠ Error while Training!!!
When training, on the time of loading data.<br>
If we getting this ERROR:<br>
**" ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor "**

This is due to Frames missing from some of the video data.<br>
For that you need to clean your data. Don't worry from a single line of code will help you to do that 🥳.

<details>
  <summary>Args</summary>
  
  `-i`, `--data_dir`: path to data dir <br>
  `-o`, `--save`: path to save dir

</details>

**Example**
```
python3 clean_data.py --data_dir data/ --save good_data
```

#### After cleaning again run the code for training with data directory path to cleaned data.

## 📺 Inference

<details>
  <summary>Args</summary>
  
  `--tflite`: path to tflite model <br>
  `-i`, `--source`: path to video or cam-id or RTSP link <br>
  `-s`, `--resolution`: Video resolution to train the model.<br>
  `-n`, `--num_frames`: Number of frame need to take to train the model from each video.<br>  
  `-d`, `--data`: path to data/test or data/train dir <br>
  `--save`: to save inferenced video, it save as ouput.mp4

</details>

**Example**
```
python3 inference.py --tflite my_model.tflite --source 'videos/sample.mp4' --num_frames 32 \
                     --data data/test --save
```
