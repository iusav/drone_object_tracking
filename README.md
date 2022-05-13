# drone_object_tracking

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

## Demo of Object Tracker on Vehicles using Drones
<p align="center"><img src="data/helpers/drone_demo.gif"\></p>

## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an pre-trained YOLOv4 object detector model of VisDrone Dataset that is able to detect 11 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained visdrone_yolo_v4_original.weights file: https://drive.google.com/file/d/1EChtoTs4_v7nbTaGYgdjdW0wb-pBIixg/view?usp=sharing 

Copy and paste visdrone_yolo_v4_original.weights from your downloads folder into the 'data' folder of this repository.

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.

```bash
# Convert darknet weights to tensorflow model
python save_model.py 

# Run yolov4 deep sort object tracker on video
python object_tracker.py 
```

The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

## Input Video (Video for processing)
Put the video for processing in the folder 'inputs'. The video must have the following name and format input.mp4. If the video should have a different name or/and format, adapt the code in the file object_tracker.py.

## Output Video (Resulting Video)
After the processing is finished, the processed output.avi video is saved in the folder 'outputs'. 
You can also adapt the name, format and/or saving path in the file object_tracker.py.

## Summary file
Summary information was saved in output_summary.csv file in the folder 'outputs'.

### References  

   Huge shoutout goes to hunglc007, nwojke and aiskyeye.com for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * [VisDrone Dataset](http://aiskyeye.com/)