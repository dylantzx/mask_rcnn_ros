# mask_rcnn_ros

## Prerequisites
---
- Ubuntu LTS 18.04

- ROS Melodic 

- [Setup PX4, QGC and MavROS](https://github.com/dylantzx/PX4)

- Python 3.7+

- TensorFlow-gpu 2.5.0

- Keras 2.3.0

- Numpy, skimage, scipy, Pillow, cython, h5py

- CUDA 11.3

## About
---

This is a ROS package of [Mask RCNN](https://github.com/akTwelve/Mask_RCNN) with [DeepSORT](https://github.com/nwojke/deep_sort) for object detection, instance segmentation and object tracking.

It contains ROS nodes for object detection and object detection with tracking.

The current repository is for a drone tracking another drone on PX4.

## Getting Started
---

[FPS.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/FPS.py) - Contains a simple FPS class for FPS calculation 

[ImageConverter.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/ImageConverter.py) - Contains ImageConverter class that converts images received via GazeboROS Plugin of `sensor_msgs` type to a usable type for object detection. 

[ObjectTracker.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/ObjectTracker.py) - Contains class to utilize DeepSORT for object tracking.

[mask_rcnn_ros_detect_node.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/mask_rcnn_ros_detect_node.py) - Main script that runs MaskRCNN with ROS for object detection

[mask_rcnn_ros_track_node.py](https://github.com/dylantzx/mask_rcnn_ros/blob/main/src/mask_rcnn_ros_track_node.py) - Main script that runs MaskRCNN with ROS for object detection with tracking.



