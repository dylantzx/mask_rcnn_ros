#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('mask_rcnn_ros')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from mask_rcnn_ros.msg import Bbox_values

from Mask_RCNN.scripts.visualize_cv2 import model, display_instances, class_names
from tensorflow.python.client import device_lib
import numpy as np
import time

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("bbox_output",Bbox_values, queue_size=10)
    self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
    self.cv_img = None

  def callback(self,data):
        
    width = data.width
    height = data.height
    channels = int(len(data.data) / (width * height))

    encoding = None
    if data.encoding.lower() in ['rgb8', 'bgr8']:
        encoding = np.uint8
    elif data.encoding.lower() == 'mono8':
        encoding = np.uint8
    elif data.encoding.lower() == '32fc1':
        encoding = np.float32
        channels = 1

    # Have to use a copy as the original image is read-only which will result in an error when
    # trying to modify the image
    self.cv_img = np.ndarray(shape=(data.height, data.width, channels), dtype=encoding, buffer=data.data).copy()

    if data.encoding.lower() == 'mono8':
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2GRAY)
    else:
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2BGR)

class FPS:

  def __init__(self):
    self.prev_frame_time = 0
    self.curr_frame_time = 0
    self.sum_of_fps = 0
    self.count = 0
    self.fps =0

  def calculateFPS(self):
    self.curr_frame_time = time.time()
    self.fps = 1/ (self.curr_frame_time - self.prev_frame_time)
    self.count+=1
    self.sum_of_fps += self.fps
    self.prev_frame_time = self.curr_frame_time

  def getAvgFPS(self):
    print(f"-----------------Avg FPS: {round(self.sum_of_fps/self.count,2)}-----------------")

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = image_converter()
  fps = FPS()

  while not rospy.is_shutdown():

    if ic.cv_img is not None:
      results = model.detect([ic.cv_img], verbose=1)

      # Calculate FPS
      fps.calculateFPS()
      fps.getAvgFPS()

      # Visualize results
      r = results[0]
      masked_image = display_instances(ic.cv_img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

      # publish bbox values when they are available
      # bbox values are in y1,x1,y2,x2
      # have to reformat to x,y,w,h
      if len(r['rois']):
        bbox_str = np.array_str(r['rois'][0])
        bbox_ls = bbox_str[1:-1].strip().replace("   ", " ").replace("  ", " ").split(" ")
        bbox = Bbox_values()
        bbox.x = int(bbox_ls[1])
        bbox.y = int(bbox_ls[0])
        bbox.w = int(bbox_ls[3]) - int(bbox_ls[1])
        bbox.h = int(bbox_ls[2]) - int(bbox_ls[0])
        ic.image_pub.publish(bbox)

      cv2.imshow("Masked Image", ic.cv_img)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)