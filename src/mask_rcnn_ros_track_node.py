#!/usr/bin/env python
from __future__ import print_function

from Mask_RCNN.scripts.visualize_cv2 import model

from FPS import *
from ObjectTracker import *

import sys
import rospy
import cv2

from sensor_msgs.msg import Image
from mask_rcnn_ros.msg import Bbox_values

from tensorflow.python.client import device_lib
import numpy as np

class ImageConverter:

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

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  ot = ObjectTracker()
  fps = FPS()

  while not rospy.is_shutdown():

    if ic.cv_img is not None:

      fps.start()
      results = model.detect([ic.cv_img], verbose=1)
      
      # Visualize results
      r = results[0]
    
      boxes, scores, names = [], [], []
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

        bbox_values=[bbox.x, bbox.y, bbox.w, bbox.h]
        boxes.append(bbox_values)
        scores = r['scores'].tolist()
        names.append('target')

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)

      ot.track_object(ic.cv_img, boxes, names, scores, r, fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)