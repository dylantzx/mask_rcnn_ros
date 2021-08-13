#!/usr/bin/env python
from __future__ import print_function

from Mask_RCNN.scripts.visualize_cv2 import model

from FPS import *
from ObjectTracker import *
from ImageConverter import *
from ExtraFunctions import *

import sys
import rospy
import cv2

from mask_rcnn_ros.msg import Bbox_values

from tensorflow.python.client import device_lib
import numpy as np

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  ot = ObjectTracker()
  fps = FPS()
  fps1 = FPS()
  detected = 0
  extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

  while not rospy.is_shutdown():

    if ic.cv_img is not None:

      fps.start()

      if detected <= 50: 
        results = model.detect([ic.cv_img], verbose=1)
        detected += 1
        print(f"\n\ndetection: {detected}\n\n")
      
      # Visualize results
      r = results[0]
    
      boxes, scores, names = [], [], []
      # publish bbox values when they are available
      # bbox values are in y1,x1,y2,x2
      # have to reformat to x,y,w,h
      if len(r['rois']):
        # To publish to rostopic
        bbox = extra.format_bbox(r['rois'])
        ic.image_pub.publish(bbox)

        # To format for object tracking
        bbox_values=[bbox.x, bbox.y, bbox.w, bbox.h]
        boxes.append(bbox_values)
        scores = r['scores'].tolist()
        names.append('target')

        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)

      fps1.start()
      ot.track_object(ic.cv_img, boxes, names, scores, r, fps)
      fps1.stop()
      print(f"Time taken to track: {fps1.elapsed()} ms")

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)