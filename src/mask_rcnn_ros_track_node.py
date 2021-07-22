#!/usr/bin/env python
from __future__ import print_function

from Mask_RCNN.scripts.visualize_cv2 import model

from FPS import *
from ObjectTracker import *
from ImageConverter import *

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