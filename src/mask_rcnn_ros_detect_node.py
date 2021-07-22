#!/usr/bin/env python
from __future__ import print_function

from FPS import *
from ImageConverter import *

import sys
import rospy
import cv2

from mask_rcnn_ros.msg import Bbox_values

from Mask_RCNN.scripts.visualize_cv2 import model, display_instances, class_names
from tensorflow.python.client import device_lib
import numpy as np

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  fps = FPS()

  while not rospy.is_shutdown():

    if ic.cv_img is not None:

      fps.start()
      results = model.detect([ic.cv_img], verbose=1)
      fps.stop()

      # Visualize results
      r = results[0]
      masked_image = display_instances(ic.cv_img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
      cv2.putText(masked_image, f"FPS: {fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

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

      cv2.imshow("Masked Image", masked_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)