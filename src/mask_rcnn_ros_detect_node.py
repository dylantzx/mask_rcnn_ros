#!/usr/bin/env python3

from FPS import *
from ImageConverter import *
from ExtraFunctions import *

import os
# 0 for GPU, -1 for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import rospy
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, display_instances, class_names
from tensorflow.python.client import device_lib
import numpy as np

import threading

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  fps = FPS()
  fps1 = FPS()
  fps2 = FPS()
  fps3 = FPS()
  fps4 = FPS()
  total_fps = FPS()
  extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")
  
  while not rospy.is_shutdown():

    # rospy.loginfo("\n\nMain Thread ID %s\n", threading.current_thread())

    if ic.cv_img is not None:

      total_fps.start()
      fps.start()
      results = model.detect([ic.cv_img], verbose=1)
      fps.stop()
      print(f"Time taken to detect: {fps.elapsed()} ms")

      fps1.start()
      # Visualize results
      r = results[0]
      masked_image = display_instances(ic.cv_img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
      fps1.stop()
      print(f"Time taken to display instances: {fps1.elapsed()} ms")

      # publish bbox values when they are available
      # bbox values are in y1,x1,y2,x2
      # have to reformat to x,y,w,h
      fps2.start()
      if len(r['rois']):
        # To publish to rostopic
        bbox = extra.format_bbox(r['rois'])
        ic.image_pub.publish(bbox)
        
        # For saving cropped images
        # extra.update()
        # extra.crop_objects(ic.cv_img, r['rois'])
      fps2.stop()
      print(f"Time taken to publish on ROS: {fps2.elapsed()} ms")

      total_fps.stop()
      cv2.putText(masked_image, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

      fps3.start()
      cv2.imshow("Masked Image", masked_image)
      fps3.stop()
      print(f"Time taken for imshow: {fps3.elapsed()} ms")

    fps4.start()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    fps4.stop()
    print(f"Time taken for waitKey: {fps4.elapsed()} ms")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)