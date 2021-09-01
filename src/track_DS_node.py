#!/usr/bin/env python3

from functions.FPS import *
from functions.DSObjectTracker import *
from functions.ImageConverter import *
from functions.ExtraFunctions import *

import sys
import rospy
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, class_dict, class_names
from tensorflow.python.client import device_lib
import numpy as np

def display_instances(image, boxes, masks, ids, names, scores):
  """
      take the image and results and apply the mask, box, and Label
  """
  n_instances = boxes.shape[0]

  if not n_instances:
      print('NO INSTANCES TO DISPLAY')
  else:
      assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

  for i in range(n_instances):
      if not np.any(boxes[i]):
          continue

      y1, x1, y2, x2 = boxes[i]
      label = names[ids[i]]
      color = class_dict[label]
      score = scores[i] if scores is not None else None
      caption = '{} {:.2f}'.format(label, score) if score else label
      mask = masks[:, :, i]

      image = apply_mask(image, mask, color)
      
      # Make detection bbox slightly bigger so that we can compare with tracked bbox
      image = cv2.rectangle(image, (x1-5, y1-5), (x2+5, y2+5), color, 2)
      image = cv2.putText(
          image, caption, (x1-10, y1-25), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
      )

  return image

def main(args):
  
  print(device_lib.list_local_devices())
  rospy.init_node('drone_detector')
  ic = ImageConverter()
  ot = DSObjectTracker()

  total_fps = FPS()
  detection_fps = FPS()
  tracking_fps = FPS()

  frame_count = 0
  detection_frame = 0

  extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

  while not rospy.is_shutdown():

    if ic.cv_img is not None:

      total_fps.start()
      detection_fps.start()

      # Have to copy otherwise, the callback thread may override the image to be displayed
      current_frame = ic.cv_img.copy()

      frame_count+=1
      print(f"\nFrame: {frame_count}")

      tracked_bboxes = ot.get_tracked_bboxes()
      print(f"\nTracked bboxes: {tracked_bboxes}\n")

      # When there is NO matched tracked bboxes, run detection
      ## If object detected: Updates bboxes
      ## If nothing detected: continue to loop since cannot track yet
      if len(tracked_bboxes) == 0:
          print("No matched tracked_bboxes... Start detection...\n")
          results = model.detect([current_frame], verbose=0)
          detection_fps.stop()

          r = results[0]
          display_instances(current_frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

          boxes, scores, names = [], [], []

          if len(r['rois']):
              detection_frame = 0
              bbox = extra.format_bbox(r['rois'])
              print(f"results: {[bbox.x, bbox.y, bbox.w + bbox.x, bbox.h + bbox.y]} {r['class_ids']}")

              # To format for object tracking
              bbox_values=[bbox.x, bbox.y, bbox.w, bbox.h]
              boxes.append(bbox_values)
              scores = r['scores'].tolist()
              names.append('target')

              boxes = np.array(boxes) 
              names = np.array(names)
              scores = np.array(scores)

              tracking_fps.start()
              ot.track_detected_object(current_frame, boxes, names, scores)
              tracked_bboxes = ot.get_tracked_bboxes()
              tracking_fps.stop()
      
      # When there is a matched tracked bboxes
      ## Check if it is the nth frame (eg. n = 10)
      ### If True: Run detection
      #### If object detected: Updates bboxes
      #### If nothing detected: run track prediction
      ### If False: Run Track prediction only
      else:
          detection_frame += 1
          print("Matched tracked_bboxes found. Check if detection is needed...\n")

          if detection_frame >= 10:
              print(f"Starting detection at {detection_frame}th frame...\n")

              results = model.detect([current_frame], verbose=0)
              detection_fps.stop()

              r = results[0]
              display_instances(current_frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

              boxes, scores, names = [], [], []

              if len(r['rois']):
                  detection_frame = 0
                  bbox = extra.format_bbox(r['rois'])
                  print(f"Detected results: {[bbox.x, bbox.y, bbox.w + bbox.x, bbox.h + bbox.y]}")

                  # To format for object tracking
                  bbox_values=[bbox.x, bbox.y, bbox.w, bbox.h]
                  boxes.append(bbox_values)
                  scores = r['scores'].tolist()
                  names.append('target')

                  boxes = np.array(boxes) 
                  names = np.array(names)
                  scores = np.array(scores)

                  tracking_fps.start()
                  ot.track_detected_object(current_frame, boxes, names, scores)
                  tracked_bboxes = ot.get_tracked_bboxes()
                  tracking_fps.stop()
              else:
                  print("Failed to detect...\n")

                  # Using Kalman filter to predict next bbox values
                  tracking_fps.start()
                  ot.tracker.predict()
                  tracked_bboxes = ot.get_tracked_bboxes()
                  tracking_fps.stop()

          else:
              print("Not Detecting...\n")
              detection_fps.stop()

              # Using Kalman filter to predict next bbox values
              tracking_fps.start()
              ot.tracker.predict()
              tracked_bboxes = ot.get_tracked_bboxes()
              tracking_fps.stop()

      print(f"Show tracked object: {tracked_bboxes}\n")    
      ot.show_tracked_object(current_frame, tracked_bboxes)
      total_fps.stop()
      cv2.putText(current_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
      cv2.imshow("Output", current_frame)

      print(f"Time taken for detection: {detection_fps.elapsed():.2f} ms")
      print(f"Time taken to track: {tracking_fps.elapsed():.2f} ms")

    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)