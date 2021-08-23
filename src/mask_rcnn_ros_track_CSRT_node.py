#!/usr/bin/env python3

from FPS import *
from ObjectTracker import *
from ImageConverter import *
from ExtraFunctions import *

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

    tracker = cv2.TrackerCSRT_create()

    total_fps = FPS()
    tracking_fps = FPS()

    frame_count = 0
    tracked = 0

    prev_p1 = (0,0)
    stationary_bbox = False

    extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

    while not rospy.is_shutdown():

        if ic.cv_img is not None:

            total_fps.start()

            # Have to copy otherwise, the callback thread may override the image to be displayed
            frame = ic.cv_img.copy()

            frame_count+=1
            print(f"\nFrame: {frame_count}")

            if not tracked or stationary_bbox:
                print("Start detection...\n")
                results = model.detect([frame], verbose=0)

                r = results[0]
                display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

                if len(r['rois']):
                    tracked =1 
                    bbox = extra.format_bbox(r['rois'])

                    # To format for object tracking
                    bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                    print(f"results: {bbox}")
                    stationary_bbox = False

                    # Initialize tracker with first frame and bounding box 
                    tracker.init(frame, bbox)

            else:
                tracking_fps.start()
                # Update tracker
                tracked , bbox = tracker.update(frame)
                tracking_fps.stop()

                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                if frame_count%90 == 0:
                    curr_p1 = p1
                    print(f"Curr: {curr_p1} Prev: {prev_p1}")
                    # Compare curr p1 with prev p1 from n frames ago
                    if prev_p1[0] - 5 <= curr_p1[0] <= prev_p1[0] + 5 \
                    and prev_p1[1] - 5 <= curr_p1[1] <= prev_p1[1] + 5:
                        stationary_bbox = True
                        print(f"Have to redetect...")

                    prev_p1 = curr_p1

                # Publish to rostopic
                bbox_ros = Bbox_values()
                bbox_ros.x, bbox_ros.y = p1
                bbox_ros.w, bbox_ros.h = int(bbox[2]), int(bbox[3])
                ic.image_pub.publish(bbox_ros)

            # Calculate Frames per second (FPS)
            total_fps.stop()

            # Display FPS on frame
            cv2.putText(frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

            # Display result
            cv2.imshow("Output", frame)
            
            print(f"Time taken to track: {tracking_fps.elapsed():.2f} ms")
            print(f"Total time taken: {total_fps.elapsed():.2f} ms\n")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)