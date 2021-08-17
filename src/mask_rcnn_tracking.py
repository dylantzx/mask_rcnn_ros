#!/usr/bin/env python3

from FPS import *
from ObjectTracker import *
from ImageConverter import *
from ExtraFunctions import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, class_dict, class_names
from tensorflow.python.client import device_lib
import numpy as np

video_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_1.avi"
output_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_7_results.mp4"


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
    ot = ObjectTracker()

    duration = FPS()
    total_fps = FPS()
    detection_fps = FPS()
    tracking_fps = FPS()

    extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")
    avg_detection_ls = []
    avg_tracking_ls = []
    avg_total_fps_ls = []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = -2 # Start from -3 because object tracker needs minimum of 3 frames

    print(f"\nVideo fps: {fps}")
    print(f"Video frame count: {total_frame_count}\n")

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    duration.start()

    while True:

        detection_fps.start()
        total_fps.start()
        _, frame = vid.read()
        
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        frame_count+=1
        print(f"\nFrame: {frame_count}")

        if frame_count <= 1 or frame_count%10==0:
            print("Detecting...\n")
            results = model.detect([original_frame], verbose=0)
            detection_fps.stop()

            r = results[0]
            display_instances(original_frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

            boxes, scores, names = [], [], []

            if len(r['rois']):
                bbox = extra.format_bbox(r['rois'])

                # To format for object tracking
                bbox_values=[bbox.x, bbox.y, bbox.w, bbox.h]
                boxes.append(bbox_values)
                scores = r['scores'].tolist()
                names.append('target')

                boxes = np.array(boxes) 
                names = np.array(names)
                scores = np.array(scores)

            tracking_fps.start()
            ot.track_detected_object(original_frame, boxes, names, scores, r)
        
        else:
            print("Not detecting...\n")
            detection_fps.stop()

            # Using Kalman filter to predict next bbox values
            tracking_fps.start()
            ot.tracker.predict()

        tracked_bboxes = ot.get_tracked_bboxes()
        tracking_fps.stop()
        
        ot.show_tracked_object(original_frame, tracked_bboxes)        
        total_fps.stop()

        cv2.putText(original_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Output", original_frame)

        print(f"Time taken for detection: {detection_fps.elapsed():.2f} ms")
        print(f"Time taken to track: {tracking_fps.elapsed():.2f} ms\n")
        avg_detection_ls.append(round(detection_fps.elapsed(),2))
        avg_tracking_ls.append(round(tracking_fps.elapsed(),2))
        avg_total_fps_ls.append(round(total_fps.getFPS(),2))

        out.write(original_frame)
                
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break   
    
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    duration.stop()
    print(f"Total time taken for video: {duration.elapsed()/1000:.2f}s")
    print(f"Average time taken for detection: {sum(avg_detection_ls[-20:])/20:.2f} ms")
    print(f"Average time taken for tracking: {sum(avg_tracking_ls[-20:])/20:.2f} ms")
    print(f"Average total fps: {sum(avg_total_fps_ls[-20:])/20:.2f} fps")

if __name__ == '__main__':
    main(sys.argv)