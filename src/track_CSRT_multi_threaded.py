#!/usr/bin/env python3

from functions.FPS import *
from functions.ObjectTracker import *
from functions.ImageConverter import *
from functions.ExtraFunctions import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2

from Mask_RCNN.scripts.visualize_cv2 import model, class_dict, class_names
from tensorflow.python.client import device_lib
import numpy as np

import threading as t
from queue import Queue
import time

##################### Set global variables #####################
# Choose tracker type
tracker = cv2.TrackerCSRT_create()

video_path   = "/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_1.avi"
output_path   = f"/home/dylan/catkin_ws/src/mask_rcnn_ros/videos/video_1_multithread_CSRT_results_2.mp4"

extra = ExtraFunctions(cropped_path = "/home/dylan/Videos/image_train/")

# Queues are thread-safe
frame_queue = Queue()
detect_queue = Queue()
display_queue = Queue()

# count is a global variable in the shared memory between threads.
# Therefore, have to use locks to prevent race condition
count = 0 
count_lock = t.Lock()
#################################################################

def frame_grabber(vid):

    print(f"[{t.current_thread().name}] Frame grabber thread starting...")
    while True:

        ok, frame = vid.read()
        
        if not ok:
            for i in range(2):
                frame_queue.put(None)
            break

        frame_queue.put(frame)

    print(f"[{t.current_thread().name}] --- End of Frame grabber thread ---\n")

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print(f"[{t.current_thread().name}] NO INSTANCES TO DISPLAY")
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

def draw_tracked_bbox(curr_frame, bbox):
    # Draw tracked bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(curr_frame, p1, p2, (255,0,0), 2, 1)

def display_frames(out):
    print(f"[{t.current_thread().name}] Display thread starting...")

    while True:
        start = time.perf_counter()
        display = display_queue.get()

        if display is None:
            break
        
        frame = display[0]
        frame_num = display[1]

        # Display result
        cv2.imshow("Output", frame)

        # out.write(frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to display frame {frame_num}: {(stop-start)*1000:.2f}ms\n")

    print(f"[{t.current_thread().name}] --- End of Display thread ---\n")

def object_tracking(tracker_init):
    global count
    print(f"[{t.current_thread().name}] Tracking thread starting...")
    total_fps = FPS()

    while True:
        total_fps.start()
        start = time.perf_counter()
        print(f'\n[{t.current_thread().name}] tracker_init: {tracker_init}')
        
        # Run tracking if tracker has been initiated before
        if tracker_init[0] == True:
            
            # Get curr frame
            curr_frame = frame_queue.get()

            if curr_frame is None:
                display_queue.put(None)
                break 

            count_lock.acquire()
            count+=1
            count_copy = count
            count_lock.release()

            print(f"[{t.current_thread().name}] Got frame from frame_queue - {count_copy}")

            if detect_queue.qsize() == 0:
                
                # Update Tracker
                _ , bbox = tracker.update(curr_frame)

            else:
                detected_results = detect_queue.get() 
            
                # Ends if no more detection
                if detected_results is None:
                    display_queue.put(None)
                    break 

                r = detected_results[1]
                detected_num = detected_results[2]

                # Check if got valid results
                if len(r['rois']):
                    bbox = extra.format_bbox(r['rois'])

                    # To format for object tracking
                    bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                    print(f"[{t.current_thread().name}] results: {bbox}")

                    # Re-init tracker
                    tracker.init(curr_frame, bbox)
                    print(f"[{t.current_thread().name}] Tracker re-initialized!")
                    print(f"[{t.current_thread().name}] Superimposing detected frame {detected_num} on frame {count_copy}...")

                else:
                    print(f"[{t.current_thread().name}] Failed to detect instance from {detected_num}, updating track on frame {count_copy} instead...")
                    
                    # Update Tracker
                    _ , bbox = tracker.update(curr_frame)
            
            total_fps.stop()

            # Display FPS on curr frame
            cv2.putText(curr_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

            # Draw tracked bounding box
            draw_tracked_bbox(curr_frame, bbox)
                
            # Display result
            display_queue.put([curr_frame, count_copy])

        else:
            # The get() will block until get first detection
            detected_results = detect_queue.get() 
            
            # Ends if no more detection
            if detected_results is None:
                display_queue.put(None)
                break
                
            detected_frame = detected_results[0]
            r = detected_results[1]
            detected_num = detected_results[2]  

            # Check if got valid results
            if len(r['rois']):

                print(f"[{t.current_thread().name}] Got First valid detection from detect_queue")

                bbox = extra.format_bbox(r['rois'])

                # To format for object tracking
                bbox=(bbox.x, bbox.y, bbox.w, bbox.h)
                print(f"[{t.current_thread().name}] results: {bbox}")

                # Init tracker
                tracker.init(detected_frame, bbox)
                tracker_init[0] = True
                print(f"[{t.current_thread().name}] Tracker initialized!")

                total_fps.stop()

                # Display FPS on detected frame
                cv2.putText(detected_frame, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)
                
                # Display result
                detected_frame = display_instances(detected_frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                display_queue.put([detected_frame, detected_num])
        
        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to track: {(stop-start)*1000:.2f}ms\n")

    print(f"[{t.current_thread().name}] --- End of Tracking thread ---\n")

def main(args):

    print(device_lib.list_local_devices())

    ############## Initialize to read and write video ##############
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker_init = [False]
    
    print(f"\nVideo fps: {fps}")
    print(f"Video frame count: {total_frame_count}\n")

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    #################################################################
    global count

    frame_grabber_thread = t.Thread(target=frame_grabber, args=(vid,))
    tracking_thread = t.Thread(target=object_tracking, args=(tracker_init,))
    display_thread = t.Thread(target=display_frames, args=(out,))

    frame_grabber_thread.start()
    tracking_thread.start()
    # display_thread.start()

    while True:
        start = time.perf_counter()
        curr_frame = frame_queue.get()

        count_lock.acquire()
        count+=1
        count_copy = count
        count_lock.release()

        print(f"[{t.current_thread().name}] Taking from frame_queue - Frame {count_copy}")

        if curr_frame is None:
            detect_queue.put(None)
            display_queue.put(None)
            break
        
        # Mask RCNN detection
        results = model.detect([curr_frame], verbose=0)
        r = results[0]

        # To reinitialize tracking
        detect_queue.put([curr_frame, r, count_copy])

        print(f"[{t.current_thread().name}] Put frame {count_copy} in detect_queue")
        
        stop = time.perf_counter()
        print(f"[{t.current_thread().name}] Time taken to detect: {(stop-start)*1000:.2f}ms\n")   

    frame_grabber_thread.join()
    tracking_thread.join()

    display_thread.start()
    display_thread.join()

    print(f'[{t.current_thread().name}] frame_queue: {frame_queue.qsize()}\n')
    print(f'[{t.current_thread().name}] detect_queue: {detect_queue.qsize()}')
    print(f'[{t.current_thread().name}] detect_queue: {list(detect_queue.queue)}\n')
    print(f'[{t.current_thread().name}] display_queue: {display_queue.qsize()}')
    print(f'[{t.current_thread().name}] display_queue: {list(display_queue.queue)}\n')
    print(f"[{t.current_thread().name}] ---End---")
    
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    # print(f"Average total fps: {total_fps.getFPS():.2f} fps")

if __name__ == '__main__':
    main(sys.argv)