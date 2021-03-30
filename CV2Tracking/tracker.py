#!/usr/bin/env python3
# multi - object tracker to improve thermal camera
# author: abhinandan.vellanki@gmail.com

# import the necessary packages

import time
import cv2

#from imutils.video import VideoStream, FPS
#from imutils import resize
#import imutils

import numpy as np
import argparse
import sys
import os

from time import sleep
from time import perf_counter

t_type = "multi, csrt, kcf, mil, tld, medianflow"
t_roi = "(513,50,41,56),(201,344,33,48)"
t_video = "face_test.mp4"

#ap = argparse.ArgumentParser()
#ap.add_argument("-n", "--num-frames", type=int, default=100,
          #help="# of frames to loop over for FPS test")
#ap.add_argument("-d", "--display", type=int, default=-1,
	#help="Whether or not frames should be displayed")
#args = vars(ap.parse_args())

class Track():
    def __init__(self, tracker_type, param_file = 'medianflow.json'):
        # Initialize the tracker
        self.tracker_type = tracker_type
        self.param_file = param_file
        
        # Initialize the video stream
        #if t_video is None: # If no video file is provided
            #print('Using webcam...\n')
            #self.vs = VideoStream(src=0).start()
        #elif not os.path.exists(t_video):
            #raise Exception(
                #f"The specified file '{t_video}' couldn't be loaded.")
        #else:
            #self.vs = cv2.VideoCapture(t_video)

        # Initial the detector
        #if detector is not None:
            #try:
                #self.detector = cv2.dnn.readNetFromCaffe(*detector)
                #self.detector.setPreferableBackend(00000)
            #except Exception as e:  # If the model fails to load
                #print(f"The detector couldn't be initialized.")
                #raise(e)
        #else:
            #self.detector = None

        #self.func = func
        #self.initBB = None  # To store bounding box coordinates
        #self.updatedBB = None  # To refresh the bounding box
        #self.fps = None  # Initialize fps (frames per second) count
        #self.frame = None  # For storing the frame to be displayed
        #self.frame_copy = None  # For storing the unmodified frame
        #self.interval = refresh_interval  # The refresh interval
        #self.frame_count = 0  # Initialize the frame counter
        #new_frame=None
        #self.vs = cv2.VideoCapture(t_video)

        # Some less important attributes
        #self.using_webcam = True if t_video is None else False
        # To track if the update of bounding box is already underway
        #self.update_in_progress = False
        #self.width = width  # The width to resize the frame to

    #def __del__(self):
        #self.stop()

    def create(self, tracker_type):
        OPENCV_TRACKERS = {  # name to function mapper, does not include GOTURN
            "multi": cv2.MultiTracker_create,
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,  # boosting
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,  # medianflow
            "mosse": cv2.TrackerMOSSE_create,
        }
        # call constructor at runtime
        tracker = OPENCV_TRACKERS[tracker_type]()
        fs = cv2.FileStorage(self.param_file, cv2.FileStorage_READ)
        tracker.read(fs.getFirstTopLevelNode())
        tracker.save('custom.json')
 
        return tracker

    #def _get_BB(self, update=False):
        #H, W = self.frame.shape[:2] # Grab the shape of the frame
        
        #if self.detector is not None:  # If a detector is provided
            # Preprocess frame to pass through the detector and create a blob
            #blob = cv2.dnn.blobFromImage(
                #cv2.resize(self.frame_copy, (300, 300)), 1.,
                #(300, 300), (104.0, 177.0, 123.0)
                #)
            #self.detector.setInput(blob)  # Set the input image for detector

            # Workaround for a cv2 error
            #try:
                # Get the detections.
                #detections = self.detector.forward()
            #except cv2.error:
                #print('error')
                # Reset the parameters and return
                #self.update_in_progress = False
                #self.frame_count = self.interval - 1
                #return

            #if detections is not None:  # If anything is detected at all
                # The returned detections are sorted according to
                # their confidences
                # Therefore, the first element is the one we want
                #newBB = detections[0, 0, 0, :].squeeze()
                #newBB_confidence = newBB[2]  # The confidence for newBB

                # Compute (x, y) coordinates for the bounding box
                #box = newBB[3:7] * np.array([W, H, W, H])
                #(startX, startY, endX, endY) = box.astype("int")

                #w = endX - startX  # Width
                #h = endY - startY  # Height
                #newBB = [startX, startY, w, h]  # The required format

                # Update the bounding box only if it has greater
                # confidence than the self.confidence threshold
                #if newBB_confidence >= self.confidence:

                    #if update:  # If an update was requested
                        #self.updatedBB = newBB  # Store the updated box

                        # Reset update parameters
                        #self.frame_count = 0
                        #self.update_in_progress = False
                        #return

                    #else:
                        #self.initBB = newBB
                        # Initialize the tracker
                        #self.tracker.init(self.frame, tuple(self.initBB))
                        #return
                #else:
                    #if update:
                        # Reset the parameters so that this function is called
                        # again on the next iteration of the main loop.
                        #self.update_in_progress = False
                        #self.frame_count = self.interval - 1
                        #return

            #else:  # If nothing is detected
                # Reset the relevant parameters so that this function is called
                # again on the next iteration of the main loop.
                #if update:  # If an update was requested
                    #self.update_in_progress = False
                    # Set frame count to one less than the interval,
                    # so that, it triggers this function on next iteration
                    #self.frame_count = self.interval - 1
                    #return
                #else:
                    # Set self.initBB to None, so that this function
                    # will be called again on next iteration
                    #self.initBB = None
                    #return
    
    #def _run_func(self):
        #pass

    #def start(self):                       
       #while True:
           #self.frame = vs.read()  # read next frame <- draw latest ROIs
           #self.frame = self.frame[1] if not self.using_webcam else self.frame

           # To reduce the processing time
           #self.frame = resize(self.frame, width=400, height=400)
           #(H, W) = self.frame.shape[:2]
           #self.frame_copy = self.frame.copy()

           #if self.frame is None: # Marks the end of the stream
               #print("Reached end of video, stopping tracker...")
               #print(
                     #"Average time per track: ", float(sum(track_times) / len(track_times))
                    #)
               #break

           #if self.intBB is None: # The bounding box is not initialized
               #self._get_BB(update=False)
               #self.fps = FPS().start() #Start recording FPS

           #elif self.updatedBB is not None:
               #self.initBB = self.updatedBB
               #self.updatedBB = None

               # Initialize the tracker
               #self.tracker.init(self.frame, tuple(self.initBB))
               # Restart the fps
               #self.fps = FPS().start()   
        
           #else:
               # Get the updated bounding box from tracker
               #success, BB = self.tracker.update(self.frame)

               #if success:  # If succeded in tracking
                       #x, y, w, h = [int(item) for item in BB]
                       #cv2.rectangle(self.frame, (x, y), (x+w, y+h),
                                     #(0, 255, 0), 2)

               # Update the FPS counter
               #self.fps.update()
               #self.fps.stop()
       #self.stop()

    #def stop(self):
        # If we were using webcam, stop it
        #if self.using_webcam:
            #self.vs.stop()

        # Otherwise, release the file pointer to tje video provided
        #else:
            #self.vs.release()

        # Close all windows
        #cv2.destroyAllWindows()

    def track(self, old_bbs, old_frame, new_frame):
        num_trackers = len(old_bbs)

        if old_frame is None or new_frame is None:
            print("Tracker did not get two frames")
            return []

        if num_trackers == 0:
            print("No Bounding Box given")
            return []
        t1 = time.time()
        try:
            trackers = cv2.MultiTracker_create()  # intialize multi-object tracker
            for i in range(num_trackers):
                tracker = self.create(self.tracker_type)
                (x, y, w, h) = [int(v) for v in old_bbs[i][0:4]]
                trackers.add(tracker, old_frame, (x, y, w, h))
        except Exception as e:
            print("Caught: ", e)
            #sleep(3)
            return []
        t2 = time.time()
        (success, boxes) = trackers.update(new_frame)
        print(t2-t1,time.time()-t2)
        if success:
            print(boxes)
            return boxes
        else:
            print("Tracker Failed!!")
            return []

if __name__ == "__main__":

    # the following block is for testing purposes without a screen

    #import sys

    print(t_type)
    tracker_type = str(input("Enter type of tracker to use: "))
    target_video = t_video
     
    # l = input("Enter coordinates of initial bounding boxes as"
    #          " \"(topleftX1, topleftY1, width1, height1),"
    #          "(topleftX2, topleftY2, width2, height2)...\" :")
    l = t_roi

    # save default parameters
    tracker = cv2.TrackerMedianFlow_create()

    # initialise multi-tracker object
    tracker = Track(tracker_type=tracker_type)

    frames = []  # list to store video frames
    latest_boxes = []  # stores coordinates of latest bounding boxes
    
    vs = cv2.VideoCapture(target_video)
    
    W = 0  # initial frame width
    H = 0  # initial frame height
    
    width = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

    track_times = []
    using_webcam = True if target_video is None else False

    #fps = None
    #fps = FPS().start()

    t1_start = perf_counter()
    
    #try:
       #tracker = Track(
             #detector=None,
             #tracker_type=str(input("Enter type of tracker to use: ")), 
             #param_file = 'medianflow.json', 
             #refresh_interval=20,
             #t_video = "face_test.mp4", 
             #width=400
            #)

    #except FileNotFoundError:
         #sys.exit(0)
                      
    #tracker.start()

    #while fps._numFrames < args["num_frames"]:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
        #(grabbed, frame) = vs.read()
        #frame = imutils.resize(frame, width=400)
        #(h, w) = frame.shape[:2]

        #if args["display"] > 0:
                #cv2.imshow("Frame", frame)
                #key = cv2.waitKey(1) & 0xFF

        #fps.update()
    #fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    while vs.isOpened():  # while videostream is open
        ret, new_frame = vs.read()
        #self.frame = vs.read()  # read next frame <- draw latest ROIs
        #self.frame = self.frame[1] if not self.using_webcam else self.frame

        # To reduce the processing time
        #self.frame = resize(self.frame, width=400, height=400)
        #(H, W) = selfframe.shape[:2]
        #self.frame_copy = self.frame.copy()

        if new_frame is None: # Marks the end of the stream
            print("Reached end of video, stopping tracker...")
            print(
                "Average time per track: ", float(sum(track_times) / len(track_times))
            )
            break

        #if self.intBB is None: # The bounding box is not initialized
                #self._get_BB(update=False)
                #self.fps = FPS().start() #Start recording FPS

        #elif self.updatedBB is not None:
           #self.initBB = self.updatedBB
           #self.updatedBB = None
           
           # Initialize the tracker
           #self.tracker.init(self.frame, tuple(self.initBB))
           # Restart the fps
           #self.fps = FPS().start()

        #else:
            # Get the updated boudning box from tracker
            #success, BB = self.tracker.update(self.frame)

            #if success:  # If succeded in tracking
                        #x, y, w, h = [int(item) for item in BB]
                        #cv2.rectangle(self.frame, (x, y), (x+w, y+h),
                                      #(0, 255, 0), 2)

            # Update the FPS counter
            #self.fps.update()
            #self.fps.stop()

        # Make sure that an update is not already in progress
        #if not self.update_in_progress:
            #self.frame_count += 1  # Increment the frame counter

        # Request a bounding box update if interval is reached.
        #if self.frame_count == self.interval:
            #self.update_in_progress = True
            #self.frame_count = 0  # Reset the frame counter
            #t = threading.Thread(target=self._get_BB, args=(True,))
            #t.start()

        #print("Created ROI, started tracking...")
        #continue
    
    #self.stop()

        if ret:  # if successfully able to read next frame
            #new_frame = new_frame[1] if not args.get(target_video, False) else new_frame
            
            (H, W) = new_frame.shape[:2]  # to set size of saved video
            
            #self.frame_copy = new_frame.copy()

            # resize all frames to Dell Inspiron 15 screen size for accurate input
            new_frame = cv2.resize(new_frame, (600,400))
            
            #frame = vs.read()

            #frame = frame[1] if args.get(target_video, False) else frame

            #if new_frame is None:
               #print('Stream has ended, exiting...')
               #break

            #if self.intBB is None:
                #self._get_BB(update=False)
                #self.fps = FPS().start()
            
            #elif self.initBB is not None:
                #self.initBB = self.updatedBB
                                  

            #while fps._numFrames < args["num_frames"]:
                  #(grabbed, frame) = vs.read()
                  #frame = imutils.resize(frame, width=400, height=400)            
                 
                  #if args["display"] > 0:
                          #cv2.imshow("Frame", frame)
                          #key = cv2.waitKey(1) & 0xFF
                  #fps.update()
            #fps.stop()
            #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            # if len(latest_boxes) == 0:  # nothing being tracked
            if latest_boxes is None or len(latest_boxes) == 0:
                if len(l) == 0:
                    print("No bounding box coordinates entered")
                    sys.exit(0)

                boxes = []
                for tup in l.split("),("):
                    tup = tup.replace(")", "").replace("(", "")
                    boxes.append(tuple(tup.split(",")))
                boxes = tuple(boxes)

                for box in boxes:  # draw initial ROIs
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(
                          new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
               
                #fps.update()
                #fps.stop()                

                latest_boxes = boxes  # storing bb coordinates
                frames.append(new_frame)  # adding first frame to list
                print("Created ROI, started tracking...")
                continue

            old_frame = frames[-1]  # fetching previous frame
            old_boxes = tuple(latest_boxes)  # fetching old bb coordinates
            before_track = time.time()
      
            new_boxes = tracker.track(
                old_bbs=old_boxes, new_frame=new_frame, old_frame=old_frame
            )  # calling multi-tracker
            after_track = time.time()
            track_times.append(after_track - before_track)
            
            try:
                for nbox in new_boxes:  # draw updated ROIs
                   (x, y, w, h) = [int(v) for v in nbox]
                   rect = cv2.rectangle(
                       new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                   )
            except Exception as e:
                print("Caught: ", e)
                #sleep(3)

            frames.append(new_frame)  # adding new frame to list
            latest_boxes = new_boxes  # setting updated bb coordinates
            #fps.stop()

        else:
            print("!!UNABLE TO READ STREAM!!")
            sys.exit(0)
    vs.release()
    t1_stop = perf_counter()
    last_time = str(round(t1_stop - t1_start, 2))
    print(last_time, "seconds")

    # combine frames and save video
     #saved_videoname = target_video[:-4]+"_tracked_" + \
     #   tracker_type + "_" + last_time + "s.avi"

    saved_videoname = "ans_" + tracker_type + "_" + last_time + "s.mp4"

    print("Saving video as: ", saved_videoname, " ...")
    out = cv2.VideoWriter(
        saved_videoname, cv2.VideoWriter_fourcc(*"mp4v"), 20, (int(width), int(height))
    )
    for i in range(len(frames)):  # iterate through frames array, write frames to video
        out.write(frames[i])
    out.release()
    print("Video saved successfully!")

    # end cv2 processing
    cv2.destroyAllWindows
