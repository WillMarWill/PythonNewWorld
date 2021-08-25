import cv2
import numpy as np 
from DarknetYolo import DetectorNetwork
from NetworkCameraReader import VideoCapture
from streaming.shm.writer import SharedMemoryFrameWriter as writer
import serial
import logging
import logging.handlers
import datetime
import time
from time import strftime
import select
import os

cap = VideoCapture('admin','bthk1234','192.168.1.201')
cap2 = VideoCapture('admin','bthk1234','192.168.1.202')


shm_w_frame = writer('frame')
network = DetectorNetwork(configPath="DarknetYolo/cfg/yolov3_tuenmun_5.cfg",
                          weightPath="DarknetYolo/weights/yolov3_tuenmun_5_best.weights",
                          metaPath="DarknetYolo/data/obj.data")
threshold = 0.1

ser = serial.Serial(port='/dev/ttyUSB0', baudrate = 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=8)
counter=0
serialString = ""
i=0
pervious_signal = ''
detected_signal = ''


# Check Folder 
if not os.path.exists('cam1_images'):
   os.mkdir('cam1_images')

if not os.path.exists('cam2_images'):
   os.mkdir('cam2_images')

if not os.path.exists('log_file'):
   os.mkdir('log_file')



while True:
    bytesToRead = ser.inWaiting()
    serialString=ser.read(bytesToRead)
    detected_signal = serialString.decode('Ascii')[18:21]

    # Detect Signal
    if len(pervious_signal) == 0 and len(detected_signal) == 0:     # No signal
       print('No Signal')
       continue


    if len(pervious_signal) != 0 and len(detected_signal) == 0:     # No New Signal; Keep Pervious Signal
#      print('Use Pervious Signal')
       detected_signal = pervious_signal
       print('Detected_signal: ' , detected_signal)




    # Signal Action
    # logging when signal changes
    if pervious_signal == '1 0' and detected_signal == '1 1':
       print('detected 1 1 signal!')
       logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       logger = logging.getLogger()
       logger.addHandler(logging.FileHandler(strftime('log_file/FODS_%d_%m_%Y.log')))
       logger.info('time: %s -- Signal changes from 1 0 to  1 1', time.asctime(time.localtime(time.time())))
       pervious_signal = detected_signal
       continue


    if pervious_signal == '1 1' and detected_signal =='1 0':
       print('detected 1 0 signal!')
       logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       logger = logging.getLogger()
       logger.addHandler(logging.FileHandler(strftime('log_file/FODS_%d_%m_%Y.log')))
       logger.info('time: %s -- Signal changes from 1 1 to  1 0', time.asctime(time.localtime(time.time())))
       pervious_signal = detected_signal
       continue


    # Update Signal
    pervious_signal = detected_signal


    # detect only whrn the signal is '0 1'
    if detected_signal != '0 1':
       print('detectd_singal: ', detected_signal)
       continue





    print('detected_signal: {}, start detection'.format(detected_signal))
    ## Detection Process
    # Camera 1
    cam1_frame = cap.read()
    print(cam1_frame)
    detections,rx,ry = network.detect(cam1_frame)


    imcaption = []

    for detection in detections:
        label = detection[0]
        confidence = float(detection[1])
        if confidence < threshold:
            continue
        pstring = str(label) + ": " + str(100*confidence) + "%"
        imcaption.append(pstring)
        #print(pstring)
        bounds = detection[2]
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)

        start_pt = (round(xCoord/rx),round(yCoord/ry))
        end_pt = (round((xCoord + xEntent)/rx),round((yCoord + yExtent)/ry))
        color = (255,0,0)
        thickness = 2
        font = cv2.FONT_HERSHEY_DUPLEX
        label_pos = (start_pt[0], start_pt[1]+10)
        confi_pos = (start_pt[0], start_pt[1]+40)
        fontColor = (255,0,0)
        #lineType = 2
        cam1_frame = cv2.rectangle(cam1_frame, start_pt, end_pt, color, thickness)
        #cv2.putText(img,label,label_pos,cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
        cv2.putText(cam1_frame, str(label), label_pos, font, 1/3, fontColor)
        cv2.putText(cam1_frame, str(confidence), confi_pos, font, 1/3, fontColor)
        #cv2.imwrite(os.path.join('./results/',ntpath.basename(image_path)),img)


#    shm_w_frame.add(cam1_frame)

    if detections:
       #print('start')
       if confidence >  0.9:
          cv2.imwrite(f'cam1_images/{i}.jpg',cam1_frame)
          i+=1
          print('CAM1 Saved Image!')
       #LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
       logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       logger = logging.getLogger()
       logger.addHandler(logging.FileHandler(strftime('log_file/detection_cam1_%d_%m_%Y.log')))
       logger.info('time: %s - class: %s - start_point: %s - end_point: %s - confidence: %s',time.asctime(time.localtime(time.time())), label, start_pt, end_pt, confidence)





    # Camera 2
    cam2_frame = cap2.read()
    cam2_detections,cam2_rx,cam2_ry = network.detect(cam2_frame)

    imcaption = []

    for detection in cam2_detections:
        label = detection[0]
        confidence = float(detection[1])
        if confidence < threshold:
            continue
        pstring = str(label) + ": " + str(100*confidence) + "%"
        imcaption.append(pstring)
        #print(pstring)
        bounds = detection[2]
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)

        start_pt = (round(xCoord/cam2_rx),round(yCoord/cam2_ry))
        end_pt = (round((xCoord + xEntent)/cam2_rx),round((yCoord + yExtent)/cam2_ry))
        color = (255,0,0)
        thickness = 2
        font = cv2.FONT_HERSHEY_DUPLEX
        label_pos = (start_pt[0], start_pt[1]+10)
        confi_pos = (start_pt[0], start_pt[1]+40)
        fontColor = (255,0,0)
        #lineType = 2
        cam2_frame = cv2.rectangle(cam2_frame, start_pt, end_pt, color, thickness)
        #cv2.putText(img,label,label_pos,cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
        cv2.putText(cam2_frame, str(label), label_pos, font, 1/3, fontColor)
        cv2.putText(cam2_frame, str(confidence), confi_pos, font, 1/3, fontColor)
        #cv2.imwrite(os.path.join('./results/',ntpath.basename(image_path)),img)


    frame = np.concatenate((cam1_frame, cam2_frame),axis=1)
    shm_w_frame.add(frame)
    #print(detections)

    if detections:
       #print('start')
       if confidence >  0.9:
          cv2.imwrite(f'cam2_images/{i}.jpg',cam2_frame)
          i+=1
          print('CAM2 Saved Image!')
       #LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
       logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
       logger = logging.getLogger()
       logger.addHandler(logging.FileHandler(strftime('log_file/detection_cam2_%d_%m_%Y.log')))
       logger.info('time: %s - class: %s - start_point: %s - end_point: %s - confidence: %s',time.asctime(time.localtime(time.time())), label, start_pt, end_pt, confidence)

 
