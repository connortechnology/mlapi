# Example of how you can use the MLAPI gateway
# with live streams

# Usage:
# python3 ./stream.py <local video file>
# if you leave out local video file, it will open your webcam


import sys
#import sys.path
#sys.path.append('/usr/local/lib/python3.5')

import cv2
import requests
import json
import imutils
import time
import argparse
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import modules.common_params as g
import modules.db as Database
import modules.utils as utils
from modules.__init__ import __version__

#--------- Change to your needs---------------
BASE_API_URL='http://localhost:5000/api/v1' 
USER='admin' 
PASSWORD='XV35me' 
FRAME_SKIP = 0

# if you want face and gender
#PARAMS = {'delete':'true', 'type':'face', 'gender':'true'}

# if you want object
PARAMS = {'delete':'true'}
# If  you want to use webcam
CAPTURE_SRC=0
# you can also point it to any media URL, like an RTSP one or a file
#CAPTURE_URL='rtsp://whatever'
#CAPTURE_URL='/home/iconnor/Downloads/PlateRecognizer/Video_N1_Street_2lanes_Fast_2min.mp4'

# If you want to use ZM
# note your URL may need /cgi-bin/zm/nph-zms - make sure you specify it correctly
# CAPTURE_SRC='https://demo.zoneminder.com/cgi-bin-zm/nph-zms?mode=jpeg&maxfps=5&buffer=1000&monitor=18&user=zmuser&pass=zmpass'
#--------- end ----------------------------

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True, help='config file with path')
ap.add_argument('-s', '--source', required=True, help='source file')

args, u = ap.parse_known_args()
args = vars(args)
utils.process_config(args)

import modules.object as ObjectDetect

od_obj = ObjectDetect.Object()

#pb_dir = '/home/iconnor/Downloads/ssd_mobilenet_v2_coco_2018_03_29'
pb_dir = '/home/iconnor/ssdlite_mobilenet_v2_coco_2018_05_09'
pb = pb_dir+'/frozen_inference_graph.pb'
pbtxt = pb_dir+'/frozen_inference_graph.pbtxt'
tfNet = cv2.dnn.readNetFromTensorflow(pb,pbtxt)

login_url = BASE_API_URL+'/login'
object_url = BASE_API_URL+'/detect/object'
access_token = None
auth_header = None

# Draws bounding box around detections
def draw_boxes(frame,data):
  color = (0,255,0) # bgr
  for item in data:
     bbox = item.get('box')
     label = item.get('type')
     gender = item.get('gender')
     if gender:
      label = label + ', '+gender
     cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
     cv2.putText(frame, label, (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

video_source = cv2.VideoCapture(args['source'])
frame_cnt = 0
api_frame_count = 0

if not video_source.isOpened():
    print("Could not open video_source")
    exit()
    
last_tick = time.time()

#if g.config['use_opencv_dnn_cuda']=='yes':
if 0:
    (maj,minor,patch) = cv2.__version__.split('.')
    min_ver = int (maj+minor)
    if min_ver < 42:
        g.logger.error ('Not setting CUDA backend for OpenCV DNN')
        g.logger.error ('You are using OpenCV version {} which does not support CUDA for DNNs. A minimum of 4.2 is required. See https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/ on how to compile and install openCV 4.2'.format(cv2.__version__))
    else:
        g.logger.debug ('Setting CUDA backend for OpenCV. If you did not set your CUDA_ARCH_BIN correctly during OpenCV compilation, you will get errors during detection related to invalid device/make_policy')
        tfNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        tfNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# read the video source, frame by frame and process it
while video_source.isOpened():
    status, frame = video_source.read()
    if not status:
        print('Error reading frame')
        exit()

    # resize width down to 800px before analysis
    # don't need more
    #frame = imutils.resize(frame,width=320)
    frame_cnt += 1
    if FRAME_SKIP and frame_cnt % FRAME_SKIP:
      continue

    frame_cnt = 0

    tick = time.time()
    if not (tick - last_tick) > 10:
        print('Analysing at {} fps'.format((api_frame_count/(tick-last_tick))))
        last_tick = tick
        api_frame_count = 0

    api_frame_count += 1
    
    # The API expects non-raw images, so lets convert to jpg
    #ret, jpeg = cv2.imencode('.jpg', frame)
    # filename is important because the API checks filename type
    #files = {'file': ('image.jpg', jpeg.tobytes())}
    #r = requests.post(url=object_url, headers=auth_header,params=PARAMS,files=files)
    #data = r.json()
    #detections = od_obj.detect(frame)

    rows, cols, channels = frame.shape
    inp = cv2.resize(frame, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    tfNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
 

    out = tfNet.forward()

    # Loop on the outputs
    for detection in out[0,0]:

        score = float(detection[2])
        if score > 0.2:

            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            #draw a red rectangle around detected objects
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)



    #cv2.imshow('Object detection via MLAPI', frame)
    cv2.imwrite(args['source']+'-output/objdetect-{}.jpg'.format(frame_cnt), frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
video_source.release()
cv2.destroyAllWindows()        
