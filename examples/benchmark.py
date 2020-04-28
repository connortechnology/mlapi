# Example of how you can use the MLAPI gateway
# with live streams

# Usage:
# python3 ./stream.py <local video file>
# if you leave out local video file, it will open your webcam

import cv2
import requests
import json
import imutils
import sys
import time
import argparse

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
args, u = ap.parse_known_args()
args = vars(args)
utils.process_config(args)

import modules.face_recognition as FaceRecog
import modules.object as ObjectDetect

face_obj = FaceRecog.Face()
od_obj = ObjectDetect.Object()

if sys.argv[1]:
    CAPTURE_SRC=sys.argv[1]

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

video_source = cv2.VideoCapture(CAPTURE_SRC)
frame_cnt = 0
api_frame_count = 0

if not video_source.isOpened():
    print("Could not open video_source")
    exit()
    
last_tick = time.time()

# read the video source, frame by frame and process it
while video_source.isOpened():
    status, frame = video_source.read()
    if not status:
        print('Error reading frame')
        exit()

    # resize width down to 800px before analysis
    # don't need more
    frame = imutils.resize(frame,width=800)
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
    ret, jpeg = cv2.imencode('.jpg', frame)
    # filename is important because the API checks filename type
    #files = {'file': ('image.jpg', jpeg.tobytes())}
    #r = requests.post(url=object_url, headers=auth_header,params=PARAMS,files=files)
    #data = r.json()
    detections = od_obj.detect(frame)

    f = draw_boxes(frame,detections)

    cv2.imshow('Object detection via MLAPI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
video_source.release()
cv2.destroyAllWindows()        
