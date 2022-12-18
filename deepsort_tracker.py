import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import json
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from core.utils import * 
from cv_tracker.utils import *


def main(
    framework: str='tf',
    weights: str='./checkpoints/yolov4-416',
    size: int=416,
    tiny: bool=False,
    model: str='yolov4',
    input_video_path: str=None,
    output_video_path: str=None,
    output_format: str='XVID',
    iou: float=0.45,
    score: float=0.5,
    dont_show: bool=False,
    info: bool=False,
    count: bool=False,
    input_json_path: str=None,
    output_json_path: str=None,
):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # Calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # Initialize tracker
    tracker = Tracker(metric)

    

    # Begin video capture
    try:
        vid = cv2.VideoCapture(int(input_video_path))
    except:
        vid = cv2.VideoCapture(input_video_path)

    out = None

    # Get video ready to save locally if flag is set
    if output_video_path:
        # By default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    frame_num = 0

    data_dict = data_reader(input_json_path)  
    
    # Initialize output data dict
    output_data_dict = {
    }
    
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]

        start_time = time.time()

        # Run detections       
        # bboxes, scores, names, features
        bboxes = np.array(data_dict["summary"][str(frame_num)]["bboxes"])
        names = np.array(data_dict["summary"][str(frame_num)]["classes"])
        scores = np.array(data_dict["summary"][str(frame_num)]["scores"])
        
        # Encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # Initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                
        detections = [detections[i] for i in indices] 

        # Call the tracker
        tracker.predict()

        tracker.update(detections)

        json_frame_data = {
            'bboxes' : [],    # [x, y, w, y]
            'classes': [],
            'idx'    : []
        }
        
        # Update tracks
        tr_count = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # bbox coordinate
            x1, y1, x2, y2, = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
             
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
        
            if x2 > frame_size[1]: x2 = frame_size[1]
            if y2 > frame_size[0]: y2 = frame_size[0]

            w = int(x2 - x1)
            h = int(y2 - y1)

            obj_bbox = [x1,y1,w,h]

            # Append obj data to json data
            json_frame_data['bboxes'].append(obj_bbox)
            json_frame_data['classes'].append(class_name)
            json_frame_data['idx'].append(track.track_id)

            
            # Draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # If enable info flag then print details about each track
            if info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            tr_count += 1
        print('tr_count: ', tr_count)

        # Calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_data_dict[frame_num] = json_frame_data
        
        if not dont_show:
            cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Output Video", result)

        # If output flag is set, save video file
        if output_video_path:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyAllWindows()

    # Write JSON data    
    data_writer(output_data_dict, output_json_path)


if __name__ == '__main__':
    
    input_video_path = r"inputs\input.mp4"
    output_video_path = r"outputs\y_cvDeepsort_tracking.avi"
    
    input_json_path = r"outputs\cvTracker_summary.json"
    output_json_path = r"outputs\tracking.json"
    
    main(input_json_path=input_json_path, output_json_path=output_json_path, input_video_path=input_video_path, output_video_path=output_video_path)