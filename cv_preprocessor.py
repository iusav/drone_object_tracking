import cv2
import time
import math

from cv_tracker.utils import *



input_video_path = r"inputs\input.mp4"
input_json_path=r"outputs\yolo_detection.json"
output_json_path=r"outputs\cvTracker_summary.json"

edge_score = 0.15


data_dict = data_reader(input_json_path)
new_data_dict = data_dict.copy()

cap = cv2.VideoCapture(input_video_path)


# Initialize frame count
count = 1

# Generate a MultiTracker object    
multi_tracker = cv2.legacy.MultiTracker_create()

false_cvTracker = {}

correct_cvTracker = {
    'prev_status'    : None,
    'current_status' : None,
    'update_status'  : None
}

while True:
    ret, frame = cap.read()
    output_frame = frame.copy()
    if not ret:
        break

    # Get Yolo data from current frame
    countYOLO, classesYOLO, scoresYOLO, bboxesYOLO, center_pointsYOLO, new_data_dict = get_YOLObbox(output_frame, new_data_dict, edge_score, count)

    start_time = time.time()
    if count == 1:
        for bbox in bboxesYOLO:
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            
            tracker = cv2.legacy.TrackerCSRT_create()
            multi_tracker.add(tracker, frame, bbox)

        if ret:
            # Update the location of the bounding boxes
            success, bboxesTRACKER = multi_tracker.update(frame)

            # Draw the bounding boxes on the video frame
            countTRACKER, bboxesTRACKER, center_pointsTRACKER, new_data_dict = get_TRACKERbbox(bboxesTRACKER, output_frame, new_data_dict, correct_cvTracker, classesYOLO, scoresYOLO, count)
    else:
        # If the object that has a yolo bbox does not have a tracker bbox that belongs to the same object, 
        # then the yolo bbox is new and create a tracker for it.

        # Check the distance from each yolo bbox to each tracker bbox
        for yolo_count, pointsYOLO in enumerate(center_pointsYOLO): 

            # Calculate the distance from the selected yolo bbox to each tracker bbox
            dist_to_trackers = np.array([ math.dist( [pointsYOLO[0], pointsYOLO[1]]  , [pointsTRACKER[0], pointsTRACKER[1]]) for pointsTRACKER in center_pointsTRACKER ])
            
            # If at least one distance is less than 15, then skip, 
            # otherwise (all distances greater than 15) add this yolo bbox to the tracker for tracking
            if any(dist_to_trackers<15):
                pass
            else:
                # A new YOLO bbox that appeared
                curent_bboxesYOLO = bboxesYOLO[yolo_count]
                bbox = (curent_bboxesYOLO[0], curent_bboxesYOLO[1], curent_bboxesYOLO[2], curent_bboxesYOLO[3] )

                tracker = cv2.legacy.TrackerCSRT_create()
                multi_tracker.add(tracker, frame, bbox)

        if ret:
            # Update the location of the bounding boxes
            success, bboxesTRACKER = multi_tracker.update(frame)

            # Draw the bounding boxes on the video frame
            countTRACKER, bboxesTRACKER, center_pointsTRACKER, new_data_dict = get_TRACKERbbox(bboxesTRACKER, output_frame, new_data_dict, correct_cvTracker, classesYOLO, scoresYOLO, count)

    print("--- %s seconds ---" % (time.time() - start_time))
    

    ##### Data visualization #####
    
    #yolo_data = True     # Activate to show predict of yolo 
    yolo_data = False

    #cv2_tracker = True   # Activate to show predict of cv tracker 
    cv2_tracker = False

    summary = True        # Activate to show summary predict of yolo and cv tracker ### BEST RESULT
    #summary = False
    
    show_data( new_data_dict, output_frame, count, yolo_data=yolo_data, cv2_tracker=cv2_tracker, summary=summary )
    

    # # Activate to visualize the path of vehicles
    # for frame_count in range(count):
    #     for bbox_count in range(len(new_data_dict["summary"][str(frame_count+1)]["bboxes"])):
    #         x, y, w, h = [int(coordinate) for coordinate in new_data_dict["summary"][str(frame_count+1)]["bboxes"][bbox_count] ]
    #         cx = int((x + x + w) / 2)
    #         cy = int((y + y + h) / 2)        
    #         pt = (cx,cy)
    #         cv2.circle(output_frame, (pt[0],pt[1]), 2, (0, 0, 255), -1)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", output_frame)

    print('frame: ', count)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    count += 1
    
cap.release()
cv2.destroyAllWindows()


# Write JSON data     
data_writer(new_data_dict, output_json_path)