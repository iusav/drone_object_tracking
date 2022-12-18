import json
import math
import cv2
import numpy as np


# Read data from the json file
def data_reader(path):
    with open(path) as json_file:
        json_data = json.load(json_file)
        
        return json_data

# Write data to a file
def data_writer(output_data, path):
    with open(path, "w", newline="") as json_file:
        json.dump(output_data, json_file)


def yolo_checker(obj_count, bboxes, names, scores, output_frame):

    for first_bbox_id in range(len(bboxes)):
        first_bbox = bboxes[first_bbox_id]

        for second_bbox_id in range(len(bboxes)):
            second_bbox = bboxes[second_bbox_id]

            
def yolo_cluster_finder(bboxes, edge_score, output_frame):
    
    # Find clusters, where yolo bboxes are next to each other. 
    # Cluster consisting of two values means that two yolo bboxes belong to the same object 
   
    cluster_list = []

    for bbox_id in range(len(bboxes)):
        cluster_list.append([bbox_id])

        bbox = bboxes[bbox_id]

        for compar_bbox_id in range(len(bboxes)):
            if bbox_id == compar_bbox_id:
                continue

            compar_bbox = bboxes[compar_bbox_id]

            edge_score_value = np.mean( [(bbox[2]+bbox[3])*2 , (compar_bbox[2]+compar_bbox[3])*2] )*edge_score

            bbox_X, bbox_Y, bbox_W, bbox_H = bbox[0], bbox[1], bbox[2], bbox[3]
            compar_bbox_X, compar_bbox_Y, compar_bbox_W, compar_bbox_H = compar_bbox[0], compar_bbox[1], compar_bbox[2], compar_bbox[3]

            a1 = math.dist( [bbox_X,        bbox_Y],        [compar_bbox_X,               compar_bbox_Y])
            a2 = math.dist( [bbox_X+bbox_W, bbox_Y],        [compar_bbox_X+compar_bbox_W, compar_bbox_Y])
            a3 = math.dist( [bbox_X+bbox_W, bbox_Y+bbox_H], [compar_bbox_X+compar_bbox_W, compar_bbox_Y+compar_bbox_H])
            a4 = math.dist( [bbox_X,        bbox_Y+bbox_H], [compar_bbox_X,               compar_bbox_Y+compar_bbox_H])

            if all(np.array([a1,a2,a3,a4])<edge_score_value):
                cluster_list[bbox_id].append(compar_bbox_id)

    cluster_list = [sorted(cluster) for cluster in cluster_list if len(cluster)<=2] 

    temp_list = []
    for cluster in cluster_list:
        if cluster not in temp_list: 
            temp_list.append(cluster) 
    cluster_list = temp_list 

    # Display yolo bboxes, of which there are several and belong to the same object
    for cluster in cluster_list:
        if len(cluster) > 1:
            for cluster_id in cluster:
                bbox = bboxes[cluster_id]

                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3]) 
    
    return cluster_list 


def yolo_updater(obj_count, bboxes, names, scores, cluster_list, output_frame):   
    
    # Update yolo data
    obj_count = len(cluster_list)
    
    temp_bboxes = []
    temp_names  = []
    temp_scores = []

    cluster_count = 0
    
    for cluster in cluster_list:
        if len(cluster)==1:
            data_id = cluster[0]
            
            temp_bboxes.append(bboxes[data_id].tolist())
            temp_names.append(names[data_id])
            temp_scores.append(scores[data_id])
        else:
            data_id = cluster
            
            max_id = max(data_id)
            
            temp_bboxes.append(bboxes[max_id].tolist())
            temp_names.append(names[max_id])
            temp_scores.append(scores[max_id])

            cluster_count += 1


    center_pointsYOLOsum = []
    for box in temp_bboxes:
        x, y, w, h = [int(coordinate) for coordinate in box]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_pointsYOLOsum.append((cx, cy))

    center_pointsYOLOsum = [list(c_pointYOLO) for c_pointYOLO in center_pointsYOLOsum]
    
    bboxes = np.array(temp_bboxes)
    names = np.array(temp_names)
    scores = np.array(temp_scores)   
    
    return obj_count, bboxes, names, scores, center_pointsYOLOsum


###### get_YOLObbox ######
def get_YOLObbox(output_frame, new_data_dict, edge_score, count):

    # Defind bbox YOLO data
    obj_count =      int(new_data_dict['yolov4'][str(count)]['tracked_objs'])
    bboxes    = np.array(new_data_dict['yolov4'][str(count)]['bboxes'])
    names     = np.array(new_data_dict['yolov4'][str(count)]['classes'])
    scores    = np.array(new_data_dict['yolov4'][str(count)]['scores'])

    # Write centre punkt of bbox YOLO to a new dict
    center_pointsYOLO = []
    for box in bboxes:
        x, y, w, h = [int(coordinate) for coordinate in box]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_pointsYOLO.append((cx, cy))

    center_pointsYOLO = [list(c_pointYOLO) for c_pointYOLO in center_pointsYOLO]
    new_data_dict['yolov4'][str(count)]['center_points']=center_pointsYOLO


    # Because yolo often falsely predicts multiple objects, 
    # when in reality there is only one object and make an average yolo bbox

    cluster_list = yolo_cluster_finder(bboxes, edge_score, output_frame)
    obj_count, bboxes, names, scores, center_pointsYOLO = yolo_updater(obj_count, bboxes, names, scores, cluster_list, output_frame)    
    
    # Write data of bbox YOLO to a new dict 'summary'
    new_data_dict['summary'][str(count)]={'tracked_objs':obj_count}
    # tracked objs
    new_data_dict['summary'][str(count)]['bboxes'] = bboxes.tolist() 
    new_data_dict["summary"][str(count)]["classes"] = names.tolist()
    new_data_dict["summary"][str(count)]["scores"] = scores.tolist()
    new_data_dict["summary"][str(count)]['center_points']=center_pointsYOLO

    return obj_count, names, scores, bboxes, center_pointsYOLO, new_data_dict

# Create True/False list for correct and not correct bbox
def correct_cvTracker_updater(bboxesTRACKER, correct_cvTracker, count):
    
    current_cvTracker_list =[]
    for bbox in bboxesTRACKER:
        if  any(np.array(bbox)<0):
            current_cvTracker_list.append(False)
        else:
            current_cvTracker_list.append(True) 
    
    if correct_cvTracker['update_status'] is None:
        correct_cvTracker['prev_status']    = current_cvTracker_list
        correct_cvTracker['current_status'] = current_cvTracker_list      

    else:
        correct_cvTracker['prev_status'] = correct_cvTracker['update_status']
        correct_cvTracker['current_status'] = current_cvTracker_list

    prev_cvTracker_list    = correct_cvTracker['prev_status']
    current_cvTracker_list = correct_cvTracker['current_status']
    update_cvTracker_list  =                  []

    for tracker_id in range(len(current_cvTracker_list)):
    
        if tracker_id not in range(len(prev_cvTracker_list)):
            update_cvTracker_list.append(current_cvTracker_list[tracker_id])
        else:
    
            update_cvTracker_list.append(all([prev_cvTracker_list[tracker_id], current_cvTracker_list[tracker_id]])) 
     
    correct_cvTracker['update_status']  = update_cvTracker_list

    return update_cvTracker_list


###### Draw the bounding boxes on the video frame ######
def get_TRACKERbbox(bboxesTRACKER, output_frame, new_data_dict, correct_cvTracker, classesYOLO, scoresYOLO, count):

    bboxesTRACKER = bboxesTRACKER.tolist()

    center_pointsTRACKER = []
    for box in bboxesTRACKER:
        x, y, w, h = [int(coordinate) for coordinate in box]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_pointsTRACKER.append([cx, cy])

    tracker_obj_count = len(bboxesTRACKER)

    ##### Update "cv2_tracker" block #####
    # Write data of bbox Tracker to y new dict 'cv2_tracker'
    new_data_dict["cv2_tracker"][str(count)]                  = {}
    new_data_dict["cv2_tracker"][str(count)]['tracked_objs']  = tracker_obj_count
    new_data_dict["cv2_tracker"][str(count)]['bboxes']        = bboxesTRACKER           
    new_data_dict["cv2_tracker"][str(count)]['center_points'] = center_pointsTRACKER          

    ##### Update "summary" block #####
    # Add to summary only those tracker bboxes that are not yet on the list 
    center_pointsYOLOsum = new_data_dict["summary"][str(count)]['center_points']
    
    # List of correct bboxes
    correct_cvTracker_list = correct_cvTracker_updater(bboxesTRACKER, correct_cvTracker, count)

    for tr_id, tr_status in enumerate(correct_cvTracker_list):
        if tr_status == False:
            pass
            
    
    # Check if there is a yolo bbox and a tracker bbox that belong to the same object at the same time,
    # If so, only the bbox from yolo is taken into account
    
    # Check the distance from each tracker bbox to each yolo bbox
    for tracker_count, pointsTRACKER in enumerate(center_pointsTRACKER):
        bboxTRACKER = bboxesTRACKER[tracker_count]
        
        # Calculate the distance from the selected tracker bbox to each yolo bbox
        dist_to_yolo = np.array([ math.dist( [pointsYOLO[0], pointsYOLO[1]]  , [pointsTRACKER[0], pointsTRACKER[1]]) for pointsYOLO in center_pointsYOLOsum ])
    
        # If the distance from the tracker bbox to all yolo bboxes is greater than or equal to 15, 
        # then calculate the Mittelpunkt and add this tracker bbox to the summary, otherwise skip
        
        if any(dist_to_yolo<15) or not(correct_cvTracker_list[tracker_count]) or not(any(bboxTRACKER)): 
            continue   

        else:            
            prev_pointsSUM = new_data_dict['summary'][str(count-1)]['center_points']  

            # Calculate the distance from the selected tracker bbox to each bbox on the previous frame from Summary
            dist_to_prev_sum = np.array([ math.dist( [prev_pointSUM[0], prev_pointSUM[1]]  , [pointsTRACKER[0], pointsTRACKER[1]]) for prev_pointSUM in prev_pointsSUM ])

            min_val_id = np.argmin(dist_to_prev_sum)

            p1 = (pointsTRACKER[0], pointsTRACKER[1])
            p2 = (new_data_dict['summary'][str(count-1)]['center_points'][min_val_id][0], new_data_dict['summary'][str(count-1)]['center_points'][min_val_id][1])
            
            cv2.line(output_frame, p1, p2, (255, 0, 0), thickness=2)

            cx = int((bboxTRACKER[0]*2 + bboxTRACKER[2]) / 2)
            cy = int((bboxTRACKER[1]*2 + bboxTRACKER[3]) / 2)
            pt = [cx,cy]
            
            new_data_dict['summary'][str(count)]['center_points'].append(pt)
            new_data_dict["summary"][str(count)]["bboxes"].append(bboxTRACKER)

            curent_class = new_data_dict['summary'][str(count-1)]['classes'][min_val_id]
            new_data_dict["summary"][str(count)]["classes"].append(curent_class)

            curent_score = new_data_dict['summary'][str(count-1)]['scores'][min_val_id]
            new_data_dict["summary"][str(count)]["scores"].append(curent_score)

    
    summary_obj_count = len(new_data_dict["summary"][str(count)]["bboxes"])
    new_data_dict["summary"][str(count)]['tracked_objs']  = summary_obj_count
     
    return tracker_obj_count, bboxesTRACKER, center_pointsTRACKER, new_data_dict



##### Data visualization #####
def show_data(new_data_dict, output_frame, count, yolo_data=True,cv2_tracker=True,summary=True):

    def show_bboxes(data_status, rectangle_color, circle_color, thickness):
        
        bboxes = new_data_dict[data_status][str(count)]['bboxes']
        for box in bboxes:
            x, y, w, h = [int(coordinate) for coordinate in box]
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), rectangle_color, thickness)
            
            pt = (cx,cy)   

    if yolo_data   == True:
        data_status     = 'yolov4'
        thickness       = 2
        rectangle_color = (0, 255, 255)
        circle_color    = (0, 255, 255)

        show_bboxes(data_status, rectangle_color, circle_color, thickness)
        
    if cv2_tracker == True:
        data_status     = 'cv2_tracker'
        thickness       = 1
        rectangle_color = (0, 255, 0)
        circle_color    = (0, 255, 0)

        show_bboxes(data_status, rectangle_color, circle_color, thickness)
    
    if summary     == True:
        data_status     = 'summary'
        thickness       = 2
        rectangle_color = (255, 0, 0)
        circle_color    = (255, 0, 0)

        show_bboxes(data_status, rectangle_color, circle_color, thickness)