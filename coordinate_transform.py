import cv2
import time
from core.utils import *


video_path = r'inputs\input.mp4'  

input_json_path=r'outputs\tracking.json'
output_json_path=r'outputs\coordinate_transforming.json'

# Select a third frame
frame_no = 3


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mark_count
        mark_count += 1  
        
        if mark_count <= 4: 
            frame_coordinate.append([int(x / (scale_percent / 100)), int(y / (scale_percent / 100))]) 
            print('x (px), y (px): ', int(x / (scale_percent / 100)), int(y / (scale_percent / 100)))
            colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (0,0,0)] # RED, GREEN, BLUE, YELlOW
            cv2.circle(frame,(x,y),5,colors[int(mark_count-1)],-1)

            if mark_count == 1:
                print('\n\nMark the second point(GREEN) in the image ...')
                
            elif  mark_count == 2:
                print('\n\nMark the third point(BLUE) in the image ...')

            elif  mark_count == 3:
                print('\n\nMark the fourth point(YELlOW) in the image ...')
                

def point_transform(center_point, trans_M):
    homg_point = [center_point[0], center_point[1], 1]  # homogeneous coords
    transf_homg_point = M.dot(homg_point)               # transform
    transf_homg_point /= transf_homg_point[2]           # scale
    transf_point = transf_homg_point[:2].tolist()       # remove Cartesian coords

    return transf_point


if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)  # Video_name is the video being called
    cap.set(1,frame_no)  # Where frame_no is the frame you want
    ret, frame = cap.read()  # Read the frame

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Read json data    
    data_dict = data_reader(input_json_path)

    frame_coordinate = []
    gis_coordinate = []
    mark_count = 0

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_ratio = frame_width / frame_height

    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)

    # The following names the window so we can access it
    cv2.namedWindow(winname='my_drawing')

    # Connects the mouse click with the callback function
    cv2.setMouseCallback('my_drawing',draw_circle)

    print(' To transform the coordinates in the image into geographic coordinates you need to mark only four points in the image. \nThen in a special field you need to enter GIS coordinates to them')
    print('\nMark four points in the image ...')        
    print('\n\nMark the first point(RED) in the image ...')

    while True:  # Runs endlessly until we interrupt with the Esc key on the keyboard
        # Displays the image window
        cv2.imshow('my_drawing',frame)

        if (cv2.waitKey(20) &  0xFF == 27) :
            break
        if (cv2.waitKey(20) &  mark_count==5):
            
            print('\n\n\nWrite GIS coordinates to the first point, \nfor example: 48.40228307637909, 9.795953379174994')
            point_1_1, point_1_2 = input('1. point(RED): ').split(',') 
            gis_coordinate.append( [float(point_1_1), float(point_1_2)] )

            print('\nWrite GIS coordinates to the second point, \nfor example: 48.40228307637909, 9.795953379174994')
            point_2_1, point_2_2 = input('2. point(GREEN): ').split(',')  
            gis_coordinate.append( [float(point_2_1), float(point_2_2)] )

            print('\nWrite GIS coordinates to the third point, \nfor example: 48.40228307637909, 9.795953379174994')
            point_3_1, point_3_2 = input('3. point(BLUE): ').split(',') 
            gis_coordinate.append( [float(point_3_1), float(point_3_2)] )

            print('\nWrite GIS coordinates to the fourth point, \nfor example: 48.40228307637909, 9.795953379174994')
            point_4_1, point_4_2 = input('4. point(YELlOW): ').split(',') 
            gis_coordinate.append( [float(point_4_1), float(point_4_2)] )

            break

    cv2.destroyAllWindows()
        
    
    frame_coordinate = np.float32(frame_coordinate)
    gis_coordinate   = np.float32(gis_coordinate)

    M = cv2.getPerspectiveTransform(frame_coordinate, gis_coordinate)

    # Coordinate transformation
    for frame_count in data_dict.keys():
        bboxes = data_dict[frame_count]['bboxes']
        frame_coordinates = [ [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)] for bbox in bboxes ]
        
        data_dict[frame_count]['frame_coordinates'] = frame_coordinates
        
        world_coordinates = [ point_transform(frame_coordinate, M) for frame_coordinate in frame_coordinates ]
        data_dict[frame_count]['world_coordinates'] = world_coordinates
        
    # Write JSON data     
    data_writer(data_dict, output_json_path)