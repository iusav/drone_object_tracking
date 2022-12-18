from core.utils import *
from math import sin, cos, sqrt, atan2, radians
import cv2
import matplotlib.pyplot as plt

input_video_path = r'inputs\input.mp4' 
input_json_path = r'outputs\coordinate_transforming.json'
velocity_dir = 'outputs/velocity_statistic/'  

# Maximum speed limit between noisy and not noisy value
max_velocity=150 # km/h

dpi_value = 150 # dpi


# Transform two points from world ccordinates to distance in km
def coordinate2dist(start_pt, end_pt):
    R = 6373.0
    lat1 = radians(start_pt[0])
    lon1 = radians(start_pt[1])
    lat2 = radians(end_pt[0])
    lon2 = radians(end_pt[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # km

    return distance


def velocity_creater(obj_data, velocity_dir, max_velocity, dpi_value):
    rows_number = 1 
    columns_number = 2
    chart_size = 15
    n_bins = 40
    
    for count, objID in enumerate(obj_data):
        fig, axs = plt.subplots(rows_number, columns_number, figsize=(chart_size*columns_number,chart_size*rows_number))

        obj_velocity = np.array(obj_data[objID]['velocity'])
       
        not_noisy_velocityValue = obj_velocity[obj_velocity<max_velocity]
        not_noisy_velocityTime = np.array(obj_data[objID]['frames'][:-1])[obj_velocity<max_velocity]

        noisy_velocityValue = obj_velocity[obj_velocity>=max_velocity]
        noisy_velocityTime = np.array(obj_data[objID]['frames'][:-1])[obj_velocity>=max_velocity]
        
        # Create histogram
        axs[0].hist([not_noisy_velocityValue, noisy_velocityValue]  , bins=n_bins, color=['blue','red'])
        axs[0].set_title(str(objID)+'. ID object. Velocity histogram')
        axs[0].legend(['Not noisy velocity', 'Noisy velocity'])
        axs[0].set_xlabel('Velocity (km/h)')
        axs[0].set_ylabel('Intensity (number)')

        # Create chart
        axs[1].scatter(not_noisy_velocityTime, not_noisy_velocityValue, color='blue')  # Not Noisy
        axs[1].scatter(noisy_velocityTime, noisy_velocityValue, color='red')  # Noisy
        axs[1].set_title(str(objID)+'. ID object. Velocity chart')
        axs[1].legend(['Not noisy velocity', 'Noisy velocity'])
        axs[1].set_xlabel('Frame number')
        axs[1].set_ylabel('Velocity (km/h)')

        # Save
        plt.savefig(velocity_dir + str(objID)+'. ID. Velocity_statistic.jpg', dpi = dpi_value)

        plt.cla()
        plt.close(fig)

        if (count % 10) == 0:
            print(str(count)+' is Done. Total is '+str(len(obj_data)))


if __name__ == '__main__':

    idx_set = set()
    obj_data = {}

    data_dict = data_reader(input_json_path) 


    # Find time per Frame
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS) # frames pro second

    object_spf     = 1/(fps-1)  # seconds per frame for object
    frame_object_t = object_spf / 3600 # hours per frame for object


    # Create a dict from the object id, the coordinates belonging to it, and an empty velocity list 
    for count, frame_count in enumerate(data_dict.keys()):
        for current_objID in data_dict[frame_count]['idx']:
            
            objID_id = data_dict[frame_count]['idx'].index(current_objID)
            
            # Make a dict. The keys are object id's, and the values to them: 'world_coordinates', 'velocity', 'distance'.
            # 'world_coordinates' - fill in from data_dict
            # 'velocity', 'distance' - leave blank. Later calculate
            if current_objID not in obj_data:
                # distance - distance between the nearest coordinates (km)
                obj_data[current_objID] = {'frames':[], 'world_coordinates':[],'velocity':None, 'distance':None}  
            obj_data[current_objID]['world_coordinates'].append(data_dict[frame_count]['world_coordinates'][objID_id])
            obj_data[current_objID]['frames'].append(int(frame_count))


    for count, id_count in enumerate(obj_data):
        
        # Initialize the world coordinate of a certain object on different frames
        world_objCoordinates = obj_data[id_count]['world_coordinates']
        # convert two or more coordinates into the distance between them on each frame 
        # of a certain object if the list contains more than one coordinate
        if len(world_objCoordinates)>=2:
            # Convert world koordinate into distances
            dist_list = [ coordinate2dist(world_objCoordinates[i],world_objCoordinates[i+1]) for  i in range(len(world_objCoordinates)-1) ]
            # Add distances and speeds in dict to each object
            obj_data[id_count]['distance'] = dist_list # The distance between the nearest coordinates (km)
            obj_data[id_count]['velocity'] = [ current_dist / frame_object_t for current_dist  in dist_list ] # Object speed (km/h)


    velocity_creater(obj_data, velocity_dir, max_velocity, dpi_value)