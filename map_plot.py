import pandas as pd
import plotly.express as px
from core.utils import *


# https://plotly.com/python/reference/scattergeo/
# https://plotly.com/python/mapbox-layers/


input_json_path=r'outputs\coordinate_transforming.json'


if __name__ == '__main__':
    class_list = []
    frame_list = []
    id_list    = []
    lat_list   = []
    lon_list   = []

    # Read the data from coordinate transformation
    data_dict = data_reader(input_json_path) 

    frame_data = data_dict.keys()

    for count, frame_count in enumerate(frame_data):
        
        obj_class = data_dict[frame_count]['classes']
        obj_id    =   [ 'id_'+str(current_id) for current_id in data_dict[frame_count]['idx'] ]
        obj_lat   =   [ coordinate[0] for coordinate in data_dict[frame_count]['world_coordinates'] ]
        obj_lon   =   [ coordinate[1] for coordinate in data_dict[frame_count]['world_coordinates'] ]
        obj_frame =   [ int(frame_count) for current_id in range(len(obj_id)) ]
        
        class_list.extend(obj_class)
        id_list.extend(obj_id)
        lat_list.extend(obj_lat)
        lon_list.extend(obj_lon)
        frame_list.extend( obj_frame )

    customdata = [ custom_id  for custom_id in range(len(class_list)) ]

    obj_data = pd.DataFrame({
        'lat'       : lat_list,
        'lon'       : lon_list,
        'customdata': customdata,
        'id'        : id_list
    })

    fig = px.scatter_mapbox(obj_data, lat='lat', lon='lon', custom_data=['customdata'],
                            color='id', zoom=3 ) 

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    fig.show()