import pandas as pd
import numpy as np
import os
#from plant_segmenting import get_joint_pixel_values, cam1_positions, cam2_positions
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import watershed, opening, square
from datetime import datetime

def get_timestamp(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return datetime.strptime(lines[4].split()[1]+' '+lines[4].split()[2],'"%Y:%m:%d %H:%M:%S",')

def get_piece(image, section):
    """
        Removes a piece of an image
        """
    x1, x2, y1, y2 = section
    return image[x1:x2, y1:y2, :]

# locaties planten CAM 1, in x1, x2, y1, y2
locations_cam1 = [#(0, 55, 50, 140),
                  #(20, 80, 120, 170),
                  #(20, 80, 200, 250),
                  #(50, 100, 270, 320),
                  (75, 140, 70, 130),
                  (90, 150, 160, 240),
                  (125, 175, 250, 320)]

# locaties planten CAM 2, in x1, x2, y1, y2
locations_cam2 = [#(10, 50, 40, 120),
                  #(50, 80, 100, 160),
                  #(30, 55, 205, 270),
                  #(60, 90, 300, 350),
                  (90, 155, 50, 110),
                  (90, 160, 180, 250),
                  (140, 210, 290, 350)]


def segment_image(image, plantid, threshold, locations, opening_size=5):
    """
        Segments an image based on simple thresholding
        
        Inputs:
        - image : the image (as a numpy array)
        - plantid : which plant on the image (int)
        - threshold : the threshold to use
        - locations : the coordinates of the plants in the
        
        Output:
        - mask of the image
        """
    image_part = get_piece(image, locations[plantid])
    mask = median_filter(image_part.mean(2), (4,4)) > threshold
    opening(mask, selem=square(opening_size), out=mask)  # open image, remove
    # clutter
    return mask  # gaussian_filter(image_part[mask, :], sigma=convolve_sigma)

def segment_image_cam1(image, plantid, threshold=450):
    return segment_image(image, plantid, threshold, locations_cam1)

def segment_image_cam2(image, plantid, threshold=450):
    return segment_image(image, plantid, threshold, locations_cam2)

def get_smoothed_pixel_values(image, location, mask, sigma=2):
    """
        Uses a Gaussian filter on the image and returns the pixels of the segmented
        plant
        """
    image_part = get_piece(image, location) + 0.0  # get part with the plant
    image_part = gaussian_filter(image_part, sigma=sigma)  # Gaus conv.
    return image_part[mask, :].ravel()

def get_joint_pixel_values(image1, cam1_positions, image2, cam2_positions, plant_id, sigma=2):
    v1 = get_smoothed_pixel_values(image1, **cam1_positions[plant_id])
    v2 = get_smoothed_pixel_values(image2, **cam2_positions[plant_id])
    return np.hstack([v1, v2])

if __name__ == '__main__':
    #STEP -1: get positions of plants in cam1 and cam2

    # load the reference images
    image1 = np.load('./cam1/000004600.npy')
    image2 = np.load('./cam2/000004600.npy')

    # get masks and locations
    cam1_positions = [{'location' : loc,
                      'mask' : segment_image_cam1(image1, i)}
                      for i, loc in enumerate(locations_cam1)]
    cam2_positions = [{'location' : loc,
                      'mask' : segment_image_cam2(image1, i)}
                      for i, loc in enumerate(locations_cam2)]
    
    # STEP 0: get number of npy-files
    no_files_cam1 = 0
    for filename in os.listdir(os.getcwd()+'/cam1/'):
        if (filename.rfind('.npy')>0) and (filename.find('000') == 0):
            # print filename
            no_files_cam1 = no_files_cam1 + 1

    no_files_cam2 = 0
    for filename in os.listdir(os.getcwd()+'/cam2/'):
        if (filename.rfind('.npy')>0) and (filename.find('000') == 0):
            # print filename
            no_files_cam2 = no_files_cam2 + 1

    if no_files_cam1 == no_files_cam2:
        print("Ok")
        no_files = no_files_cam1
    else:
        print("ERROR: number of files cam1 and cam2 not equal")

    # STEP 1: initialise a np-array of the appropriate sizes and initialise the index with time stamp
    index = []

    ## Get mask sizes for each plant and camera from the segmentation script
    no_points_plant1 = 67941
    no_points_plant2 = 57771
    no_points_plant3 = 30006
    no_points_reference = 1

    ##
    data_plant1 = np.zeros((no_files,no_points_plant1))
    data_plant2 = np.zeros((no_files,no_points_plant2))
    data_plant3 = np.zeros((no_files,no_points_plant3))
    data_reference = np.zeros((no_files,no_points_reference))

    # STEP 2: read each image + json and save it to array
    cnt = 0
    for filename in os.listdir(os.getcwd()+'/cam1/'):
        if (filename.rfind('.npy')>0) and (filename.find('000') == 0):
            print(filename)
        
            # Get and check json data
            tmp_json_cam1 = get_timestamp('./cam1/'+os.path.splitext(filename)[0]+'.json')
            tmp_json_cam2 = get_timestamp('./cam2/'+os.path.splitext(filename)[0]+'.json')
            if tmp_json_cam1 == tmp_json_cam2:
                index.append(tmp_json_cam1)
            else:
                print('ERROR: timestamps for cam1 and cam2 are not different')

            # Get raw plant data
            tmp_cam1 = np.load('./cam1/'+filename)
            tmp_cam2 = np.load('./cam2/'+filename)

            # Extract plant data
            data_plant1[cnt,:] = get_joint_pixel_values(tmp_cam1,cam1_positions,tmp_cam2,cam2_positions,0)
            data_plant2[cnt,:] = get_joint_pixel_values(tmp_cam1,cam1_positions,tmp_cam2,cam2_positions,1)
            data_plant3[cnt,:] = get_joint_pixel_values(tmp_cam1,cam1_positions,tmp_cam2,cam2_positions,2)
            #data_reference[cnt,:] = get_joint_pixel_values(tmp_cam1,cam1_positions,tmp_cam2,cam2_positions,3)
            
            # Count
            cnt = cnt + 1

    # STEP 3: convert to pandas object
    dataframe_plant1 = pd.DataFrame(data_plant1,index=index)
    dataframe_plant2 = pd.DataFrame(data_plant2,index=index)
    dataframe_plant3 = pd.DataFrame(data_plant3,index=index)
#dataframe_reference = pd.DataFrame(data_reference,index=index)

    # STEP 4: save pandas objects to hard drive
    dataframe_plant1.to_pickle('plant1.pkl')
    dataframe_plant2.to_pickle('plant2.pkl')
    dataframe_plant3.to_pickle('plant3.pkl')
#dataframe_reference.to_pickle('reference.pkl')






