import os
import csv
import cv2
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

#from PIL import Image, ImageDraw  # for bounding boxes
from pathlib import Path
from tqdm import tqdm


def load_data(path):
    """"Load Data and Construct a Dataframe holding the bounding box coordinates"""
    # check if the CSV file exists
    if not os.path.exists(path):
        print(f"Error: CSV file not found at {path}")
        return None
    # loads data in dataframe called df (needed later to extract the bounding boxes for all images)
    df = pd.read_csv(path) 
    return df

def debris_info(df, image_id, image_path):
    """In order to generate debris Trajectorie, firstly acquire the debris' bounding box coordinates
    and generate randomized direction, speed and depth"""    
    
    # select row with matching image id
    row = df[df["ImageID"] == image_id] # inner [df[] == id] creates boolean mask such that only the row with macthing id is true, then df[df[..]] selects rows    
    
    # check if bounding boxes columns exits, if yes, extract
    bbox_str = row.iloc[0]["bboxes"] # iloc for indexing, [0] selects first row, [bboxes] select bboex column -> we get a bounding box string!
    try:
        objects = []
        # load image
        img = cv2.imread(str(image_path))
        
        # convert bbox string to list
        bboxes = ast.literal_eval(bbox_str) if isinstance(bbox_str, str) else bbox_str # ast.literal_eval to convert string of list to list object

        # loop over each bounding box
        for bbox in bboxes: 
            
            # skip invalid ones (true if they dont have xmin, xmax, ymin, ymax entries)
            if len(bbox) != 4:            
                print(f"Skipping invalid bbox format: {bbox}")
                continue
            
            # extract coord
            x_min, x_max, y_min, y_max = bbox
            
            # generate random directions, speed, depth
            direction = np.random.uniform(0, 2 * np.pi)  # Random direction in radians
            #velocity = np.abs(np.random.normal(mean_speed, std_dev_speed))  # Speed magnitude
            velocity = np.random.uniform(40, 100)
            dx = velocity * np.cos(direction)
            dy = velocity * np.sin(direction)
            depth = np.random.uniform(0.001, 1)
            
            # crop object image using the coordinates 
            object_img = img[y_min:y_max, x_min:x_max].copy()
            # store all relevant info 
            objects.append({
                'image': object_img,
                'original_pos': (x_min, x_max, y_min, y_max),
                'direction': (dx, dy),
                'speed': velocity,
                'depth': depth})
        return objects
    
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing bboxes for image {image_id}: {e}")
        return []


def draw_with_bboxes(image, objects):
    for i, obj in enumerate(objects):
        x_min, x_max, y_min, y_max = obj['original_pos'] 
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max) , color=(0, 0, 255), thickness = 1)
        cv2.putText(image, f"Debris {i+1}", (x_min, max(y_min - 15, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image


def get_background_patch(image, bbox):
    """Extracts a background patch from an area not covered by bounding boxes."""
    h, w, _ = image.shape
    x_min, x_max, y_min, y_max = bbox 
    patch_height = y_max - y_min
    patch_width = x_max - x_min

    for _ in range(100):  # Try multiple times to find a background patch
        rand_x = np.random.randint(0, w - patch_width)
        rand_y = np.random.randint(0, h - patch_height)
        
        if not (rand_x < x_max and rand_x + patch_width > x_min and rand_y < y_max and rand_y + patch_height > y_min):
            return image[rand_y:rand_y + patch_height, rand_x:rand_x + patch_width]
    
    # Fallback in case no patch is found
    print("Did not find a patch, use black as background")
    return np.zeros(patch_size, dtype=np.uint8)  

    
def save_bboxes_to_csv(image_id, csv_rows, output_dir, frame):
    """Create a CSV file for each Image which collects the bounding box coordinates for each frame and for each debris object"""
    # create csv file
    csv_file = os.path.join(output_dir, f"{image_id}_bboxes.csv")
    
    # append entries for each object in each frame
    with open(csv_file, mode='a', newline='') as file: # mode a appends new data
        writer = csv.writer(file)
        
        # if file doesnt exist, create
        #if not os.path.exists(csv_file):
        #writer.writerow(["Frame", "Debris_Index", "speed", "x_min", "x_max", "y_min", "y_max"])
    
        for row in csv_rows:
            frame, debris, speed, x_min, x_max, y_min, y_max = row
            writer.writerow([frame, debris, speed, x_min, x_max, y_min, y_max])


def generate_frames(image_id, image_path, num_frames, objects, output_dir, draw_bbox):
    """Generate new frames based on the provided debris information, same frames in folder, save new coordination in csv file"""
    try:
        # Load the original image, copy it which will be drawn onto later
        img = cv2.imread(str(image_path))
        h, w, _ = img.shape
        
        # save first frame
        img_copy = img.copy()
        if draw_bbox == 1:
            img_copy = draw_with_bboxes(img_copy, objects)
        first_frame_path = os.path.join(output_dir, f"frame_00.jpg") # how to save frame in output directory
        cv2.imwrite(first_frame_path, img_copy)
        
        # for further images, get the background which hides the orginal debris object (so no remnants remain of any debris object)
        img_cleaned = img.copy()
        for obj in objects:
            bbox = obj['original_pos']
            x_min, x_max, y_min, y_max = bbox
            background_patch = get_background_patch(img, bbox)
            img_cleaned[y_min:y_max, x_min:x_max] = background_patch
                                  
        # Generate movement frames
        for frame in range(1, num_frames): # loop over frames
            csv_rows = []
            # Start with a copy of the original image as background
            frame_img = img_cleaned.copy() # this copy will now get the shifted bboxes drawn onto
            
            # Calculate (fractional value) progress (0 to 1)
            progress = frame / (num_frames - 1)
            
            # for each object at its new position
            for i, obj in enumerate(objects):
                obj_img = obj['image']
                org_x_min, org_x_max, org_y_min, org_y_max = obj['original_pos']
                org_height = org_y_max - org_y_min
                org_width = org_x_max - org_x_min
                
                dx, dy = obj['direction']
                depth = obj['depth']
                eff_speed = obj['speed'] * (1 / depth) # farer objects move slower
                eff_dx = int(progress * dx / depth)
                eff_dy = int(progress * dy / depth)
                
                # displacemenrt 
                new_x_min = org_x_min + eff_dx
                new_y_min = org_y_min + eff_dy
                new_x_max = new_x_min + org_width
                new_y_max = new_y_min + org_height

                # ensure new position is within image boundaries
                if (new_x_min >= 0 and new_y_min >= 0 and
                    org_width > 0 and org_height > 0 and
                    new_x_max <= w and 
                    new_y_max <= h):
                    
                    # Paste the extracted object at the new position
                    frame_img[new_y_min:new_y_max, new_x_min:new_x_max] = obj['image']
                    
                    if draw_bbox == 1:
                        # draw bounding box around the moved object
                        cv2.rectangle(frame_img, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 0, 255), 1)
                        cv2.putText(frame_img, f"Debris {i+1}", (new_x_min, max(new_y_min - 15, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # update & save new bbox coord
                    csv_rows.append([frame, i+1, eff_speed, new_x_min, new_x_max, new_y_min, new_y_max])

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame:02d}.jpg")
            cv2.imwrite(frame_path, frame_img)           
            # save bbox coord to csv file
            save_bboxes_to_csv(image_id, csv_rows, output_dir, frame)
            
    except Exception as e: # error stored in e, should catch any error type 
        print(f"Error generating trajectory frames: {e}")
        import traceback
        traceback.print_exc()

def process_vid_no_img(df, image_dir, output_dir, num_frames, num_images, draw_bbox, fps):
    """Generate randomized trajectories for given amount of images for given amount of frames"""
    # loop through all in the dataframe
    for image_id in tqdm(df["ImageID"], total=num_images):
        if(image_id >= num_images):
            break
            
        #print("processing image", image_id)
        
        # construct image and output folder paths
        image_path = image_dir / f"{image_id}.jpg" # construct path
        output_img_dir = output_dir / f"image_{image_id}" 
        os.makedirs(output_img_dir, exist_ok=True)
    
        # before generating the trajectories, remove any csv file with old trajectory information
        csv_file = os.path.join(output_img_dir, f"{image_id}_bboxes.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        
        # before generating the trajectories, collect debris information:
        # extract bounding boxes,
        # assign randomized directions, speed, depth
        objects = debris_info(df, image_id, image_path) # =={'image', 'possition(x_min, x_max, y_min, y_max), 'direction','speed','depth'}   

        # now, generate trajectories
        generate_frames(image_id, image_path, num_frames, objects, output_img_dir, draw_bbox)

        
        path = output_dir / f"image_{image_id}"
        #print("processing video", image_id)
        
        # need to sort images (such eg 11 does not appear before 2), and to mind the numeric sorting in the name "frame_10.jpg" (split string at _, . to get only the number) 
        images = sorted([img for img in os.listdir(path) if img.endswith(".jpg")],
                       key = lambda x: int(x.split('_')[1].split('.')[0]))
        if not images:
            print(f"No images found in folder: {folder_path}")
            continue
        
        frame = cv2.imread(os.path.join(path, images[0])) 
        height, width, layers = frame.shape

        # defines codec, creates videowriter object
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #video_name = os.path.join(path, f"video_image_{image_id}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = os.path.join(path, f"video_image_{image_id}.mp4")        
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # writes videos to frames
        for img in images:
            img_path = os.path.join(path, img)
            frame = cv2.imread(img_path)
            video.write(frame)
            os.remove(img_path)

        video.release()
        cv2.destroyAllWindows()
    print("Finished processing")


def process(df, image_dir, output_dir, num_frames, num_images, draw_bbox):
    """Generate randomized trajectories for given amount of images for given amount of frames"""
    # loop through all in the dataframe
    for image_id in df["ImageID"]:
        if(image_id >= num_images):
            break
            
        print("processing image", image_id)
        
        # construct image and output folder paths
        image_path = image_dir / f"{image_id}.jpg" # construct path
        output_img_dir = output_dir / f"image_{image_id}" 
        os.makedirs(output_img_dir, exist_ok=True)
    
        # before generating the trajectories, remove any csv file with old trajectory information
        csv_file = os.path.join(output_img_dir, f"{image_id}_bboxes.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        
        # before generating the trajectories, collect debris information:
        # extract bounding boxes,
        # assign randomized directions, speed, depth
        objects = debris_info(df, image_id, image_path) # =={'image', 'possition(x_min, x_max, y_min, y_max), 'direction','speed','depth'}   

        # now, generate trajectories
        generate_frames(image_id, image_path, num_frames, objects, output_img_dir, draw_bbox)


def frames_to_video(df, output_folder, fps, num_images): # fps= # frames per second
    for image_id in df["ImageID"]:
        if(image_id >= num_images):
            break
        path = output_folder / f"image_{image_id}"
        print("processing video", image_id)
        
        # need to sort images (such eg 11 does not appear before 2), and to mind the numeric sorting in the name "frame_10.jpg" (split string at _, . to get only the number) 
        images = sorted([img for img in os.listdir(path) if img.endswith(".jpg")],
                       key = lambda x: int(x.split('_')[1].split('.')[0]))
        if not images:
            print(f"No images found in folder: {folder_path}")
            continue
        
        frame = cv2.imread(os.path.join(path, images[0])) 
        height, width, layers = frame.shape

        # defines codec, creates videowriter object
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #video_name = os.path.join(path, f"video_image_{image_id}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = os.path.join(path, f"video_image_{image_id}.mp4")        
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # writes videos to frames
        for img in images:
            img_path = os.path.join(path, img)
            frame = cv2.imread(img_path)
            video.write(frame)
            os.remove(img_path)

        video.release()
        cv2.destroyAllWindows()
        