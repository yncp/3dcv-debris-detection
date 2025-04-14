import os
import sys
import cv2
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from deep_sort import nn_matching # calc detection-track distances
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
#based on https://github.com/nwojke/deep_sort

from tqdm import tqdm



def load_gen_csv(csv_path):
    """Load csv file with detections without header:(Frame, Debris_Index, speed, x_min, x_max, y_min, y_max)"""
    if not os.path.exists(csv_path):
        print(f"Error: No CSV file at {csv_path}")
        return None
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["Frame", "Debris_Index", "speed", "x_min", "x_max", "y_min", "y_max"]
    return df


def process_video(input_video_path, output_video_path, csv_path, tracked_csv_path, fps,
                 nFrames=450, eval_interval=100, tolerance=5, feature_dim=128, nMaxObj=8):
    """Process a single video using Deep SORT. Get for each frame info from the CSV file, run tracker,
    draw tracked bounding boxes, save the output video and tracked CSV. 
    Plus compare tracked bounding boxes with ground truth within a tolerance for every interval (if frame has expected number of objects)
    --> return tuple (id, accuracy) for later evaluation (accurcacy = step, amount objects, 0=correctly tracked or 1 = total objects)"""
    try:
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            print(f"Error opening video file: {input_video_path}")
            return
    
        # dimensions
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup evaluation array:
        n_steps = int((nFrames - eval_interval/2) // eval_interval) + 1
        # Shape: (n_steps, nMaxObj+1, 2) where for each step and each possible object count, get [correct, total]
        accuracy = np.zeros((n_steps, nMaxObj + 1, 2))
        
        # get new VideoWriter object (for tracked video)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # get Deepsort tracker
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 200)
        # (distance metric, matching threshold, max sample size)
        tracker = Tracker(metric)
        
        # get csv
        df = load_gen_csv(csv_path)
        if df is None:
            cap.release()
            video_writer.release()
            return
        
        
        # initial detection data from the first frame
        first_frame = df[df["Frame"] == df["Frame"].min()]
        num_objects = first_frame["Debris_Index"].nunique()
        bboxes = first_frame[["x_min", "x_max", "y_min", "y_max"]].to_numpy()
        
        # catch object flying out of frame before first frame
        if (first_frame["Debris_Index"].nunique() != first_frame["Debris_Index"].max()):
                print("First frame error in Video ", input_video_path)
                return(input_video_path,accuracy)
        
        # mind: only evaluate on frames where all objects are present
        frame_counts = df.groupby('Frame')['Debris_Index'].nunique()
        complete_frames = frame_counts[frame_counts == num_objects].index
        if len(complete_frames) == 0:
            cap.release()
            video_writer.release()
            return
        last_frame = int(complete_frames.max())
        
    
    
        # write tracked bounding boxes
        with open(tracked_csv_path, mode='w', newline='') as csvfile:
            # create new csv writer object for tracked coordinates
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame", "Track_id", "x_min", "x_max", "y_min", "y_max"])
        
            frame_index = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
    
                # filter df to get rows with the wanted frame
                curr_detections = df[df["Frame"] == frame_index]
                detection_list = []
                # iterrate over each detection or row in the filtered df
                for _, row in curr_detections.iterrows():
                    x_min = row.x_min
                    y_min = row.y_min
                    w = row.x_max - row.x_min
                    h = row.y_max - row.y_min
                    tlwh = [x_min, y_min, w, h]  # deepsort required tlwh format
                    conf = 1.0  # default confidence.
                    bbox = [row.x_min, row.y_min, row.x_max, row.y_max]
                    feature = np.ones(feature_dim, dtype=np.float32)#extract_appearance_feature(frame, bbox) #np.ones(feature_dim, dtype=np.float32)
                    detection_list.append(Detection(tlwh, conf, feature))
                
                # get tracker prediction & update tracker
                tracker.predict()
                tracker.update(detection_list)
        
                # draw tracked bounding boxes
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    track_id = track.track_id
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # save tracked bounding box to a csv
                    csv_writer.writerow([frame_index, track_id, int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])])
        
                # frame saved to output video.
                video_writer.write(frame)
                
                
                # on each frames interval (centered at eval_interval/2), do accuracy evaluation
                if frame_index % eval_interval == int(eval_interval/2):
                    # proceed if ground truth for this frame has all objects.
                    if curr_detections["Debris_Index"].nunique() == num_objects:
                        # create array from tracked bounding boxes: collect only confirmed tracks 
                        tracked_bboxes = []
                        for track in tracker.tracks:
                            if track.is_confirmed() and track.time_since_update <= 1:
                                # convert track bbox to [x_min, x_max, y_min, y_max]
                                bbox = track.to_tlbr()
                                tracked_bboxes.append([int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])])
                        if len(tracked_bboxes) < num_objects:
                            # if not all objects are tracked; skip
                            pass
                        else:
                            tracked_bboxes = np.array(tracked_bboxes)
                            # in order to match current tracked boxes to ground truth, sort both arrays by the x-coordinate of the box centers.
                            gt_boxes = np.zeros_like(bboxes)
                            # fill ground truth for current frame
                            indices = curr_detections["Debris_Index"].to_numpy() - 1
                            gt_boxes[indices] = curr_detections[["x_min", "x_max", "y_min", "y_max"]].to_numpy()
                            
                            # horizontal centers.
                            gt_centers = np.mean(gt_boxes, axis=1)
                            track_centers = np.mean(tracked_bboxes, axis=1)
                            gt_order = np.argsort(gt_centers)
                            track_order = np.argsort(track_centers)
                            gt_boxes_sorted = gt_boxes[gt_order]
                            tracked_boxes_sorted = tracked_bboxes[track_order]
                            
                            # check per object if absolute difference is within tolerance
                            differences = np.abs(gt_boxes_sorted - tracked_boxes_sorted)
                            matching = np.all(differences <= tolerance, axis=1)
                            correct = int(np.sum(matching))
                            
                            step = int((frame_index - eval_interval/2) // eval_interval)
                            if num_objects < accuracy.shape[1]:
                                # remembre number of correctly tracked objects and the total objects
                                accuracy[step, num_objects, 0] += correct
                                accuracy[step, num_objects, 1] += num_objects
                
                
                frame_index += 1
    
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        #print(f"Finished processing video: {input_video_path}")
        return input_video_path, accuracy
    except:
        print("unkown error in ",input_video_path)
        n_steps = int((nFrames - eval_interval/2) // eval_interval) + 1
        return(input_video_path,  np.zeros((n_steps, nMaxObj + 1, 2)))




def process_videos(df, output_dir, fps, num_images=1, **kwargs):
    """Process multiple videos using Deep SORT"""
    accuracy_total = None
    for image_id in tqdm(df["ImageID"], total=num_images):
        #print("Processing video for image", image_id)
        video_path = output_dir / f"image_{image_id}" / f"video_image_{image_id}.mp4"
        csv_path = output_dir / f"image_{image_id}" / f"{image_id}_bboxes.csv"
        output_video_path = output_dir / f"image_{image_id}" / f"tracked_deepsort_video_image_{image_id}.mp4"
        tracked_csv_path = output_dir / f"image_{image_id}" / f"tracked_deepsort_bboxes_image_{image_id}.csv"
        
        vid_id, acc = process_video(video_path, output_video_path, csv_path, tracked_csv_path, fps, **kwargs)
        
        # stop early, if you want to process only that many images
        if accuracy_total is None:
            accuracy_total = acc
        else:
            accuracy_total += acc
        
        if image_id >= num_images-1:
            break
    return accuracy_total


def plot_histos_DS_all(accuracy_total, eval_interval=100):
    n_steps = accuracy_total.shape[0]
    n_objects_range = np.arange(1, 8)  # if nMaxObj is 8, this represents counts 1..7
    
    for step in range(n_steps):
        plt.figure(figsize=(7, 3))
        y_values = np.zeros(len(n_objects_range))
        n_total = np.zeros(9)  # since nMaxObj+1 = 9
        for i, nobj in enumerate(n_objects_range):
            if nobj < accuracy_total.shape[1]:
                if accuracy_total[step, nobj, 1] > 0:
                    y_values[i] = (accuracy_total[step, nobj, 0] / accuracy_total[step, nobj, 1]) * 100
                    n_total[nobj] = accuracy_total[step, nobj, 1] / nobj
        bars = plt.bar(n_objects_range, y_values, alpha=0.7, color='skyblue', edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
            plt.text(bar.get_x() + bar.get_width()/2., 70,
                     f'{height:.1f}%', ha='center', va='center', fontsize=9)
            plt.text(bar.get_x() + bar.get_width()/2., 20,
                     f'n={n_total[int(bar.xy[0])+1]:.1f}', ha='center', va='center', fontsize=9)
        plt.xlabel('Number of Objects')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Frame {step*100+50} - Accuracy Histogram')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(n_objects_range)
        plt.ylim(0, 100)
        plt.show()

# Second set: for a different range (e.g. 2 to 6)
def plot_histos_DS(accuracy_total, eval_interval=100):
    n_objects_range = np.arange(2, 7)
    n_steps = accuracy_total.shape[0]
    for step in range(n_steps - 1):
        plt.figure(figsize=(5, 4))
        y_values = np.zeros(len(n_objects_range))
        n_total = np.zeros(9)
        for i, nobj in enumerate(n_objects_range):
            if nobj < accuracy_total.shape[1]:
                if accuracy_total[step, nobj, 1] > 0:
                    y_values[i] = (accuracy_total[step, nobj, 0] / accuracy_total[step, nobj, 1]) * 100
                    n_total[nobj] = accuracy_total[step, nobj, 1] / nobj
        bars = plt.bar(n_objects_range, y_values, alpha=0.7, color='skyblue', edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            if height == 0:
                continue
            plt.text(bar.get_x() + bar.get_width()/2., 70,
                     f'{height:.1f}%', ha='center', va='center', fontsize=9)
            plt.text(bar.get_x() + bar.get_width()/2., 20,
                     f'n={int(n_total[int(bar.xy[0])+1])}', ha='center', va='center', fontsize=9)
        plt.xlabel('Number of Objects')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Frame {step*100+50} - Accuracy Histogram')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(n_objects_range)
        plt.ylim(0, 100)
        plt.savefig(f'Histo_DS_Frame_{step*eval_interval+0.5*eval_interval}.png')
        plt.show()