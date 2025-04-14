import cv2
import pandas as pd
import ast
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from multiprocessing.pool import ThreadPool as Pool
import os
from tqdm import tqdm
import functools


nMaxObj = 8


def tracker_single(video_id, showVid=False, nFrames=450, eval_interval=100, tolerance=5,  output_folder=""):
    
    #try:
    if(True):
        #showVid=False
        #nFrames=450
        #eval_interval=100
        #tolerance=5
        n_steps = int((nFrames - eval_interval / 2) // eval_interval) + 1
    
        accuracy = np.zeros((n_steps,nMaxObj+1,2))
    
        path = output_folder / f"image_{video_id}"
    
        # Open video file
        video = cv2.VideoCapture(path/f"video_image_{video_id}.mp4")
    
        # Get frame dimensions and FPS
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(video.get(cv2.CAP_PROP_FPS) or 30)  # fallback if fps = 0
        
        # Define the codec and create VideoWriter object
        if(showVid):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
            out = cv2.VideoWriter(path/f"video_tracking_{video_id}.mp4", fourcc, fps, (width, height))
    
    
        # Read the first frame
        ret, frame = video.read()
        if not ret:
            print("Failed to read video")
            return(video_id,accuracy)
            
        
        df = pd.read_csv(path/f"{video_id}_bboxes.csv", header=None)
        df.columns = ["Frame", "Debris_Index", "speed", "x_min", "x_max", "y_min", "y_max"]
        
        # Filter data for the first frame
        first_frame = df[df["Frame"] == df["Frame"].min()]
        
        # Get unique object count
        num_objects = first_frame["Debris_Index"].nunique()
        
        # Extract coordinates for each object
        bboxes = first_frame[["x_min", "x_max", "y_min", "y_max"]].to_numpy()
    
        # catch object flying out of frame before first frame
        if (first_frame["Debris_Index"].nunique() != first_frame["Debris_Index"].max()):
            print("First frame error in Video ", video_id)
            return(video_id,accuracy)
    
    
    
        # Find frames where all objects are present
        frame_counts = df.groupby('Frame')['Debris_Index'].nunique()
        complete_frames = frame_counts[frame_counts == num_objects].index
        if len(complete_frames) == 0:
            return(video_id,accuracy)
        last_frame = complete_frames.max()
    
        #print("Last Frame: ",last_frame)
    
        
    
        
    
    
        tracker = []
        bbox = []
        
    
        #init tracker for each object
        for obj in range(num_objects):
            #convert from xmin, xmax, ymin, ymax to x, y, w, h
            x, y, w, h = int(bboxes[obj,0]), int(bboxes[obj,2]), int(bboxes[obj,1]-bboxes[obj,0]), int(bboxes[obj,3]-bboxes[obj,2])
            box_conv= x, y, w, h
            bbox.append(box_conv)
            
            # Create KCF Tracker
            tracker_single = cv2.TrackerKCF_create()
    
            # Initialize the tracker with the selected region
            tracker_single.init(frame, bbox[obj])
    
            tracker.append(tracker_single)
    
    
        active_tracker = [True] * num_objects
        n_frame = 0
    
        
        while (n_frame<last_frame and n_frame<nFrames):
            n_frame = n_frame+1
            ret, frame = video.read()
            if not ret:
                break
    
            #print(active_tracker)
    
            for obj in range(num_objects):
                if not active_tracker[obj]:
                    continue
                
                # Update tracker
                success, bbox[obj] = tracker[obj].update(frame)
           
                if success:
                    #x, y, w, h = map(int, bbox[obj])
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if showVid:
                        x, y, w, h = map(int, bbox[obj])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    active_tracker[obj] = False
                    #print("trk error")
                    if showVid:
                        cv2.putText(frame, "Tracking Failure", (50, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
                     
            # Display result
            if(showVid):
                cv2.imshow("KCF Tracker", frame)
                out.write(frame)
        
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            if n_frame % eval_interval == int(eval_interval/2):
                correct = 0
                total = num_objects
    
                
                track_bboxes = np.zeros((num_objects,4))
                for obj in range(num_objects):
                    x_min, y_min, w_2, h_2 = map(int, bbox[obj])
                    x_max = x_min + w_2
                    y_max = y_min+h_2
                    track_bboxes[obj] = [x_min,x_max,y_min,y_max]
                    
                # Filter data for the frame
                current_frame = df[df["Frame"] == n_frame]
                #print(current_frame)
                
                # Extract coordinates for each object, 0 for out of frame
                bboxes_gt = np.zeros_like(bboxes)
                bboxes_gt[ current_frame["Debris_Index"].to_numpy() -1 ] = current_frame[["x_min", "x_max", "y_min", "y_max"]].to_numpy()
                matching = np.all(np.abs(bboxes_gt - track_bboxes) <= tolerance, axis=1) 
                correct = matching.sum()
    
                #print("Frame ",n_frame)
                #print(int((n_frame - eval_interval / 2) // eval_interval))
                
                
                if(num_objects>nMaxObj):
                    num_objects = 0
                accuracy[int((n_frame - eval_interval / 2) // eval_interval)][num_objects][0] = \
                    accuracy[int((n_frame - eval_interval / 2) // eval_interval)][num_objects][0] + correct
                accuracy[int((n_frame - eval_interval / 2) // eval_interval)][num_objects][1] = \
                    accuracy[int((n_frame - eval_interval / 2) // eval_interval)][num_objects][1] + num_objects
                
        
                
                #print("GT: ", bboxes_gt) 
                #print("Tracker: ", track_bboxes)
                #print(matching)
                #print(correct, "\n\n")
    
        
        video.release()
        
        if(showVid):
            out.release()
            cv2.destroyAllWindows()

        return(video_id,accuracy)
        
'''    except:
        print("Unexpected error in video ", video_id)
        return(video_id,np.zeros((n_steps,nMaxObj+1,2)))'''

    

def tracker_single_thread(n_videos=20, n_start=0, showVid=False, nFrames=450, eval_interval=30, tolerance=5):
    accuracy_total = np.zeros((n_steps,nMaxObj+1,2))
    n_videos_processed = 0
    for video_id in range(n_start, n_videos+n_start):
        video_id, accuracy = tracker_single(video_id, showVid=showVid, nFrames=nFrames, eval_interval=eval_interval, tolerance=tolerance)
        accuracy_total = accuracy_total + accuracy
    return accuracy_total


def tracker_multi_thread(n_videos=20, n_start=0, reset_stats=True, showVid=False, nFrames=450, eval_interval=30, tolerance=5, output_folder="./"):
    
    video_ids = list(range(n_start, n_start + n_videos))

    n_steps = int((nFrames - eval_interval / 2) // eval_interval) + 1

    if(reset_stats):
        accuracy_total = np.zeros((n_steps,nMaxObj+1,2))
    #if __name__ == "__main__": 
    if(True):
        with Pool(processes=min(len(video_ids), os.cpu_count()-2)) as pool:
            print("starting videos")
            #n_videos_processed = 0
            #args_list = [(vid, showVid, nFrames, eval_interval, tolerance) for vid in video_ids]
            #results = (pool.starmap(tracker_single, args_list))
            results = list(
                tqdm(
                    pool.imap(
                        functools.partial(
                            tracker_single, showVid=showVid, nFrames=nFrames, eval_interval=eval_interval, tolerance=tolerance, output_folder=output_folder
                        ),
                        video_ids
                    ),
                total=len(video_ids))
            )
   
        # Gather accuracy results
        for video_id, accuracy in results:
            accuracy_total = accuracy_total + accuracy
            
    return accuracy_total




def plot_histos_kcf(accuracy_total,eval_interval=100):
    n_steps = accuracy_total.shape[0]
    #n_objects_range = np.arange(accuracy.shape[1])  # X-axis: Number of objects
    n_objects_range = np.arange(2, 7)
    
    for step in range(n_steps -1):
        plt.figure(figsize=(5, 3))
        
        # Compute accuracy percentage: (correct / total_objects) * 100
        #accuracy_percentage = (accuracy[step, :, 0] / accuracy[step, :, 1]) * 100
        
        # Mask invalid entries (where total_objects = 0)
        #valid_mask = accuracy[step, :, 1] > 0
        #x_values = n_objects_range[valid_mask]
        #y_values = accuracy_percentage[valid_mask]
        y_values = np.zeros(len(n_objects_range))
        n_total= np.zeros(nMaxObj+1)
        # Fill in available data
        for i, nobj in enumerate(n_objects_range):
            if nobj < accuracy_total.shape[1]:  # Check if nobj exists in your data
                if accuracy_total[step, nobj, 1] > 0:  # Avoid division by zero
                    y_values[i] = (accuracy_total[step, nobj, 0] / accuracy_total[step, nobj, 1]) * 100
                    
                    n_total[nobj] = accuracy_total[step, nobj, 1]/nobj
        if(n_total[0] != 0):
            print(n_total)
        # Plot histogram (bar plot)
        bars = plt.bar(n_objects_range, y_values, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Annotate each bar with its height (accuracy %)
        for bar in bars:
            height = bar.get_height()
            if(height == 0):
                continue
            plt.text(bar.get_x() + bar.get_width()/2., 70,
                     f'{height:.1f}%',
                     ha='center', va='center', fontsize=9)
            plt.text(bar.get_x() + bar.get_width()/2., 20,
                     f'n={int(n_total[int(bar.xy[0])+1])}',
                     ha='center', va='center', fontsize=9)
            i +=1
        
        plt.xlabel('Number of Objects')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Frame {step*eval_interval+0.5*eval_interval} - Accuracy Histogram')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(n_objects_range)  # Ensure all object counts are labeled
        plt.ylim(0, 100)  # Percentage scale
        plt.savefig(f'Histo_KCF_Frame_{step*eval_interval+0.5*eval_interval}.png')
        plt.show()