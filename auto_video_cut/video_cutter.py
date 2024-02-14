import subprocess

import cv2, sys
from transformers import YolosImageProcessor, YolosForObjectDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch, threading, shutil
from pathlib import Path
import requests, datetime

# Much faster, but not as accurate
#model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
#processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Much slower, but more accurate
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def detect_ppl( image: Image ):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    #logits = outputs.logits
    #bboxes = outputs.pred_boxes


    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] in ("person", "dog", "animal"):
            #print(f"{round(score.item(), 3)}")
            return True
        #box = [round(i, 2) for i in box.tolist()]
        #print(
        #    f"Detected {model.config.id2label[label.item()]} with confidence "
        #    f"{round(score.item(), 3)} at location {box}"
        #)

    return False


# This is actually 3x faster than skipping to the frame directly
global_frame = None
def skip_frames( cap, frame_num ):
    global global_frame
    ret, frame = (True, None)
    for _ in range(frame_num):
        ret, frame = cap.read()
        if not ret:
            global_frame = None
            return None

    global_frame = frame
    return frame


# Open the video file
cap = cv2.VideoCapture(sys.argv[1])

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
    
# Use Path.mkdir() to create the directory
try:
    shutil.rmtree('cuts')
except:
    pass
path = Path('cuts')
path.mkdir(parents=True, exist_ok=True)

# Frame extraction interval (1 frame per second)
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) + 1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

hit_start = -1
misses = 0

miss_max = 3
start_stop_time = 10

print("")
print("")

clip_files = []
debug = []

# Start running the script
script_path = f"cuts/{sys.argv[1]}_ffmpeg.sh"
with open(script_path, "w") as f:
    # Setup my start and stop
    runtime = start_stop_time
    total_runtime = round(total_frames / frame_interval)
    print(f"Total Runtime: {total_runtime} seconds")

    # Skip The Frames to our start runtime
    skip_frames( cap, runtime * frame_interval )

    # Always add the first start/stop time
    clip_files.append(f"cuts/{sys.argv[1]}_cut_video{len(clip_files):04d}.mp4")
    action = f"ffmpeg -i {sys.argv[1]} -ss 0 -t {start_stop_time} -c:v copy -c:a copy {clip_files[-1]}"
    print(action)
    f.write(f"{action}\n")

    # Move to the next frame based on the frame interval
    while runtime < total_runtime - start_stop_time - miss_max:
        # Pull in the global frame from the thread
        if (frame := global_frame) is None:
            print("Encountered an error. Truncating the video.")
            break

        # Start the next Frame while we start the detect ppl
        thread = threading.Thread(target=skip_frames, args=(cap, frame_interval))
        thread.start()

        # Convert the frame to an Image object
        opencv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(opencv_image_rgb)

        # Detect people in the frame (Slow)
        ppl_detected = detect_ppl(pil_image)

        # Combine the threads
        sys.stdout.write(f"Frame {round((runtime * 100)/total_runtime, 1)}%: {ppl_detected}   \r")

        # Shit logic to dump out the videos
        if ppl_detected:
            debug.append( pil_image )
            if hit_start == -1:
                hit_start = runtime
            misses = 0
        elif hit_start >= 0:
            duration = runtime - hit_start
            misses += 1
            if misses >= miss_max:
                if duration >= 5:
                    clip_files.append( f"cuts/{sys.argv[1]}_cut_video{len(clip_files):04d}.mp4" )
                    action = f"ffmpeg -i {sys.argv[1]} -ss {hit_start-miss_max} -t {duration+miss_max+2} -c:v copy -c:a copy {clip_files[-1]}"
                    print(f"\r{action}")
                    f.write(f"{action}\n")

                    # Debug dump
                    for idx, img in enumerate(debug):
                        img.save(clip_files[-1]+ f"{idx}.png")

                # Reset
                debug = []
                hit_start = -1

            # Save the frame as an image (e.g., in PNG format)
            #output_frame_filename = f'frame_{runtime:04d}.png'
            #cv2.imwrite(output_frame_filename, frame)

        # Step the runtime forward
        thread.join()
        runtime += 1

    # Always write out the last 10 seconds
    clip_files.append( f"cuts/{sys.argv[1]}_cut_video{len(clip_files):04d}.mp4" )
    action = f"ffmpeg -i {sys.argv[1]} -ss {total_runtime - start_stop_time} -t {start_stop_time} -c:v copy -c:a copy {clip_files[-1]}"
    print(action)
    f.write(f"{action}\n\n")

    # finally, write out the concat command
    action = f"mencoder -ovc copy -oac pcm {' '.join(clip_files)} -o cuts/{sys.argv[1]}_edited.mp4"
    print(action)
    f.write(f"{action}\n")

    # The last thing to reencode so youtube is happy
    action = f"ffmpeg -i cuts/{sys.argv[1]}_edited.mp4  -c:v libx264 -c:a aac {sys.argv[1]}_reencoded.mp4"
    print(action)
    f.write(f"{action}\n")

print()
print()

# Release the video capture object and close the video file
cap.release()
cv2.destroyAllWindows()

# Use subprocess.run() to run the Bash script
try:
    subprocess.run(['bash', script_path], check=True, text=True)
    print("Video edited uccessfully")
except subprocess.CalledProcessError as e:
    print(f"Error executing Bash script: {e}")
