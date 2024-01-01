import cv2, sys
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests, datetime

# Setup the model
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

def detect_ppl( image: Image ):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    #logits = outputs.logits
    #bboxes = outputs.pred_boxes


    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "person":
            #print(f"{round(score.item(), 3)}")
            return True
        #box = [round(i, 2) for i in box.tolist()]
        #print(
        #    f"Detected {model.config.id2label[label.item()]} with confidence "
        #    f"{round(score.item(), 3)} at location {box}"
        #)

    return False

# Open the video file
cap = cv2.VideoCapture(sys.argv[1])

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Frame extraction interval (1 frame per second)
frame_interval = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

hit_start = -1
misses = 0

miss_max = 3
tail_len = 10

print("")
print("")

frame_count = 0
clips = 0
file_list = f"{sys.argv[1]}_input.txt"
with open(file_list, "w") as filez:
    with open(f"{sys.argv[1]}_ffmpeg.sh", "w") as f:
        while (frame_count + tail_len) * frame_interval < total_frames:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to an Image object
            opencv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(opencv_image_rgb)

            # Detect people in the frame
            ppl_detected = detect_ppl(pil_image)
            sys.stdout.write(f"Frame {round((frame_count*frame_interval * 100)/total_frames, 1)}%: {ppl_detected}   \r")

            # Shit logic to dump out the videos
            if ppl_detected:
                if hit_start == -1:
                    hit_start = frame_count
                misses = 0
            elif hit_start >= 0:
                duration = frame_count - hit_start
                misses += 1
                if misses >= miss_max:
                    if duration >= 5:
                        filename = f"{sys.argv[1]}_cut_video{clips:04d}.mp4"
                        action = f"ffmpeg -i {sys.argv[1]} -ss {hit_start-miss_max*2-2} -t {duration+misses} -c:v copy -c:a copy {filename}"
                        print(f"\r{action}")
                        f.write(f"{action}\n")
                        filez.write(f"{filename}\n")

                        clips += 1
                    hit_start = -1


                # Save the frame as an image (e.g., in PNG format)
                #output_frame_filename = f'frame_{frame_count:04d}.png'
                #cv2.imwrite(output_frame_filename, frame)

            # Move to the next frame based on the frame interval
            frame_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)

        # Always write out the last 10 seconds
        filename = f"{sys.argv[1]}_cut_video{clips:04d}.mp4"
        action = f"ffmpeg -i {sys.argv[1]} -ss {frame_count} -t {tail_len} -c:v copy -c:a copy {filename}"
        f.write(f"{action}\n")
        f.write(f"ffmpeg -f concat -i {file_list} -c:v copy -c:a copy {sys.argv[1]}_combined.mp4\n")
        filez.write(f"{filename}\n")

print()
print()

# Release the video capture object and close the video file
cap.release()
cv2.destroyAllWindows()

