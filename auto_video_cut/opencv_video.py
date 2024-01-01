import cv2, sys

# Open the video file
cap = cv2.VideoCapture(sys.argv[1])

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Frame extraction interval (1 frame per second)
frame_interval = frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Save the frame as an image (e.g., in PNG format)
    output_frame_filename = f'frame_{frame_count:04d}.png'
    cv2.imwrite(output_frame_filename, frame)

    # Increment the frame count
    frame_count += 1

    # Move to the next frame based on the frame interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
    print(output_frame_filename)

# Release the video capture object and close the video file
cap.release()
cv2.destroyAllWindows()

