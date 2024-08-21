import cv2
import yt_dlp
import os

meteorTime = [[0,36,26],
 [1,4,44],
 [1,10,28],
 [1,17,39],
 [1,17,54],
 [1,20,49],
 [1,23,19],
 [1,23,49],
 [1,29,53],
 [1,30,19],
 [1,35,37],
 [1,36,27],
 [1,39,27],
 [1,42,1],
 [1,47,59],
 [1,50,54],
 [1,51,51],
 [1,52,28],
 [1,53,4],
 [1,58,13],
 [2,0,25],
 [2,2,49],
 [2,6,36],
 [2,6,42],
 [2,7,36],
 [2,9,42],
 [2,11,58],
 [2,16,44],
 [2,17,44],
 [2,25,23],
 [2,28,29],
 [2,28,48],
 [2,34,55],
 [2,41,15],
 [2,43,39],
 [2,44,54],
 [2,45,7],
 [2,48,4],
 [2,51,35],
 [2,52,9],
 [2,52,41],
 [2,54,34],
 [2,55,5],
 [2,58,12],
 [2,58,50],
 [3,4,20],
 [3,4,33],
 [3,5,5],
 [3,6,2],
 [3,7,30],
 [3,10,50],
 [3,11,51],
 [3,14,57],
 [3,16,54],
 [3,17,51],
 [3,18,44],
 [3,19,14],
 [3,23,46],
 [3,27,43],
 [3,28,49],
 [3,30,13],
 [3,31,45],
 [3,35,30],
 [3,37,27],
 [3,39,18],
 [3,39,39],
 [3,40,51],
 [3,43,19],
 [3,48,5],
 [3,48,48],
 [3,54,23],
 [3,53,30],
 [3,54,16],
 [3,55,0],
 [3,55,43],
 [3,56,16],
 [3,59,20],
 [4,1,49],
 [4,2,13],
 [4,2,31],
 [4,6,47],
 [4,9,35],
 [4,10,15],
 [4,14,45],
 [4,16,42],
 [4,19,14],
 [4,19,28],
 [4,20,26],
 [4,21,5],
 [4,24,15],
 [4,29,18],
 [4,30,13],
 [4,35,5],
 [4,35,23],
 [4,35,59]]


def get_streaming_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Get the best available MP4 format
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) qas ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']

    return video_url

def process_frame(previous_frame, current_frame, roi, threshold=30, min_area=500):
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Apply the ROI to both frames
    x1, y1, x2, y2 = roi
    roi_current = gray_current[y1:y2, x1:x2]
    roi_previous = gray_previous[y1:y2, x1:x2]

    # Compute the absolute difference within the ROI
    diff = cv2.absdiff(roi_previous, roi_current)

    # Check if the maximum grayscale value in the difference exceeds the threshold
    if diff.max() > threshold:
        
        # Apply a threshold to get binary image
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        meteor_frames = []
        for contour in contours:
            if cv2.contourArea(contour) > min_area:  # Filter out small areas
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust coordinates back to the full frame context
                cv2.rectangle(current_frame, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                meteor_frames.append(current_frame[y + y1:y + h + y1, x + x1:x + w + x1])

        return current_frame, meteor_frames        
    
    return current_frame, []

def isMeteor(previous_frame, current_frame, roi, threshold):
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Apply the ROI to both frames
    x1, y1, x2, y2 = roi
    roi_current = gray_current[y1:y2, x1:x2]
    roi_previous = gray_previous[y1:y2, x1:x2]

    # Compute the absolute difference within the ROI
    diff = cv2.absdiff(roi_previous, roi_current)

    # Check if the maximum grayscale value in the difference exceeds the threshold
    if diff.max() > threshold:
        return True
    else:
        return False

def process_video_segment(video_url, start_time, end_time, roi, threshold=30):
    cap = cv2.VideoCapture(video_url)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame to use as the initial previous frame
    ret, previous_frame = cap.read()

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(current_frame)
        ret, frame = cap.read()

        if not ret or current_frame > end_frame:
            break

        # Process the frame within the ROI using the previous frame
        # processed_frame, meteor_frames = process_frame(previous_frame, frame, roi, threshold)
        # Display the frame        
        if isMeteor(previous_frame, frame, roi, threshold):
            #cv2.imshow('Meteor Detection', frame)
            cv2.imwrite(output_dir + f'meteor_{current_frame}.png', frame)            

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Optional: Save or process detected meteor frames
        '''
        for meteor_frame in meteor_frames:
            cv2.imwrite(f'meteor_{current_frame}.png', meteor_frame)
        '''
        # Update the previous frame
        previous_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

url = "https://www.youtube.com/live/2sjlviZZB94?si=bJSe23PGgeb-XZMt" 
video_stream_url = get_streaming_url(url)



# Define ROI as (x1, y1, x2, y2) where (x1, y1) is the top-left and (x2, y2) is the bottom-right
roi = (50, 50, 500, 270)

# Threshold value
threshold = 100
output_dir = 'images/'
os.makedirs(output_dir, exist_ok=True)

for time in meteorTime:
    # Define the start and end time in seconds
    h,m,s = time[0],time[1],time[2]
    duration = 10

    start_time = h*60*60 + m*60 + s
    end_time = start_time + duration
    process_video_segment(video_stream_url, start_time, end_time, roi, threshold)