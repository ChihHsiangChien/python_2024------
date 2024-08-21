##
# 擷取一段影片，尋找片段中的最大值疊合在一起，也就是重複曝光
#
import cv2
import yt_dlp
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def get_streaming_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Get the best available MP4 format
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']

    return video_url

def process_video_segment(video_url, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_url)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read the first frame to use as the initial previous frame
    ret, first_frame = cap.read()

    # Initialize a black canvas (same size as the images)
    canvas = np.zeros_like(first_frame)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame  # Total number of frames to process
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize the progress bar
    with tqdm(total=total_frames, desc=f"Processing segment {start_time // fps}s", unit="frame") as pbar:
        while cap.isOpened():
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret or current_frame > end_frame:
                break

            canvas = np.maximum(canvas, frame)

            # Update the progress bar
            pbar.update(1)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cv2.imwrite(output_path, canvas)
    print(f"Image saved: {output_path}")
    cap.release()
    cv2.destroyAllWindows()
# 臺北天文館
# url = "https://www.youtube.com/live/2sjlviZZB94?si=bJSe23PGgeb-XZMt" 
# weathernews
url = "https://www.youtube.com/live/yC_cQYkoBRA?si=wVoUfYFyHtp78vgO"

video_stream_url = get_streaming_url(url)


output_dir = "test/"
os.makedirs(output_dir, exist_ok=True)

# Define the start and end time in seconds
h0, m0, s0 = 1, 5, 0  # 影片開始擷取時間點
h1, m1, s1 = 1, 10, 0  # 影片結束擷取時間點
time_offset = (0, 4, 00)  # 1個segment有多長

start_seconds = h0 * 3600 + m0 * 60 + s0
max_seconds   = h1 * 3600 + m1 * 60 + s1  # 最後擷取到哪個時間

while start_seconds < max_seconds:

    start_seconds = h0 * 3600 + m0 * 60 + s0
    print(h0, m0, s0)
    # 運算下一個時間點(是此段的終點，也是下一段起點)
    s0 += time_offset[2]
    m0 += s0 // 60  # Add carry from seconds
    s0 %= 60       # Keep seconds within 0-59
    
    m0 += time_offset[1]
    h0 += m0 // 60  # Add carry from minutes
    m0 %= 60       # Keep minutes within 0-59
    
    h0 += time_offset[0]
    # ============   
    
    end_seconds = h0 * 3600 + m0 *60 + s0

    output_path = os.path.join(output_dir, f'{h0:02d}_{m0:02d}_{s0:02d}.png')   
    process_video_segment(video_stream_url, start_seconds, end_seconds, output_path)
