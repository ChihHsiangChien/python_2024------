import cv2
import os

# Path to the folder containing images
image_folder = 'overlaps'
output_filename = "satellite.mp4"
fps = 10


# Get a list of all files in the folder, sorted by filename
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

# Get the size of the first image to determine the video size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Define the codec (MPEG-4) and create a VideoWriter object
video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Loop through all images and add them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# Release the VideoWriter
video.release()

print("Video creation complete!")
