import cv2
import os
import numpy as np
from datetime import time

# Path to the folder containing images
image_folder = 'overlaps'
output_filename = 'output.jpg'

# Define the start and end times (hh, mm, ss)
start_time = time(1, 40, 0)
end_time = time(1, 45, 0)

# Function to convert filename to time object
def filename_to_time(filename):
    hh, mm, ss = map(int, filename[:-4].split('_'))
    return time(hh, mm, ss)

# Get a list of all files in the folder, sorted by filename
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

# Filter images based on the specified time range
selected_images = []
for img in images:
    img_time = filename_to_time(img)
    if start_time <= img_time <= end_time:
        selected_images.append(img)

# Check if any images were found
if not selected_images:
    print("No images found in the specified time range.")
    exit()

# Load the first image to get the size and type
first_image = cv2.imread(os.path.join(image_folder, selected_images[0]))
if first_image is None:
    print("Failed to load the first image.")
    exit()

# Initialize a black canvas (same size as the images)
canvas = np.zeros_like(first_image)

# Loop through the selected images and take the maximum pixel value for each position
for image_name in selected_images:
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}, skipping.")
        continue
    
    # Take the maximum value between the current image and the canvas
    canvas = np.maximum(canvas, img)

# Save the final combined image
cv2.imwrite(output_filename, canvas)
print(f"Image combining complete! Combined image saved as '{output_filename}'.")

# Display the final combined image
cv2.imshow('Combined Image', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
