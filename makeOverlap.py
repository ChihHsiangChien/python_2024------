#
# 將整個資料夾的照片疊成一張圖
#
import cv2
import os
import numpy as np

# Path to the folder containing images
image_folder = 'overlaps'
image_filename = 'satellite_combined_image.jpg'

# Get a list of all files in the folder, sorted by filename
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

# Check if any images were found
if not images:
    print("No images found in the specified folder.")
    exit()

# Load the first image to get the size and type
first_image = cv2.imread(os.path.join(image_folder, images[0]))
if first_image is None:
    print("Failed to load the first image.")
    exit()

# Initialize a black canvas (same size as the images)
canvas = np.zeros_like(first_image)

# Loop through the images and take the maximum pixel value for each position
for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}, skipping.")
        continue
    
    # Take the maximum value between the current image and the canvas
    canvas = np.maximum(canvas, img)

# Save the final combined image

cv2.imwrite(image_filename, canvas)
print("Image combining complete! Combined image saved as 'combined_image.jpg'.")

# Display the final combined image
cv2.imshow('Combined Image', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
