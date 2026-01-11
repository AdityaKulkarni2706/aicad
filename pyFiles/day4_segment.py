import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image_kmeans(image_path, k=3):
    img = cv2.imread(image_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    plt.figure(figsize=(10, 5))

    random_colors = np.random.randint(0, 255, (k, 3))
    
    # Map the labels to these random bright colors
    false_color_image = random_colors[labels.flatten()]
    
    # Reshape back to image dimensions
    false_color_image = false_color_image.reshape(img.shape)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.imshow(false_color_image)
    plt.title(f'False Color Map (K={k})')
    plt.axis('off')
    plt.show()

segment_image_kmeans("../images/cup_1.jpeg", k=3)