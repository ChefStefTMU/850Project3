import cv2
import numpy as np

# Load image
img = cv2.imread("data/motherboard_image.jpeg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge Detection - Canny
edges = cv2.Canny(blur, 50, 150)

# Dilation + Closing
kernel = np.ones((5, 5), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# Contour Detection
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter large contours
large_contours = [c for c in contours if cv2.contourArea(c) > 4000]

# Largest contour = PCB
largest = max(large_contours, key=cv2.contourArea)

# Mask creation
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

# Extract PCB
extracted = cv2.bitwise_and(img, img, mask=mask)

# Save Step 1 outputs
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("edges.jpg", edges)
cv2.imwrite("mask.jpg", mask)
cv2.imwrite("extracted.jpg", extracted)


