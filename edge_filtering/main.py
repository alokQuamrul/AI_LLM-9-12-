import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles, rows, cols, figsize=(15, 10)):
    """Helper function to display multiple images"""
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load the image
    image_path = 'your_image.jpg'  # Replace with your image path
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("Error: Image not found")
        return
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # 1. Canny Edge Detection
    canny_edges = cv2.Canny(blurred_image, 50, 150)
    
    # 2. Sobel Edge Detection
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined / np.max(sobel_combined) * 255)
    
    # 3. Laplacian Edge Detection
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 4. Prewitt Edge Detection (using Sobel with kernel size 3)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_x = cv2.filter2D(blurred_image, -1, kernelx)
    prewitt_y = cv2.filter2D(blurred_image, -1, kernely)
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt_combined = np.uint8(prewitt_combined / np.max(prewitt_combined) * 255)
    
    # 5. Adaptive Thresholding for edge-like effect
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred_image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 6. Bilateral Filter (edge-preserving smoothing)
    bilateral = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # 7. Non-local Means Denoising (advanced filtering)
    denoised = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)
    
    # Display results
    images = [
        gray_image, blurred_image, canny_edges, 
        sobel_combined, laplacian, prewitt_combined,
        adaptive_thresh, bilateral, denoised
    ]
    
    titles = [
        "Original Grayscale", "Gaussian Blur", "Canny Edges",
        "Sobel Combined", "Laplacian", "Prewitt Combined",
        "Adaptive Threshold", "Bilateral Filter", "Denoised"
    ]
    
    display_images(images, titles, 3, 3)

if __name__ == "__main__":
    main()