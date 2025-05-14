import cv2
import numpy as np

def apply_filter(frame, filter_name):
    """Apply the selected color filter to the frame"""
    if filter_name == "Grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_name == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)
    elif filter_name == "Invert":
        return cv2.bitwise_not(frame)
    elif filter_name == "Red Channel":
        b, g, r = cv2.split(frame)
        return cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    elif filter_name == "Green Channel":
        b, g, r = cv2.split(frame)
        return cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    elif filter_name == "Blue Channel":
        b, g, r = cv2.split(frame)
        return cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
    elif filter_name == "Warm":
        # Increase red and green channels for warm effect
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 30)
        g = cv2.add(g, 10)
        return cv2.merge([b, g, r])
    elif filter_name == "Cool":
        # Increase blue channel for cool effect
        b, g, r = cv2.split(frame)
        b = cv2.add(b, 30)
        return cv2.merge([b, g, r])
    elif filter_name == "Cartoon":
        # Edge mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
        
        # Color quantization
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        quantized = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        quantized = quantized.astype("float32") / 255.0
        (h, w) = quantized.shape[:2]
        quantized = quantized.reshape((h * w, 3))
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(quantized, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = (centers * 255.0).astype("uint8")
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape((h, w, 3))
        quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2BGR)
        
        # Combine edges with quantized image
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(quantized, edges)
    elif filter_name == "Pencil Sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    else:
        return frame

def main():
    # Available filters
    filters = [
        "None", "Grayscale", "Sepia", "Invert", 
        "Red Channel", "Green Channel", "Blue Channel",
        "Warm", "Cool", "Cartoon", "Pencil Sketch"
    ]
    current_filter = 0
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Real-Time Color Filters with OpenCV")
    print("Press:")
    print("  'n' - Next filter")
    print("  'p' - Previous filter")
    print("  'q' - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Apply current filter
        filtered = apply_filter(frame, filters[current_filter])
        
        # Display the original and filtered frames
        combined = np.hstack((frame, filtered))
        
        # Add filter name text
        cv2.putText(combined, f"Filter: {filters[current_filter]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Real-Time Color Filters (Original | Filtered)', combined)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_filter = (current_filter + 1) % len(filters)
        elif key == ord('p'):
            current_filter = (current_filter - 1) % len(filters)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()