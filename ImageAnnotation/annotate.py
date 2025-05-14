import cv2
import numpy as np

def annotate_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    # Make a copy of the original image
    annotated = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # 1. Draw a rectangle around an object
    cv2.rectangle(annotated, (50, 50), (200, 200), (0, 255, 0), 2)
    
    # 2. Draw a filled circle
    cv2.circle(annotated, (300, 150), 50, (255, 0, 0), -1)
    
    # 3. Draw a line
    cv2.line(annotated, (400, 50), (500, 200), (0, 0, 255), 3)
    
    # 4. Draw an arrow
    cv2.arrowedLine(annotated, (600, 100), (700, 100), (255, 255, 0), 2)
    
    # 5. Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, 'OpenCV Annotation', (50, height - 50), 
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 6. Draw a polygon
    pts = np.array([[100, 300], [200, 250], [300, 300], [250, 400]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
    
    # 7. Draw an ellipse
    cv2.ellipse(annotated, (400, 350), (100, 50), 45, 0, 270, (255, 0, 255), 2)
    
    # 8. Add a watermark
    watermark = np.zeros_like(image)
    cv2.putText(watermark, 'SAMPLE', (width-200, height-30), 
                font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.5
    annotated = cv2.addWeighted(annotated, 1, watermark, alpha, 0)
    
    # 9. Draw a bounding box with label
    bbox_top_left = (500, 300)
    bbox_bottom_right = (700, 400)
    label = "Object"
    
    # Draw rectangle
    cv2.rectangle(annotated, bbox_top_left, bbox_bottom_right, (0, 200, 200), 2)
    
    # Draw label background
    text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
    text_width, text_height = text_size[0], text_size[1]
    cv2.rectangle(annotated, 
                 (bbox_top_left[0], bbox_top_left[1] - text_height - 10),
                 (bbox_top_left[0] + text_width + 10, bbox_top_left[1]),
                 (0, 200, 200), -1)
    
    # Draw label text
    cv2.putText(annotated, label, 
                (bbox_top_left[0] + 5, bbox_top_left[1] - 5), 
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to {output_path}")
    
    # Display the image (optional)
    cv2.imshow('Annotated Image', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    input_image = "input.jpg"  # Replace with your image path
    output_image = "annotated_output.jpg"
    annotate_image(input_image, output_image)