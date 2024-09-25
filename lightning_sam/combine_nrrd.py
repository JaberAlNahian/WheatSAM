import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nrrd

def visualize_images_with_masks(data_folder, output_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(data_folder, file_name)
            mask_path = os.path.join(data_folder, file_name.rsplit('.', 1)[0] + '.nrrd')
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Read the mask
            mask, header = nrrd.read(mask_path)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # Use the first slice if the mask has more than 2 dimensions
            
            # Convert mask to single-channel image with binary values (0 or 255)
            mask = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)[1]
            
            # Get contours from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding boxes and overlay masks
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Overlay mask
                mask_overlay = np.zeros_like(image)
                cv2.drawContours(mask_overlay, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
                image = cv2.addWeighted(image, 0.5, mask_overlay, 0.5, 0)
            
            # Save the visualized image
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, image)

            # Display the visualized image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()



# Example usage
data_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi'  # Folder containing both images and masks
output_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_NRRD_Mask'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

visualize_images_with_masks(data_folder, output_folder)
