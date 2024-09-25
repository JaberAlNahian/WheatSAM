import os
import cv2
import nrrd
import numpy as np
import SimpleITK as sitk


# def convert_nrrd_to_png(nrrd_folder, output_folder):
#     for file_name in os.listdir(nrrd_folder):
#         if file_name.endswith('.nrrd'):
#             nrrd_path = os.path.join(nrrd_folder, file_name)
            
#             # Read the NRRD file
#             nrrd_data, _ = nrrd.read(nrrd_path)
            
#             # Convert to uint8 for saving as PNG
#             nrrd_data_uint8 = (nrrd_data.astype(float) / np.max(nrrd_data) * 255).astype(np.uint8)
            
#             # Save as PNG
#             output_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + '.png')
#             cv2.imwrite(output_path, nrrd_data_uint8)

def convert_nrrd_to_png(nrrd_folder, output_folder):
    for file_name in os.listdir(nrrd_folder):
        if file_name.endswith('.nrrd'):
            nrrd_path = os.path.join(nrrd_folder, file_name)
            # Read the NRRD file
            img = sitk.ReadImage(nrrd_path)
            image_array = sitk.GetArrayFromImage(img)
            reshaped_array = np.transpose(image_array, (1, 2, 0))
            reduced_array = np.squeeze(reshaped_array, axis=-1)
            image = (reduced_array * 255).astype(np.uint8)
        
            # save as jpg
            output_path = os.path.join(output_folder, file_name.rsplit('.', 1)[0] + '.jpg')
            cv2.imwrite(output_path, image)

# Example usage
nrrd_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi'  # Folder containing NRRD files
output_folder = '/home/mdjaberal.nahian/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Semi-Self-SupervisedLearningForSemanticSegmentationInImagesWithDensePatterns-Paper/Psi_NRRDtoPNG'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

convert_nrrd_to_png(nrrd_folder, output_folder)
