import os
import nibabel as nib
import numpy as np

# Input and output directories
input_dir = "path_to_input_folder"
output_dir = "path_to_output_folder"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all .nii.gz files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".nii.gz"):
        # Construct full file path
        nii_file_path = os.path.join(input_dir, file_name)

        # Load the NIfTI file
        nii_data = nib.load(nii_file_path)

        # Get the image data as a NumPy array
        numpy_array = nii_data.get_fdata()

        # Save the array as a .npy file
        output_file_name = os.path.splitext(os.path.splitext(file_name)[0])[0] + ".npy"
        npy_file_path = os.path.join(output_dir, output_file_name)
        np.save(npy_file_path, numpy_array)

        print(f"Converted {file_name} to {output_file_name}")

print("All files have been processed.")
