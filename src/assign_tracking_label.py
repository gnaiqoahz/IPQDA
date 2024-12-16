import tifffile as tiff
import numpy as np
from skimage import segmentation
from skimage.morphology import disk, erosion, dilation
from matplotlib import pyplot as plt
import time
import os
import json

# Root directory
inputdir1 = "../data/corrected_masks"
inputdir2 = "../data/tracked_masks"
outputdir1 = "../data/labeled_masks"
outputdir2 = "../result/labeled_mask_example"
# Parameters
params = {
    "control_nucleus_masks": os.path.join(inputdir1, "control/control_nucleus_masks.tif"),
    "control_label_stack": os.path.join(inputdir2, "control/CorrectedTracking_control_cell.tif"),
    "control_output_cytoplasm": os.path.join(outputdir1, "control/labelled_control_cytoplasm.tif"),
    "control_output_nucleus": os.path.join(outputdir1, "control/labelled_control_nucleus.tif"),
    "sample_nucleus_masks": os.path.join(inputdir1, "sample/sample_nucleus_masks.tif"),
    "sample_label_stack": os.path.join(inputdir2, "sample/CorrectedTracking_sample_cell.tif"),
    "sample_output_cytoplasm": os.path.join(outputdir1, "sample/labelled_sample_cytoplasm.tif"),
    "sample_output_nucleus": os.path.join(outputdir1, "sample/labelled_sample_nucleus.tif"),
}

# Save parameters as JSON
output_base_dir = os.path.dirname(params["control_output_cytoplasm"])
os.makedirs(output_base_dir, exist_ok=True)

with open(os.path.join(output_base_dir, 'params.json'), 'w') as json_file:
    json.dump(params, json_file, indent=4)

# Start the timer
start = time.perf_counter()

def process_stacks(label_stack_path, nucleus_stack_path, output_cytoplasm_path, output_nucleus_path):
    """Segment the labeled masks into nucleus and cytoplasm masks."""
    label_stack = tiff.imread(label_stack_path).astype(np.uint32)
    nucleus_stack = tiff.imread(nucleus_stack_path).astype(np.uint32)

    # Ensure the stacks have the same number of frames
    if label_stack.shape[0] != nucleus_stack.shape[0]:
        raise ValueError("Label stack and nucleus stack must have the same number of frames")

    num_frames = label_stack.shape[0]
    cytoplasm_label_stack = np.zeros_like(label_stack)
    nucleus_label_stack = np.zeros_like(label_stack)

    selem = disk(4)  # Structuring element for dilation and erosion

    for frame in range(num_frames):
        # Remove border cells
        label_image = segmentation.clear_border(label_stack[frame])
        nucleus_image = segmentation.clear_border(nucleus_stack[frame])

        # Convert to binary masks
        binary_label_image = (label_image > 0).astype(np.uint8)
        binary_nucleus_image = (nucleus_image > 0).astype(np.uint8)

        # Create final label images
        label_cytoplasm_image = np.zeros_like(label_image)
        label_nucleus_image = np.zeros_like(label_image)

        unique_labels = np.unique(label_image)
        for label in unique_labels:
            if label == 0:
                continue

            cell_mask = (label_image == label)
            nucleus_mask = cell_mask & (binary_nucleus_image > 0)

            if np.any(nucleus_mask):
                # Expand the nucleus mask by 4 pixels
                expanded_nucleus_mask = dilation(nucleus_mask, selem)
                # For cytoplasm
                label_cytoplasm_image[cell_mask & (binary_label_image > 0) & (expanded_nucleus_mask == 0)] = label
                # For nucleus
                label_nucleus_image[nucleus_mask] = label

        # Apply erosion to remove small artifacts
        eroded_cytoplasm = erosion(label_cytoplasm_image, selem)
        eroded_nucleus = erosion(label_nucleus_image, selem)

        # Store results
        cytoplasm_label_stack[frame] = eroded_cytoplasm
        nucleus_label_stack[frame] = eroded_nucleus

    tiff.imwrite(output_cytoplasm_path, cytoplasm_label_stack)
    tiff.imwrite(output_nucleus_path, nucleus_label_stack)
    return cytoplasm_label_stack, nucleus_label_stack

def main(params):
    """Main function to process control and sample groups."""
    # Process control group
    cytoplasm_label_stack, nucleus_label_stack = process_stacks(
        params["control_label_stack"],
        params["control_nucleus_masks"],
        params["control_output_cytoplasm"],
        params["control_output_nucleus"]
    )
    print("Control group processing completed.")

    # Save specific frames to verify the results
    save_frame(cytoplasm_label_stack, 0, "Control_Cytoplasm_Frame_0", outputdir2)
    save_frame(nucleus_label_stack, 0, "Control_Nucleus_Frame_0", outputdir2)

    # Process sample group
    cytoplasm_label_stack, nucleus_label_stack = process_stacks(
        params["sample_label_stack"],
        params["sample_nucleus_masks"],
        params["sample_output_cytoplasm"],
        params["sample_output_nucleus"]
    )
    print("Sample group processing completed.")

    # Save specific frames to verify the results
    save_frame(cytoplasm_label_stack, 0, "Sample_Cytoplasm_Frame_0", outputdir2)
    save_frame(nucleus_label_stack, 0, "Sample_Nucleus_Frame_0", outputdir2)

    # End the timer
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken: {elapsed:.2f} seconds")

def save_frame(stack, frame_idx, title, output_folder):
    """Save a specific frame from a stack as a PNG file."""
    plt.imshow(stack[frame_idx], cmap='tab20')
    plt.title(title)
    plt.axis('off')
    output_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main(params)
