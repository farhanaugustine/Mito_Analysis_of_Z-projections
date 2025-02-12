# Fluorescence Image Analysis Pipeline for Mitochondrial Z-Projections

## Overview

This Python script provides an enhanced image analysis pipeline for fluorescence microscopy images. It leverages the [napari](https://napari.org/) viewer and the [napari-sam](https://github.com/MIC-DKFZ/napari-sam) plugin (note: the *repository* is named `napari-sam`, and the *installable package* is `napari-sam`) to facilitate interactive segmentation of cells and intracellular objects (e.g., mitochondria) using the Segment Anything Model (SAM). Following segmentation, the script extracts a comprehensive set of features for each segmented object, including advanced morphology descriptors and intensity measurements, enabling detailed quantitative analysis of cellular structures.

|napari-sam Operations | Commonly Used Mouse Keys|
|---|---|
| Select Intensity Points | Middle Mouse Button|
| Deselect Intensity Points | Ctrl + Middle Mouse Button|
|Undo Selection | Ctrl + Z |

## Key Features

*   **Interactive Segmentation with SAM:** Integrates with the `napari-sam` plugin for user-friendly, AI-assisted segmentation of cells and objects directly within Napari.
*   **Comprehensive Feature Extraction:** Calculates a wide range of features for each segmented object.
*   **Cell-Specific Object Analysis:** Optionally performs cell-specific analysis.
*   **Robust Error Handling and Debugging:** Includes checks for image and mask validity and error handling.
*   **Clean Output to CSV:** Exports extracted features to a well-structured CSV file (`object_features_enhanced.csv`).
*   **Napari Integration:** Keeps the Napari viewer open after analysis.

## Installation

Before running this script, ensure you have the following installed:

*   **Python:** Python 3.7 or higher is recommended.

*   **PyTorch:** Required by `napari-sam`. Install PyTorch following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

    *   **If you have an NVIDIA GPU:** Determine your CUDA version (`nvidia-smi`) and select the appropriate PyTorch version.
    *   **If you do *not* have an NVIDIA GPU:** Select the CPU-only version.

    Example (GPU, CUDA 11.8 - adjust as needed):

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    Example (CPU):
    ```bash
    pip install torch torchvision torchaudio
    ```

*   **Napari:**

    ```bash
    pip install napari[all]
    ```

*   **napari-sam Plugin:**

    Method 1: Using pip:

    ```bash
    pip install napari-sam
    ```

    Method 2: Using Napari GUI:

    1.  Open Napari.
    2.  **Plugins > Install/Uninstall Plugins...**.
    3.  Search for `segment-anything` and install `napari-sam`.
    4.  Restart Napari if prompted.
    5.  Activate via **Plugins > Segment Anything > Segment Anything** and download a SAM model.

*   **Other Libraries:**

    ```bash
    pip install numpy>=1.20 pandas>=1.3 scikit-image>=0.18 scipy>=1.7 napari[all] scikit-network networkx
    ```

## Usage Instructions

1.  **Download the Script:** Download the `fluorescence_analyzer_enhanced.py` script.

2.  **Open Napari via the Script:**

    ```bash
    python fluorescence_analyzer_enhanced.py
    ```

3.  **Load Your Image:** In Napari, drag and drop your image, or use **File > Open File(s)...**.

4.  **Segment Cells (Optional):**

    *   Use `napari-sam` (**Plugins > Segment Anything > Segment Anything**).
    *   Create cell segmentations.
    *   **Rename the layer to `cell_masks`.**

5.  **Segment Objects:**

    *   Use `napari-sam`.
    *   **Rename this layer to `object_masks`.**

6.  **Trigger Feature Extraction:** In the terminal, press **Enter**.

7.  **Feature Extraction and Output:** The script will:

    *   Extract features.
    *   Save to `object_features_enhanced.csv`.
    *   Display a summary in the terminal.
    *   Keep Napari open.

8.  **Analyze Results:** Open `object_features_enhanced.csv` in your preferred data analysis tool.

**Note:**

*   No `cell_masks` layer skips cell-specific analysis.
*   Layer names must be *exactly* `cell_masks` and `object_masks`.
*   Adjust `image_path` in the script to auto-load an image.

## Features Extracted

### 1. Basic Region Properties

These features provide fundamental descriptors of object shape and location.

| Feature             | Description                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `label`             | **Object Identifier.** A unique integer ID for each segmented object.                                                       |
| `object_area`       | **Object Area.** The number of pixels within the object's mask.                                                              |
| `centroid-0`, `centroid-1` | **Object Centroid.** (y, x) or (row, column) coordinates of the object's center.                                          |
| `length`            | **Major Axis Length.** Length of the major axis of the best-fitting ellipse.                                                 |
| `width`             | **Minor Axis Length.** Length of the minor axis of the best-fitting ellipse.                                                 |
| `shape_eccentricity` | **Eccentricity.**  A measure of how elongated the object is (0 = circle, 1 = line).                                         |
| `shape_solidity`    | **Solidity.** Ratio of the object's area to its convex hull area (closer to 1 = more convex, less concave).                 |
| `shape_extent`      | **Extent.** Ratio of the object's area to the area of its bounding box.                                                        |
| `shape_orientation` | **Orientation.** Angle (in radians) of the major axis of the best-fitting ellipse, relative to the horizontal axis.          |
| `object_perimeter`  | **Perimeter.** The total length of the object's boundary.                                                                     |

### 2. Intensity Features

These features quantify the fluorescence intensity within each object.

| Feature               | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `intensity_mean`      | **Mean Intensity.** The average pixel intensity within the object.                                |
| `intensity_integrated` | **Integrated Intensity.** The sum of all pixel intensities within the object.                     |
| `intensity_raw_min`   | **Minimum Intensity.** The lowest pixel intensity within the object.                              |
| `intensity_raw_max`   | **Maximum Intensity.** The highest pixel intensity within the object.                             |
| `intensity_raw_std`   | **Intensity Standard Deviation.** The standard deviation of pixel intensities within the object. |

### 3. Enhanced Morphology Features

These features provide more sophisticated measures of object shape, derived from the basic properties.

| Feature                  | Description                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aspect_ratio`           | **Aspect Ratio.**  Calculated as `length / width`.  Indicates object elongation.                                                                                                                                                    |
| `form_factor`            | **Form Factor (Circularity).** Calculated as `(4 * π * object_area) / (object_perimeter²)`. Values closer to 1 indicate a more circular shape.                                                                                        |
| `perimeter_area_ratio`   | **Perimeter-Area Ratio.** Calculated as `object_perimeter / object_area`.  A 2D proxy for surface area-to-volume ratio; higher values indicate more complex shapes or smaller objects of the same shape.                           |
| `convexity`              | **Convexity.** Calculated as `object_area / shape_solidity`.  Effectively the inverse of solidity, emphasizing deviation from a perfectly convex shape.                                                                              |
| `num_branches`           | **Number of Branches.** The total number of branches detected in the object's skeleton.                                                                                                                                            |
| `total_branch_length`    | **Total Branch Length.** The sum of the lengths of all branches in the object's skeleton.                                                                                                                                          |
| `avg_branch_length`      | **Average Branch Length.** The average length of the branches in the object's skeleton.                                                                                                                                               |
| `num_junctions`          | **Number of Junctions.** The number of junction points (where branches meet) in the object's skeleton.                                                                                                                               |

### 4. Summary & Contextual Information

| Feature        | Description                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `object_count` | **Object Count.** The total number of objects analyzed.                                                                                  |
| `cell_label`   | **Cell Label.**  Indicates the cell to which an object belongs (`"cell_[label]"`) or `"outside_cell"` if not assigned to a segmented cell. |

## Dependencies

*   Python (>=3.7)
*   napari
*   napari-sam
*   PyTorch
*   numpy (>=1.20)
*   pandas (>=1.3)
*   scikit-image (skimage) (>=0.18)
*   scipy (>=1.7)
