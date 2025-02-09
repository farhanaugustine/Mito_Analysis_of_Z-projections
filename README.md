# Fluorescence Image Analysis Pipeline for Mitochondrial Z-Projections

## Overview

This Python script provides an enhanced image analysis pipeline for fluorescence microscopy images. It leverages the [napari](https://napari.org/) viewer and the [napari-sam](https://github.com/napari/napari-segment-anything) plugin (note: the *repository* is named `napari-segment-anything`, but the *installable package* is `napari-sam`) to facilitate interactive segmentation of cells and intracellular objects (e.g., mitochondria) using the Segment Anything Model (SAM). Following segmentation, the script extracts a comprehensive set of features for each segmented object, including advanced morphology descriptors and intensity measurements, enabling detailed quantitative analysis of cellular structures.

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
    pip install numpy pandas scikit-image scipy
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

| Feature             | Description                                                                                                                                                                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label`             | **Object Identifier.** A unique number to identify each segmented object.                                                                                                                                                                                            |
| `object_area`       | **Size of the Object.** The number of pixels within the object.                                                                                                                                                                                                       |
| `centroid-0`, `centroid-1` | **Object Location.** (y, x) or (row, column) coordinates of the object's center.                                                                                                                                                                                    |
| `length`            | **Object Elongation (Major Axis).** Length of the major axis of the best-fitting ellipse.                                                                                                                                                                              |
| `width`             | **Object Elongation (Minor Axis).** Length of the minor axis of the best-fitting ellipse.                                                                                                                                                                              |
| `shape_eccentricity` | **Object Elongation (Ratio).**  How much the shape deviates from a circle (0 = circle, 1 = line).                                                                                                                                                                    |
| `shape_solidity`    | **Object Convexity.** Ratio of object area to its convex hull area (closer to 1 = more convex).                                                                                                                                                                      |
| `shape_extent`      | **Object Compactness within Bounding Box.** Ratio of object area to bounding box area.                                                                                                                                                                                |
| `shape_orientation` | **Object Orientation.** Angle of the major axis of the best-fitting ellipse (relative to horizontal).                                                                                                                                                                |
| `object_perimeter`  | **Object Boundary Length.** Length of the object's outer boundary.                                                                                                                                                                                                    |

### 2. Intensity Features

These features quantify the fluorescence intensity.

| Feature               | Description                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `intensity_mean`      | **Average Intensity.** Mean pixel intensity within the object.                                             |
| `intensity_integrated` | **Total Intensity.** Sum of pixel intensities within the object.                                           |
| `intensity_raw_min`   | **Minimum Intensity.** Lowest intensity value within the object.                                            |
| `intensity_raw_max`   | **Maximum Intensity.** Highest intensity value within the object.                                           |
| `intensity_raw_std`   | **Intensity Variation.** Standard deviation of pixel intensities within the object.                        |

### 3. Enhanced Morphology Features

These features provide more sophisticated measures of object shape.

| Feature                  | Description                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aspect_ratio`           | **Elongation Ratio.** Calculated as `length / width`.                                                                                                                                                                                 |
| `form_factor`            | **Circularity Proxy.** Calculated as `(4 * π * object_area) / (object_perimeter²)`. Closer to 1 = rounder.                                                                                                                               |
| `perimeter_area_ratio`   | **Surface Area to Volume Proxy (2D).** Calculated as `object_perimeter / object_area`.                                                                                                                                               |
| `convexity`              | Calculated as `object_area / shape_solidity`. Measures deviation from convex hull.                                                                                                                                                   |

### 4. Summary & Contextual Information

| Feature        | Description                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `object_count` | **Number of Objects Analyzed.** Total count of analyzed objects.                                                                           |
| `cell_label`   | **Cellular Context.**  Indicates if an object is within a segmented cell (`"cell_[label]"`) or outside cells (`"outside_cell"`). |

## Dependencies

*   Python (>=3.7)
*   napari
*   napari-sam
*   PyTorch
*   numpy (>=1.20)
*   pandas (>=1.3)
*   scikit-image (skimage) (>=0.18)
*   scipy (>=1.7)
