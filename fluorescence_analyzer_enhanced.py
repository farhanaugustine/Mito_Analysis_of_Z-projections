import napari
import numpy as np
import pandas as pd
from skimage import measure, io, morphology
from scipy.spatial import ConvexHull
import skimage  # For version checking


def extract_enhanced_object_features(image, mask, cell_mask=None):
    """
    Extracts features from segmented objects.
    """
    print("Calculating enhanced object features...")

    # --- Debugging Prints ---
    print("\n--- Debugging Information before regionprops_table ---")
    print(f"Scikit-image version: {skimage.__version__}")  # Keep this!
    print(f"Type of 'image': {type(image)}, Shape of 'image': {image.shape}, dtype of 'image': {image.dtype}")
    print(f"Type of 'mask': {type(mask)}, Shape of 'mask': {mask.shape}, dtype of 'mask': {mask.dtype}")
    print("--- End Debugging Information ---")

    # --- Mask Checks and Preprocessing ---
    mask = measure.label(mask)  # Relabel the mask
    print("\n--- Mask Value Check (After Relabeling) ---")
    print("Unique values in mask:", np.unique(mask))
    assert np.issubdtype(mask.dtype, np.integer), "Mask must contain integer values"

    if not np.any(mask):
        print("Error: The mask contains only zeros. No objects to analyze.")
        return pd.DataFrame()

    # --- Image Checks and Preprocessing ---
    print("\n--- Image Value Check ---")
    print("Minimum image value:", np.min(image))
    print("Maximum image value:", np.max(image))
    print("Number of NaN values in image:", np.isnan(image).sum())
    print("Number of Inf values in image:", np.isinf(image).sum())

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, 65535.0)  # Adjust range as needed

    print("--- Image Value Check (After Cleaning) ---")
    print("Minimum image value:", np.min(image))
    print("Maximum image value:", np.max(image))
    print("Number of NaN values in image:", np.isnan(image).sum())
    print("Number of Inf values in image:", np.isinf(image).sum())

    # --- Properties for regionprops_table (WITHOUT integrated_intensity) ---
    properties = ('label', 'area', 'centroid', 'major_axis_length',
                  'minor_axis_length', 'eccentricity', 'solidity', 'extent',
                  'orientation', 'perimeter', 'mean_intensity',
                  'min_intensity', 'max_intensity', 'std_intensity')  # REMOVED integrated_intensity

    try:
        regions = measure.regionprops_table(mask, intensity_image=image,
                                            properties=properties)
        features_df = pd.DataFrame(regions)
        print(features_df.head())  # Print first few rows
    except Exception as e:  # Catch-all for errors during regionprops
        print(f"\n--- Error during regionprops_table: {e} ---")
        return pd.DataFrame()

    # --- MANUAL Integrated Intensity Calculation ---
    integrated_intensities = []
    for label in features_df['label']:
        object_mask = (mask == label)
        object_pixels = image[object_mask]
        integrated_intensity = np.sum(object_pixels)
        integrated_intensities.append(integrated_intensity)
    features_df['integrated_intensity'] = integrated_intensities


    # --- Morphology Ratios ---
    features_df['aspect_ratio'] = features_df['major_axis_length'] / features_df['minor_axis_length']
    features_df['form_factor'] = (4 * np.pi * features_df['area']) / (features_df['perimeter']**2)
    features_df['perimeter_area_ratio'] = features_df['perimeter'] / features_df['area']
    features_df['convexity'] = features_df['area'] / features_df['solidity']

    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    # --- Object Count ---
    object_count = len(features_df)
    print(f"Number of objects segmented and analyzed: {object_count}")
    features_df['object_count'] = object_count

    # --- Cell-Specific Analysis ---
    if cell_mask is not None:
        cell_labels = measure.label(cell_mask)
        cell_ids = np.unique(cell_labels[cell_labels > 0])
        cell_assignments = []

        for _, row in features_df.iterrows():
            object_centroid = tuple(int(c) for c in row[['centroid-0', 'centroid-1']].values)
            try:
                cell_label_at_centroid = cell_labels[object_centroid]
            except IndexError:
                cell_label_at_centroid = 0

            if cell_label_at_centroid > 0:
                cell_assignments.append(f"cell_{cell_label_at_centroid}")
            else:
                cell_assignments.append("outside_cell")

        features_df['cell_label'] = cell_assignments
    else:
        features_df['cell_label'] = "all_objects"

    # --- Rename Columns ---
    features_df.rename(columns={
        'major_axis_length': 'length',
        'minor_axis_length': 'width',
        'solidity': 'shape_solidity',
        'eccentricity': 'shape_eccentricity',
        'extent': 'shape_extent',
        'orientation': 'shape_orientation',
        'perimeter': 'object_perimeter',
        'mean_intensity': 'intensity_mean',
        'integrated_intensity': 'intensity_integrated',  # Keep this consistent
        'min_intensity': 'intensity_raw_min',
        'max_intensity': 'intensity_raw_max',
        'std_intensity': 'intensity_raw_std',
        'area': 'object_area'
    }, inplace=True)

    # --- Reorder Columns ---
    column_order = ['label', 'object_count', 'cell_label', 'object_area', 'object_perimeter', 'centroid-0', 'centroid-1',
                    'length', 'width', 'aspect_ratio', 'form_factor', 'perimeter_area_ratio', 'convexity',
                    'shape_solidity', 'shape_eccentricity', 'shape_extent', 'shape_orientation',
                    'intensity_raw_min', 'intensity_raw_max', 'intensity_raw_std', 'intensity_mean', 'intensity_integrated']

    for col in column_order:
        if col not in features_df.columns:
            features_df[col] = np.nan

    features_df = features_df[column_order]
    print("Enhanced feature extraction completed (or failed, check output).")

    return features_df

def display_results_in_napari_enhanced(viewer, cell_mask, object_mask, features_df):
    """
    Displays segmentation masks in Napari and prints feature summary.
    """
    print("\n--- Displaying Segmentation Masks in Napari ---")

    if cell_mask is not None:
        viewer.add_labels(cell_mask, name='cell_masks_display')
        print("'cell_masks_display' layer (cell segmentations) is displayed in Napari.")
    else:
        print("Note: No 'cell_masks' provided, cell segmentation display skipped.")

    viewer.add_labels(object_mask, name='object_masks_display')
    print("'object_masks_display' layer (object segmentations) is displayed in Napari.")

    print("\n--- Enhanced Feature Summary ---")
    print("Extracted enhanced features overview (first 5 objects):")
    print(features_df.head())
    print("\nFor detailed feature analysis, see 'object_features_enhanced.csv' file.")

def save_features_to_csv(features_df, output_filename="object_features_enhanced.csv"):
    """
    Saves the extracted features DataFrame to a CSV file.
    """
    print(f"Saving object features to CSV file: '{output_filename}'...")
    features_df.to_csv(output_filename, index=False)
    print(f"Features saved successfully to '{output_filename}'.")

def analyze_fluorescence_images_enhanced(image_path):
    """
    Analyzes fluorescence images.
    """
    print("--- Enhanced Fluorescence Image Analysis Pipeline ---")
    print("1. Napari will open with your image.")
    print("2. Use 'napari-segment-anything' to segment Cells and Objects.")
    print("3. Rename layers to 'cell_masks' and 'object_masks'.")
    print("4. Press Enter in the terminal to proceed.")

    viewer = napari.Viewer()
    try:
        original_image = io.imread(image_path)
        viewer.add_image(original_image, name='Original Image')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input("Press Enter after segmenting and renaming layers...")

    try:
        cell_masks_layer = viewer.layers['cell_masks']
        object_masks_layer = viewer.layers['object_masks']
        cell_segmentation_mask = cell_masks_layer.data
        object_segmentation_mask = object_masks_layer.data

        # --- CRITICAL DEBUGGING CHECKS HERE ---
        print("\n--- Checking Layer Names ---")
        print("Available layer names:", [layer.name for layer in viewer.layers])

        print("\n--- Checking Mask Data Types and Shapes ---")
        print(f"Type of cell_segmentation_mask: {type(cell_segmentation_mask)}, Shape: {cell_segmentation_mask.shape if cell_segmentation_mask is not None else None}, dtype: {cell_segmentation_mask.dtype if cell_segmentation_mask is not None else None}")
        print(f"Type of object_segmentation_mask: {type(object_segmentation_mask)}, Shape: {object_segmentation_mask.shape}, dtype: {object_segmentation_mask.dtype}")

        print("\n--- Checking Unique Values in Masks ---")
        print("Unique values in cell_segmentation_mask:", np.unique(cell_segmentation_mask) if cell_segmentation_mask is not None else None)
        print("Unique values in object_segmentation_mask:", np.unique(object_segmentation_mask))

        # --- Check if masks are all zeros ---
        if cell_segmentation_mask is not None and not np.any(cell_segmentation_mask):
            print("WARNING: cell_segmentation_mask contains only zeros.")
        if not np.any(object_segmentation_mask):
            print("WARNING: object_segmentation_mask contains only zeros.")


        viewer.add_labels(cell_segmentation_mask, name='cell_masks_display')
        viewer.add_labels(object_segmentation_mask, name='object_masks_display')


    except KeyError:
        print("Error: 'cell_masks' or 'object_masks' layer not found.  Make sure you renamed the layers correctly.")
        napari.run()
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        napari.run()
        return

    original_image = original_image.astype(np.float64)
    object_segmentation_mask = object_segmentation_mask.astype(int)
    if cell_segmentation_mask is not None:
        cell_segmentation_mask = cell_segmentation_mask.astype(int)

    features_df_enhanced = extract_enhanced_object_features(original_image, object_segmentation_mask, cell_segmentation_mask)
    if not features_df_enhanced.empty:
        save_features_to_csv(features_df_enhanced)

    display_results_in_napari_enhanced(viewer, cell_segmentation_mask, object_segmentation_mask, features_df_enhanced)
    print("Analysis complete. Results saved. Napari viewer remains open.")

    napari.run()


if __name__ == "__main__":
    image_path = r"C:\Users\Aegis-MSI\Videos\Mitochondria_Analysis\MAX_PreFA_MitoImg1_tRPM5_Female_08_12_24_Tastebud-1.tif"  # Replace with your image path!
    analyze_fluorescence_images_enhanced(image_path)