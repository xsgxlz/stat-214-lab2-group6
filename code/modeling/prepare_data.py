import os
import numpy as np

def get_img_mask(data):
    """
    Convert coordinate-based data into image and mask arrays.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array with shape (n_points, n_features) where first two columns are coordinates
        
    Returns:
    --------
    img : numpy.ndarray
        3D array containing image data with shape (x_size, y_size, n_features-2)
    mask : numpy.ndarray
        2D binary mask array with shape (x_size, y_size)
    """
    # Extract and normalize coordinates
    idx = data[:, 0:2].astype(int)
    idx = idx - idx.min(axis=0)
    
    x_coords = idx[:, 1]
    y_coords = idx[:, 0]
    
    # Calculate image dimensions
    x_size = x_coords.max() - x_coords.min() + 1
    y_size = y_coords.max() - y_coords.min() + 1
    
    # Create image and mask arrays
    img = np.zeros((x_size, y_size, data.shape[1] - 2))
    mask = np.zeros((x_size, y_size))
    
    # Fill arrays with data
    img[x_coords, y_coords] = data[:, 2:]
    mask[x_coords, y_coords] = 1
    
    return img, mask

def trans_to_array(data_path, save_path=None):
    """
    Process image data from .npz files into structured numpy arrays.
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing .npz files
    save_path : str, optional
        If provided, save the resulting arrays to this .npz file
        
    Returns:
    --------
    tuple : (unlabeled_images_array, unlabeled_masks_array, 
             labeled_images_array, labeled_masks_array, labels_array)
        Arrays containing processed image data
        
    Raises:
    -------
    ValueError
        If image data has invalid number of channels (not 8 or 9)
    """
    # Initialize storage lists
    unlabeled_images = []
    unlabeled_masks = []
    labeled_images = []
    labeled_masks = []
    labels = []
    
    # Process each file in directory
    files = os.listdir(data_path)
    for fn in files:
        fp = os.path.join(data_path, fn)
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        
        # Convert data to image and mask
        img, mask = get_img_mask(data)
        
        # Sort into labeled or unlabeled based on channel count
        if img.shape[2] == 9:
            labeled_images.append(img[:, :, :-1])  # All but last channel
            labeled_masks.append(mask)
            labels.append(img[:, :, -1])          # Last channel is label
        elif img.shape[2] == 8:
            unlabeled_images.append(img)
            unlabeled_masks.append(mask)
        else:
            raise ValueError(f"Invalid number of channels in image data: {img.shape[2]}")
    
    # Find maximum dimensions for padding
    x_shape_max = max([img.shape[0] for img in labeled_images + unlabeled_images])
    y_shape_max = max([img.shape[1] for img in labeled_images + unlabeled_images])
    
    # Create final arrays with padding
    n_unlabeled = len(unlabeled_images)
    n_labeled = len(labeled_images)
    
    # Initialize arrays for unlabeled data
    unlabeled_images_array = np.zeros((n_unlabeled, x_shape_max, y_shape_max, 8))
    unlabeled_masks_array = np.zeros((n_unlabeled, x_shape_max, y_shape_max))
    
    # Fill unlabeled arrays
    for i, (img, mask) in enumerate(zip(unlabeled_images, unlabeled_masks)):
        x_size, y_size, _ = img.shape
        unlabeled_images_array[i, :x_size, :y_size] = img
        unlabeled_masks_array[i, :x_size, :y_size] = mask
    
    # Initialize arrays for labeled data
    labeled_images_array = np.zeros((n_labeled, x_shape_max, y_shape_max, 8))
    labeled_masks_array = np.zeros((n_labeled, x_shape_max, y_shape_max))
    labels_array = np.zeros((n_labeled, x_shape_max, y_shape_max))
    
    # Fill labeled arrays
    for i, (img, mask, label) in enumerate(zip(labeled_images, labeled_masks, labels)):
        x_size, y_size, _ = img.shape
        labeled_images_array[i, :x_size, :y_size] = img
        labeled_masks_array[i, :x_size, :y_size] = mask
        labels_array[i, :x_size, :y_size] = label
    
    # Save results if path provided
    if save_path is not None:
        np.savez(save_path, 
                 unlabeled_images=unlabeled_images_array,
                 unlabeled_masks=unlabeled_masks_array,
                 labeled_images=labeled_images_array,
                 labeled_masks=labeled_masks_array,
                 labels=labels_array)
    
    return (unlabeled_images_array, unlabeled_masks_array,
            labeled_images_array, labeled_masks_array, labels_array)

if __name__ == "__main__":
    """
    Example usage of the script
    """
    # Example data path
    data_path = "/jet/home/azhang19/stat 214/stat-214-lab2-group6/data/image_data"
    
    # Process the data
    unlabeled_images, unlabeled_masks, labeled_images, labeled_masks, labels = trans_to_array(data_path)
    
    # Print array shapes for verification
    print(f"Unlabeled images shape: {unlabeled_images.shape}")
    print(f"Unlabeled masks shape: {unlabeled_masks.shape}")
    print(f"Labeled images shape: {labeled_images.shape}")
    print(f"Labeled masks shape: {labeled_masks.shape}")
    print(f"Labels shape: {labels.shape}")
