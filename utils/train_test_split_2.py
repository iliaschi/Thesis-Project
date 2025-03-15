import os
import random
import shutil
from tqdm import tqdm
import numpy as np

def create_test_split(
    source_dir: str, 
    train_dir: str,
    test_dir: str, 
    test_ratio: float = 0.2,
    seed: int = 42,
    file_ext: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
):
    """
    Randomly select a percentage of images from each emotion subfolder in the source directory
    and copy them to train and test directories while maintaining the folder structure.

    Parameters
    ----------
    source_dir : str
        Path to the source directory containing processed images, e.g.:
        "Thesis-Project/data/synthetic/processed"
    
    train_dir : str
        Path to the directory where training images will be copied, e.g.:
        "Thesis-Project/data/synthetic/train"
    
    test_dir : str
        Path to the directory where test images will be copied, e.g.:
        "Thesis-Project/data/synthetic/test"
    
    test_ratio : float
        Fraction of files to allocate to the test set, e.g. 0.2 for 20%.
        
    seed : int
        Random seed for reproducibility.
    
    file_ext : tuple
        A tuple of file extensions to consider as images.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Ensure the train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Expected structure: source_dir/gender/emotion
    gender_folders = ['men', 'women']
    
    # Get list of emotion folders (assuming both men and women have the same emotions)
    emotion_folders = []
    for gender in gender_folders:
        gender_path = os.path.join(source_dir, gender)
        if os.path.exists(gender_path):
            # Get emotions from first gender folder that exists
            emotion_folders = [f for f in os.listdir(gender_path) 
                              if os.path.isdir(os.path.join(gender_path, f))]
            break
    
    if not emotion_folders:
        print(f"No emotion folders found in {source_dir}")
        return
    
    print(f"Found emotions: {emotion_folders}")
    
    # Create emotion folders in train and test directories for each gender
    for gender in gender_folders:
        for emotion in emotion_folders:
            os.makedirs(os.path.join(train_dir, gender, emotion), exist_ok=True)
            os.makedirs(os.path.join(test_dir, gender, emotion), exist_ok=True)
    
    # Statistics to track
    stats = {
        'total_images': 0,
        'train_images': 0,
        'test_images': 0
    }
    
    # Process each gender and emotion folder
    for gender in gender_folders:
        gender_path = os.path.join(source_dir, gender)
        if not os.path.exists(gender_path):
            print(f"Warning: {gender_path} does not exist. Skipping.")
            continue
        
        for emotion in tqdm(emotion_folders, desc=f"Processing {gender}"):
            emotion_path = os.path.join(gender_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} does not exist. Skipping.")
                continue
            
            # Get all image files
            all_files = [f for f in os.listdir(emotion_path) 
                        if f.lower().endswith(file_ext)]
            
            if not all_files:
                print(f"No images found in {emotion_path}. Skipping.")
                continue
            
            # Shuffle files
            random.shuffle(all_files)
            
            # Calculate split
            num_test = int(len(all_files) * test_ratio)
            test_files = all_files[:num_test]
            train_files = all_files[num_test:]
            
            # Copy files to train directory
            for file in train_files:
                src_file = os.path.join(emotion_path, file)
                dst_file = os.path.join(train_dir, gender, emotion, file)
                shutil.copy2(src_file, dst_file)
                stats['train_images'] += 1
            
            # Copy files to test directory
            for file in test_files:
                src_file = os.path.join(emotion_path, file)
                dst_file = os.path.join(test_dir, gender, emotion, file)
                shutil.copy2(src_file, dst_file)
                stats['test_images'] += 1
            
            stats['total_images'] += len(all_files)
            
            print(f"Split {emotion} {gender}: {len(train_files)} train, {len(test_files)} test")
    
    print("\nData splitting complete:")
    print(f"Total images: {stats['total_images']}")
    print(f"Train images: {stats['train_images']} ({stats['train_images']/stats['total_images']*100:.1f}%)")
    print(f"Test images: {stats['test_images']} ({stats['test_images']/stats['total_images']*100:.1f}%)")
    
    return stats

def main():
    # Base directory containing processed data
    base_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic"
    
    # Source directory with processed images
    source_dir = os.path.join(base_dir, "processed_python_2.0")
    
    # Output directories for train and test splits
    train_dir = os.path.join(base_dir, "synth_train")
    test_dir = os.path.join(base_dir, "synth_test")
    
    # Create the train/test split with 20% test data
    stats = create_test_split(
        source_dir=source_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        test_ratio=0.1,
        seed=42  # Set a random seed for reproducibility
    )

if __name__ == "__main__":
    main()