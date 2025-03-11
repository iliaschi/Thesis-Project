import os
import random
import shutil

def create_test_split(
    train_dir: str, 
    test_dir: str, 
    test_ratio: float = 0.1, 
    file_ext: tuple = ('.jpg', '.jpeg', '.png')
):
    """
    Randomly select a percentage of images from each subfolder in 'train_dir'
    and move them to a corresponding subfolder in 'test_dir'.

    Parameters
    ----------
    train_dir : str
        Path to the directory containing subfolders, e.g.:
        "Thesis-Project/data/synthetic/synth_train"
    
    test_dir : str
        Path to the directory where test images will be moved, e.g.:
        "Thesis-Project/data/synthetic/synth_test"
    
    test_ratio : float
        Fraction of files to move from each subfolder, e.g. 0.1 for 10%.
    
    file_ext : tuple
        A tuple of file extensions to consider as images. 
        Default is ('.jpg', '.jpeg', '.png').
    """
    # 1. Ensure the test_dir exists
    os.makedirs(test_dir, exist_ok=True)
    
    # 2. Iterate through each subfolder in the train directory
    subfolders = [f for f in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(train_dir, subfolder)
        
        # 2a. Gather all image files in the subfolder
        all_files = [
            f for f in os.listdir(subfolder_path) 
            if f.lower().endswith(file_ext)
        ]
        
        # 2b. Determine how many files to move (10% or "test_ratio" fraction)
        if len(all_files) == 0:
            print(f"No images found in: {subfolder_path}. Skipping.")
            continue
        
        num_to_move = int(len(all_files) * test_ratio)
        
        # 2c. Randomly sample that many files
        random.shuffle(all_files)
        files_to_move = all_files[:num_to_move]
        
        # 2d. Create the corresponding test subfolder name
        # For example, if subfolder == "angry_men_proc", 
        # name it "angry_men_test". Adjust as needed:
        if subfolder.endswith('_proc'):
            test_subfolder_name = subfolder.replace('_proc', '_test')
        else:
            # Or just append "_test" if you prefer
            test_subfolder_name = subfolder + "_test"
        
        test_subfolder_path = os.path.join(test_dir, test_subfolder_name)
        os.makedirs(test_subfolder_path, exist_ok=True)
        
        # 2e. Move the sampled files
        for filename in files_to_move:
            src_file = os.path.join(subfolder_path, filename)
            dst_file = os.path.join(test_subfolder_path, filename)
            
            # Move the file
            shutil.move(src_file, dst_file)
        
        print(f"Moved {len(files_to_move)} images from "
              f"{subfolder_path} to {test_subfolder_path}.")

def main():
    # Example usage
    base_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic"
    synth_train_dir = os.path.join(base_dir, "synth_train")
    synth_test_dir = os.path.join(base_dir, "synth_test")
    
    # Create a 10% test split
    create_test_split(train_dir=synth_train_dir, 
                      test_dir=synth_test_dir, 
                      test_ratio=0.1)

if __name__ == "__main__":
    main()
