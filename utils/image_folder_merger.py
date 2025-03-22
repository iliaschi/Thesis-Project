import os
import shutil

def combine_images(parent_folder, destination_folder):
    """
    Combines images from subfolders of parent_folder into destination_folder.
    Images are renamed to include the subfolder name and a running counter.
    
    Parameters:
        parent_folder (str): The path to the parent folder (e.g., 'synth_100').
        destination_folder (str): The folder where combined images will be stored.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    # List all subdirectories (assumed to be emotion categories)
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        count = 1  # Counter for each emotion subfolder
        # Process image files with common image extensions.
        for filename in sorted(os.listdir(subfolder_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src_path = os.path.join(subfolder_path, filename)
                ext = os.path.splitext(filename)[1]
                new_filename = f"{subfolder}_{count}{ext}"
                dst_path = os.path.join(destination_folder, new_filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} -> {dst_path}")
                count += 1

if __name__ == "__main__":
    # Change these paths as needed.
    parent_folder = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_merged"
    destination_folder = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_combined"
    
    combine_images(parent_folder, destination_folder)
    print("Image combination complete!")
