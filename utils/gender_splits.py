import os
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_gender_splits(base_dir, emotions, output_dir, seed=42):
    """
    Create gender-based splits for emotion recognition testing
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing processed emotion folders for men and women
    emotions : list
        List of emotion names to process
    output_dir : str
        Directory to save the splits
    seed : int
        Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the split ratios to create
    splits = {
        '100M_100W': (1.0, 1.0),  # 100% men, 100% women
        '100M_75W': (1.0, 0.75),   # 100% men, 75% women
        '100M_50W': (1.0, 0.5),    # 100% men, 50% women
        '100M_25W': (1.0, 0.25),   # 100% men, 25% women
        '100M_0W': (1.0, 0.0),     # 100% men, 0% women
        '100W_100M': (1.0, 1.0),   # 100% women, 100% men (same as 100M_100W)
        '100W_75M': (0.75, 1.0),   # 75% men, 100% women
        '100W_50M': (0.5, 1.0),    # 50% men, 100% women
        '100W_25M': (0.25, 1.0),   # 25% men, 100% women
        '100W_0M': (0.0, 1.0),     # 0% men, 100% women
    }
    
    # Create directories for each split
    for split_name in splits.keys():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create emotion directories within each split
        for emotion in emotions:
            os.makedirs(os.path.join(split_dir, emotion), exist_ok=True)
    
    # Log file to track what we're doing
    log_file = os.path.join(output_dir, "split_details.csv")
    log_entries = []
    
    # Process each emotion
    # for emotion in tqdm(emotions, desc="Processing emotions"):
    #     # Correct folder naming pattern - make sure these match your actual folder names
    #     men_dir = os.path.join(base_dir, f"{emotion}")
    #     women_dir = os.path.join(base_dir, f"{emotion}")
        
    #     # Skip if directories don't exist
    #     if not os.path.exists(men_dir):
    #         print(f"Warning: Men directory for {emotion} not found at {men_dir}. Skipping.")
    #         continue
            
    #     if not os.path.exists(women_dir):
    #         print(f"Warning: Women directory for {emotion} not found at {women_dir}. Skipping.")
    #         continue
    for emotion in tqdm(emotions, desc="Processing emotions"):
        # Corrected folder structure based on your actual paths
        men_dir = os.path.join(base_dir, "men", emotion)
        women_dir = os.path.join(base_dir, "women", emotion)
        
        # Skip if directories don't exist
        if not os.path.exists(men_dir):
            print(f"Warning: Men directory for {emotion} not found at {men_dir}. Skipping.")
            continue
            
        if not os.path.exists(women_dir):
            print(f"Warning: Women directory for {emotion} not found at {women_dir}. Skipping.")
            continue

        # Get all image files
        men_images = [f for f in os.listdir(men_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        women_images = [f for f in os.listdir(women_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(men_images)} men and {len(women_images)} women images for {emotion}")
        
        # Process each split
        for split_name, (men_ratio, women_ratio) in splits.items():
            split_dir = os.path.join(output_dir, split_name, emotion)
            
            # Determine how many images to include
            num_men = int(len(men_images) * men_ratio)
            num_women = int(len(women_images) * women_ratio)
            
            # Handle edge case: if ratio is not 0 but would result in 0 images due to rounding
            if men_ratio > 0 and num_men == 0 and len(men_images) > 0:
                num_men = 1
            if women_ratio > 0 and num_women == 0 and len(women_images) > 0:
                num_women = 1
            
            # Randomly select images based on ratios
            selected_men = random.sample(men_images, num_men) if num_men > 0 else []
            selected_women = random.sample(women_images, num_women) if num_women > 0 else []
            
            # Copy selected images to the split directory
            for img in selected_men:
                src = os.path.join(men_dir, img)
                dst = os.path.join(split_dir, f"m_{img}")  # Add prefix to avoid name collisions
                shutil.copy2(src, dst)
            
            for img in selected_women:
                src = os.path.join(women_dir, img)
                dst = os.path.join(split_dir, f"w_{img}")  # Add prefix to avoid name collisions
                shutil.copy2(src, dst)
            
            # Log this split
            log_entries.append({
                'split': split_name,
                'emotion': emotion,
                'men_total': len(men_images),
                'women_total': len(women_images),
                'men_selected': num_men,
                'women_selected': num_women,
                'men_ratio': men_ratio,
                'women_ratio': women_ratio,
                'total_images': num_men + num_women
            })
    
    # Save the log
    pd.DataFrame(log_entries).to_csv(log_file, index=False)
    print(f"Split details saved to {log_file}")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "split_summary.csv")
    summary_data = []
    
    for split_name in splits.keys():
        split_dir = os.path.join(output_dir, split_name)
        total_images = 0
        
        # Count images per emotion in this split
        emotion_counts = {}
        for emotion in emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            if os.path.exists(emotion_dir):
                count = len([f for f in os.listdir(emotion_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                emotion_counts[emotion] = count
                total_images += count
        
        # Add to summary
        summary_data.append({
            'split': split_name,
            'total_images': total_images,
            **emotion_counts
        })
    
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"Split summary saved to {summary_file}")

def main():
    # Direct input variables instead of using argparse
    base_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_test"
    output_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\test_splits"
    seed = 42
    
    # List of emotions to process
    emotions = [
        'angry', 'contempt','disgust', 'fear', 'happy', 
        'neutral', 'sad', 'surprised'
    ]
    
    create_gender_splits(
        base_dir=base_dir,
        emotions=emotions,
        output_dir=output_dir,
        seed=seed
    )

if __name__ == "__main__":
    main()