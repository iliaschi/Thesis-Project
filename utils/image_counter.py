#!/usr/bin/env python3

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def count_images_in_subfolders(base_dir):
    """
    Scans all subfolders (assumed to be emotion labels) in 'base_dir'
    and returns a dictionary mapping emotion -> number_of_images.
    """
    emotion_counts = {}
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # Update as needed
    
    for emotion_name in os.listdir(base_dir):
        emotion_path = os.path.join(base_dir, emotion_name)
        if not os.path.isdir(emotion_path):
            continue  # Skip files, only process directories
        
        # Count image files inside this emotion folder
        all_files = os.listdir(emotion_path)
        image_files = [
            f for f in all_files 
            if f.lower().endswith(valid_extensions)
        ]
        emotion_counts[emotion_name] = len(image_files)
    
    return emotion_counts

def main():
    # 1) Adjust these paths to match your folder structure:
    # e.g. synthetic_data/men/<emotion>, synthetic_data/women/<emotion>
    synthetic_base_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\processed_python_2.0\synth_train"
    men_dir   = os.path.join(synthetic_base_dir, "men")
    women_dir = os.path.join(synthetic_base_dir, "women")
    
    # 2) Count images in each emotion folder for men and women
    men_counts   = count_images_in_subfolders(men_dir)
    women_counts = count_images_in_subfolders(women_dir)
    
    # Gather all unique emotion names from both dicts
    all_emotions = sorted(set(men_counts.keys()) | set(women_counts.keys()))
    
    # 3) Build a DataFrame for combined analysis
    data_records = []
    for emotion in all_emotions:
        count_m = men_counts.get(emotion, 0)
        count_w = women_counts.get(emotion, 0)
        data_records.append({
            "emotion": emotion,
            "men_count": count_m,
            "women_count": count_w
        })
    df = pd.DataFrame(data_records)
    
    # 4) Add a "Total" row at the bottom
    men_total   = df["men_count"].sum()
    women_total = df["women_count"].sum()
    df.loc[len(df)] = ["Total", men_total, women_total]
    
    # 5) Create a bar plot to compare men vs. women per emotion (excluding "Total" row)
    #    We'll melt the DataFrame for Seaborn, ignoring the last row if "emotion" == "Total".
    df_filtered = df[df["emotion"] != "Total"].copy()
    df_melt = df_filtered.melt(id_vars="emotion", 
                               value_vars=["men_count","women_count"],
                               var_name="group", 
                               value_name="count")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="emotion", y="count", hue="group")
    plt.title("Image Count per Emotion (Men vs. Women)")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # 6) Save and show the plot
    output_plot = "synthetic_data_barplot.png"
    plt.savefig(output_plot, dpi=300)
    plt.show()
    print(f"[INFO] Bar plot saved to {output_plot}")
    
    # 7) Write a LaTeX table summarizing the counts (including the 'Total' row)
    latex_table_str = df.to_latex(
        index=False,
        columns=["emotion","men_count","women_count"],
        header=["Emotion","Men","Women"],
        caption="Number of images per emotion (Men and Women), including a total count.",
        label="tab:synthetic_data_counts"
    )
    
    latex_file = "synthetic_data_counts.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_table_str)
    
    print(f"[INFO] LaTeX table saved to {latex_file}")

if __name__ == "__main__":
    main()



# def main():
#     # 1) Adjust these paths to your actual folder structure.
#     # For example: synthetic_base_dir/men/<emotion> and synthetic_base_dir/women/<emotion>
#     synthetic_base_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\processed_python_2.0\synth_train"
#     men_dir = os.path.join(synthetic_base_dir, "men")
#     women_dir = os.path.join(synthetic_base_dir, "women")
    
#     # 2) Count images in each emotion folder for men and women
#     men_counts   = count_images_in_subfolders(men_dir)
#     women_counts = count_images_in_subfolders(women_dir)
    
#     # 3) Build a DataFrame for plotting and analysis
#     data_records = []
    
#     # Convert men's dictionary into a list of dicts
#     for emotion, count_m in men_counts.items():
#         data_records.append({
#             "emotion": emotion,
#             "count": count_m,
#             "group": "men"
#         })
    
#     # Convert women's dictionary into a list of dicts
#     for emotion, count_w in women_counts.items():
#         data_records.append({
#             "emotion": emotion,
#             "count": count_w,
#             "group": "women"
#         })
    
#     df = pd.DataFrame(data_records)
    
#     # 4) Create a bar plot to compare men vs. women per emotion
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=df, x="emotion", y="count", hue="group")
#     plt.title("Image Count per Emotion (Men vs. Women)")
#     plt.xlabel("Emotion")
#     plt.ylabel("Number of Images")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
    
#     # 5) Save the plot to a file and/or display
#     output_plot = "synthetic_data_barplot.png"
#     plt.savefig(output_plot, dpi=300)
#     print(f"[INFO] Bar plot saved to {output_plot}")
#     plt.show()

# if __name__ == "__main__":
#     main()
