import os
import shutil

def gather_for_latex_report(base_folder):
    """
    Create a 'latex_report' folder inside `base_folder`.
    Copy relevant files from each 'evaluation_*' subfolder:
      - single_confusion_matrix.png
      - reliability_diagram.png
      - results_*.csv
      - classification_report.json
      - metadata.json
    Also copy the final combined confusion matrix if present in the base_folder.
    """
    
    # 1) Create the latex_report folder if not exists
    latex_dir = os.path.join(base_folder, "latex_report")
    os.makedirs(latex_dir, exist_ok=True)
    print(f"[INFO] Created (or found) latex_report folder: {latex_dir}")
    
    # 2) For each subfolder named evaluation_*, gather the relevant files
    for item in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, item)
        if os.path.isdir(subfolder_path) and item.startswith("evaluation_"):
            # inside subfolder, we might have single_confusion_matrix, reliability_diagram, CSV, etc.
            for filename in os.listdir(subfolder_path):
                # We decide which files to copy
                if (filename.endswith(".png") and ("single_confusion_matrix" in filename or "reliability_diagram" in filename)) \
                   or (filename.startswith("results_") and filename.endswith(".csv")) \
                   or filename in ["metadata.json", "classification_report.json"]:
                    
                    src_path = os.path.join(subfolder_path, filename)
                    dst_path = os.path.join(latex_dir, f"{item}_{filename}") 
                    # rename the file by prefixing the subfolder name so they're unique
                    shutil.copy2(src_path, dst_path)
                    print(f"[INFO] Copied {src_path} => {dst_path}")
    
    # 3) Also check if combined_confusion_matrix.* or aggregated_results.* are in the base_folder
    #    so we can copy them to latex_report as well
    possible_main_files = [
        "combined_confusion_matrix.png", "combined_confusion_matrix.csv",
        "aggregated_results.csv", "aggregated_emotion_results.csv"
    ]
    for file_check in possible_main_files:
        main_file_path = os.path.join(base_folder, file_check)
        if os.path.exists(main_file_path):
            # copy to latex_report
            dst_main = os.path.join(latex_dir, file_check)
            shutil.copy2(main_file_path, dst_main)
            print(f"[INFO] Copied {main_file_path} => {dst_main}")
    
    print("[INFO] Gathering completed. All files ready in:", latex_dir)

def main():
    
        # e.g. "C:\\Users\\ilias\\Python\\Thesis-Project\\results\\Results_2.0\\finetuned_3_results_20250314_025819"
    
    base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\RAFDB_results_20250314_004534" # RAFDB with pretrained
    # base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\synth_results_20250314_020656" # synthetic tested with pretrained
    # base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\synth_finetuned_3_results_20250314_025953" # synthetic tested with fine tuned

    # base_folder = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\finetuned_3_results_20250314_025819"
    gather_for_latex_report(base_results_dir)

if __name__ == "__main__":
    main()
