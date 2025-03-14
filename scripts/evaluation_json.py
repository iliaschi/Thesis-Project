import os
import json
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def gather_metrics_from_report(json_path):
    """
    Loads 'classification_report.json' from 'json_path'
    and returns a dictionary of key metrics for the primary emotion 
    (extracted from the folder name or the 'true_emotion' in the JSON).
    """
    with open(json_path, 'r') as f:
        report_data = json.load(f)
    
    # The file has structure for all 8 classes plus 'accuracy', 'macro avg', etc.
    # Typically you want:
    # - overall accuracy => report_data['accuracy']
    # - for the main emotion (like 'Angry'), parse 'precision','recall','f1-score','false positives'
    # 
    # We'll guess the folder name from the parent directory. You can also store it in the JSON, 
    # or parse from the 'true_emotion' if you included that. Adjust as needed.

    # Sample approach: 
    # if you already have e.g. 'Angry' dictionary for that folder's main class:
    # main_emotion_metrics = report_data.get("Angry", {})
    # Or parse from 'report_data['true_emotion']' if you stored it in JSON 
    # but your snippet doesn't show a top-level 'true_emotion'. 
    # We'll do a fallback approach.

    # We'll read the 'accuracy' from the top-level:
    overall_accuracy = report_data.get("accuracy", 0.0)
    
    # Identify which emotion actually has support > 0 in this folder 
    # (since the entire folder might be 'Angry' or 'Disgust', etc.)
    main_emotion = None
    for cls_name, cls_metrics in report_data.items():
        # skip these dictionary keys
        if cls_name in ["accuracy","macro avg","weighted avg"]:
            continue
        if isinstance(cls_metrics, dict):
            support = cls_metrics.get("support", 0)
            if support > 0:
                main_emotion = cls_name
                break

    # If we found an emotion with support:
    if main_emotion is None:
        main_emotion = "Unknown"
        main_metrics = {}
    else:
        main_metrics = report_data.get(main_emotion, {})

    precision = main_metrics.get("precision", 0.0)
    recall = main_metrics.get("recall", 0.0)
    f1score = main_metrics.get("f1-score", 0.0)
    fp = main_metrics.get("false positives", 0)

    # Return in a standard format
    return {
        "emotion": main_emotion,
        "accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "false_positives": fp
    }

def main():
    # The top-level results directory 
    # e.g. "C:\\Users\\ilias\\Python\\Thesis-Project\\results\\Results_2.0\\finetuned_3_results_20250314_025819"
    # base_results_dir = r"C:\\Users\\ilias\\Python\\Thesis-Project\\results\\Results_2.0\\finetuned_3_results_20250314_025819" # RafDB with fine tuned
    # base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\RAFDB_results_20250314_004534" # RAFDB with pretrained
    # base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\synth_results_20250314_020656" # synthetic tested with pretrained
    base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\synth_finetuned_3_results_20250314_025953" # synthetic tested with fine tuned

    # We'll scan subfolders named 'evaluation_angry_6_2025...' or similar
    # and look for classification_report.json in each.
    rows = []
    for folder_name in os.listdir(base_results_dir):
        subfolder_path = os.path.join(base_results_dir, folder_name)
        if not os.path.isdir(subfolder_path):
            continue
        
        # The classification_report.json path
        json_path = os.path.join(subfolder_path, "classification_report.json")
        if os.path.exists(json_path):
            # parse the metrics
            metrics = gather_metrics_from_report(json_path)
            
            # Also let's store a guess about the subfolder 
            # to help identify which emotion it's for
            # e.g. 'evaluation_angry_6_20250314_025819'
            # parse the folder name from that
            folder_emotion_match = re.match(r'evaluation_([a-zA-Z]+)_?\d*_?\d*', folder_name)
            found_emotion = folder_emotion_match.group(1) if folder_emotion_match else 'unknown'
            
            # Add subfolder & found emotion to the row
            row = {
                "folder": folder_name,
                "parsed_emotion": found_emotion,
                "reported_emotion": metrics["emotion"],  # from the JSON
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1score": metrics["f1score"],
                "false_positives": metrics["false_positives"],
            }
            rows.append(row)
            
            # Also check if confusion matrix or reliability diagram is there:
            cm_path = os.path.join(subfolder_path, "single_confusion_matrix.png")
            rel_path = os.path.join(subfolder_path, "reliability_diagram.png")
            if os.path.exists(cm_path):
                print(f"[INFO] Found confusion matrix in: {cm_path}")
            if os.path.exists(rel_path):
                print(f"[INFO] Found reliability diagram in: {rel_path}")
        else:
            print(f"[WARNING] No classification_report.json in {subfolder_path}")

    # Summarize rows in a CSV
    df = pd.DataFrame(rows)
    out_csv = os.path.join(base_results_dir, "aggregated_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote aggregated CSV to {out_csv}")


### conf matrix all
    base_results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\finetuned_3_results_20250314_025819"
    
    # Suppose we have the same 'class_to_idx' in each subfolder or you know the fixed order
    # E.g. 'Angry'=0, 'Contempt'=1, 'Disgust'=2, 'Fear'=3, 'Happiness'=4, 'Neutral'=5, 'Sadness'=6, 'Surprise'=7
    class_names = ["Angry","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
    class_to_idx = {cls: i for i,cls in enumerate(class_names)}

    # We'll accumulate a single 8x8 matrix
    confusion_matrix_8x8 = np.zeros((8,8), dtype=int)

    # 1) Walk through each subfolder like 'evaluation_angry_6_20250314_025819'
    for folder_name in os.listdir(base_results_dir):
        subfolder_path = os.path.join(base_results_dir, folder_name)
        if not os.path.isdir(subfolder_path):
            continue
        
        # 2) We want the metadata.json
        meta_path = os.path.join(subfolder_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            # 'true_emotion': e.g. 'Angry'
            true_emo = meta_data.get("true_emotion","Unknown")
            # get row index for the confusion matrix
            row_idx = class_to_idx.get(true_emo, None)
            if row_idx is None:
                print(f"[WARNING] The true_emotion='{true_emo}' not recognized in class_to_idx.")
                continue
            
            # 'prediction_counts': for each predicted emotion => {count, percentage}
            pred_counts = meta_data.get("prediction_counts", {})
            for pred_emo, info in pred_counts.items():
                col_idx = class_to_idx.get(pred_emo, None)
                if col_idx is None:
                    continue
                count_val = info.get("count", 0)
                
                # fill the single row in the matrix
                confusion_matrix_8x8[row_idx, col_idx] += count_val

        else:
            print(f"[WARNING] No metadata.json in {subfolder_path}")

    # 3) Now confusion_matrix_8x8 is built. Let's plot it in a heatmap.
    plt.figure(figsize=(8,6))
    sns.heatmap(
        confusion_matrix_8x8, 
        annot=True, 
        fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title("Combined 8x8 Confusion Matrix (All Folders)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    out_png = os.path.join(base_results_dir, "combined_confusion_matrix.png")
    plt.savefig(out_png)
    plt.close()
    print(f"[INFO] Saved combined confusion matrix to: {out_png}")
    
    # 4) Optionally print or store numeric matrix as CSV
    out_csv = os.path.join(base_results_dir, "combined_confusion_matrix.csv")
    np.savetxt(out_csv, confusion_matrix_8x8, fmt='%d', delimiter=',')
    print(f"[INFO] Also saved matrix as CSV to: {out_csv}")


if __name__ == "__main__":
    main()
