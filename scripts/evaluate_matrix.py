"""
evaluate_confusions.py

Loads JSON files from a directory structure like:
  C:/Users/ilias/Python/Thesis-Project/results/Results_Initial_1.0/evaluation_<emotion>_<timestamp>/metadata.json
Each JSON references a single 'true_emotion' and a 'prediction_counts' distribution across 8 classes.

Steps:
1) Gather all JSON files recursively.
2) For each file:
   - Parse 'true_emotion', 'accuracy', 'prediction_counts'.
   - Create a bar plot showing the predicted distribution (like a single row confusion matrix).
   - Collect everything into a single CSV row in 'aggregated_emotion_results.csv'
3) (Optional) If you want one 8x8 confusion matrix for all data combined, we can do it at the end.

Author: ChatGPT
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The main directory of results
RESULTS_DIR = r"C:\Users\ilias\Python\Thesis-Project\results\Results_Initial_1.0"
# Where to save aggregated results
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "aggregated_results_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We assume 8 classes, same order as your class_to_idx_8
ALL_EMOTIONS = ["Angry","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
emotion_to_idx = {emo:i for i,emo in enumerate(ALL_EMOTIONS)}


def load_all_json_metrics(base_dir):
    """
    Recursively find all .json files in 'base_dir' and parse them into a list of dicts.
    Each dict typically has keys like:
      'true_emotion', 'accuracy', 'prediction_counts' (which itself is a dict of emotions).
    """
    json_data_list = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, 'r') as jf:
                        data = json.load(jf)
                        # We'll store the 'full_path' for reference if needed
                        data["json_path"] = full_path
                        json_data_list.append(data)
                except Exception as e:
                    print(f"[WARNING] Error reading {full_path}: {e}")
    return json_data_list


def plot_confusion_row_for_json(data, output_dir=OUTPUT_DIR):
    """
    For a single JSON data dict (with 'true_emotion' and 'prediction_counts'), 
    create a bar plot of predictions distribution => a 'row' of the confusion matrix.
    We'll name the plot e.g. 'confusion_Angry.png' if true_emotion='Angry'.
    """
    true_emotion = data.get("true_emotion","Unknown")
    prediction_counts = data.get("prediction_counts", {})
    
    # We gather the predicted classes from 'prediction_counts' in the same order as ALL_EMOTIONS
    # If an emotion is missing from 'prediction_counts', treat it as 0.
    predicted_values = []
    for emo in ALL_EMOTIONS:
        pred_info = prediction_counts.get(emo, {"count": 0})
        predicted_values.append(pred_info["count"])
    
    total = sum(predicted_values) if predicted_values else 1
    
    # Bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(ALL_EMOTIONS, predicted_values, color='orange')
    plt.title(f"Confusion Row for True={true_emotion}\n(total={total})")
    plt.xlabel("Predicted Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    # Save figure
    fig_name = f"confusion_{true_emotion}.png"
    plt_path = os.path.join(output_dir, fig_name)
    plt.savefig(plt_path)
    plt.close()
    print(f"[INFO] Saved single-row confusion plot to {plt_path}")


def main():
    # 1) Load all JSON
    all_json_data = load_all_json_metrics(RESULTS_DIR)
    if not all_json_data:
        print("[INFO] No JSON files found. Exiting.")
        return
    
    print(f"[INFO] Found {len(all_json_data)} JSON metric files.")
    
    # 2) For each JSON dict, plot the single-row confusion matrix
    #    Also build a row for the aggregated CSV
    csv_rows = []
    for data in all_json_data:
        # basic info
        emotion = data.get("true_emotion","Unknown")
        accuracy = data.get("accuracy", 0.0)
        total_images = data.get("total_images", 0)
        time_stamp = data.get("timestamp", "")
        path = data.get("json_path","")
        
        # create bar plot of predictions distribution
        plot_confusion_row_for_json(data, output_dir=OUTPUT_DIR)
        
        # store row
        row = {
            "Emotion": emotion,
            "Accuracy": accuracy,
            "TotalImages": total_images,
            "Timestamp": time_stamp,
            "JSON_path": path
        }
        csv_rows.append(row)
    
    # 3) Save CSV summarizing all these results
    df = pd.DataFrame(csv_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "aggregated_emotion_results.csv")
    df.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote CSV summary to {summary_csv}")
    
    # 4) (OPTIONAL) If you want to combine everything into one global 8x8 confusion matrix:
    #    We'll build an 8x8 by reading the 'prediction_counts' from each data, 
    #    assigning row = emotion_to_idx[true_emotion], columns = distribution from 'prediction_counts'.
    
    # Initialize matrix
    combined_matrix = np.zeros((8,8), dtype=int)
    
    for data in all_json_data:
        true_emotion = data.get("true_emotion","Unknown")
        row_idx = emotion_to_idx.get(true_emotion, None)
        if row_idx is None:
            # skip unknown
            continue
        
        prediction_counts = data.get("prediction_counts", {})
        for emo_pred, info_dict in prediction_counts.items():
            col_idx = emotion_to_idx.get(emo_pred, None)
            if col_idx is not None:
                combined_matrix[row_idx, col_idx] += info_dict["count"]
    
    # Convert matrix to DataFrame
    cm_df = pd.DataFrame(combined_matrix, index=ALL_EMOTIONS, columns=ALL_EMOTIONS)
    # Save as CSV
    cm_csv_path = os.path.join(OUTPUT_DIR, "combined_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"[INFO] Saved combined 8x8 confusion matrix to {cm_csv_path}")

    # We can also do a quick heatmap if you want:
    plt.figure(figsize=(8,6))
    plt.imshow(combined_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Combined Confusion Matrix (all JSON)")
    plt.colorbar()
    tick_marks = np.arange(len(ALL_EMOTIONS))
    plt.xticks(tick_marks, ALL_EMOTIONS, rotation=45)
    plt.yticks(tick_marks, ALL_EMOTIONS)
    plt.tight_layout()
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    matrix_png = os.path.join(OUTPUT_DIR, "combined_confusion_matrix.png")
    plt.savefig(matrix_png)
    plt.close()
    print(f"[INFO] Also saved a heatmap of the combined confusion matrix to {matrix_png}")


if __name__ == "__main__":
    main()
