"""
evaluate_confusions.py (updated)

Loads JSON files from a directory, each describing a single 'true_emotion'
plus 'prediction_counts', but now we differentiate subfolders like:
  'angry_m' or 'angry_w'
to record in our CSV that the 'gender' is 'M' or 'W'.

Steps:
  1) gather all .json files (like 'metadata.json'),
  2) parse 'folder' name from the JSON to detect whether it's men/women,
  3) plot a single bar chart row for the predicted distribution,
  4) store results in a CSV with columns: [Emotion, Gender, Accuracy, TotalImages, ...].

If you want a single 8x8 confusion matrix ignoring men/women for the entire dataset, we provide an optional step.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The main directory of results
RESULTS_DIR = r"C:\Users\ilias\Python\Thesis-Project\results_synthetic"
# Where to save aggregated results
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "aggregated_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_EMOTIONS = ["Angry","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
emotion_to_idx = {emo: i for i,emo in enumerate(ALL_EMOTIONS)}

def load_all_json_metrics(base_dir):
    """
    Recursively find all .json files in 'base_dir' and parse them into a list of dicts.
    Each dict typically has:
      'true_emotion', 'accuracy', 'prediction_counts', 'folder' (the path),
      'json_path' => we add ourselves for reference.
    """
    json_data_list = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, 'r') as jf:
                        data = json.load(jf)
                        data["json_path"] = full_path
                        json_data_list.append(data)
                except Exception as e:
                    print(f"[WARNING] Error reading {full_path}: {e}")
    return json_data_list

def extract_emotion_and_gender(folder_path):
    """
    Attempt to parse the 'folder' field, which might look like:
      'C:\\...\\angry_m' or 'C:\\...\\angry_w' or 'C:\\...\\disgust_m' etc.
    We'll return (emotion, gender).

    If we can't parse, default to (Unknown, Unknown).

    We'll do simple logic:
      - if folder_path contains 'angry_m' => (Angry, M)
      - if folder_path ends with '_m' => gender=M
      - or ends with '_w' => gender=W
    etc.

    For a robust approach, you can do more advanced parsing.
    """
    folder_name = os.path.basename(folder_path)  # e.g. 'angry_m' or 'disgust_w'
    folder_name_lower = folder_name.lower()

    # Default
    emotion, gender = "Unknown", "Unknown"

    # 1) Identify gender from suffix
    if folder_name_lower.endswith("_m"):
        gender = "M"
        # remove the _m to parse emotion
        base = folder_name_lower[:-2]  # 'angry'
    elif folder_name_lower.endswith("_w"):
        gender = "W"
        base = folder_name_lower[:-2]  # e.g. 'angry'
    else:
        # no _m or _w in the last 2 chars, keep entire
        base = folder_name_lower

    # 2) Identify emotion by matching base to known or partial
    # A simple approach: if 'angr' in base => 'Angry' etc.
    # or you can do exact matches. We'll do partial matches:
    if "angry" in base:
        emotion = "Angry"
    elif "contempt" in base:
        emotion = "Contempt"
    elif "disgust" in base:
        emotion = "Disgust"
    elif "fear" in base:
        emotion = "Fear"
    elif "happy" in base:
        emotion = "Happiness"
    elif "neutr" in base:
        emotion = "Neutral"
    elif "sad" in base:
        emotion = "Sadness"
    elif "surpris" in base:
        emotion = "Surprise"

    return (emotion, gender)

def plot_confusion_row_for_json(data, output_dir=OUTPUT_DIR, label_suffix=""):
    """
    For a single JSON data dict, create a bar plot of 'prediction_counts'.
    We'll label the figure with true_emotion + optional gender.

    label_suffix might be e.g. 'M' or 'W'.
    """
    true_emotion = data.get("true_emotion","Unknown")
    pred_counts  = data.get("prediction_counts", {})
    
    # build array in order ALL_EMOTIONS
    pred_array = []
    for emo in ALL_EMOTIONS:
        info = pred_counts.get(emo, {"count": 0})
        pred_array.append(info["count"])
    
    total = sum(pred_array) if len(pred_array) else 1
    
    # bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(ALL_EMOTIONS, pred_array, color='orange')
    plt.title(f"True={true_emotion}{label_suffix}  (n={total})")
    plt.xlabel("Predicted Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    fig_name = f"confusion_{true_emotion}{label_suffix}.png"
    plt_path = os.path.join(output_dir, fig_name)
    plt.savefig(plt_path)
    plt.close()
    print(f"[INFO] Saved single-row confusion to {plt_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = load_all_json_metrics(RESULTS_DIR)
    if not all_data:
        print("[INFO] No JSON files found.")
        return

    print(f"[INFO] Found {len(all_data)} JSON metrics.")
    
    # We'll also create a single 8x8 matrix if desired
    combined_cm = np.zeros((8,8), dtype=int)

    # For CSV
    aggregated_rows = []
    
    for data in all_data:
        folder_path = data.get("folder","")
        accuracy    = data.get("accuracy",0.0)
        total_imgs  = data.get("total_images",0)
        time_stamp  = data.get("timestamp","")
        pred_counts = data.get("prediction_counts",{})
        
        # parse out emotion, gender
        emotion, gender = extract_emotion_and_gender(folder_path)
        data["true_emotion"] = emotion  # override if the JSON had something else
        label_suffix = f"_{gender}" if gender!="Unknown" else ""
        
        # plot single row confusion
        plot_confusion_row_for_json(data, OUTPUT_DIR, label_suffix=label_suffix)
        
        # Save CSV row
        row = {
            "Emotion": emotion,
            "Gender": gender,
            "Accuracy": accuracy,
            "TotalImages": total_imgs,
            "Timestamp": time_stamp,
            "Folder": folder_path,
        }
        aggregated_rows.append(row)

        # Optionally fill global 8x8 confusion
        row_idx = emotion_to_idx.get(emotion,None)
        if row_idx is not None:
            # accumulate
            for emo_pred, info in pred_counts.items():
                col_idx = emotion_to_idx.get(emo_pred, None)
                if col_idx is not None:
                    combined_cm[row_idx,col_idx] += info["count"]

    # Save aggregated CSV
    df = pd.DataFrame(aggregated_rows)
    csv_path = os.path.join(OUTPUT_DIR, "aggregated_emotion_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote CSV to {csv_path}")

    # (Optional) Save combined 8x8 confusion ignoring men/women
    cm_df = pd.DataFrame(combined_cm, index=ALL_EMOTIONS, columns=ALL_EMOTIONS)
    cm_csv = os.path.join(OUTPUT_DIR, "combined_confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    print(f"[INFO] Wrote combined confusion matrix CSV to {cm_csv}")

    # also do a quick heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.imshow(combined_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("All Data Combined (Ignoring M/W)")
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
    print(f"[INFO] Also saved heatmap to {matrix_png}")

if __name__ == "__main__":
    main()
