"""
evaluate_metrics.py

Loads JSON metric files from a directory structure like:
C:/Users/ilias/Python/Thesis-Project/results/Results_Initial_1.0/<evaluation_emotion_timestamp>/metadata.json

Then collects each file’s accuracy, emotion, etc., and optionally plots or saves a CSV summary.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1) Directory of results
results_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_Initial_1.0"

def load_metrics(results_dir):
    """
    Recursively walk through 'results_dir', look for any .json files,
    and parse them into a list of dictionaries.
    """
    metrics_list = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                # Load the JSON
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        # data is a dict with keys like 'accuracy', 'true_emotion', etc.
                        metrics_list.append(data)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return metrics_list

def plot_accuracy_per_emotion(metrics_list, output_dir=None):
    """
    Plot accuracy per emotion from the loaded metric dicts.
    Output a PNG if output_dir is provided.
    """
    emotions = []
    accuracies = []
    
    for metric in metrics_list:
        # We assume each JSON has 'true_emotion' and 'accuracy'
        emotions.append(metric.get("true_emotion", "Unknown"))
        accuracies.append(metric.get("accuracy", 0.0))
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(emotions, accuracies, color='skyblue')
    plt.title("Accuracy per Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Accuracy (%)")
    plt.ylim([0, 100])
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "accuracy_per_emotion.png")
        plt.savefig(plot_path)
        print(f"Saved accuracy plot to {plot_path}")
    else:
        plt.show()

def save_metrics_to_csv(metrics_list, output_dir=None):
    """
    Summarize the relevant metrics into a CSV file.
    We gather each JSON's 'true_emotion', 'accuracy', etc.
    """
    data_rows = []
    for metric in metrics_list:
        row = {
            "Emotion": metric.get("true_emotion", "Unknown"),
            "Accuracy": metric.get("accuracy", 0.0),
            "TotalImages": metric.get("total_images", 0),
            "ProcessingTime": metric.get("processing_time", 0.0),
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "emotion_metrics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics summary to {csv_path}")
    else:
        print(df)

def main(results_dir):
    # 1) Load metric data from all subfolders
    all_metrics = load_metrics(results_dir)
    if not all_metrics:
        print("No JSON metric files found in", results_dir)
        return
    
    print(f"Found {len(all_metrics)} JSON metrics files.")
    
    # 2) Create an output directory to store charts/csv
    out_dir = os.path.join(results_dir, "aggregated_results")
    os.makedirs(out_dir, exist_ok=True)
    
    # 3) Plot accuracy
    plot_accuracy_per_emotion(all_metrics, output_dir=out_dir)
    
    # 4) Save a CSV summary
    save_metrics_to_csv(all_metrics, output_dir=out_dir)

if __name__ == "__main__":
    main(results_dir)
