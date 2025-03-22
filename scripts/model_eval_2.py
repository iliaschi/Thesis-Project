import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.calibration import calibration_curve

def load_predictions(csv_path):
    """
    Loads the predictions CSV. 
    Expects columns: 
      true_emotion, true_idx, predicted_emotion, predicted_idx, correct, confidence, prob_Angry, prob_Disgust, ...
    """
    df = pd.read_csv(csv_path)
    return df

def build_confusion_matrix(df, class_names, out_path):
    y_true = df['true_idx'].to_numpy()
    y_pred = df['predicted_idx'].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def classification_report_dict(df, class_names, out_json):
    y_true = df['true_idx'].to_numpy()
    y_pred = df['predicted_idx'].to_numpy()
    report = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Wrote classification report => {out_json}")

def reliability_diagram(df, class_names, out_path, n_bins=10):
    y_true = df['true_idx'].to_numpy()
    prob_array = []
    for row in df.itertuples():
        row_probs = []
        for cname in class_names:
            col_name = f"prob_{cname}"
            row_probs.append(getattr(row, col_name))
        prob_array.append(row_probs)
    prob_array = np.array(prob_array)  # shape(N, C)

    correct_probs = []
    correct_or_not = []
    for i in range(len(prob_array)):
        gt = y_true[i]
        p  = prob_array[i, gt]
        correct_probs.append(p)
        pred_idx = np.argmax(prob_array[i])
        correct_or_not.append(1 if pred_idx == gt else 0)

    correct_probs = np.array(correct_probs)
    correct_or_not = np.array(correct_or_not)
    prob_true, prob_pred = calibration_curve(correct_or_not, correct_probs, n_bins=n_bins)

    plt.figure()
    plt.plot(prob_pred, prob_true, 'o-', label='Model')
    plt.plot([0,1],[0,1], '--', color='gray', label='Perfect')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_accuracy_metrics(df, class_names):
    """
    Compute overall accuracy and per-emotion accuracy.
    """
    overall_accuracy = 100.0 * df['correct'].sum() / len(df) if len(df) > 0 else np.nan
    emotion_accuracy = {}
    for emotion in class_names:
        emotion_df = df[df['true_emotion'] == emotion]
        if len(emotion_df) > 0:
            accuracy = 100.0 * emotion_df['correct'].sum() / len(emotion_df)
        else:
            accuracy = np.nan
        emotion_accuracy[emotion] = accuracy
    return overall_accuracy, emotion_accuracy

def main():
    # 1) Root folder for all prediction outputs.
    predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\predictions_on_real_20250319_174124"
    
    # The classes in the same order as during training.
    class_names = ["Angry", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

    # Create a new folder for evaluation outputs.
    eval_out_dir = os.path.join(predictions_root, f"evaluation_{datetime.now().strftime('%m%d_%H%M%S')}")
    os.makedirs(eval_out_dir, exist_ok=True)

    # List to hold accuracy metrics for each folder.
    all_metrics = []

    # 2) Iterate over each model subfolder.
    for model_folder in os.listdir(predictions_root):
        model_path = os.path.join(predictions_root, model_folder)
        if not os.path.isdir(model_path):
            continue
        
        combined_csv = os.path.join(model_path, "combined_results.csv")
        if not os.path.exists(combined_csv):
            print(f"[WARNING] No combined_results.csv found in {model_path}, skipping.")
            continue

        print(f"\n============================")
        print(f"[INFO] Evaluating {model_folder}'s combined_results.csv")
        df = load_predictions(combined_csv)
        if len(df) == 0:
            print(f"[WARNING] CSV is empty => {combined_csv}")
            continue

        # Create a subfolder for evaluation outputs for this model.
        model_eval_dir = os.path.join(eval_out_dir, model_folder)
        os.makedirs(model_eval_dir, exist_ok=True)

        # (a) Confusion matrix.
        cm_path = os.path.join(model_eval_dir, "confusion_matrix.png")
        build_confusion_matrix(df, class_names, cm_path)
        
        # (b) Classification report.
        cr_json = os.path.join(model_eval_dir, "classification_report.json")
        classification_report_dict(df, class_names, cr_json)
        
        # (c) Reliability diagram.
        rd_path = os.path.join(model_eval_dir, "reliability_diagram.png")
        reliability_diagram(df, class_names, rd_path, n_bins=10)

        # (d) Optional: Confidence distribution plot.
        df_correct = df[df['correct'] == True]
        df_wrong   = df[df['correct'] == False]
        if len(df_correct) > 0 and len(df_wrong) > 0:
            plt.figure()
            sns.kdeplot(df_correct['confidence'], label='Correct')
            sns.kdeplot(df_wrong['confidence'], label='Incorrect')
            plt.title("Confidence Distribution")
            plt.legend()
            confdist_path = os.path.join(model_eval_dir, "confidence_distribution.png")
            plt.savefig(confdist_path)
            plt.close()

        # 3) Compute and save overall and per-emotion accuracy.
        overall_acc, emotion_acc = compute_accuracy_metrics(df, class_names)
        metrics_entry = {'model_folder': model_folder, 'overall_accuracy': overall_acc}
        for emotion, acc in emotion_acc.items():
            metrics_entry[f'accuracy_{emotion}'] = acc
        all_metrics.append(metrics_entry)
        
        # Save accuracy metrics for this model in its own CSV.
        folder_metrics_df = pd.DataFrame([metrics_entry])
        folder_metrics_csv = os.path.join(model_eval_dir, "accuracy_metrics.csv")
        folder_metrics_df.to_csv(folder_metrics_csv, index=False)
        print(f"[INFO] Saved accuracy metrics for {model_folder} at {folder_metrics_csv}")

        # --- NEW: Create individual per-emotion bar plot for this model ---
        emotions = class_names
        accuracies = [metrics_entry[f'accuracy_{emotion}'] for emotion in class_names]
        plt.figure(figsize=(8, 6))
        # Use a vibrant palette, e.g., "bright"
        sns.barplot(x=emotions, y=accuracies, palette=sns.color_palette("bright", len(emotions)))
        plt.title(f'Per-Emotion Accuracy for {model_folder}')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Emotion')
        plt.tight_layout()
        individual_plot_path = os.path.join(model_eval_dir, f"per_emotion_accuracy_{model_folder}.png")
        plt.savefig(individual_plot_path)
        plt.close()
        print(f"[INFO] Individual per-emotion bar plot saved at {individual_plot_path}")
        
        print(f"[INFO] Evaluation artifacts for {model_folder} saved in {model_eval_dir}")

    # 4) Save global accuracy metrics across all models.
    if all_metrics:
        global_metrics_df = pd.DataFrame(all_metrics)
        global_metrics_csv = os.path.join(eval_out_dir, "global_accuracy_metrics.csv")
        global_metrics_df.to_csv(global_metrics_csv, index=False)
        print(f"[INFO] Global accuracy metrics saved at {global_metrics_csv}")

        # 5) Create bar plot for overall accuracy (global).
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_folder', y='overall_accuracy', data=global_metrics_df,
                    palette=sns.color_palette("bright", len(global_metrics_df)))
        plt.xticks(rotation=45)
        plt.title('Overall Accuracy Comparison')
        plt.ylabel('Overall Accuracy (%)')
        plt.xlabel('Model Folder')
        plt.tight_layout()
        overall_barplot_path = os.path.join(eval_out_dir, "overall_accuracy_barplot.png")
        plt.savefig(overall_barplot_path)
        plt.close()
        print(f"[INFO] Overall accuracy bar plot saved at {overall_barplot_path}")

        # 6) Create a grouped bar plot for per-emotion accuracy (global).
        emotion_columns = [f'accuracy_{emotion}' for emotion in class_names]
        global_metrics_long = pd.melt(global_metrics_df, id_vars=['model_folder', 'overall_accuracy'],
                                      value_vars=emotion_columns,
                                      var_name='emotion', value_name='accuracy')
        global_metrics_long['emotion'] = global_metrics_long['emotion'].str.replace('accuracy_', '')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model_folder', y='accuracy', hue='emotion', data=global_metrics_long, palette="bright")
        plt.xticks(rotation=45)
        plt.title('Per-Emotion Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Model Folder')
        plt.tight_layout()
        per_emotion_barplot_path = os.path.join(eval_out_dir, "per_emotion_accuracy_barplot.png")
        plt.savefig(per_emotion_barplot_path)
        plt.close()
        print(f"[INFO] Per-emotion accuracy bar plot saved at {per_emotion_barplot_path}")
    else:
        print("[WARNING] No accuracy metrics collected.")

    print("[INFO] Done with all evaluations.")

if __name__ == "__main__":
    main()



# def main():
#     # 1) Root folder for all prediction outputs.
#     predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\predictions_on_real_20250319_174124"
    
#     # The classes in the same order as during training.
#     class_names = ["Angry", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

#     # Create a new folder for evaluation outputs.
#     eval_out_dir = os.path.join(predictions_root, f"evaluation_{datetime.now().strftime('%m%d_%H%M%S')}")
#     os.makedirs(eval_out_dir, exist_ok=True)

#     # List to hold accuracy metrics for each folder.
#     all_metrics = []

#     # 2) Iterate over each model subfolder.
#     for model_folder in os.listdir(predictions_root):
#         model_path = os.path.join(predictions_root, model_folder)
#         if not os.path.isdir(model_path):
#             continue
        
#         combined_csv = os.path.join(model_path, "combined_results.csv")
#         if not os.path.exists(combined_csv):
#             print(f"[WARNING] No combined_results.csv found in {model_path}, skipping.")
#             continue

#         print(f"\n============================")
#         print(f"[INFO] Evaluating {model_folder}'s combined_results.csv")
#         df = load_predictions(combined_csv)
#         if len(df) == 0:
#             print(f"[WARNING] CSV is empty => {combined_csv}")
#             continue

#         # Create a subfolder for evaluation outputs for this model.
#         model_eval_dir = os.path.join(eval_out_dir, model_folder)
#         os.makedirs(model_eval_dir, exist_ok=True)

#         # (a) Confusion matrix.
#         cm_path = os.path.join(model_eval_dir, "confusion_matrix.png")
#         build_confusion_matrix(df, class_names, cm_path)
        
#         # (b) Classification report.
#         cr_json = os.path.join(model_eval_dir, "classification_report.json")
#         classification_report_dict(df, class_names, cr_json)
        
#         # (c) Reliability diagram.
#         rd_path = os.path.join(model_eval_dir, "reliability_diagram.png")
#         reliability_diagram(df, class_names, rd_path, n_bins=10)

#         # (d) (Optional) Confidence distribution plot.
#         df_correct = df[df['correct'] == True]
#         df_wrong   = df[df['correct'] == False]
#         if len(df_correct) > 0 and len(df_wrong) > 0:
#             plt.figure()
#             sns.kdeplot(df_correct['confidence'], label='Correct')
#             sns.kdeplot(df_wrong['confidence'], label='Incorrect')
#             plt.title("Confidence Distribution")
#             plt.legend()
#             confdist_path = os.path.join(model_eval_dir, "confidence_distribution.png")
#             plt.savefig(confdist_path)
#             plt.close()

#         # 3) Compute and save overall and per-emotion accuracy.
#         overall_acc, emotion_acc = compute_accuracy_metrics(df, class_names)
#         metrics_entry = {'model_folder': model_folder, 'overall_accuracy': overall_acc}
#         for emotion, acc in emotion_acc.items():
#             metrics_entry[f'accuracy_{emotion}'] = acc
#         all_metrics.append(metrics_entry)
        
#         # Save accuracy metrics for this model in its own CSV.
#         folder_metrics_df = pd.DataFrame([metrics_entry])
#         folder_metrics_csv = os.path.join(model_eval_dir, "accuracy_metrics.csv")
#         folder_metrics_df.to_csv(folder_metrics_csv, index=False)
#         print(f"[INFO] Saved accuracy metrics for {model_folder} at {folder_metrics_csv}")
        
#         print(f"[INFO] Evaluation artifacts for {model_folder} saved in {model_eval_dir}")

#     # 4) Save global accuracy metrics across all models.
#     if all_metrics:
#         global_metrics_df = pd.DataFrame(all_metrics)
#         global_metrics_csv = os.path.join(eval_out_dir, "global_accuracy_metrics.csv")
#         global_metrics_df.to_csv(global_metrics_csv, index=False)
#         print(f"[INFO] Global accuracy metrics saved at {global_metrics_csv}")

#         # 5) Create bar plot for overall accuracy.
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='model_folder', y='overall_accuracy', data=global_metrics_df)
#         plt.xticks(rotation=45)
#         plt.title('Overall Accuracy Comparison')
#         plt.ylabel('Overall Accuracy (%)')
#         plt.xlabel('Model Folder')
#         plt.tight_layout()
#         overall_barplot_path = os.path.join(eval_out_dir, "overall_accuracy_barplot.png")
#         plt.savefig(overall_barplot_path)
#         plt.close()
#         print(f"[INFO] Overall accuracy bar plot saved at {overall_barplot_path}")

#         # 6) Create a grouped bar plot for per-emotion accuracy.
#         # Melt the DataFrame to long format.
#         emotion_columns = [f'accuracy_{emotion}' for emotion in class_names]
#         global_metrics_long = pd.melt(global_metrics_df, id_vars=['model_folder', 'overall_accuracy'],
#                                       value_vars=emotion_columns,
#                                       var_name='emotion', value_name='accuracy')
#         # Remove the "accuracy_" prefix for clarity.
#         global_metrics_long['emotion'] = global_metrics_long['emotion'].str.replace('accuracy_', '')
        
#         plt.figure(figsize=(12, 8))
#         sns.barplot(x='model_folder', y='accuracy', hue='emotion', data=global_metrics_long)
#         plt.xticks(rotation=45)
#         plt.title('Per-Emotion Accuracy Comparison')
#         plt.ylabel('Accuracy (%)')
#         plt.xlabel('Model Folder')
#         plt.tight_layout()
#         per_emotion_barplot_path = os.path.join(eval_out_dir, "per_emotion_accuracy_barplot.png")
#         plt.savefig(per_emotion_barplot_path)
#         plt.close()
#         print(f"[INFO] Per-emotion accuracy bar plot saved at {per_emotion_barplot_path}")
#     else:
#         print("[WARNING] No accuracy metrics collected.")

#     print("[INFO] Done with all evaluations.")

# if __name__ == "__main__":
#     main()
