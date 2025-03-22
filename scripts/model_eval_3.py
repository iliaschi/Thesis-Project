import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc

def load_predictions(csv_path):
    """
    Loads the predictions CSV. 
    Expects columns: 
      true_emotion, true_idx, predicted_emotion, predicted_idx, correct, confidence, prob_Angry, prob_Disgust, ...
    """
    df = pd.read_csv(csv_path)
    return df

def build_confusion_matrix(df, class_names, out_path):
    """
    Build confusion matrix from df's 'true_idx' and 'predicted_idx',
    then plot & save. Each cell shows count and (percentage%).
    """
    y_true = df['true_idx'].to_numpy()
    y_pred = df['predicted_idx'].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Compute row-wise percentages
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm[i] / row_sum * 100
        else:
            cm_normalized[i] = 0

    # Annotate each cell with count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def classification_report_dict(df, class_names, out_json):
    """
    Compute classification_report(...) + also attach per-emotion accuracy in the same JSON.
    """
    y_true = df['true_idx'].to_numpy()
    y_pred = df['predicted_idx'].to_numpy()

    # 1) Build the standard classification report.
    report = classification_report(
        y_true, 
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    # 2) Also compute per-emotion accuracy using your existing function.
    overall_acc, emotion_acc = compute_accuracy_metrics(df, class_names)

    # 3) Attach these metrics to the classification report under a new key.
    # e.g. "per_emotion_accuracy" or some other name you like.
    # We can store both overall and per-class accuracy if we want:
    report["overall_accuracy"] = overall_acc
    report["per_emotion_accuracy"] = emotion_acc

    # 4) Write out to JSON.
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Wrote classification report => {out_json}")


# def classification_report_dict(df, class_names, out_json):
#     y_true = df['true_idx'].to_numpy()
#     y_pred = df['predicted_idx'].to_numpy()
#     report = classification_report(
#         y_true, y_pred,
#         labels=range(len(class_names)),
#         target_names=class_names,
#         zero_division=0,
#         output_dict=True
#     )
#     with open(out_json, 'w') as f:
#         json.dump(report, f, indent=4)
#     print(f"[INFO] Wrote classification report => {out_json}")


def reliability_diagram(df, class_names, out_path, n_bins=10):
    """
    Build a reliability diagram across the entire dataset.
    """
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

def plot_precision_recall_curves(df, class_names, out_path):
    """
    Plot Precision-Recall curves for each emotion using a one-vs-rest approach.
    """
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(class_names):
        # Binary ground truth for the emotion: 1 if true label equals class index, else 0.
        y_true = (df['true_idx'] == i).astype(int).to_numpy()
        y_scores = df[f"prob_{emotion}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f'{emotion} (AP={ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Emotion')
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

def compute_top_k_accuracy(df, class_names, k=2):
    """
    Compute overall and per-emotion Top-K accuracy (here k=2).
    A prediction is considered correct if the true class is among the top k highest probabilities.
    """
    total_samples = len(df)
    if total_samples == 0:
        return np.nan, {}
    overall_topk_correct = 0
    emotion_topk = {emotion: {'correct': 0, 'total': 0} for emotion in class_names}
    for idx, row in df.iterrows():
        # Get predicted probabilities for each class in the order of class_names.
        probs = np.array([row[f"prob_{emotion}"] for emotion in class_names])
        topk_indices = probs.argsort()[-k:][::-1]  # Top-k indices.
        true_idx = row['true_idx']
        is_topk = true_idx in topk_indices
        if is_topk:
            overall_topk_correct += 1
        # Count per emotion.
        true_emotion = row['true_emotion']
        emotion_topk[true_emotion]['total'] += 1
        if is_topk:
            emotion_topk[true_emotion]['correct'] += 1
    overall_topk_acc = 100.0 * overall_topk_correct / total_samples
    emotion_topk_acc = {}
    for emotion in class_names:
        tot = emotion_topk[emotion]['total']
        corr = emotion_topk[emotion]['correct']
        emotion_topk_acc[emotion] = 100.0 * corr / tot if tot > 0 else np.nan
    return overall_topk_acc, emotion_topk_acc


def plot_confidence_histograms(df, class_names, out_path, bins=20):
    """
    Create subplots for each emotion showing histograms of confidence values.
    For each emotion, we plot the confidence for correct and incorrect predictions.
    """
    num_classes = len(class_names)
    # Create a subplot grid: for 8 classes, 2 rows x 4 columns
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, emotion in enumerate(class_names):
        ax = axes[i]
        # Filter rows for this emotion (using true_emotion column)
        df_emotion = df[df['true_emotion'] == emotion]
        if df_emotion.empty:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
            ax.set_title(emotion)
            continue
        
        # Plot histogram with hue for correct (True/False)
        sns.histplot(data=df_emotion, x="confidence", hue="correct",
                     bins=bins, element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(emotion)
        ax.set_xlim(0, 1)
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Confidence Histograms per Emotion", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confidence histogram saved at {out_path}")

#########################
# NEW: ROC Curves and AUC per Emotion
#########################
def plot_roc_curves(df, class_names, out_path):
    """
    Plot ROC curves for each emotion (one-vs-rest) along with the AUC.
    """
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(class_names):
        # Create binary labels: 1 if true label equals current class index, else 0.
        y_true = (df['true_idx'] == i).astype(int).to_numpy()
        y_scores = df[f"prob_{emotion}"].to_numpy()
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{emotion} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Emotion')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curves saved at {out_path}")


def main():
    # 1) Root folder for all prediction outputs.

    # ----- Baseline Model on full test data -----
    # Baseline on real
    predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK"
    # Baseline on synt
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_synth_0319_2343_OK"

    # ----- Finetuned Model on full test data -----
    # Finetuned on real # C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\predictions_on_real_20250319_174546
    # Finetuned on Synthetic 
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\predictions_on_synth_20250319_181628"

    # ----- Synthetic Model on full test data -----
    # Synthetic on synth
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\predictions_combined_synth_on_synth_0320_023947"
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\predictions_combined_synth_on_synth_0320_023947"
    # Synthetic on real
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\predictions_combined_synth_on_real_0320_024137"

    # ----- Gender Splits -----
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_0316_1223synthetic_on_vggface2_gender\gender_splits_m_100_real"
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_0316_1223synthetic_on_vggface2_gender\gender_splits_w_100_real"
    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_0316_013917fine_real_gender\pred_fine_on_real_2_0320_1139\gen_m_f_fine_real"

    # predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\predictions_combined_synth_on_real_2"
    
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

        # (a) Confusion matrix with normalization.
        cm_path = os.path.join(model_eval_dir, "confusion_matrix.png")
        build_confusion_matrix(df, class_names, cm_path)
        
        # (b) Classification report.
        cr_json = os.path.join(model_eval_dir, "classification_report.json")
        classification_report_dict(df, class_names, cr_json)
        
        # (c) Reliability diagram.
        rd_path = os.path.join(model_eval_dir, "reliability_diagram.png")
        reliability_diagram(df, class_names, rd_path, n_bins=10)
        
        # (d) Precision-Recall curves.
        pr_curve_path = os.path.join(model_eval_dir, "precision_recall_curves.png")
        plot_precision_recall_curves(df, class_names, pr_curve_path)

        # (e) Confidence Histogram (new).
        conf_hist_path = os.path.join(model_eval_dir, "confidence_histograms.png")
        plot_confidence_histograms(df, class_names, conf_hist_path, bins=20)
        
        # (f) ROC Curves and AUC (new).
        roc_curve_path = os.path.join(model_eval_dir, "roc_curves.png")
        plot_roc_curves(df, class_names, roc_curve_path)

        # (e) Optional: Confidence distribution plot.
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

        # 3) Compute overall and per-emotion accuracy.
        overall_acc, emotion_acc = compute_accuracy_metrics(df, class_names)
        # 3b) Compute Top-2 accuracy.
        overall_top2, emotion_top2 = compute_top_k_accuracy(df, class_names, k=2)

        # New: Compute average negative log likelihood (NLL) per emotion from the CSV.
        # Group by 'true_emotion' and compute the mean of 'neg_log_likelihood'
        nll_by_emotion = df.groupby('true_emotion')['neg_log_likelihood'].mean().to_dict()

        metrics_entry = {'model_folder': model_folder, 'overall_accuracy': overall_acc, 'top2_overall_accuracy': overall_top2}
        for emotion in class_names:
            metrics_entry[f'accuracy_{emotion}'] = emotion_acc.get(emotion, np.nan)
            metrics_entry[f'top2_accuracy_{emotion}'] = emotion_top2.get(emotion, np.nan)
            metrics_entry[f'nll_{emotion}'] = nll_by_emotion.get(emotion, np.nan)
        all_metrics.append(metrics_entry)
        
        # Save accuracy metrics for this model in its own CSV.
        folder_metrics_df = pd.DataFrame([metrics_entry])
        folder_metrics_csv = os.path.join(model_eval_dir, "accuracy_metrics.csv")
        folder_metrics_df.to_csv(folder_metrics_csv, index=False)
        print(f"[INFO] Saved accuracy metrics for {model_folder} at {folder_metrics_csv}")
        
        # --- Individual per-emotion bar plot for this model ---
        emotions = class_names
        accuracies = [metrics_entry[f'accuracy_{emotion}'] for emotion in class_names]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=emotions, y=accuracies, palette=sns.color_palette("Paired", len(emotions)))
        plt.title(f'Per-Emotion Accuracy for {model_folder}')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Emotion')
        plt.tight_layout()

        individual_plot_path = os.path.join(model_eval_dir, f"per_emotion_accuracy_{model_folder}.png")

        # # Ensure the directory exists before saving
        # os.makedirs(os.path.dirname(individual_plot_path), exist_ok=True)
        # plt.savefig(individual_plot_path)

        # individual_plot_path = os.path.join(model_eval_dir, f"per_emotion_accuracy_{model_folder}.png")
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

        # 5) Create global bar plot for overall accuracy.
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_folder', y='overall_accuracy', data=global_metrics_df,
                    palette=sns.color_palette("Paired", len(global_metrics_df)))
        plt.xticks(rotation=90)
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
        
        # test per emotion bar plot.
        # Assuming global_metrics_long and eval_out_dir are already defined
        plt.figure(figsize=(16, 16))
        ax = sns.barplot(
            x='model_folder', 
            y='accuracy', 
            hue='emotion', 
            data=global_metrics_long,
            palette=sns.color_palette("Paired", len(global_metrics_long['emotion'].unique()))
        )
        plt.xticks(rotation=90)
        plt.title('Per-Emotion Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Model Folder')

        ax.set_ylim(0, 100)
        # Place the legend at the top left inside the plot area.
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), title="Emotion")

        plt.tight_layout()
        per_emotion_barplot_path = os.path.join(eval_out_dir, "per_emotion_accuracy_barplot.png")
        plt.savefig(per_emotion_barplot_path, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Per-emotion accuracy bar plot saved at {per_emotion_barplot_path}")
        

        # 7) Create a grouped bar plot for per-emotion Top-2 accuracy (if not already done)
        # Melt the DataFrame for Top-2 accuracy columns.
        emotion_columns_top2 = [f'top2_accuracy_{emotion}' for emotion in class_names]
        global_metrics_long_top2 = pd.melt(global_metrics_df,
                                        id_vars=['model_folder'],
                                        value_vars=emotion_columns_top2,
                                        var_name='emotion', value_name='top2_accuracy')
        # Remove the prefix from emotion names.
        global_metrics_long_top2['emotion'] = global_metrics_long_top2['emotion'].str.replace('top2_accuracy_', '')
        
        plt.figure(figsize=(16, 16))
        ax = sns.barplot(
            x='model_folder',
            y='top2_accuracy',
            hue='emotion',
            data=global_metrics_long_top2,
            palette=sns.color_palette("Paired", len(global_metrics_long_top2['emotion'].unique()))
        )
        plt.xticks(rotation=90)
        plt.title('Per-Emotion Top-2 Accuracy Comparison')
        plt.ylabel('Top-2 Accuracy (%)')
        plt.xlabel('Model Folder')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), title="Emotion")
        plt.tight_layout()
        top2_barplot_path = os.path.join(eval_out_dir, "per_emotion_top2_accuracy_barplot.png")
        plt.savefig(top2_barplot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Per-emotion Top-2 accuracy bar plot saved at {top2_barplot_path}")
        
        # 8) Create a grouped bar plot for per-emotion average Negative Log Likelihood (NLL).
        # Melt the DataFrame for NLL columns.
        emotion_columns_nll = [f'nll_{emotion}' for emotion in class_names]
        global_metrics_long_nll = pd.melt(global_metrics_df,
                                        id_vars=['model_folder'],
                                        value_vars=emotion_columns_nll,
                                        var_name='emotion', value_name='avg_nll')
        # Remove the prefix from emotion names.
        global_metrics_long_nll['emotion'] = global_metrics_long_nll['emotion'].str.replace('nll_', '')
        
        plt.figure(figsize=(16, 16))
        ax = sns.barplot(
            x='model_folder',
            y='avg_nll',
            hue='emotion',
            data=global_metrics_long_nll,
            palette=sns.color_palette("Paired", len(global_metrics_long_nll['emotion'].unique()))
        )
        plt.xticks(rotation=90)
        plt.title('Per-Emotion Average Negative Log Likelihood (NLL) Comparison')
        plt.ylabel('Average NLL')
        plt.xlabel('Model Folder')
        # Adjust y-axis limit if needed; for instance, if NLL values are around 0-3, you can set:
        # ax.set_ylim(0, 3)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), title="Emotion")
        plt.tight_layout()
        nll_barplot_path = os.path.join(eval_out_dir, "per_emotion_nll_barplot.png")
        plt.savefig(nll_barplot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Per-emotion NLL bar plot saved at {nll_barplot_path}")

    else:
        print("[WARNING] No accuracy metrics collected.")

    print("[INFO] Done with all evaluations.")

if __name__ == "__main__":
    main()
