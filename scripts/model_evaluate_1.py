import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    """
    Build confusion matrix from df's 'true_idx' and 'predicted_idx', then plot & save.
    """
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
    import json
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Wrote classification report => {out_json}")

def reliability_diagram(df, class_names, out_path, n_bins=10):
    """
    Build a reliability diagram across the entire dataset:
    For each row, we have:
     - 'true_idx'
     - 'prob_className' for each class
    We find the probability assigned to the correct class, then measure calibration.
    """
    y_true = df['true_idx'].to_numpy()
    # build array of shape(N, C)
    # we must gather columns like prob_Angry, prob_Disgust, ...
    # assume the order of class_names => the columns are prob_{class_name}
    prob_array = []
    for row in df.itertuples():
        row_probs = []
        for cname in class_names:
            col_name = f"prob_{cname}"
            row_probs.append(getattr(row, col_name))
        prob_array.append(row_probs)
    prob_array = np.array(prob_array)  # shape(N,8)

    correct_probs = []
    correct_or_not = []
    for i in range(len(prob_array)):
        gt = y_true[i]
        p  = prob_array[i, gt]
        correct_probs.append(p)
        pred_idx = np.argmax(prob_array[i])
        correct_or_not.append(1 if pred_idx==gt else 0)

    correct_probs = np.array(correct_probs)
    correct_or_not= np.array(correct_or_not)

    from sklearn.calibration import calibration_curve
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

def main():
    # 1) Where we look for the 'combined_results.csv' for each model
    # e.g. "predictions_real_20250317_111412"
    predictions_root = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0\predictions_real_20250317_111412"
    
    # The classes in the same order as during training
    class_names = ["Angry","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]

    # We'll create a new folder for these evaluation outputs
    eval_out_dir = os.path.join(predictions_root, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_out_dir, exist_ok=True)

    # 2) We iterate each model subfolder
    for model_folder in os.listdir(predictions_root):
        model_path = os.path.join(predictions_root, model_folder)
        if not os.path.isdir(model_path):
            continue
        
        # Check if there's a 'combined_results.csv'
        combined_csv = os.path.join(model_path, "combined_results.csv")
        if not os.path.exists(combined_csv):
            print(f"[WARNING] No combined_results.csv found in {model_path}, skipping.")
            continue

        print(f"\n============================")
        print(f"[INFO] Evaluating {model_folder}'s combined_results.csv")
        df = load_predictions(combined_csv)
        if len(df)==0:
            print(f"[WARNING] CSV is empty => {combined_csv}")
            continue

        # 3) We create a subdir for storing evaluation plots
        model_eval_dir = os.path.join(eval_out_dir, model_folder)
        os.makedirs(model_eval_dir, exist_ok=True)

        # (a) confusion matrix
        cm_path = os.path.join(model_eval_dir, "confusion_matrix.png")
        build_confusion_matrix(df, class_names, cm_path)
        
        # (b) classification report
        cr_json = os.path.join(model_eval_dir, "classification_report.json")
        classification_report_dict(df, class_names, cr_json)
        
        # (c) reliability diagram
        rd_path = os.path.join(model_eval_dir, "reliability_diagram.png")
        reliability_diagram(df, class_names, rd_path, n_bins=10)

        # (d) you can do more: e.g. top-k accuracy, confidence distribution, etc.
        # For example, confidence distribution:
        #   * correct vs. incorrect
        #   * we can do a quick plot:
        df_correct = df[df['correct']==True]
        df_wrong   = df[df['correct']==False]
        if len(df_correct)>0 and len(df_wrong)>0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure()
            sns.kdeplot(df_correct['confidence'], label='Correct')
            sns.kdeplot(df_wrong['confidence'], label='Incorrect')
            plt.title("Confidence distribution")
            plt.legend()
            confdist_path = os.path.join(model_eval_dir, "confidence_distribution.png")
            plt.savefig(confdist_path)
            plt.close()

        print(f"[INFO] Evaluation artifacts for {model_folder} => {model_eval_dir}")

    print("[INFO] Done with all evaluations.")


if __name__=="__main__":
    main()
