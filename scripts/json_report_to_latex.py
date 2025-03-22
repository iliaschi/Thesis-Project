import json

# Hard-code the input JSON file name.
# C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\predictions_on_real_20250319_174546_ok\evaluation_0320_021246_ok\Best-Epoch\classification_report.json

INPUT_JSON_FILE = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK\evaluation_0322_012325\enet_b0_8_best_afew_state_dict\classification_report.json"
# r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\predictions_combined_synth_on_real_2\evaluation_0320_033313_synth_on_real\Best Validation Epoch\classification_report.json" 
#r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\predictions_on_real_20250319_174546_ok\evaluation_0320_021246_ok\Best-Epoch\classification_report.json" 
#r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK\evaluation_0320_023204\enet_b0_8_best_afew_state_dict\classification_report.json"  # Update this path if necessary.

def json_to_latex(json_file, table_caption="Classification Report", table_label="tab:classification_report"):
    with open(json_file, 'r') as f:
        report = json.load(f)

    # Attempt to fetch the per_emotion_accuracy dictionary, if it exists.
    per_emo_acc = report.get("per_emotion_accuracy", {})  # e.g. {"Angry": 78.33, "Contempt": 12.5, ...}
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")

    # We now have 6 columns: Class, Acc, Precision, Recall, F1, Support
    latex_lines.append("\\begin{tabular}{lrrrrr}")
    latex_lines.append("\\hline")
    latex_lines.append("Class & Acc (\\%) & Precision & Recall & F1-Score & Support \\\\")
    latex_lines.append("\\hline")

    # These are top-level keys that we skip from iterating in the classification_report
    skip_keys = ["accuracy", "macro avg", "weighted avg", "overall_accuracy", "per_emotion_accuracy"]

    # 1) Print rows for each emotion/class
    for key, metrics in report.items():
        if key in skip_keys:
            continue
        
        if isinstance(metrics, dict):
            # The classification_report has entries for precision/recall/f1/support
            precision = f"{metrics.get('precision', 0):.3f}"
            recall    = f"{metrics.get('recall', 0):.3f}"
            f1        = f"{metrics.get('f1-score', 0):.3f}"
            support   = f"{metrics.get('support', 0):.0f}"
            
            # Now fetch the per-emotion accuracy from the dictionary, if it exists
            # (the dictionary keys for per_emotion_accuracy presumably match the same class name)
            this_acc = per_emo_acc.get(key, float('nan'))  # If missing, we default to nan
            if not isinstance(this_acc, float):
                # ensure it's numeric
                try:
                    this_acc = float(this_acc)
                except:
                    this_acc = float('nan')
            
            # Convert to float with e.g. 1 decimal place
            acc_str = f"{this_acc:.1f}" if not (this_acc is float('nan')) else "NaN"

            latex_lines.append(f"{key} & {acc_str} & {precision} & {recall} & {f1} & {support} \\\\")
    latex_lines.append("\\hline")

    # 2) Macro avg row
    if "macro avg" in report:
        metrics = report["macro avg"]
        precision = f"{metrics.get('precision', 0):.3f}"
        recall    = f"{metrics.get('recall', 0):.3f}"
        f1        = f"{metrics.get('f1-score', 0):.3f}"
        support   = f"{metrics.get('support', 0):.0f}"
        # For macro avg, we don't have a single per-class accuracy. 
        # We'll just put a dash or empty for the Acc(%) column.
        latex_lines.append(f"Macro Avg & - & {precision} & {recall} & {f1} & {support} \\\\")

    # 3) Weighted avg row
    if "weighted avg" in report:
        metrics = report["weighted avg"]
        precision = f"{metrics.get('precision', 0):.3f}"
        recall    = f"{metrics.get('recall', 0):.3f}"
        f1        = f"{metrics.get('f1-score', 0):.3f}"
        support   = f"{metrics.get('support', 0):.0f}"
        latex_lines.append(f"Weighted Avg & - & {precision} & {recall} & {f1} & {support} \\\\")

    # 4) If classification_report has "accuracy", show it
    if "accuracy" in report:
        accuracy_val = f"{report['accuracy']:.3f}"
        # Merge columns for clarity
        latex_lines.append(f"Accuracy & \\multicolumn{{5}}{{c}}{{{accuracy_val}}} \\\\")
    
    # 5) If you want overall_accuracy from your custom dictionary:
    if "overall_accuracy" in report:
        overall_acc_str = f"{report['overall_accuracy']:.3f}"
        latex_lines.append(f"Overall Acc & \\multicolumn{{5}}{{c}}{{{overall_acc_str}}} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{table_caption}}}")
    latex_lines.append(f"\\label{{{table_label}}}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def main():
    latex_code = json_to_latex(INPUT_JSON_FILE)
    output_file = "classification_report_extended.txt"
    with open(output_file, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table code saved to {output_file}")

if __name__ == "__main__":
    main()

# def json_to_latex(json_file, table_caption="Classification Report", table_label="tab:classification_report"):
#     with open(json_file, 'r') as f:
#         report = json.load(f)
    
#     latex_lines = []
#     latex_lines.append("\\begin{table}[htbp]")
#     latex_lines.append("\\centering")
#     latex_lines.append("\\begin{tabular}{lrrrr}")
#     latex_lines.append("\\hline")
#     latex_lines.append("Class & Precision & Recall & F1-Score & Support \\\\")
#     latex_lines.append("\\hline")
    
#     # Iterate over keys that are class names.
#     for key in report:
#         if key in ["accuracy", "macro avg", "weighted avg"]:
#             continue
#         metrics = report[key]
#         precision = f"{metrics.get('precision', 0):.3f}"
#         recall    = f"{metrics.get('recall', 0):.3f}"
#         f1        = f"{metrics.get('f1-score', 0):.3f}"
#         support   = f"{metrics.get('support', 0):.0f}"
#         latex_lines.append(f"{key} & {precision} & {recall} & {f1} & {support} \\\\")
    
#     latex_lines.append("\\hline")
#     # Add macro average row
#     if "macro avg" in report:
#         metrics = report["macro avg"]
#         precision = f"{metrics.get('precision', 0):.3f}"
#         recall    = f"{metrics.get('recall', 0):.3f}"
#         f1        = f"{metrics.get('f1-score', 0):.3f}"
#         support   = f"{metrics.get('support', 0):.0f}"
#         latex_lines.append(f"Macro Avg & {precision} & {recall} & {f1} & {support} \\\\")
    
#     # Add weighted average row
#     if "weighted avg" in report:
#         metrics = report["weighted avg"]
#         precision = f"{metrics.get('precision', 0):.3f}"
#         recall    = f"{metrics.get('recall', 0):.3f}"
#         f1        = f"{metrics.get('f1-score', 0):.3f}"
#         support   = f"{metrics.get('support', 0):.0f}"
#         latex_lines.append(f"Weighted Avg & {precision} & {recall} & {f1} & {support} \\\\")
    
#     # Optionally add overall accuracy
#     if "accuracy" in report:
#         accuracy = f"{report['accuracy']:.3f}"
#         latex_lines.append(f"Accuracy & \\multicolumn{{4}}{{c}}{{{accuracy}}} \\\\")
    
#     latex_lines.append("\\hline")
#     latex_lines.append("\\end{tabular}")
#     latex_lines.append(f"\\caption{{{table_caption}}}")
#     latex_lines.append(f"\\label{{{table_label}}}")
#     latex_lines.append("\\end{table}")
    
#     return "\n".join(latex_lines)

# def main():
#     latex_code = json_to_latex(INPUT_JSON_FILE)
#     output_file = "classification_report_syn_real.txt"
#     with open(output_file, "w") as f:
#         f.write(latex_code)
#     print(f"LaTeX table code saved to {output_file}")

# if __name__ == "__main__":
#     main()
