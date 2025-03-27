import json
import os

# Hard-code the input JSON file name.
# C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\predictions_on_real_20250319_174546_ok\evaluation_0320_021246_ok\Best-Epoch\classification_report.json

INPUT_JSON_FILE = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK\evaluation_0325_004321\enet_b0_8_best_afew_state_dict\classification_report_full.json"
# r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK\evaluation_0322_012325\enet_b0_8_best_afew_state_dict\classification_report.json"

evaluation_folder_path = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\predictions_combined_real_19_2330_OK\evaluation_0322_012325\enet_b0_8_best_afew_state_dict"

input_json = os.path.join(evaluation_folder_path, "classification_report.json")

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
    latex_lines.append("Class & Acc (\\%) & Precision & F1-Score & Support \\\\")
    # latex_lines.append("Class & Acc (\\%) & Precision & Recall & F1-Score & Support \\\\")
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
            #recall    = f"{metrics.get('recall', 0):.3f}"
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

            # latex_lines.append(f"{key} & {acc_str} & {precision} & {recall} & {f1} & {support} \\\\")
            latex_lines.append(f"{key} & {acc_str} & {precision} & {f1} & {support} \\\\")
    latex_lines.append("\\hline")

    # 2) Macro avg row
    if "macro avg" in report:
        metrics = report["macro avg"]
        precision = f"{metrics.get('precision', 0):.3f}"
        # recall    = f"{metrics.get('recall', 0):.3f}"
        f1        = f"{metrics.get('f1-score', 0):.3f}"
        support   = f"{metrics.get('support', 0):.0f}"
        # For macro avg, we don't have a single per-class accuracy. 
        # We'll just put a dash or empty for the Acc(%) column.
        # latex_lines.append(f"Macro Avg & - & {precision} & {recall} & {f1} & {support} \\\\")
        latex_lines.append(f"Macro Avg & - & {precision} & {f1} & {support} \\\\")

    # 3) Weighted avg row
    if "weighted avg" in report:
        metrics = report["weighted avg"]
        precision = f"{metrics.get('precision', 0):.3f}"
        # recall    = f"{metrics.get('recall', 0):.3f}"
        f1        = f"{metrics.get('f1-score', 0):.3f}"
        support   = f"{metrics.get('support', 0):.0f}"
        latex_lines.append(f"Weighted Avg & - & {precision} & {f1} & {support} \\\\")
        # latex_lines.append(f"Weighted Avg & - & {precision} & {recall} & {f1} & {support} \\\\")

    # 4) If classification_report has "accuracy", show it
    if "accuracy" in report:
        accuracy_val = f"{report['accuracy']:.3f}"
        # Merge columns for clarity
        latex_lines.append(f"Accuracy & \\multicolumn{{4}}{{c}}{{{accuracy_val}}} \\\\")
        # latex_lines.append(f"Accuracy & \\multicolumn{{5}}{{c}}{{{accuracy_val}}} \\\\")
    
    # 5) If you want overall_accuracy from your custom dictionary:
    if "overall_accuracy" in report:
        overall_acc_str = f"{report['overall_accuracy']:.3f}"
        latex_lines.append(f"Overall Acc & \\multicolumn{{4}}{{c}}{{{overall_acc_str}}} \\\\")
        # latex_lines.append(f"Overall Acc & \\multicolumn{{5}}{{c}}{{{overall_acc_str}}} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{table_caption}}}")
    latex_lines.append(f"\\label{{{table_label}}}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def json_to_latex_scratch(json_file, table_caption="Classification Metrics", table_label="tab:classification_metrics"):
    
    with open(json_file, 'r') as f:
        report = json.load(f)
    
    # We'll produce two tables:
    # 1) Per-Class Table (Accuracy%, Precision%, Recall%, F1%, Support, TP, FP, FN, TN)
    # 2) Overall Metrics Table (Accuracy%, Macro/Weighted Precision%, Recall%, F1%, Support)
    
    # ------------------------
    # Per-Class Table
    # ------------------------
    per_class_lines = []
    per_class_lines.append("\\begin{table}[htbp]")
    per_class_lines.append("\\centering")
    per_class_lines.append("\\begin{tabular}{lrrrrrrrrr}")
    per_class_lines.append("\\hline")
    per_class_lines.append(
        "Class & Accuracy\\% & Precision & Recall & F1-Score & Support & True Pos. & False Pos. & False Neg. & True Neg. \\\\"
    )
    per_class_lines.append("\\hline")

    # Identify all class entries (everything except the "overall" key).
    class_names = [k for k in report.keys() if k != "overall"]
    # Sort them for a stable output (optional).
    class_names.sort()

    for cls in class_names:
        metrics = report[cls]
        # Multiply relevant metrics by 100 (accuracy, precision, recall, f1-score).
        acc = metrics.get("accuracy", 0.0) * 100
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        support = metrics.get("support", 0)
        tp = metrics.get("True Positives", 0)
        fp = metrics.get("False Positives", 0)
        fn = metrics.get("False Negatives", 0)
        tn = metrics.get("True Negatives", 0)
        per_class_lines.append(
            f"{cls} & {acc:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & "
            f"{support} & {tp} & {fp} & {fn} & {tn} \\\\"
        )

    per_class_lines.append("\\hline")
    per_class_lines.append("\\end{tabular}")
    per_class_lines.append(f"\\caption{{Per-Class Classification Metrics}}")
    per_class_lines.append(f"\\label{{{table_label}_per_class}}")
    per_class_lines.append("\\end{table}")
    per_class_lines.append("")

    # ------------------------
    # Overall Metrics Table
    # ------------------------
    overall_lines = []
    overall_lines.append("\\begin{table}[htbp]")
    overall_lines.append("\\centering")
    overall_lines.append("\\begin{tabular}{lr}")
    overall_lines.append("\\hline")
    overall_lines.append("Metric & Value \\\\")
    overall_lines.append("\\hline")

    overall = report.get("overall", {})
    # Overall accuracy
    if "accuracy" in overall:
        overall_accuracy = overall["accuracy"] * 100
        overall_lines.append(f"Accuracy\\% & {overall_accuracy:.2f} \\\\")
    # Macro avg
    if "macro avg" in overall:
        macro = overall["macro avg"]
        macro_prec = macro.get("precision", 0.0)
        macro_recall = macro.get("recall", 0.0)
        macro_f1 = macro.get("f1-score", 0.0)
        macro_support = macro.get("support", 0)
        overall_lines.append(f"Macro Precision & {macro_prec:.2f} \\\\")
        overall_lines.append(f"Macro Recall & {macro_recall:.2f} \\\\")
        overall_lines.append(f"Macro F1-Score & {macro_f1:.2f} \\\\")
        overall_lines.append(f"Macro Support & {macro_support} \\\\")
    # Weighted avg
    if "weighted avg" in overall:
        weighted = overall["weighted avg"]
        w_prec = weighted.get("precision", 0.0)
        w_recall = weighted.get("recall", 0.0)
        w_f1 = weighted.get("f1-score", 0.0)
        w_support = weighted.get("support", 0)
        overall_lines.append(f"Weighted Precision & {w_prec:.2f} \\\\")
        overall_lines.append(f"Weighted Recall & {w_recall:.2f} \\\\")
        overall_lines.append(f"Weighted F1-Score & {w_f1:.2f} \\\\")
        overall_lines.append(f"Weighted Support & {w_support} \\\\")

    overall_lines.append("\\hline")
    overall_lines.append("\\end{tabular}")
    overall_lines.append(f"\\caption{{Overall Classification Metrics}}")
    overall_lines.append(f"\\label{{{table_label}_overall}}")
    overall_lines.append("\\end{table}")

    # Combine everything into one output: first per-class table, then overall table.
    return "\n".join(per_class_lines + overall_lines)   
    # latex_lines = []
    
    # # Overall metrics table
    # latex_lines.append("\\begin{table}[htbp]")
    # latex_lines.append("\\centering")
    # latex_lines.append("\\begin{tabular}{lr}")
    # latex_lines.append("\\hline")
    # latex_lines.append("Metric & Value \\\\")
    # latex_lines.append("\\hline")
    # if "overall" in report:
    #     overall = report["overall"]
    #     if "accuracy" in overall:
    #         latex_lines.append(f"Accuracy & {overall['accuracy']:.3f} \\\\")
    #     if "macro avg" in overall:
    #         macro = overall["macro avg"]
    #         latex_lines.append(f"Macro Precision & {macro.get('precision', 0):.3f} \\\\")
    #         latex_lines.append(f"Macro Recall & {macro.get('recall', 0):.3f} \\\\")
    #         latex_lines.append(f"Macro F1-Score & {macro.get('f1-score', 0):.3f} \\\\")
    #         latex_lines.append(f"Macro Support & {macro.get('support', 0)} \\\\")
    #     if "weighted avg" in overall:
    #         weighted = overall["weighted avg"]
    #         latex_lines.append(f"Weighted Precision & {weighted.get('precision', 0):.3f} \\\\")
    #         latex_lines.append(f"Weighted Recall & {weighted.get('recall', 0):.3f} \\\\")
    #         latex_lines.append(f"Weighted F1-Score & {weighted.get('f1-score', 0):.3f} \\\\")
    #         latex_lines.append(f"Weighted Support & {weighted.get('support', 0)} \\\\")
    # latex_lines.append("\\hline")
    # latex_lines.append("\\end{tabular}")
    # latex_lines.append(f"\\caption{{Overall Classification Metrics}}")
    # latex_lines.append(f"\\label{{{table_label}_overall}}")
    # latex_lines.append("\\end{table}")
    # latex_lines.append("")

    # # Per-class metrics table
    # # Define columns: Class, Precision, Recall, F1-Score, Accuracy, Support, TP, FP, FN, TN
    # latex_lines.append("\\begin{table}[htbp]")
    # latex_lines.append("\\centering")
    # latex_lines.append("\\begin{tabular}{lrrrrrrrrr}")
    # latex_lines.append("\\hline")
    # latex_lines.append("Class & Accuracy & Precision & Recall & F1-Score & Support & True Positives & False Positives & False Negatives & True Negatives \\\\")
    # latex_lines.append("\\hline")
    # for key in report:
    #     if key == "overall":
    #         continue
    #     metrics = report[key]
    #     precision = metrics.get("precision", 0)
    #     recall = metrics.get("recall", 0)
    #     f1 = metrics.get("f1-score", 0)
    #     accuracy = metrics.get("accuracy", 0)
    #     Support = metrics.get("support", 0)
    #     TruePositive = metrics.get("True Positives", 0)
    #     FalsePositive = metrics.get("False Positives", 0)
    #     FalseNegative = metrics.get("False Negatives", 0)
    #     TrueNegative = metrics.get("True Negatives", 0)
    #     latex_lines.append(f"{key} & {accuracy:.4f} & {precision:.4f} & {recall:.4f} & {f1:.4f} &  {Support} & {TruePositive} & {FalsePositive} & {FalseNegative} & {TrueNegative} \\\\")
    # latex_lines.append("\\hline")
    # latex_lines.append("\\end{tabular}")
    # latex_lines.append(f"\\caption{{Per-Class Classification Metrics}}")
    # latex_lines.append(f"\\label{{{table_label}_per_class}}")
    # latex_lines.append("\\end{table}")
    
    # return "\n".join(latex_lines)

# def json_to_latex_scratch(json_file, table_caption="Metrics (From Scratch)", table_label="tab:metrics_scratch"):
#     """
#     Convert a JSON file produced by 'compute_classification_metrics_from_scratch' 
#     into a LaTeX table.

#     Parameters
#     ----------
#     json_file : str
#         Path to the JSON file that has 'overall' plus per-class entries (Angry, Disgust, etc.)
#     table_caption : str
#         The caption to place under the LaTeX table
#     table_label : str
#         The label to reference this table in LaTeX

#     Returns
#     -------
#     str
#         A string containing the LaTeX table code.
#     """
#     with open(json_file, 'r') as f:
#         report = json.load(f)
    
#     # The structure of 'report' is:
#     # {
#     #    "overall": {
#     #        "accuracy": float,
#     #        "macro avg": {...},
#     #        "weighted avg": {...}
#     #    },
#     #    "Angry": {
#     #        "precision": float,
#     #        "recall": float,
#     #        "f1-score": float,
#     #        "accuracy": float,
#     #        "support": int,
#     #        "True Positives": int, ...
#     #    },
#     #    "Disgust": { ... },
#     #    ...
#     # }

#     # 1) Extract overall metrics
#     overall_data = report["overall"]
#     overall_accuracy = overall_data.get("accuracy", 0.0)  # e.g. 0.68359
#     macro_avg = overall_data.get("macro avg", {})
#     weighted_avg = overall_data.get("weighted avg", {})

#     # 2) Build up the table lines for each class
#     # We'll define columns: Class, Acc(%), Precision, Recall, F1-Score, Support
#     # where Acc(%) is 100 * "accuracy" if you want it as percent
#     #   or you can just store "accuracy" as fraction if you want to keep it that way.

#     # Prepare lines:
#     table_lines = []
#     header = (
#         "Class & Acc (\\%) & Precision & Recall & F1-Score & Support \\\\ \\hline\n"
#     )
#     table_lines.append(header)

#     # We know each top-level key except "overall" is a class name
#     # so let's collect them in a list, skipping "overall"
#     class_names = []
#     for key in report.keys():
#         if key != "overall":
#             class_names.append(key)
    
#     # Sort them if you like, or keep them in the order they appear
#     # class_names.sort()

#     for class_name in class_names:
#         class_stats = report[class_name]
#         # class_stats["accuracy"] is a fraction [0..1], so we multiply by 100 if we want percentage
#         acc_percent = 100.0 * class_stats.get("accuracy", 0.0)
#         prec = class_stats.get("precision", 0.0)
#         rec  = class_stats.get("recall", 0.0)
#         f1   = class_stats.get("f1-score", 0.0)
#         sup  = class_stats.get("support", 0)

#         row = (
#             f"{class_name} & "
#             f"{acc_percent:.1f} & "
#             f"{prec:.3f} & "
#             f"{rec:.3f} & "
#             f"{f1:.3f} & "
#             f"{sup} \\\\"
#         )
#         table_lines.append(row + "\n")

#     # Now we place macro avg, weighted avg, etc. at the bottom
#     table_lines.append("\\hline\n")

#     # Macro Avg row
#     macro_precision = macro_avg.get("precision", 0.0)
#     macro_recall    = macro_avg.get("recall", 0.0)
#     macro_f1        = macro_avg.get("f1-score", 0.0)
#     macro_sup       = macro_avg.get("support", 0)
#     # We won't have a single 'accuracy' for macro, so we put a dash or empty
#     row_macro = (
#         f"Macro Avg & - & "
#         f"{macro_precision:.3f} & "
#         f"{macro_recall:.3f} & "
#         f"{macro_f1:.3f} & "
#         f"{macro_sup} \\\\"
#     )
#     table_lines.append(row_macro + "\n")

#     # Weighted Avg row
#     w_precision = weighted_avg.get("precision", 0.0)
#     w_recall    = weighted_avg.get("recall", 0.0)
#     w_f1        = weighted_avg.get("f1-score", 0.0)
#     w_sup       = weighted_avg.get("support", 0)
#     row_weighted = (
#         f"Weighted Avg & - & "
#         f"{w_precision:.3f} & "
#         f"{w_recall:.3f} & "
#         f"{w_f1:.3f} & "
#         f"{w_sup} \\\\"
#     )
#     table_lines.append(row_weighted + "\n")

#     # Then overall accuracy on its own line
#     overall_acc_percent = 100.0 * overall_accuracy
#     table_lines.append(
#         f"Accuracy & \\multicolumn{{5}}{{c}}{{{overall_accuracy:.3f}}} \\\\ \n"
#     )
#     table_lines.append(
#         f"Overall Acc & \\multicolumn{{5}}{{c}}{{{overall_acc_percent:.2f}\\%}} \\\\ \n"
#     )

#     # 3) Wrap it all up in LaTeX code
#     latex_str = (
#         "\\begin{table}[htbp]\n"
#         "\\centering\n"
#         "\\begin{tabular}{lrrrrr}\n"
#         "\\hline\n"
#         + "".join(table_lines) +
#         "\\hline\n"
#         "\\end{tabular}\n"
#         f"\\caption{{{table_caption}}}\n"
#         f"\\label{{{table_label}}}\n"
#         "\\end{table}\n"
#     )

#     return latex_str


def json_to_latex_single_table(json_file, table_caption="Classification Metrics", table_label="tab:classification_metrics"):
    with open(json_file, "r") as f:
        report = json.load(f)
    
    # We'll produce ONE LaTeX table with:
    # Rows = Each class + (macro avg + weighted avg + accuracy)
    # Columns = Class, Acc(%), Precision(%), F1(%), Support, TP, FP, FN, TN
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrrrrrrr}")
    lines.append("\\hline")
    lines.append("Class & Acc(\\%) & Prec & F1 & Support & Recall & True P. & False P. & False N. & True N. \\\\")
    lines.append("\\hline")
    
    # Identify class keys (everything except "overall")
    class_names = [k for k in report.keys() if k != "overall"]
    class_names.sort()  # Sort them for stable output (optional)
    
    # 1) Per-class rows
    for cls in class_names:
        metrics = report[cls]
        acc = metrics.get("accuracy", 0.0) * 100
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)  # not shown as separate column, only F1
        f1 = metrics.get("f1-score", 0.0)
        support = metrics.get("support", 0)
        tp = metrics.get("True Positives", 0)
        fp = metrics.get("False Positives", 0)
        fn = metrics.get("False Negatives", 0)
        tn = metrics.get("True Negatives", 0)
        
        lines.append(
            f"{cls} & {acc:.2f} & {precision:.3f} & {f1:.3f} & {support} & {recall:.3f} & {tp} & {fp} & {fn} & {tn} \\\\"
        )
    lines.append("\\hline")

    # If "overall" block is missing, just end
    overall = report.get("overall", {})
    if not overall:
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{table_caption}}}")
        lines.append(f"\\label{{{table_label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)
    
    # 2) Optional: add rows for Macro Avg, Weighted Avg, etc. (omitting the per-class Acc)
    # We'll “pretend” we have columns: Class, Acc(%), Prec(%), F1(%), Support, ...
    # so for Macro/Weighted we put a dash (or empty) in the Acc(%) column.
    
    if "macro avg" in overall:
        metrics = overall["macro avg"]
        precision = metrics.get("precision", 0)
        a_recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        lines.append(
            f"Macro Avg & - & {precision:.3f} & {f1:.3f} & {support:.0f} & {a_recall:.3f} & - & - & - & - \\\\"
        )
    
    if "weighted avg" in overall:
        metrics = overall["weighted avg"]
        precision = metrics.get("precision", 0)
        w_recall = metrics.get("recall", 0.0) 
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        lines.append(
            f"Weighted Avg & - & {precision:.3f} & {f1:.3f} & {support:.0f} & {w_recall:.3f} & - & - & - & - \\\\"
        )
    
    # 3) If the JSON has "accuracy", print an “Accuracy” row
    if "accuracy" in overall:
        accuracy_val = overall["accuracy"] * 100
        lines.append(f"Accuracy & \\multicolumn{{8}}{{c}}{{{accuracy_val:.1f}}} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{table_caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def main():
    latex_code = json_to_latex_single_table(INPUT_JSON_FILE)
    with open(OUTPUT_FILE, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table code saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()


def main():
    latex_code = json_to_latex_scratch(INPUT_JSON_FILE)
    output_file = "classification_report_extended_AFreal.txt"
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
