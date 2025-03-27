import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def plot_individual_per_emotion_accuracy(model_eval_dir, model_folder, metrics_entry, class_names):
def plot_individual_per_emotion_accuracy(model_eval_dir, metrics_entry, class_names):
    """
    Creates and saves a single bar plot of per-emotion accuracy for one model.
    The plot filename includes the number of classes to avoid overwriting
    when switching between 7- and 8-class scenarios.
    """
    # Extract accuracies for each emotion
    accuracies = [metrics_entry[f'accuracy_{emotion}'] for emotion in class_names]

    # Create the figure
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_names, y=accuracies, palette=sns.color_palette("Paired", len(class_names)))
    plt.title(f'Per-Emotion Accuracy for {len(class_names)} Classes')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Emotion')
    plt.tight_layout()

    # Build unique filename using len(class_names)
    filename = f"per_emot_acc_{len(class_names)}cls.png"
    individual_plot_path = os.path.join(model_eval_dir, filename)

    plt.savefig(individual_plot_path)
    plt.close()
    print(f"[INFO] Individual per-emotion bar plot saved at {individual_plot_path}")

def plot_global_overall_accuracy(eval_out_dir, global_metrics_df, class_names):
    """
    Creates and saves a bar plot showing overall accuracy for each model in global_metrics_df.
    The plot filename includes the number of classes to avoid overwriting
    when switching between 7- and 8-class scenarios.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model_folder', y='overall_accuracy', data=global_metrics_df,
                palette=sns.color_palette("Paired", len(global_metrics_df)))
    plt.xticks(rotation=90)
    plt.title('Overall Accuracy Comparison')
    plt.ylabel('Overall Accuracy (%)')
    plt.xlabel('Model Folder')
    plt.tight_layout()

    filename = f"overall_acc_bar_{len(class_names)}cls.png"
    overall_barplot_path = os.path.join(eval_out_dir, filename)
    plt.savefig(overall_barplot_path)
    plt.close()
    print(f"[INFO] Overall accuracy bar plot saved at {overall_barplot_path}")

def plot_global_per_emotion_accuracy(eval_out_dir, global_metrics_df, class_names):
    """
    Creates and saves a grouped bar plot comparing per-emotion accuracy across models.
    Includes the number of classes in the filename.
    """
    emotion_columns = [f'accuracy_{emotion}' for emotion in class_names]
    # Melt into long form
    global_metrics_long = pd.melt(global_metrics_df,
                                  id_vars=['model_folder', 'overall_accuracy'],
                                  value_vars=emotion_columns,
                                  var_name='emotion', value_name='accuracy')
    # Remove the prefix from 'accuracy_'
    global_metrics_long['emotion'] = global_metrics_long['emotion'].str.replace('accuracy_', '')

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
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), title="Emotion")
    plt.tight_layout()

    filename = f"per_emot_accu_bar_{len(class_names)}cls.png"
    per_emotion_barplot_path = os.path.join(eval_out_dir, filename)
    plt.savefig(per_emotion_barplot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-emotion accuracy bar plot saved at {per_emotion_barplot_path}")

def plot_global_top2_accuracy(eval_out_dir, global_metrics_df, class_names):
    """
    Creates and saves a grouped bar plot for per-emotion Top-2 accuracy across models.
    Includes the number of classes in the filename.
    """
    emotion_columns_top2 = [f'top2_accuracy_{emotion}' for emotion in class_names]
    global_metrics_long_top2 = pd.melt(global_metrics_df,
                                       id_vars=['model_folder'],
                                       value_vars=emotion_columns_top2,
                                       var_name='emotion', value_name='top2_accuracy')
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

    filename = f"per_emot_top2_acc_bar_{len(class_names)}cls.png"
    top2_barplot_path = os.path.join(eval_out_dir, filename)
    plt.savefig(top2_barplot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-emotion Top-2 accuracy bar plot saved at {top2_barplot_path}")

def plot_global_nll(eval_out_dir, global_metrics_df, class_names):
    """
    Creates and saves a grouped bar plot for per-emotion average Negative Log Likelihood (NLL).
    Includes the number of classes in the filename.
    """
    emotion_columns_nll = [f'nll_{emotion}' for emotion in class_names]
    global_metrics_long_nll = pd.melt(global_metrics_df,
                                      id_vars=['model_folder'],
                                      value_vars=emotion_columns_nll,
                                      var_name='emotion', value_name='avg_nll')
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
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), title="Emotion")
    plt.tight_layout()

    filename = f"per_emot_nll_bar_{len(class_names)}cls.png"
    nll_barplot_path = os.path.join(eval_out_dir, filename)
    plt.savefig(nll_barplot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-emotion NLL bar plot saved at {nll_barplot_path}")

# Now an if name == main plot the script name
if __name__ == "__main__":
    print("This script is not meant to be run directly.")
    print(" file name = ", os.path.basename(__file__))