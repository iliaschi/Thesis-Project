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



# Class accuracy line plot
def plot_global_per_emotion_accuracy_line_1(eval_out_dir, global_metrics_df, class_names, custom_folder_order=None):
    """
    Creates and saves a line plot comparing per-emotion accuracy across models.
    The DataFrame is expected to have columns like 'accuracy_{emotion}' for each emotion in 'class_names'.
    
    :param eval_out_dir: output directory for saving the plot
    :param global_metrics_df: DataFrame with at least the columns: 
        'model_folder' and 'accuracy_{emotion}' for each emotion in class_names
    :param class_names: list of strings, e.g. ["Angry","Disgust",...]
    :param custom_folder_order: optional list specifying the exact x-axis order 
        (e.g. ["0","25","50","75","100"]). If provided, any folder outside this list is placed at the end.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1) Build a list of columns corresponding to each emotion
    emotion_columns = [f'accuracy_{emotion}' for emotion in class_names]
    
    # 2) Melt the DataFrame into long format
    #    so we have columns: model_folder, overall_accuracy, emotion, accuracy
    global_metrics_long = pd.melt(
        global_metrics_df,
        id_vars=['model_folder', 'overall_accuracy'],
        value_vars=emotion_columns,
        var_name='emotion',
        value_name='accuracy'
    )
    # remove the "accuracy_" prefix
    global_metrics_long['emotion'] = global_metrics_long['emotion'].str.replace('accuracy_', '')
    
    # 3) If a custom folder order is given, sort the DataFrame accordingly
    if custom_folder_order is not None:
        # build a dict => folder to sort index
        folder_to_sortidx = {val: i for i, val in enumerate(custom_folder_order)}
        # add a sort_idx col, default 999 if not in order
        global_metrics_long['sort_idx'] = global_metrics_long['model_folder'].apply(lambda x: folder_to_sortidx.get(x, 999))
        global_metrics_long.sort_values('sort_idx', inplace=True)
    else:
        # default: just let them appear in whatever order they appear
        pass

    # 4) Create the line plot
    plt.figure(figsize=(16, 16))
    ax = sns.lineplot(
        x='model_folder',
        y='accuracy',
        hue='emotion',
        data=global_metrics_long,
        marker='o',
        palette=sns.color_palette("Paired", len(class_names)),
        sort=False  # we rely on the DF sort order
    )
    plt.xticks(rotation=90)
    plt.title('Per-Emotion Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model Folder')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), title="Emotion")
    plt.tight_layout()

    # 5) Save the plot
    filename = f"per_emot_accu_line_{len(class_names)}cls.png"
    line_plot_path = os.path.join(eval_out_dir, filename)
    plt.savefig(line_plot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-emotion accuracy line plot saved at {line_plot_path}")

def plot_global_per_emotion_accuracy_line(eval_out_dir, global_metrics_df, class_names, custom_folder_order=None):

        emotion_columns = [f'accuracy_{emotion}' for emotion in class_names]
        long_df = pd.melt(
            global_metrics_df,
            id_vars=['model_folder', 'overall_accuracy'],
            value_vars=emotion_columns,
            var_name='emotion',
            value_name='accuracy'
        )
        long_df['emotion'] = long_df['emotion'].str.replace('accuracy_', '')

        if custom_folder_order is not None:
            folder_to_sortidx = {val: i for i, val in enumerate(custom_folder_order)}
            long_df['sort_idx'] = long_df['model_folder'].apply(lambda x: folder_to_sortidx.get(x, 999))
            long_df.sort_values('sort_idx', inplace=True)

        plt.figure(figsize=(10, 6))  # a moderate size
        palette = sns.color_palette("Set2", len(class_names))  # choose a palette
        ax = sns.lineplot(
            x='model_folder',
            y='accuracy',
            hue='emotion',
            data=long_df,
            marker='o',
            markersize=8,
            linewidth=2,
            palette=palette,
            sort=False
        )
        plt.xticks(rotation=45)
        plt.title('Emotion Class Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Data Fraction')
        ax.set_ylim(0, 100)
        # place legend outside
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Emotion")
        plt.grid(axis='y', alpha=0.6)
        plt.tight_layout()

        filename = f"per_emot_accu_line_{len(class_names)}cls.png"
        line_plot_path = os.path.join(eval_out_dir, filename)
        plt.savefig(line_plot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Per-emotion accuracy line plot saved at {line_plot_path}")



# Now an if name == main plot the script name
if __name__ == "__main__":
    print("This script is not meant to be run directly.")
    print(" file name = ", os.path.basename(__file__))