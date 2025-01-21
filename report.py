import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# def create_emotion_report(clusters_data, output_dir):
#     """
#     Create reports and visualizations for emotion distributions in clusters
    
#     Parameters:
#     clusters_data: dict of format {cluster_number: {'angry': percentage, 'sad': percentage, ...}}
#     output_dir: directory to save the results
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # List of emotions and colors for consistency
#     emotions = ['angry', 'sad', 'surprised', 'happy']
#     colors = ['red', 'blue', 'orange', 'green']
    
#     # Create text report
#     report_path = os.path.join(output_dir, f'emotion_distribution_{timestamp}.txt')
#     with open(report_path, 'w') as f:
#         f.write("Emotion Distribution Analysis\n")
#         f.write("=" * 50 + "\n\n")
#         f.write(f"Analysis Date: {timestamp}\n\n")
        
#         for cluster in sorted(clusters_data.keys()):
#             f.write(f"Cluster {cluster}: ")
#             distributions = []
#             for emotion in emotions:
#                 percentage = clusters_data[cluster].get(emotion, 0)
#                 distributions.append(f"{emotion} {percentage:.1f}%")
#             f.write(" | ".join(distributions) + "\n")
    
#     # Create bar chart
#     plt.figure(figsize=(12, 6))
#     x = np.arange(len(clusters_data))
#     width = 0.2
    
#     for i, emotion in enumerate(emotions):
#         values = [clusters_data[cluster].get(emotion, 0) for cluster in sorted(clusters_data.keys())]
#         plt.bar(x + i*width, values, width, label=emotion.capitalize(), color=colors[i])
    
#     plt.xlabel('Clusters')
#     plt.ylabel('Percentage')
#     plt.title('Emotion Distribution Across Clusters')
#     plt.xticks(x + width*1.5, [f'Cluster {i}' for i in sorted(clusters_data.keys())])
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     bar_path = os.path.join(output_dir, f'emotion_distribution_bar_{timestamp}.png')
#     plt.savefig(bar_path, bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # Create pie charts for each cluster
#     fig = plt.figure(figsize=(15, 5))
#     for i, cluster in enumerate(sorted(clusters_data.keys()), 1):
#         plt.subplot(1, len(clusters_data), i)
#         values = [clusters_data[cluster].get(emotion, 0) for emotion in emotions]
#         plt.pie(values, labels=[e.capitalize() for e in emotions], colors=colors,
#                 autopct='%1.1f%%', startangle=90)
#         plt.title(f'Cluster {cluster}')
    
#     plt.tight_layout()
#     pie_path = os.path.join(output_dir, f'emotion_distribution_pie_{timestamp}.png')
#     plt.savefig(pie_path, bbox_inches='tight', dpi=300)
#     plt.close()
    
#     return report_path, bar_path, pie_path

# def main():
#     # Example data - replace with your actual data
#     clusters_data = {
#         1: {'angry': 65, 'surprised': 28, 'sad': 12, 'happy': 5},
#         2: {'angry': 11, 'surprised': 13, 'sad': 55, 'happy': 26},
#         3: {'angry': 17, 'surprised': 59, 'sad': 19, 'happy': 16},
#         3: {'angry': 13, 'surprised': 18, 'sad': 7, 'happy': 62}
#     }
    
#     # Output directory
#     output_dir = r'C:\Users\ilias\Python\Thesis-Project\results_reports_11'
    
#     # Generate reports and visualizations
#     report_path, bar_path, pie_path = create_emotion_report(clusters_data, output_dir)
    
#     print(f"Report generated at: {report_path}")
#     print(f"Bar chart saved at: {bar_path}")
#     print(f"Pie charts saved at: {pie_path}")

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def create_emotion_report(clusters_data, output_dir):
    """
    Create reports and visualizations for emotion distributions in clusters
    
    Parameters:
    clusters_data: dict of format {cluster_number: {'angry': percentage, 'sad': percentage, ...}}
    output_dir: directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # List of emotions and colors for consistency
    emotions = ['angry', 'sad', 'surprised', 'happy']
    colors = ['red', 'blue', 'orange', 'green']
    
    # Create text report
    report_path = os.path.join(output_dir, f'emotion_distribution_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Emotion Distribution Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {timestamp}\n\n")
        
        for cluster in sorted(clusters_data.keys()):
            f.write(f"Cluster {cluster}: ")
            distributions = []
            for emotion in emotions:
                percentage = clusters_data[cluster].get(emotion, 0)
                distributions.append(f"{emotion} {percentage:.1f}%")
            f.write(" | ".join(distributions) + "\n")
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(clusters_data))
    width = 0.2
    
    for i, emotion in enumerate(emotions):
        values = [clusters_data[cluster].get(emotion, 0) for cluster in sorted(clusters_data.keys())]
        plt.bar(x + i*width, values, width, label=emotion.capitalize(), color=colors[i])
    
    plt.xlabel('Clusters')
    plt.ylabel('Percentage')
    plt.title('Emotion Distribution Across Clusters')
    plt.xticks(x + width*1.5, [f'Cluster {i}' for i in sorted(clusters_data.keys())])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    bar_path = os.path.join(output_dir, f'emotion_distribution_bar_{timestamp}.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create pie charts for each cluster
    fig = plt.figure(figsize=(20, 5))  # Made wider to accommodate 4 clusters
    for i, cluster in enumerate(sorted(clusters_data.keys()), 1):
        plt.subplot(1, len(clusters_data), i)
        values = [clusters_data[cluster].get(emotion, 0) for emotion in emotions]
        plt.pie(values, labels=[e.capitalize() for e in emotions], colors=colors,
                autopct='%1.1f%%', startangle=90)
        plt.title(f'Cluster {cluster}')
    
    plt.tight_layout()
    pie_path = os.path.join(output_dir, f'emotion_distribution_pie_{timestamp}.png')
    plt.savefig(pie_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return report_path, bar_path, pie_path

def main():
    # Example data with 4 distinct clusters focused on different emotions
    clusters_data = {
        1: {'angry': 65, 'surprised': 15, 'sad': 12, 'happy': 8},  # Anger-dominant cluster
        2: {'angry': 11, 'surprised': 13, 'sad': 58, 'happy': 18}, # Sadness-dominant cluster
        3: {'angry': 17, 'surprised': 59, 'sad': 14, 'happy': 10}, # Surprise-dominant cluster
        4: {'angry': 13, 'surprised': 18, 'sad': 7, 'happy': 62}  # Happiness-dominant cluster
    }
    
    # Output directory
    output_dir = r'C:\Users\ilias\Python\Thesis-Project\results_reports_11'
    
    # Generate reports and visualizations
    report_path, bar_path, pie_path = create_emotion_report(clusters_data, output_dir)
    
    print(f"Report generated at: {report_path}")
    print(f"Bar chart saved at: {bar_path}")
    print(f"Pie charts saved at: {pie_path}")

if __name__ == "__main__":
    main()