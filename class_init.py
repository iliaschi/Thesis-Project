import os
import torch
from torchvision import models, transforms
from PIL import Image


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from datetime import datetime

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image



class ImageEmotionClustering:
    def __init__(self, folder_path, eps=0.7, min_samples=3):
        """
        Initialize image emotion clustering
        
        Parameters:
        -----------
        folder_path : str
            Path to folder containing images
        eps : float
            The maximum distance between two samples for them to be considered in the same cluster
        min_samples : int
            The number of samples in a neighborhood for a point to be considered as a core point
        """
        self.folder_path = os.path.normpath(folder_path)
        self.eps = eps
        self.min_samples = min_samples
        self.model = self._load_model()
        self.image_files = self._get_image_files()
        self.features = None
        self.features_2d = None
        self.clusters = None
    
    def _load_model(self):
        """Load EfficientNet model for feature extraction"""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        return base_model
    
    def _get_image_files(self):
        """Get list of image files in the folder"""
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
        image_files = [f for f in os.listdir(self.folder_path) 
                       if f.lower().endswith(supported_extensions)]
        print(f"Total image files found: {len(image_files)}")
        return image_files
    
    def extract_features(self):
        """Extract features from all images"""
        features = []
        processed_images = []
        
        print("Extracting emotional features from images...")
        for filename in self.image_files:
            try:
                image_path = os.path.join(self.folder_path, filename)
                print(f"Processing {filename}")
                
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_resized = img.resize((224, 224), Image.LANCZOS)
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                
                feature = self.model.predict(img_array, verbose=0)
                features.append(feature.flatten())
                processed_images.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        self.features = np.array(features)
        self.image_files = processed_images
        print(f"Features extracted for {len(self.features)} images")
        return self.features
    
    def cluster_emotions(self):
        """Cluster images based on emotional similarity"""
        if self.features is None:
            self.extract_features()
        
        print("\nClustering images based on emotional similarity...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Perform clustering using DBSCAN
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.clusters = clusterer.fit_predict(features_scaled)
        
        # Get number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        print(f"Found {n_clusters} emotion clusters")
        if -1 in self.clusters:
            n_noise = list(self.clusters).count(-1)
            print(f"Number of images not clearly belonging to any cluster: {n_noise}")
        
        # Reduce dimensions for visualization
        print("Creating emotional similarity map...")
        tsne = TSNE(n_components=2, random_state=42)
        self.features_2d = tsne.fit_transform(features_scaled)
        
        return self.clusters
    
    def visualize_emotion_clusters(self):
        """Create and save visualization of the emotion clusters"""
        if self.clusters is None:
            self.cluster_emotions()
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(
            self.features_2d[:, 0],
            self.features_2d[:, 1],
            c=self.clusters,
            cmap='viridis'
        )
        
        plt.title('Emotion Similarity Map')
        plt.colorbar(scatter, label='Emotion Cluster')
        plt.xlabel('Emotional Feature Dimension 1')
        plt.ylabel('Emotional Feature Dimension 2')
        
        # Add cluster statistics
        unique_clusters = sorted(set(self.clusters))
        cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
                        for i in unique_clusters if i != -1]
        if -1 in unique_clusters:
            cluster_stats.append(f"Unclustered: {sum(self.clusters == -1)} images")
        
        plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.folder_path, 'emotion_clusters_visualization.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def save_clustering_results(self):
        """Save clustering results to a text file"""
        if self.clusters is None:
            self.cluster_emotions()
        
        output_path = os.path.join(self.folder_path, 'emotion_clustering_results.txt')
        
        with open(output_path, 'w') as f:
            f.write("Emotion Clustering Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Write summary
            unique_clusters = sorted(set(self.clusters))
            f.write("Summary:\n")
            for cluster in unique_clusters:
                count = sum(self.clusters == cluster)
                if cluster == -1:
                    f.write(f"Unclustered Images: {count}\n")
                else:
                    f.write(f"Emotion Cluster {cluster}: {count} images\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Group images by cluster
            cluster_groups = {}
            for img, cluster in zip(self.image_files, self.clusters):
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(img)
            
            # Write detailed results
            f.write("Detailed Results:\n\n")
            for cluster in sorted(cluster_groups.keys()):
                if cluster == -1:
                    f.write("Unclustered Images:\n")
                else:
                    f.write(f"Emotion Cluster {cluster}:\n")
                for img in sorted(cluster_groups[cluster]):
                    f.write(f"  - {img}\n")
                f.write("\n")
        
        print(f"Clustering results saved to: {output_path}")

def main():
    # Configuration
    folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    
    # Create clustering instance with custom parameters
    # You can adjust these parameters to get more or fewer clusters:
    # - Lower eps value = more clusters
    # - Higher eps value = fewer clusters
    # - Higher min_samples = requires more similar images to form a cluster
    clustering = ImageEmotionClustering(folder_path, eps=0.5, min_samples=3)
    
    # Run clustering
    clustering.cluster_emotions()
    
    # Generate visualization
    clustering.visualize_emotion_clusters()
    
    # Save results
    clustering.save_clustering_results()

if __name__ == '__main__':
    main()


# def classify_images_in_folder(folder_path, top_k=3):
#     """
#     Classify images in a given folder using EfficientNet B0 model.
    
#     Parameters:
#     -----------
#     folder_path : str
#         Path to the folder containing images to classify
#     top_k : int, optional
#         Number of top predictions to return for each image (default is 3)
    
#     Returns:
#     --------
#     dict
#         A dictionary with image filenames as keys and classification results as values
#     """
#     # Load pre-trained EfficientNet B0 model
#     model = EfficientNetB0(weights='imagenet')
    
#     # Dictionary to store classification results
#     classification_results = {}
    
#     # Iterate through image files in the folder
#     for filename in os.listdir(folder_path):
#         # Check if file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             # Full path to the image
#             image_path = os.path.join(folder_path, filename)
            
#             try:
#                 # Load and preprocess the image
#                 img = load_img(image_path, target_size=(224, 224))
#                 img_array = img_to_array(img)
#                 img_array = np.expand_dims(img_array, axis=0)
#                 img_array = preprocess_input(img_array)
                
#                 # Predict
#                 predictions = model.predict(img_array)
                
#                 # Decode and store top-k predictions
#                 decoded_predictions = decode_predictions(predictions, top=top_k)[0]
#                 classification_results[filename] = [
#                     {'class': pred[1], 'confidence': float(pred[2])} 
#                     for pred in decoded_predictions
#                 ]
                
#             except Exception as e:
#                 classification_results[filename] = f"Error processing image: {str(e)}"
    
#     return classification_results

# def main():
#     # Specify the folder path containing your images
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    
#     # Classify images
#     results = classify_images_in_folder(folder_path)
    
#     # Print results
#     for filename, predictions in results.items():
#         print(f"\nImage: {filename}")
#         if isinstance(predictions, list):
#             for pred in predictions:
#                 print(f"  - Class: {pred['class']}, Confidence: {pred['confidence']:.2%}")
#         else:
#             print(f"  {predictions}")

# if __name__ == '__main__':
#     main()

# ################# try 2
# def classify_images_in_folder(folder_path, top_k=3):
#     """
#     Classify images in a given folder using EfficientNet B0 model.
    
#     Parameters:
#     -----------
#     folder_path : str
#         Path to the folder containing images to classify
#     top_k : int, optional
#         Number of top predictions to return for each image (default is 3)
    
#     Returns:
#     --------
#     dict
#         A dictionary with image filenames as keys and classification results as values
#     """
#     # Normalize path to handle different path formats
#     folder_path = os.path.normpath(folder_path)
    
#     # Load pre-trained EfficientNet B0 model
#     model = EfficientNetB0(weights='imagenet')
    
#     # Dictionary to store classification results
#     classification_results = {}
    
#     # Iterate through image files in the folder
#     for filename in os.listdir(folder_path):
#         # Check if file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             # Full path to the image
#             image_path = os.path.join(folder_path, filename)
            
#             try:
#                 # Load and preprocess the image
#                 img = load_img(image_path, target_size=(224, 224))
#                 img_array = img_to_array(img)
#                 img_array = np.expand_dims(img_array, axis=0)
#                 img_array = preprocess_input(img_array)
                
#                 # Predict
#                 predictions = model.predict(img_array)
                
#                 # Decode and store top-k predictions
#                 decoded_predictions = decode_predictions(predictions, top=top_k)[0]
#                 classification_results[filename] = [
#                     {'class': pred[1], 'confidence': float(pred[2])} 
#                     for pred in decoded_predictions
#                 ]
                
#             except Exception as e:
#                 classification_results[filename] = f"Error processing image: {str(e)}"
    
#     return classification_results

# def log_results(results, output_dir=None):
#     """
#     Log classification results to a timestamped text file.
    
#     Parameters:
#     -----------
#     results : dict
#         Dictionary of classification results
#     output_dir : str, optional
#         Directory to save the log file. If None, uses current directory.
    
#     Returns:
#     --------
#     str
#         Path to the created log file
#     """
#     # Create output directory if it doesn't exist
#     if output_dir is None:
#         output_dir = os.getcwd()
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate timestamp for filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_filename = os.path.join(output_dir, f"classification_results_{timestamp}.txt")
    
#     # Write results to file
#     with open(log_filename, 'w') as log_file:
#         log_file.write(f"Classification Results - {timestamp}\n")
#         log_file.write("=" * 50 + "\n\n")
        
#         for filename, predictions in results.items():
#             log_file.write(f"Image: {filename}\n")
#             if isinstance(predictions, list):
#                 for pred in predictions:
#                     log_file.write(f"  - Class: {pred['class']}, Confidence: {pred['confidence']:.2%}\n")
#             else:
#                 log_file.write(f"  {predictions}\n")
#             log_file.write("\n")
    
#     print(f"Results logged to: {log_filename}")
#     return log_filename

# def main():
#     # Specify the folder path containing your images
#     folder_path = os.path.normpath(r'C:\Users\ilias\Python\Thesis-Project\synthetic_images')
    
#     # Optional: Specify a different output directory for logs
#     # If not specified, logs will be saved in the current working directory
#     output_dir = os.path.normpath(r'C:\Users\ilias\Python\Thesis-Project\classification_logs')
    
#     # Classify images
#     results = classify_images_in_folder(folder_path)
    
#     # Log results to a text file
#     log_results(results, output_dir)
    
#     # Optional: Print results to console as well
#     for filename, predictions in results.items():
#         print(f"\nImage: {filename}")
#         if isinstance(predictions, list):
#             for pred in predictions:
#                 print(f"  - Class: {pred['class']}, Confidence: {pred['confidence']:.2%}")
#         else:
#             print(f"  {predictions}")

# if __name__ == '__main__':
#     main()