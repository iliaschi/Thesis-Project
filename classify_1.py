import os
import torch
from torchvision import models, transforms
from PIL import Image


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime


################## try 3

# import os
# import numpy as np
# import tensorflow as tf
# from datetime import datetime
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from PIL import Image

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
#     tuple
#         (classification results dictionary, total files count, image files count)
#     """
#     # Normalize path to handle different path formats
#     folder_path = os.path.normpath(folder_path)
    
#     # Load pre-trained EfficientNet B0 model
#     model = EfficientNetB0(weights='imagenet')
    
#     # Dictionary to store classification results
#     classification_results = {}
    
#     # Get list of all files 
#     all_files = os.listdir(folder_path)
#     total_files_count = len(all_files)
    
#     # Filter and process image files
#     image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif'))]
#     image_files_count = len(image_files)
    
#     print(f"Total files in folder: {total_files_count}")
#     print(f"Total image files: {image_files_count}")
    
#     # Iterate through image files in the folder
#     for filename in image_files:
#         # Full path to the image
#         image_path = os.path.join(folder_path, filename)
        
#         try:
#             # Print debugging info for each image
#             print(f"Processing image: {filename}")
            
#             # Open image and resize if necessary
#             with Image.open(image_path) as img:
#                 # Check image size and resize if larger than model's input size
#                 original_size = img.size
#                 print(f"Original image size: {original_size}")
                
#                 # Resize image maintaining aspect ratio if larger than 224x224
#                 if original_size[0] > 224 or original_size[1] > 224:
#                     print("Resizing image to 224x224")
#                     img = tf.keras.preprocessing.image.smart_resize(
#                         tf.keras.preprocessing.image.img_to_array(img), 
#                         (224, 224)
#                     )
#                 else:
#                     img = tf.keras.preprocessing.image.img_to_array(img)
            
#             # Preprocess the image
#             img_array = np.expand_dims(img, axis=0)
#             img_array = preprocess_input(img_array)
            
#             # Predict
#             predictions = model.predict(img_array)
            
#             # Decode and store top-k predictions
#             decoded_predictions = decode_predictions(predictions, top=top_k)[0]
            
#             # Print raw predictions for debugging
#             print("Raw predictions:")
#             for pred in decoded_predictions:
#                 print(f"  - Class: {pred[1]}, Confidence: {pred[2]:.2%}")
            
#             classification_results[filename] = [
#                 {'class': pred[1], 'confidence': float(pred[2])} 
#                 for pred in decoded_predictions
#             ]
            
#         except Exception as e:
#             print(f"Error processing {filename}: {str(e)}")
#             classification_results[filename] = f"Error processing image: {str(e)}"
    
#     # Print total number of processed images
#     print(f"Total images processed: {len(classification_results)}")
    
#     return classification_results, total_files_count, image_files_count

# def log_results(results, total_files, image_files, output_dir=None):
#     """
#     Log classification results to a timestamped text file.
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
        
#         # Write file count information
#         log_file.write(f"Total files in folder: {total_files}\n")
#         log_file.write(f"Total image files: {image_files}\n")
#         log_file.write(f"Images processed: {len(results)}\n\n")
        
#         if not results:
#             log_file.write("No images were processed successfully.\n")
        
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
#     output_dir = os.path.normpath(r'C:\Users\ilias\Python\Thesis-Project\classification_logs')
    
#     # Classify images
#     results, total_files, image_files = classify_images_in_folder(folder_path)
    
#     # Log results to a text file
#     log_results(results, total_files, image_files, output_dir)

# if __name__ == '__main__':
#     main()




### test 4

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from PIL import Image

# class ImageSimilarityClustering:
#     def __init__(self, folder_path, model='efficientnet'):
#         """
#         Initialize image similarity clustering
        
#         Parameters:
#         -----------
#         folder_path : str
#             Path to folder containing images
#         model : str, optional
#             Model to use for feature extraction (default: 'efficientnet')
#         """
#         self.folder_path = os.path.normpath(folder_path)
#         self.model = self._load_model(model)
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.clusters = None
    
#     def _load_model(self, model_name):
#         """
#         Load pre-trained model for feature extraction
#         """
#         if model_name == 'efficientnet':
#             # Load EfficientNet B0 without top layers to extract features
#             base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#             return base_model
#         else:
#             raise ValueError("Unsupported model")
    
#     def _get_image_files(self):
#         """
#         Get list of image files in the folder
#         """
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                        if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         """
#         Extract features from images using the pre-trained model
        
#         Returns:
#         --------
#         numpy.ndarray
#             Array of extracted features
#         """
#         features = []
#         processed_images = []
        
#         for filename in self.image_files:
#             try:
#                 # Full path to the image
#                 image_path = os.path.join(self.folder_path, filename)
                
#                 # Open and preprocess image
#                 with Image.open(image_path) as img:
#                     # Convert to RGB if needed
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
                    
#                     # Resize image
#                     img_resized = img.resize((224, 224), Image.LANCZOS)
                    
#                     # Convert to array
#                     img_array = img_to_array(img_resized)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input(img_array)
                
#                 # Extract features
#                 feature = self.model.predict(img_array)
#                 features.append(feature.flatten())
#                 processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
        
#         print(f"Features extracted for {len(self.features)} images")
#         return self.features
    
#     def cluster_images(self, method='kmeans', n_clusters=5):
#         """
#         Cluster images based on extracted features
        
#         Parameters:
#         -----------
#         method : str, optional
#             Clustering method ('kmeans' or 'dbscan')
#         n_clusters : int, optional
#             Number of clusters for K-means
        
#         Returns:
#         --------
#         numpy.ndarray
#             Cluster labels for each image
#         """
#         if self.features is None:
#             self.extract_features()
        
#         # Standardize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(self.features)
        
#         # Dimensionality reduction for visualization
#         tsne = TSNE(n_components=2, random_state=42)
#         features_2d = tsne.fit_transform(features_scaled)
        
#         # Clustering
#         if method == 'kmeans':
#             clusterer = KMeans(n_clusters=n_clusters, random_state=42)
#             self.clusters = clusterer.fit_predict(features_scaled)
#         elif method == 'dbscan':
#             clusterer = DBSCAN(eps=0.5, min_samples=3)
#             self.clusters = clusterer.fit_predict(features_scaled)
#         else:
#             raise ValueError("Unsupported clustering method")
        
#         return self.clusters
    
#     def visualize_clusters(self):
#         """
#         Visualize clusters using t-SNE
#         """
#         if self.clusters is None:
#             self.cluster_images()
        
#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(
#             self.features_2d[:, 0], 
#             self.features_2d[:, 1], 
#             c=self.clusters, 
#             cmap='viridis'
#         )
#         plt.title('Image Clusters Visualization')
#         plt.colorbar(scatter)
#         plt.xlabel('t-SNE Feature 1')
#         plt.ylabel('t-SNE Feature 2')
#         plt.tight_layout()
        
#         # Save the plot
#         output_path = os.path.join(
#             self.folder_path, 
#             'cluster_visualization.png'
#         )
#         plt.savefig(output_path)
#         print(f"Cluster visualization saved to {output_path}")
    
#     def save_cluster_results(self):
#         """
#         Save clustering results to a text file
#         """
#         if self.clusters is None:
#             self.cluster_images()
        
#         output_path = os.path.join(
#             self.folder_path, 
#             'clustering_results.txt'
#         )
        
#         with open(output_path, 'w') as f:
#             f.write("Image Clustering Results\n")
#             f.write("=" * 30 + "\n\n")
            
#             # Group images by cluster
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             # Write clusters
#             for cluster, images in cluster_groups.items():
#                 f.write(f"Cluster {cluster}:\n")
#                 for img in images:
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Clustering results saved to {output_path}")

# def main():
#     # Specify the folder path containing your images
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    
#     # Create clustering instance
#     clustering = ImageSimilarityClustering(folder_path)
    
#     # Extract features
#     clustering.extract_features()
    
#     # Cluster images (you can choose 'kmeans' or 'dbscan')
#     clustering.cluster_images(method='kmeans', n_clusters=5)
    
#     # Visualize clusters
#     clustering.visualize_clusters()
    
#     # Save clustering results
#     clustering.save_cluster_results()

# if __name__ == '__main__':
#     main()



################## try 5

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from PIL import Image

# class ImageSimilarityClustering:
#     def __init__(self, folder_path, model='efficientnet'):
#         self.folder_path = os.path.normpath(folder_path)
#         self.model = self._load_model(model)
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.features_2d = None  # Added this line
#         self.clusters = None
    
#     def _load_model(self, model_name):
#         if model_name == 'efficientnet':
#             base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#             return base_model
#         else:
#             raise ValueError("Unsupported model")
    
#     def _get_image_files(self):
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                        if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         features = []
#         processed_images = []
        
#         for filename in self.image_files:
#             try:
#                 image_path = os.path.join(self.folder_path, filename)
#                 print(f"Processing {filename}")  # Added progress indicator
                
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
#                     img_resized = img.resize((224, 224), Image.LANCZOS)
#                     img_array = img_to_array(img_resized)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input(img_array)
                
#                 feature = self.model.predict(img_array, verbose=0)  # Reduced verbosity
#                 features.append(feature.flatten())
#                 processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
        
#         print(f"Features extracted for {len(self.features)} images")
#         return self.features
    
#     def cluster_images(self, method='kmeans', n_clusters=5):
#         if self.features is None:
#             self.extract_features()
        
#         # Standardize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(self.features)
        
#         print("Performing t-SNE dimensionality reduction...")
#         # Dimensionality reduction for visualization
#         tsne = TSNE(n_components=2, random_state=42)
#         self.features_2d = tsne.fit_transform(features_scaled)  # Store the 2D features
        
#         print(f"Clustering images using {method}...")
#         # Clustering
#         if method == 'kmeans':
#             clusterer = KMeans(n_clusters=n_clusters, random_state=42)
#             self.clusters = clusterer.fit_predict(features_scaled)
#         elif method == 'dbscan':
#             clusterer = DBSCAN(eps=0.5, min_samples=3)
#             self.clusters = clusterer.fit_predict(features_scaled)
#         else:
#             raise ValueError("Unsupported clustering method")
        
#         print(f"Found {len(np.unique(self.clusters))} clusters")
#         return self.clusters
    
#     def visualize_clusters(self):
#         if self.clusters is None:
#             self.cluster_images()
        
#         plt.figure(figsize=(10, 8))
#         scatter = plt.scatter(
#             self.features_2d[:, 0], 
#             self.features_2d[:, 1], 
#             c=self.clusters, 
#             cmap='viridis'
#         )
#         plt.title('Image Clusters Visualization')
#         plt.colorbar(scatter)
#         plt.xlabel('t-SNE Feature 1')
#         plt.ylabel('t-SNE Feature 2')
#         plt.tight_layout()
        
#         output_path = os.path.join(
#             self.folder_path, 
#             'cluster_visualization.png'
#         )
#         plt.savefig(output_path)
#         print(f"Cluster visualization saved to {output_path}")
#         plt.close()  # Added to free memory
    
#     def save_cluster_results(self):
#         if self.clusters is None:
#             self.cluster_images()
        
#         output_path = os.path.join(
#             self.folder_path, 
#             'clustering_results.txt'
#         )
        
#         with open(output_path, 'w') as f:
#             f.write("Image Clustering Results\n")
#             f.write("=" * 30 + "\n\n")
            
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             for cluster in sorted(cluster_groups.keys()):  # Sort clusters for better readability
#                 f.write(f"Cluster {cluster}:\n")
#                 for img in sorted(cluster_groups[cluster]):  # Sort images within clusters
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Clustering results saved to {output_path}")

# def main():
#     # Specify the folder path containing your images
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    
#     # Create clustering instance
#     clustering = ImageSimilarityClustering(folder_path)
    
#     # Extract features
#     clustering.extract_features()
    
#     # Cluster images (you can adjust number of clusters based on your needs)
#     clustering.cluster_images(method='kmeans', n_clusters=5)
    
#     # Visualize clusters
#     clustering.visualize_clusters()
    
#     # Save clustering results
#     clustering.save_cluster_results()

# if __name__ == '__main__':
#     main()


################## try 6

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from PIL import Image

# class ImageEmotionClustering:
#     def __init__(self, folder_path, eps=0.7, min_samples=3):
#         """
#         Initialize image emotion clustering
        
#         Parameters:
#         -----------
#         folder_path : str
#             Path to folder containing images
#         eps : float
#             The maximum distance between two samples for them to be considered in the same cluster
#         min_samples : int
#             The number of samples in a neighborhood for a point to be considered as a core point
#         """
#         self.folder_path = os.path.normpath(folder_path)
#         self.eps = eps
#         self.min_samples = min_samples
#         self.model = self._load_model()
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.features_2d = None
#         self.clusters = None
    
#     def _load_model(self):
#         """Load EfficientNet model for feature extraction"""
#         base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#         return base_model
    
#     def _get_image_files(self):
#         """Get list of image files in the folder"""
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                        if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         """Extract features from all images"""
#         features = []
#         processed_images = []
        
#         print("Extracting emotional features from images...")
#         for filename in self.image_files:
#             try:
#                 image_path = os.path.join(self.folder_path, filename)
#                 print(f"Processing {filename}")
                
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
#                     img_resized = img.resize((224, 224), Image.LANCZOS)
#                     img_array = img_to_array(img_resized)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input(img_array)
                
#                 feature = self.model.predict(img_array, verbose=0)
#                 features.append(feature.flatten())
#                 processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
#         print(f"Features extracted for {len(self.features)} images")
#         return self.features
    
#     def cluster_emotions(self):
#         """Cluster images based on emotional similarity"""
#         if self.features is None:
#             self.extract_features()
        
#         print("\nClustering images based on emotional similarity...")
        
#         # Standardize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(self.features)
        
#         # Perform clustering using DBSCAN
#         clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
#         self.clusters = clusterer.fit_predict(features_scaled)
        
#         # Get number of clusters (excluding noise points labeled as -1)
#         n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
#         print(f"Found {n_clusters} emotion clusters")
#         if -1 in self.clusters:
#             n_noise = list(self.clusters).count(-1)
#             print(f"Number of images not clearly belonging to any cluster: {n_noise}")
        
#         # Reduce dimensions for visualization
#         print("Creating emotional similarity map...")
#         tsne = TSNE(n_components=2, random_state=42)
#         self.features_2d = tsne.fit_transform(features_scaled)
        
#         return self.clusters
    
#     def visualize_emotion_clusters(self):
#         """Create and save visualization of the emotion clusters"""
#         if self.clusters is None:
#             self.cluster_emotions()
        
#         plt.figure(figsize=(12, 8))
        
#         # Create scatter plot
#         scatter = plt.scatter(
#             self.features_2d[:, 0],
#             self.features_2d[:, 1],
#             c=self.clusters,
#             cmap='viridis'
#         )
        
#         plt.title('Emotion Similarity Map')
#         plt.colorbar(scatter, label='Emotion Cluster')
#         plt.xlabel('Emotional Feature Dimension 1')
#         plt.ylabel('Emotional Feature Dimension 2')
        
#         # Add cluster statistics
#         unique_clusters = sorted(set(self.clusters))
#         cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
#                         for i in unique_clusters if i != -1]
#         if -1 in unique_clusters:
#             cluster_stats.append(f"Unclustered: {sum(self.clusters == -1)} images")
        
#         plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8)
        
#         plt.tight_layout()
        
#         output_path = os.path.join(self.folder_path, 'emotion_clusters_visualization.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         print(f"\nVisualization saved to: {output_path}")
#         plt.close()
    
#     def save_clustering_results(self):
#         """Save clustering results to a text file"""
#         if self.clusters is None:
#             self.cluster_emotions()
        
#         output_path = os.path.join(self.folder_path, 'emotion_clustering_results.txt')
        
#         with open(output_path, 'w') as f:
#             f.write("Emotion Clustering Results\n")
#             f.write("=" * 50 + "\n\n")
            
#             # Write summary
#             unique_clusters = sorted(set(self.clusters))
#             f.write("Summary:\n")
#             for cluster in unique_clusters:
#                 count = sum(self.clusters == cluster)
#                 if cluster == -1:
#                     f.write(f"Unclustered Images: {count}\n")
#                 else:
#                     f.write(f"Emotion Cluster {cluster}: {count} images\n")
#             f.write("\n" + "=" * 50 + "\n\n")
            
#             # Group images by cluster
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             # Write detailed results
#             f.write("Detailed Results:\n\n")
#             for cluster in sorted(cluster_groups.keys()):
#                 if cluster == -1:
#                     f.write("Unclustered Images:\n")
#                 else:
#                     f.write(f"Emotion Cluster {cluster}:\n")
#                 for img in sorted(cluster_groups[cluster]):
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Clustering results saved to: {output_path}")

# def main():
#     # Configuration
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    
#     # Create clustering instance with custom parameters
#     # You can adjust these parameters to get more or fewer clusters:
#     # - Lower eps value = more clusters
#     # - Higher eps value = fewer clusters
#     # - Higher min_samples = requires more similar images to form a cluster
#     clustering = ImageEmotionClustering(folder_path, eps=0.5, min_samples=3)
    
#     # Run clustering
#     clustering.cluster_emotions()
    
#     # Generate visualization
#     clustering.visualize_emotion_clusters()
    
#     # Save results
#     clustering.save_clustering_results()

# if __name__ == '__main__':
#     main()


################## try 7

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from PIL import Image

# class XGBoostImageClustering:
#     def __init__(self, folder_path, n_clusters=3):
#         self.folder_path = os.path.normpath(folder_path)
#         self.n_clusters = n_clusters
#         self.feature_extractor = self._load_feature_extractor()
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.features_2d = None
#         self.clusters = None
#         self.xgb_features = None
    
#     def _load_feature_extractor(self):
#         return EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    
#     def _get_image_files(self):
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                       if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         features = []
#         processed_images = []
        
#         print("Extracting initial features...")
#         for filename in self.image_files:
#             try:
#                 image_path = os.path.join(self.folder_path, filename)
#                 print(f"Processing {filename}")
                
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
#                     img_resized = img.resize((224, 224), Image.LANCZOS)
#                     img_array = img_to_array(img_resized)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input(img_array)
                
#                 feature = self.feature_extractor.predict(img_array, verbose=0)
#                 features.append(feature.flatten())
#                 processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
#         print(f"Initial features extracted for {len(self.features)} images")
#         return self.features
    
#     def enhance_features_with_xgboost(self):
#         if self.features is None:
#             self.extract_features()
        
#         print("\nEnhancing features with XGBoost...")
        
#         # Create artificial targets for self-supervised learning
#         # We'll use the first feature dimension as target
#         target = self.features[:, 0]
        
#         # Prepare data for XGBoost
#         dtrain = xgb.DMatrix(self.features, label=target)
        
#         # XGBoost parameters
#         params = {
#             'objective': 'reg:squarederror',
#             'eval_metric': 'rmse',
#             'max_depth': 3,
#             'eta': 0.1,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'tree_method': 'hist'  # Added for better performance
#         }
        
#         # Train XGBoost model
#         model = xgb.train(params, dtrain, num_boost_round=100)
        
#         # Get enhanced features using XGBoost's leaf indices
#         self.xgb_features = model.predict(dtrain, pred_leaf=True)
        
#         print("Feature enhancement completed")
#         return self.xgb_features
    
#     def perform_clustering(self):
#         if self.xgb_features is None:
#             self.enhance_features_with_xgboost()
        
#         print("\nPerforming clustering...")
        
#         # Standardize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(self.xgb_features)
        
#         # Perform clustering
#         clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
#         self.clusters = clusterer.fit_predict(features_scaled)
        
#         # Create 2D representation for visualization
#         print("Creating 2D visualization...")
#         tsne = TSNE(n_components=2, random_state=42)
#         self.features_2d = tsne.fit_transform(features_scaled)
        
#         print(f"Clustering completed. Found {self.n_clusters} clusters")
#         return self.clusters
    
#     def visualize_clusters(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         plt.figure(figsize=(12, 8))
        
#         scatter = plt.scatter(
#             self.features_2d[:, 0],
#             self.features_2d[:, 1],
#             c=self.clusters,
#             cmap='viridis'
#         )
        
#         plt.title('Image Clusters')
#         plt.colorbar(scatter, label='Cluster')
#         plt.xlabel('Feature Dimension 1')
#         plt.ylabel('Feature Dimension 2')
        
#         cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
#                         for i in range(self.n_clusters)]
#         plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8)
        
#         plt.tight_layout()
        
#         output_path = os.path.join(self.folder_path, 'xgboost_clusters_visualization.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         print(f"\nVisualization saved to: {output_path}")
#         plt.close()
    
#     def save_results(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         output_path = os.path.join(self.folder_path, 'xgboost_clustering_results.txt')
        
#         with open(output_path, 'w') as f:
#             f.write("XGBoost Enhanced Clustering Results\n")
#             f.write("=" * 50 + "\n\n")
            
#             # Write summary
#             f.write("Summary:\n")
#             for i in range(self.n_clusters):
#                 count = sum(self.clusters == i)
#                 f.write(f"Cluster {i}: {count} images\n")
#             f.write("\n" + "=" * 50 + "\n\n")
            
#             # Group images by cluster
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             # Write detailed results
#             f.write("Detailed Results:\n\n")
#             for cluster in sorted(cluster_groups.keys()):
#                 f.write(f"Cluster {cluster}:\n")
#                 for img in sorted(cluster_groups[cluster]):
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Results saved to: {output_path}")

# def main():
#     # Configuration
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
#     n_clusters = 3
    
#     # Create clustering instance
#     clustering = XGBoostImageClustering(folder_path, n_clusters)
    
#     # Perform clustering
#     clustering.perform_clustering()
    
#     # Generate visualization
#     clustering.visualize_clusters()
    
#     # Save results
#     clustering.save_results()

# if __name__ == '__main__':
#     main()



################## try 8

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from PIL import Image

# class XGBoostImageClustering:
#     def __init__(self, folder_path, n_clusters=3):
#         self.folder_path = os.path.normpath(folder_path)
#         self.n_clusters = n_clusters
#         self.feature_extractor = self._load_feature_extractor()
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.features_2d = None
#         self.clusters = None
#         self.xgb_features = None
    
#     def _load_feature_extractor(self):
#         return EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    
#     def _get_image_files(self):
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                       if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         features = []
#         processed_images = []
        
#         print("Extracting initial features...")
#         for filename in self.image_files:
#             try:
#                 image_path = os.path.join(self.folder_path, filename)
#                 print(f"Processing {filename}")
                
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
#                     img_resized = img.resize((224, 224), Image.LANCZOS)
#                     img_array = img_to_array(img_resized)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input(img_array)
                
#                 feature = self.feature_extractor.predict(img_array, verbose=0)
#                 features.append(feature.flatten())
#                 processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
#         print(f"Initial features extracted for {len(self.features)} images")
#         return self.features
    
#     def enhance_features_with_xgboost(self):
#         if self.features is None:
#             self.extract_features()
        
#         print("\nEnhancing features with XGBoost...")
        
#         # Create artificial targets for self-supervised learning
#         # We'll use the first feature dimension as target
#         target = self.features[:, 0]
        
#         # Prepare data for XGBoost
#         dtrain = xgb.DMatrix(self.features, label=target)
        
#         # XGBoost parameters
#         params = {
#             'objective': 'reg:squarederror',
#             'eval_metric': 'rmse',
#             'max_depth': 3,
#             'eta': 0.1,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'tree_method': 'hist'  # Added for better performance
#         }
        
#         # Train XGBoost model
#         model = xgb.train(params, dtrain, num_boost_round=100)
        
#         # Get enhanced features using XGBoost's leaf indices
#         self.xgb_features = model.predict(dtrain, pred_leaf=True)
        
#         print("Feature enhancement completed")
#         return self.xgb_features
    
#     def perform_clustering(self):
#         if self.xgb_features is None:
#             self.enhance_features_with_xgboost()
        
#         print("\nPerforming clustering...")
        
#         # Standardize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(self.xgb_features)
        
#         # Perform clustering
#         clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
#         self.clusters = clusterer.fit_predict(features_scaled)
        
#         # Create 2D representation for visualization
#         print("Creating 2D visualization...")
#         tsne = TSNE(n_components=2, random_state=42)
#         self.features_2d = tsne.fit_transform(features_scaled)
        
#         print(f"Clustering completed. Found {self.n_clusters} clusters")
#         return self.clusters
    
#     def visualize_clusters(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         plt.figure(figsize=(12, 8))
        
#         scatter = plt.scatter(
#             self.features_2d[:, 0],
#             self.features_2d[:, 1],
#             c=self.clusters,
#             cmap='viridis'
#         )
        
#         plt.title('Image Clusters')
#         plt.colorbar(scatter, label='Cluster')
#         plt.xlabel('Feature Dimension 1')
#         plt.ylabel('Feature Dimension 2')
        
#         cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
#                         for i in range(self.n_clusters)]
#         plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8)
        
#         plt.tight_layout()
        
#         output_path = os.path.join(self.folder_path, 'xgboost_clusters_visualization.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         print(f"\nVisualization saved to: {output_path}")
#         plt.close()
    
#     def save_results(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         output_path = os.path.join(self.folder_path, 'xgboost_clustering_results.txt')
        
#         with open(output_path, 'w') as f:
#             f.write("XGBoost Enhanced Clustering Results\n")
#             f.write("=" * 50 + "\n\n")
            
#             # Write summary
#             f.write("Summary:\n")
#             for i in range(self.n_clusters):
#                 count = sum(self.clusters == i)
#                 f.write(f"Cluster {i}: {count} images\n")
#             f.write("\n" + "=" * 50 + "\n\n")
            
#             # Group images by cluster
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             # Write detailed results
#             f.write("Detailed Results:\n\n")
#             for cluster in sorted(cluster_groups.keys()):
#                 f.write(f"Cluster {cluster}:\n")
#                 for img in sorted(cluster_groups[cluster]):
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Results saved to: {output_path}")

# def main():
#     # Configuration
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
#     n_clusters = 4
    
#     # Create clustering instance
#     clustering = XGBoostImageClustering(folder_path, n_clusters)
    
#     # Perform clustering
#     clustering.perform_clustering()
    
#     # Generate visualization
#     clustering.visualize_clusters()
    
#     # Save results
#     clustering.save_results()

# if __name__ == '__main__':
#     main()


######## test 9

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# import xgboost as xgb
# import matplotlib.pyplot as plt
# from PIL import Image
# from datetime import datetime

# class XGBoostImageClustering:
#     def __init__(self, folder_path, n_clusters=3, results_dir='results_reports'):
#         self.folder_path = os.path.normpath(folder_path)
#         self.n_clusters = n_clusters
#         self.results_dir = os.path.normpath(results_dir)
#         os.makedirs(self.results_dir, exist_ok=True)
        
#         self.feature_extractor = self._load_feature_extractor()
#         self.image_files = self._get_image_files()
#         self.features = None
#         self.features_2d = None
#         self.clusters = None
#         self.xgb_features = None
        
#         # Create timestamped subfolder for this run
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.run_dir = os.path.join(self.results_dir, f"run_{self.timestamp}")
#         os.makedirs(self.run_dir, exist_ok=True)
    
#     def _load_feature_extractor(self):
#         base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#         return base_model
    
#     def _get_image_files(self):
#         supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
#         image_files = [f for f in os.listdir(self.folder_path) 
#                       if f.lower().endswith(supported_extensions)]
#         print(f"Total image files found: {len(image_files)}")
#         return image_files
    
#     def extract_features(self):
#         features = []
#         processed_images = []
        
#         print("Extracting features with enhanced processing...")
#         for filename in self.image_files:
#             try:
#                 image_path = os.path.join(self.folder_path, filename)
#                 print(f"Processing {filename}")
                
#                 # Enhanced image processing
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
                    
#                     # Multiple size processing for better feature extraction
#                     sizes = [(224, 224), (299, 299)]
#                     all_features = []
                    
#                     for size in sizes:
#                         img_resized = img.resize(size, Image.LANCZOS)
#                         img_array = img_to_array(img_resized)
#                         img_array = np.expand_dims(img_array, axis=0)
#                         img_array = preprocess_input(img_array)
                        
#                         feature = self.feature_extractor.predict(img_array, verbose=0)
#                         all_features.append(feature.flatten())
                    
#                     # Combine features from different sizes
#                     combined_feature = np.concatenate(all_features)
#                     features.append(combined_feature)
#                     processed_images.append(filename)
                
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
        
#         self.features = np.array(features)
#         self.image_files = processed_images
#         print(f"Features extracted for {len(self.features)} images")
#         return self.features
    
#     def enhance_features_with_xgboost(self):
#         if self.features is None:
#             self.extract_features()
        
#         print("\nEnhancing features with XGBoost (improved parameters)...")
        
#         # Create multiple target variables for better feature learning
#         targets = np.column_stack([
#             self.features[:, 0],  # First feature
#             np.mean(self.features, axis=1),  # Mean of features
#             np.std(self.features, axis=1)   # Standard deviation of features
#         ])
        
#         enhanced_features = []
        
#         # Train multiple XGBoost models for different aspects
#         for i, target in enumerate(targets.T):
#             dtrain = xgb.DMatrix(self.features, label=target)
            
#             # Enhanced XGBoost parameters
#             params = {
#                 'objective': 'reg:squarederror',
#                 'eval_metric': ['rmse', 'mae'],
#                 'max_depth': 6,
#                 'eta': 0.05,
#                 'subsample': 0.8,
#                 'colsample_bytree': 0.8,
#                 'min_child_weight': 3,
#                 'gamma': 0.2,
#                 'tree_method': 'hist',
#                 'num_parallel_tree': 3
#             }
            
#             # Train model
#             model = xgb.train(
#                 params, 
#                 dtrain, 
#                 num_boost_round=200,
#                 verbose_eval=False
#             )
            
#             # Get leaf features
#             leaf_features = model.predict(dtrain, pred_leaf=True)
#             enhanced_features.append(leaf_features)
        
#         # Combine all enhanced features
#         self.xgb_features = np.concatenate(enhanced_features, axis=1)
#         print("Feature enhancement completed with improved parameters")
#         return self.xgb_features
    
#     def perform_clustering(self):
#         if self.xgb_features is None:
#             self.enhance_features_with_xgboost()
        
#         print("\nPerforming enhanced clustering...")
        
#         # Advanced feature scaling
#         scaler = MinMaxScaler()
#         features_scaled = scaler.fit_transform(self.xgb_features)
        
#         # Try different numbers of clusters and select the best
#         best_score = -1
#         best_clusters = None
        
#         for n_clusters in range(max(2, self.n_clusters - 1), self.n_clusters + 2):
#             clusterer = AgglomerativeClustering(n_clusters=n_clusters)
#             labels = clusterer.fit_predict(features_scaled)
#             score = silhouette_score(features_scaled, labels)
            
#             print(f"Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
            
#             if score > best_score:
#                 best_score = score
#                 best_clusters = labels
#                 self.n_clusters = n_clusters
        
#         self.clusters = best_clusters
        
#         # Create 2D representation for visualization
#         print("Creating enhanced 2D visualization...")
#         tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.image_files)-1))
#         self.features_2d = tsne.fit_transform(features_scaled)
        
#         print(f"Clustering completed. Found {self.n_clusters} optimal clusters")
#         return self.clusters
    
#     def visualize_clusters(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         plt.figure(figsize=(12, 8))
        
#         scatter = plt.scatter(
#             self.features_2d[:, 0],
#             self.features_2d[:, 1],
#             c=self.clusters,
#             cmap='viridis',
#             s=100,  # Larger points
#             alpha=0.6  # Some transparency
#         )
        
#         plt.title('Image Clusters (Enhanced Visualization)')
#         plt.colorbar(scatter, label='Cluster')
#         plt.xlabel('t-SNE Feature 1')
#         plt.ylabel('t-SNE Feature 2')
        
#         # Add cluster statistics
#         cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
#                         for i in range(self.n_clusters)]
#         plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8, 
#                    bbox=dict(facecolor='white', alpha=0.8))
        
#         plt.tight_layout()
        
#         output_path = os.path.join(self.run_dir, 'cluster_visualization.png')
#         plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         print(f"\nVisualization saved to: {output_path}")
#         plt.close()
    
#     def save_results(self):
#         if self.clusters is None:
#             self.perform_clustering()
        
#         output_path = os.path.join(self.run_dir, 'clustering_results.txt')
        
#         with open(output_path, 'w') as f:
#             f.write("Enhanced Clustering Results\n")
#             f.write("=" * 50 + "\n\n")
            
#             f.write(f"Analysis Date: {self.timestamp}\n")
#             f.write(f"Total Images Analyzed: {len(self.image_files)}\n")
#             f.write(f"Optimal Number of Clusters: {self.n_clusters}\n\n")
            
#             # Write summary
#             f.write("Summary:\n")
#             for i in range(self.n_clusters):
#                 count = sum(self.clusters == i)
#                 percentage = (count / len(self.image_files)) * 100
#                 f.write(f"Cluster {i}: {count} images ({percentage:.1f}%)\n")
#             f.write("\n" + "=" * 50 + "\n\n")
            
#             # Group images by cluster
#             cluster_groups = {}
#             for img, cluster in zip(self.image_files, self.clusters):
#                 if cluster not in cluster_groups:
#                     cluster_groups[cluster] = []
#                 cluster_groups[cluster].append(img)
            
#             # Write detailed results
#             f.write("Detailed Results:\n\n")
#             for cluster in sorted(cluster_groups.keys()):
#                 f.write(f"Cluster {cluster}:\n")
#                 for img in sorted(cluster_groups[cluster]):
#                     f.write(f"  - {img}\n")
#                 f.write("\n")
        
#         print(f"Detailed results saved to: {output_path}")

# def main():
#     # Configuration
#     folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
#     results_dir = r'C:\Users\ilias\Python\Thesis-Project\results_reports'
#     n_clusters = 4
    
#     # Create clustering instance
#     clustering = XGBoostImageClustering(folder_path, n_clusters, results_dir)
    
#     # Perform clustering
#     clustering.perform_clustering()
    
#     # Generate visualization
#     clustering.visualize_clusters()
    
#     # Save results
#     clustering.save_results()

# if __name__ == '__main__':
#     main()



#### test 10

# Save this as classify_1.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

class XGBoostImageClustering:
    def __init__(self, folder_path, n_clusters=3, results_dir='results_reports'):
        """Initialize the clustering class with paths and parameters"""
        self.folder_path = os.path.normpath(folder_path)
        self.n_clusters = n_clusters
        self.results_dir = os.path.normpath(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.feature_extractor = self._load_feature_extractor()
        self.image_files = self._get_image_files()
        self.features = None
        self.features_2d = None
        self.clusters = None
        self.xgb_features = None
        
        # Create timestamped subfolder for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Initialized clustering with {n_clusters} clusters")
        print(f"Results will be saved in: {self.run_dir}")
    
    def _load_feature_extractor(self):
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
        
        print("Extracting features with enhanced processing...")
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
                
                feature = self.feature_extractor.predict(img_array, verbose=0)
                features.append(feature.flatten())
                processed_images.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        self.features = np.array(features)
        self.image_files = processed_images
        print(f"Features extracted for {len(self.features)} images")
        return self.features
    
    def enhance_features_with_xgboost(self):
        """Enhance features using XGBoost"""
        if self.features is None:
            self.extract_features()
        
        print("\nEnhancing features with XGBoost...")
        
        # Create target for XGBoost (using first feature dimension)
        target = self.features[:, 0]
        
        # Prepare data for XGBoost
        dtrain = xgb.DMatrix(self.features, label=target)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'
        }
        
        # Train XGBoost model
        model = xgb.train(params, dtrain, num_boost_round=200)
        
        # Get enhanced features
        self.xgb_features = model.predict(dtrain, pred_leaf=True)
        
        print("Feature enhancement completed")
        return self.xgb_features
    
    def perform_clustering(self):
        """Perform clustering on the enhanced features"""
        if self.xgb_features is None:
            self.enhance_features_with_xgboost()
        
        print("\nPerforming clustering...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.xgb_features)
        
        # Perform clustering
        clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.clusters = clusterer.fit_predict(features_scaled)
        
        # Create 2D representation for visualization
        print("Creating 2D visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        self.features_2d = tsne.fit_transform(features_scaled)
        
        print(f"Clustering completed. Found {self.n_clusters} clusters")
        return self.clusters
    
    def visualize_clusters(self):
        """Create and save visualization of the clusters"""
        if self.clusters is None:
            self.perform_clustering()
        
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            self.features_2d[:, 0],
            self.features_2d[:, 1],
            c=self.clusters,
            cmap='viridis'
        )
        
        plt.title('Image Clusters')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Feature Dimension 1')
        plt.ylabel('Feature Dimension 2')
        
        cluster_stats = [f"Cluster {i}: {sum(self.clusters == i)} images" 
                        for i in range(self.n_clusters)]
        plt.figtext(0.02, 0.02, '\n'.join(cluster_stats), fontsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.run_dir, 'cluster_visualization.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def save_results(self):
        """Save clustering results with emotion analysis"""
        if self.clusters is None:
            self.perform_clustering()
        
        output_path = os.path.join(self.run_dir, 'clustering_results.txt')
        
        # Define emotions to look for
        emotions = ['happy', 'sad', 'angry', 'surprised']
        
        with open(output_path, 'w') as f:
            f.write("Enhanced Clustering Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {self.timestamp}\n")
            f.write(f"Total Images Analyzed: {len(self.image_files)}\n")
            f.write(f"Optimal Number of Clusters: {self.n_clusters}\n\n")
            
            # Write summary
            f.write("Summary:\n")
            for i in range(self.n_clusters):
                count = sum(self.clusters == i)
                percentage = (count / len(self.image_files)) * 100
                f.write(f"Cluster {i}: {count} images ({percentage:.1f}%)\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Group images by cluster
            cluster_groups = {}
            for img, cluster in zip(self.image_files, self.clusters):
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(img)
            
            # Analyze emotion distribution per cluster
            f.write("Emotion Distribution per Cluster:\n")
            f.write("-" * 30 + "\n\n")
            
            for cluster in sorted(cluster_groups.keys()):
                f.write(f"Cluster {cluster}:\n")
                # Count emotions in this cluster
                emotion_counts = {emotion: 0 for emotion in emotions}
                total_cluster_images = len(cluster_groups[cluster])
                
                for img in cluster_groups[cluster]:
                    for emotion in emotions:
                        if emotion in img.lower():
                            emotion_counts[emotion] += 1
                
                # Write emotion distribution
                for emotion, count in emotion_counts.items():
                    percentage = (count / total_cluster_images) * 100 if total_cluster_images > 0 else 0
                    f.write(f"  - {emotion.capitalize()}: {count} images ({percentage:.1f}%)\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n\n")
            
            # Write detailed results
            f.write("Detailed Results:\n\n")
            for cluster in sorted(cluster_groups.keys()):
                f.write(f"Cluster {cluster}:\n")
                # Group by emotion within cluster
                emotion_groups = {emotion: [] for emotion in emotions}
                
                for img in sorted(cluster_groups[cluster]):
                    for emotion in emotions:
                        if emotion in img.lower():
                            emotion_groups[emotion].append(img)
                
                # Write images grouped by emotion
                for emotion in emotions:
                    if emotion_groups[emotion]:
                        f.write(f"  {emotion.capitalize()}:\n")
                        for img in emotion_groups[emotion]:
                            f.write(f"    - {img}\n")
                f.write("\n")
        
        print(f"Detailed results with emotion analysis saved to: {output_path}")

def main():
    # Configuration
    folder_path = r'C:\Users\ilias\Python\Thesis-Project\synthetic_images'
    results_dir = r'C:\Users\ilias\Python\Thesis-Project\results_reports'
    n_clusters = 5
    
    # Create clustering instance
    clustering = XGBoostImageClustering(folder_path, n_clusters, results_dir)
    
    # Perform clustering
    clustering.perform_clustering()
    
    # Generate visualization
    clustering.visualize_clusters()
    
    # Save results
    clustering.save_results()

if __name__ == '__main__':
    main()