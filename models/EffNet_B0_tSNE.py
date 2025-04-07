import os
import re
import json
import torch
import collections
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

import timm
import torch.nn as nn


from sklearn.cluster import KMeans
import random

from scipy.spatial import ConvexHull

# 1) CREATE MODEL
def create_efficientnet_b0(num_classes, device='cuda'):
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(1280, num_classes)
    )
    model.to(device)
    model.eval()
    return model

# 2) LOAD WEIGHTS
def load_weights(model, weight_path, device='cuda', strict=False):
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, collections.OrderedDict):
        model.load_state_dict(checkpoint, strict=strict)
    elif hasattr(checkpoint, 'state_dict'):
        model.load_state_dict(checkpoint.state_dict(), strict=strict)
    else:
        print(f"[WARNING] Could not interpret checkpoint at {weight_path}")
    return model

# 3) TRANSFORM FOR IMAGES
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

# 4) PREDICT + CAPTURE EMBEDDINGS FOR t-SNE
def predict_and_embed(model, folder_path, transform, class_to_idx, device='cuda', batch_size=32):
    """
    Predict on a folder that encodes the label in each filename (e.g. "Angry_1.jpg").
    Also store final-layer embeddings (or penultimate-layer embeddings) for t-SNE.
    """

    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if len(all_images) == 0:
        print(f"[WARNING] No images found in {folder_path}")
        return pd.DataFrame([]), None, None
    
    # We'll hook into the model to grab penultimate embeddings
    # For EfficientNet B0 in timm, the penultimate layer is model.bn2 or model.act2 or .global_pool
    # We'll choose the outputs right before the final classifier
    embedding_layer_outputs = []
    
    def hook_fn(module, input_, output_):
        # Flatten to 1D
        embedding_layer_outputs.append(output_.detach().cpu().numpy())

    # Register a forward hook at the last conv or global pool
    hook_handle = model.global_pool.register_forward_hook(hook_fn)
    
    # We might need a mapping from raw str => “true_emotion”
    emotion_mapping = {
        'angry': 'Angry','anger':'Angry',
        'contempt':'Contempt','disgust':'Disgust',
        'fear':'Fear','happy':'Happiness','happiness':'Happiness',
        'neutral':'Neutral','sad':'Sadness','sadness':'Sadness',
        'surprise':'Surprise','surprised':'Surprise'
    }
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    eps = 1e-12

    results = []
    buffer_imgs = []
    buffer_paths = []
    buffer_true_emotions = []
    buffer_true_idxs = []

    def run_batch(buf_imgs, buf_paths, buf_true_emotions, buf_true_idxs):
        batch = torch.stack(buf_imgs, dim=0).to(device)
        with torch.no_grad():
            _ = model(batch)  
            # predictions via classifier -> triggers the forward hook
            # embeddings are now in embedding_layer_outputs

        # Convert the last chunk of embeddings
        chunk_size = len(buf_imgs)
        # Each forward pass appends chunk_size embeddings => slice them
        # each shape might be [batch_size, 1280] depending on global_pool
        embed_chunk = embedding_layer_outputs[-1]
        # embed_chunk shape => (chunk_size, 1280)
        
        # Then we do classification from the final logits if we want
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        for i in range(chunk_size):
            row = {}
            row['file'] = os.path.basename(buf_paths[i])
            row['path'] = buf_paths[i]
            row['true_emotion'] = buf_true_emotions[i]
            row['true_idx'] = int(buf_true_idxs[i])
            pred_idx = int(np.argmax(probs[i]))
            row['predicted_idx'] = pred_idx
            row['predicted_emotion'] = idx_to_class.get(pred_idx, "Unknown")
            row['correct'] = (pred_idx == buf_true_idxs[i])
            row['confidence'] = float(probs[i, pred_idx])
            
            true_prob = max(probs[i, buf_true_idxs[i]], eps)
            row['neg_log_likelihood'] = float(-np.log(true_prob))

            # store class probabilities
            for cix, cname in idx_to_class.items():
                row[f"prob_{cname}"] = float(probs[i, cix])
            
            # store embeddings
            # we can store them in a separate array for t-SNE
            row['embedding'] = embed_chunk[i]
            results.append(row)

    for fname in sorted(all_images):
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            im_t = transform(img)
            buffer_imgs.append(im_t)
            buffer_paths.append(fpath)

            # parse the label
            base = os.path.basename(fname)
            match = re.match(r'([a-zA-Z]+)', base)
            if match:
                raw_emotion = match.group(1).lower()
                true_emotion = emotion_mapping.get(raw_emotion, raw_emotion.capitalize())
            else:
                true_emotion = "Unknown"
            buffer_true_emotions.append(true_emotion)
            true_idx = class_to_idx.get(true_emotion, -1)
            buffer_true_idxs.append(true_idx)

            if len(buffer_imgs) >= batch_size:
                run_batch(buffer_imgs, buffer_paths, buffer_true_emotions, buffer_true_idxs)
                buffer_imgs.clear()
                buffer_paths.clear()
                buffer_true_emotions.clear()
                buffer_true_idxs.clear()

        except Exception as e:
            print(f"[WARNING] Could not read {fpath}: {str(e)}")

    # leftover
    if len(buffer_imgs) > 0:
        run_batch(buffer_imgs, buffer_paths, buffer_true_emotions, buffer_true_idxs)
        buffer_imgs.clear()

    # remove the hook
    hook_handle.remove()

    df = pd.DataFrame(results)
    if len(df) > 0:
        # separate out embeddings
        X = np.vstack(df['embedding'].to_numpy())  # shape (N, 1280)
        # get integer labels
        y = df['true_idx'].to_numpy()
    else:
        X, y = None, None
    
    return df, X, y

def compute_classification_report(df, class_names):
    """
    Build a classification report from the DataFrame, returning a dictionary.
    """
    from sklearn.metrics import classification_report

    y_true = df['true_idx'].to_numpy()
    y_pred = df['predicted_idx'].to_numpy()
    report_dict = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    return report_dict

def run_tsne_and_plot(X, y, class_names, out_path):
    """
    Run t-SNE on the embeddings array X (shape NxD), and color by y (shape N).
    Save the resulting scatter plot.
    """
    if X is None or y is None or len(X) == 0:
        print("[WARNING] No embeddings to run t-SNE.")
        return
    
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for idx, cls_label in enumerate(class_names):
        mask = (y == idx)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=cls_label,
            s=10
        )
    plt.title("t-SNE Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] t-SNE plot saved => {out_path}")

def run_tsne(X, n_components=2, perplexity=30.0, random_state=42):
    """
    Run t-SNE on embeddings X (shape: (N, D)) -> returns X_2d (shape: (N, 2))
    """
    if X is None or len(X) == 0:
        print("[WARNING] run_tsne: No data provided.")
        return None
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_2d = tsne.fit_transform(X)
    return X_2d

def cluster_and_plot_centroids(X_2d, y, class_names, cluster_size_list, out_dir):
    """
    For each cluster_size in cluster_size_list:
      1) Determine K = N / cluster_size (rounded).
      2) Run KMeans on X_2d.
      3) For each cluster, compute centroid and the majority label.
      4) Plot only the centroids, colored by majority label.
      5) Save the plot.

    :param X_2d: (N, 2) t-SNE embedded points
    :param y: (N,) integer array of class labels
    :param class_names: list of strings, class names
    :param cluster_size_list: e.g. [5, 10, 15, 20]
    :param out_dir: directory to save the cluster plot
    """
    # Ensure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    N = X_2d.shape[0]
    for cluster_size in cluster_size_list:
        # number of clusters K
        K = max(1, N // cluster_size)  # ensure at least 1 cluster

        print(f"[INFO] Clustering {N} points into K={K} clusters (approx {cluster_size} per cluster).")
        kmeans = KMeans(n_clusters=K, random_state=42)
        labels = kmeans.fit_predict(X_2d)  # shape (N,)
        cluster_centers = kmeans.cluster_centers_  # shape (K,2)

        # For each cluster, find the majority class
        # gather each cluster's points
        cluster_majorities = []
        for k in range(K):
            mask = (labels == k)
            cluster_labels = y[mask]  # y is ground-truth integer labels
            if len(cluster_labels) == 0:
                # theoretically shouldn't happen, but just in case
                majority_label = -1
            else:
                # majority label
                counts = np.bincount(cluster_labels)
                majority_label = np.argmax(counts)
            cluster_majorities.append(majority_label)

        # Now cluster_centers is (K,2)
        # cluster_majorities is length K
        # We can scatter each centroid, colored by that cluster's majority label.
        plt.figure(figsize=(8,6))
        for k in range(K):
            lbl = cluster_majorities[k]
            cx, cy = cluster_centers[k, 0], cluster_centers[k, 1]
            plt.scatter(cx, cy,
                        color=sns.color_palette("tab10")[lbl % 10],  # or any palette
                        s=80,
                        label=None,
                        alpha=0.8,
                        edgecolors='black')
        # We'll also add a small legend indicating which color belongs to which class
        # But since multiple clusters can have the same label, let's show the classes in a separate legend
        patches = []
        unique_lbls = np.unique(cluster_majorities)
        for ul in unique_lbls:
            if ul >= 0 and ul < len(class_names):
                color = sns.color_palette("tab10")[ul % 10]
                patches.append(plt.Line2D([0], [0], marker='o', color='w',
                                          label=class_names[ul],
                                          markerfacecolor=color, markersize=8))
        plt.legend(handles=patches, title="Majority Label", loc='best')

        plt.title(f"K-Means Centroids (cluster_size={cluster_size}, K={K})")
        plt.tight_layout()
        fname = f"tsne_centroids_size{cluster_size}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved cluster-centroids plot => {out_path}")

def cluster_per_class_and_plot_centroids(
    X_2d, y, class_names, cluster_size, out_path
):
    """
    Clusters each class's t-SNE points separately, then plots ONLY the centroids of each class.
    Produces a single plot with all classes' centroids, color-coded by class.

    :param X_2d: (N,2) array, the t-SNE embeddings
    :param y: (N,) array of integer labels
    :param class_names: list of str, e.g. ['Angry','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
    :param cluster_size: integer, about how many points per cluster
    :param out_path: full path (including .png) to save the resulting plot
    """

    # We'll gather all centroids + their class label in these lists
    centroid_coords = []
    centroid_labels = []

    N = X_2d.shape[0]
    num_classes = len(class_names)

    for c_idx, c_name in enumerate(class_names):
        # select only the points for this class
        mask = (y == c_idx)
        X_2d_class = X_2d[mask]
        n_c = X_2d_class.shape[0]

        if n_c < 2:
            # not enough points to do any meaningful clustering
            continue

        # compute K for this class
        if n_c < cluster_size:
            # if fewer than cluster_size points => treat as 1 cluster
            K_c = 1
        else:
            K_c = max(1, n_c // cluster_size)

        print(f"[INFO] Clustering class '{c_name}' with {n_c} points "
              f"into K_c={K_c} clusters ~ size={cluster_size} each.")

        # run K-means on X_2d_class
        kmeans = KMeans(n_clusters=K_c, random_state=42)
        labels_class = kmeans.fit_predict(X_2d_class)
        centers_class = kmeans.cluster_centers_  # shape (K_c, 2)

        for k in range(K_c):
            cx, cy = centers_class[k]
            centroid_coords.append([cx, cy])
            centroid_labels.append(c_idx)

    # Convert to arrays
    centroid_coords = np.array(centroid_coords)
    centroid_labels = np.array(centroid_labels)

    # Plot
    plt.figure(figsize=(8,6))
    for c_idx, c_name in enumerate(class_names):
        mask = (centroid_labels == c_idx)
        plt.scatter(
            centroid_coords[mask, 0],
            centroid_coords[mask, 1],
            s=80,
            alpha=0.8,
            edgecolors='black',
            label=c_name
        )
    plt.title(f"Per-Class Prediction Outcomes")
    plt.legend(loc='best')
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved single combined cluster-centroids plot => {out_path}")

def cluster_per_class_with_hull(
    X_2d, 
    y, 
    class_names, 
    cluster_size, 
    out_path,
    min_hull_points=30
):
    """
    Clusters each class's t-SNE points separately, then for each cluster:
      - If it has at least 'min_hull_points' points, draw a polygon (convex hull) around them.
      - Plot the cluster's centroid as well, colored by class.

    This creates a single figure merging all classes, but guaranteeing no cross-class clusters.

    :param X_2d: (N,2) array of 2D coordinates (e.g. from t-SNE)
    :param y: (N,) array of integer labels (one label per point)
    :param class_names: list of str, your class labels
    :param cluster_size: integer ~ how many points per cluster (like 5,10,etc.)
    :param out_path: path to save the final PNG
    :param min_hull_points: threshold (default 30). If a cluster has >= this many points,
                            we draw a hull around them.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # ensure dir exists
    plt.figure(figsize=(8,6))

    for c_idx, c_name in enumerate(class_names):
        # select the points for this class
        mask = (y == c_idx)
        Xc = X_2d[mask]
        n_c = len(Xc)
        if n_c < 2:
            # skip
            continue

        # compute K for this class
        if n_c < cluster_size:
            K_c = 1
        else:
            K_c = max(1, n_c // cluster_size)

        print(f"[INFO] Class '{c_name}': {n_c} points => {K_c} clusters (size ~ {cluster_size}).")

        if K_c < 1:
            continue
        
        # run K-means on just this class
        kmeans = KMeans(n_clusters=K_c, random_state=42)
        labels_class = kmeans.fit_predict(Xc)
        centers_class = kmeans.cluster_centers_  # shape (K_c,2)

        # color for this class
        # A standard approach is to pick from a palette by index c_idx
        # If you have many classes, you might want a bigger palette
        class_color = sns.color_palette("tab10")[c_idx % 10]

        for k in range(K_c):
            # points in cluster k
            mask_k = (labels_class == k)
            Xk = Xc[mask_k]
            n_k = len(Xk)
            # cluster centroid
            cx, cy = centers_class[k]

            # plot centroid
            plt.scatter(
                cx, cy,
                color=class_color,
                s=80, alpha=0.9,
                edgecolors='black'
            )

            # if cluster has enough points, draw a hull around them
            if n_k >= min_hull_points:
                try:
                    hull = ConvexHull(Xk)
                    # hull.vertices is the ordering for the boundary
                    verts = hull.vertices
                    # close the polygon by repeating the first vertex at the end
                    poly_x = Xk[verts, 0]
                    poly_y = Xk[verts, 1]
                    
                    # fill or just outline
                    plt.fill(
                        np.append(poly_x, poly_x[0]),
                        np.append(poly_y, poly_y[0]),
                        color=class_color,
                        alpha=0.2
                    )
                except Exception as e:
                    print(f"[WARNING] Could not compute hull for cluster size {n_k}: {str(e)}")

    # Create a legend. We'll do this by making a patch for each class
    from matplotlib.lines import Line2D
    legend_patches = []
    for i, c_name in enumerate(class_names):
        col = sns.color_palette("tab10")[i % 10]
        legend_patches.append(Line2D(
            [0], [0],
            marker='o', color='w',
            label=c_name,
            markerfacecolor=col,
            markersize=8,
            alpha=0.9
        ))
    plt.legend(handles=legend_patches, loc='best')

    plt.title(f"Per-Class Clusters, hull if >= {min_hull_points} pts")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved cluster/hull plot => {out_path}")


def example_tsne_clustering_flow(
    X_2d, y, class_names, model_out_dir,
    cluster_sizes=[5, 10, 15, 20]
):
    """
    Example flow:
      - For each cluster_size in cluster_sizes, cluster each class separately and
        produce a single t-SNE centroid plot that merges all classes.
      - Each plot is saved to the model_out_dir (e.g. one plot per cluster_size).
    """
    for size in cluster_sizes:
        plot_name = f"tsne_centroids_size{size}.png"
        out_path = os.path.join(model_out_dir, plot_name)
        cluster_per_class_and_plot_centroids(X_2d, y, class_names, size, out_path)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # ----- Model Weights Folder -----
    # Affect Net : C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model"


    # 1) Where your model weights are stored
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\..."

    # --- Part A ---
    # Affect Net
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\model"

    # Best Finetuned classification layer only dropout 0
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230_drop0_OK\best_model_only"

    # Best Synthetic
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models_best"

    # --- Gender Splits ---
    # Men best
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0327_2011gender\gend_models_best\models_best_men"
    # Women best
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0327_2011gender\gend_models_best\models_best_wom"

    # --- Fractions ---
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fractions_best\models"



    # ----- Test Folders -----
    # The RAFDB real data test combined test folder
    test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_combined"

    # The SYNTH real data test combined test folder.
    # test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_100_combined"



    # 3) Output folder
    model_parent_dir = os.path.dirname(model_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    overall_dir = os.path.join(model_parent_dir, f"tsne_fine_on_real_{timestamp}") # for real
    # overall_dir = os.path.join(model_parent_dir, f"tsne_analysis_synth_{timestamp}") # for synth
    os.makedirs(overall_dir, exist_ok=True)

    # 4) Class mapping
    class_to_idx = {
        'Angry': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3,
        'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7
    }
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_names = list(class_to_idx.keys())  # e.g. ["Angry","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]

    # 5) Loop over each checkpoint in model_dir
    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(('.pth', '.pt')):
            continue
        
        weight_path = os.path.join(model_dir, fname)
        model_name = os.path.splitext(fname)[0]

        # create subfolder for this model
        model_out_dir = os.path.join(overall_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        print(f"\n============================")
        print(f"[INFO] t-SNE + Predict for model: {fname}")
        print(f"[INFO] Results => {model_out_dir}")

        # 6) Create + load model
        num_classes = len(class_names)
        model = create_efficientnet_b0(num_classes, device=device)
        load_weights(model, weight_path, device=device, strict=False)

        transform = get_transform(224)
        
        # 7) Predict + get embeddings
        df, X, y = predict_and_embed(model, test_folder, transform, class_to_idx, device=device, batch_size=32)
        if len(df) == 0:
            print(f"[WARNING] No predictions. Skipping {model_name}.")
            continue

        # Save CSV of predictions
        combined_csv = os.path.join(model_out_dir, "combined_results.csv")
        df.to_csv(combined_csv, index=False)
        print(f"[INFO] Wrote combined CSV => {combined_csv}")

        # 8) Classification report
        report_dict = compute_classification_report(df, class_names)

        # 9) Save classification report to JSON
        out_json = os.path.join(model_out_dir, "classification_report.json")
        with open(out_json, 'w') as f:
            json.dump(report_dict, f, indent=4)
        print(f"[INFO] Classification report => {out_json}")

        # 10) Save classification metrics also as CSV
        rows = []
        for key, val in report_dict.items():
            if isinstance(val, dict):
                # e.g. "0": {...}, "macro avg": {...}, "weighted avg": {...}
                row = {"class_name": key}
                row.update(val)
                rows.append(row)
            else:
                # e.g. "accuracy": 0.123
                # store it in a simpler format
                rows.append({"class_name": key, "value": val})

        metrics_csv = os.path.join(model_out_dir, "classification_report.csv")
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"[INFO] Classification metrics CSV => {metrics_csv}")
        

        # 11) t-SNE + Plot
        # old for all points
        # tsne_png = os.path.join(model_out_dir, "tsne_plot.png")
        # run_tsne_and_plot(X, y, class_names, tsne_png)

        # 1) Run t-SNE
        X_2d = run_tsne(X)

        # 2) Choose a few cluster_size values
        cluster_sizes = [1, 5, 10]
        # cluster_and_plot_centroids(X_2d, y, class_names, cluster_sizes, "output_plots")

        for size in cluster_sizes:
            plot_name = f"tsne_centroids_size{size}.png"
            out_path = os.path.join(model_out_dir, plot_name)
            cluster_per_class_and_plot_centroids(X_2d, y, class_names, size, out_path)
            out_path_hull = os.path.join(model_out_dir, "hull_plot.png")

            cluster_per_class_with_hull(
                X_2d, y, class_names, size,out_path_hull, min_hull_points=10)

    print("\n[INFO] All done with t-SNE analysis.")

if __name__=="__main__":
    main()
