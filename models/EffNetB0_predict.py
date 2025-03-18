import os
import torch
import torch.nn as nn
import timm
import collections
import json
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve


import torchvision.transforms as transforms

import os
import torch
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms

def create_efficientnet_b0(num_classes, device='cuda'):
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(1280, num_classes)
    )
    model.to(device)
    model.eval()
    return model

def load_weights(model, weight_path, device='cuda', strict=False):
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=strict)
    return model

def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

def predict_folder(model, folder_path, true_emotion, transform, class_to_idx, device='cuda', batch_size=32):
    """
    Predict on a single subfolder that presumably all belongs to the given 'true_emotion'.
    Return a DataFrame with columns:
      file, path, true_emotion, true_idx, predicted_emotion, predicted_idx, correct, confidence, prob_..., ...
    """
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    true_idx = class_to_idx.get(true_emotion, -1)

    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if len(all_images)==0:
        print(f"[WARNING] No images found in {folder_path}")
        return pd.DataFrame([])

    results = []
    buffer_imgs = []
    buffer_paths= []

    def run_batch(buffer_imgs, buffer_paths):
        batch = torch.stack(buffer_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        for i in range(len(buffer_imgs)):
            row = {}
            row['file'] = os.path.basename(buffer_paths[i])
            row['path'] = buffer_paths[i]
            row['true_emotion'] = true_emotion
            row['true_idx'] = int(true_idx)
            pred_idx = np.argmax(probs[i])
            row['predicted_idx'] = int(pred_idx)
            row['predicted_emotion'] = idx_to_class[pred_idx] if pred_idx in idx_to_class else "Unknown"
            row['correct'] = (pred_idx==true_idx)
            row['confidence'] = float(probs[i,pred_idx])
            # store all class probabilities
            for cix, cname in idx_to_class.items():
                row[f"prob_{cname}"] = float(probs[i,cix])
            results.append(row)

        buffer_imgs.clear()
        buffer_paths.clear()

    for fname in sorted(all_images):
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            im_t= transform(img)
            buffer_imgs.append(im_t)
            buffer_paths.append(fpath)
            if len(buffer_imgs)>=batch_size:
                run_batch(buffer_imgs, buffer_paths)
        except Exception as e:
            print(f"[WARNING] could not read {fpath}: {str(e)}")
    # leftover
    if len(buffer_imgs)>0:
        run_batch(buffer_imgs, buffer_paths)

    df = pd.DataFrame(results)
    return df

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Where your model weights are stored
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_013917finetuned_real\models"
    
    # 2) Your test root that has subfolders for each emotion
    #    e.g. test_root/Angry_6, test_root/Disgust_12, etc.
    # rafdb C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test"

    # 3) A global output folder
    global_out_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_dir = os.path.join(global_out_dir, f"predictions_real_{timestamp}")
    os.makedirs(overall_dir, exist_ok=True)

    # 4) class mapping
    class_to_idx = {
        'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
        'Happiness':4,'Neutral':5,'Sadness':6,'Surprise':7
    }

    # 5) We loop over each .pth or .pt file in model_dir
    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(('.pth','.pt')):
            continue
        
        weight_path = os.path.join(model_dir, fname)
        model_name  = os.path.splitext(fname)[0]  # e.g. 'enet_b0_finetuned' etc.

        # Make subfolder for this model
        model_out_dir = os.path.join(overall_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        print(f"\n============================")
        print(f"[INFO] Evaluating model: {fname} on test set => {test_root}")
        print(f"[INFO] Results subfolder => {model_out_dir}")

        # Create model fresh
        num_classes = len(class_to_idx)
        model = create_efficientnet_b0(num_classes, device=device)

        # Load checkpoint
        try:
            checkpoint = torch.load(weight_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"[INFO] Loaded checkpoint from {fname}")
        except Exception as e:
            print(f"[ERROR] Could not load {fname}: {str(e)}")
            continue

        transform = get_transform(224)

        # We'll unify a global DF for all subfolders
        global_df = []

        # 6) For each subfolder (like 'angry_6'), we guess the emotion from name
        import re
        emotion_mapping = {
            'angry':'Angry','anger':'Angry','contempt':'Contempt','disgust':'Disgust',
            'fear':'Fear','happy':'Happiness','happiness':'Happiness','neutral':'Neutral',
            'sad':'Sadness','sadness':'Sadness','surprise':'Surprise','surprised':'Surprise'
        }

        # subfolders
        for subf in os.listdir(test_root):
            subf_path = os.path.join(test_root, subf)
            if not os.path.isdir(subf_path):
                continue
            
            # parse the folder name for emotion
            match = re.match(r'([a-zA-Z]+)_?\d*', subf)
            raw_emotion = match.group(1).lower() if match else subf.lower()
            true_emotion = emotion_mapping.get(raw_emotion, raw_emotion)
            # Evaluate
            df = predict_folder(
                model, subf_path, true_emotion,
                transform=transform, class_to_idx=class_to_idx,
                device=device, batch_size=32
            )
            if len(df)>0:
                # Save per-emotion CSV
                out_csv = os.path.join(model_out_dir, f"results_{true_emotion}.csv")
                df.to_csv(out_csv, index=False)
                print(f"[INFO] Wrote {len(df)} rows => {out_csv}")
                
                global_df.append(df)
            else:
                print(f"[WARNING] No results for {subf_path}")

        # 7) Combine into a single CSV for all subfolders
        if len(global_df)>0:
            combined = pd.concat(global_df, ignore_index=True)
            combined_csv = os.path.join(model_out_dir, "combined_results.csv")
            combined.to_csv(combined_csv, index=False)
            print(f"[INFO] Wrote combined CSV => {combined_csv}")
        else:
            print("[WARNING] No data from any subfolders?")

    print("[INFO] All done.")


if __name__=="__main__":
    main()




# def create_model(num_classes, weights_path, device='cuda'):
#     """
#     Creates an EfficientNet-B0, sets final layer, and loads weights with strict=False 
#     (in case final layer doesn't match).
#     """
#     model = timm.create_model('tf_efficientnet_b0', pretrained=False)
#     model.classifier = torch.nn.Sequential(
#         torch.nn.Linear(1280, num_classes)
#     )
#     model.to(device)
    
#     # load weights with strict=False in case final layer doesn't match
#     checkpoint = torch.load(weights_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=False)
#     model.eval()
#     return model


# def get_transform(img_size=224):
#     """
#     Basic transform to resize and normalize images for an EfficientNet-B0.
#     """
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485,0.456,0.406],
#             std =[0.229,0.224,0.225]
#         )
#     ])


# def predict_and_save_results(
#     model,
#     root_dir,
#     class_to_idx,
#     output_dir,
#     img_size=224,
#     batch_size=32,
#     device='cuda'
# ):
#     """
#     Iterates over each subfolder of 'root_dir', 
#     assuming each subfolder is a single ground-truth emotion folder.

#     For each folder:
#       - loads images
#       - runs inference in mini-batches
#       - saves a CSV with columns:
#          [file, path, true_emotion, true_idx, predicted_emotion, predicted_idx,
#           correct, confidence, prob_Emotion0, ..., prob_EmotionN]
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     idx_to_class = {v:k for k,v in class_to_idx.items()}
    
#     transform = get_transform(img_size)
    
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
        
#         # guess the 'true emotion' from folder name
#         # e.g. "angry_6" => "angry"
#         # or "sad" => "sad"
#         import re
#         match = re.match(r'([a-zA-Z]+)_?\d*', folder_name)
#         raw_emotion = match.group(1).lower() if match else folder_name.lower()
        
#         # map to official name if needed
#         emotion_mapping = {
#             'angry':'Angry','anger':'Angry','contempt':'Contempt','disgust':'Disgust',
#             'fear':'Fear','happy':'Happiness','happiness':'Happiness','neutral':'Neutral',
#             'sad':'Sadness','sadness':'Sadness','surprise':'Surprise','surprised':'Surprise'
#         }
#         true_emotion = emotion_mapping.get(raw_emotion, raw_emotion)
        
#         # get the numeric label
#         if true_emotion in class_to_idx:
#             true_idx = class_to_idx[true_emotion]
#         else:
#             print(f"[WARNING] {true_emotion} not found in class_to_idx => using -1.")
#             true_idx = -1
        
#         # gather images
#         all_images = [
#             f for f in os.listdir(folder_path) 
#             if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))
#         ]
#         if len(all_images)==0:
#             print(f"[WARNING] No images found in {folder_path}")
#             continue
        
#         # Prepare for inference
#         results = []
#         images_buffer = []
#         paths_buffer  = []
        
#         def run_inference(images_buffer, paths_buffer):
#             batch = torch.stack(images_buffer, dim=0).to(device)
#             with torch.no_grad():
#                 logits = model(batch)
#                 # convert to probabilities
#                 probs = F.softmax(logits, dim=1).cpu().numpy()
                
#             # build results
#             for i in range(len(images_buffer)):
#                 p = probs[i]
#                 pred_idx = np.argmax(p)
#                 confidence = float(p[pred_idx])
#                 correct = (pred_idx==true_idx)
                
#                 row = {
#                     'file': os.path.basename(paths_buffer[i]),
#                     'path': paths_buffer[i],
#                     'true_emotion': true_emotion,
#                     'true_idx': int(true_idx),
#                     'predicted_emotion': idx_to_class[pred_idx] if pred_idx in idx_to_class else "Unknown",
#                     'predicted_idx': int(pred_idx),
#                     'correct': correct,
#                     'confidence': confidence
#                 }
#                 # add all probabilities
#                 for cidx, cname in idx_to_class.items():
#                     row[f'prob_{cname}'] = float(p[cidx])
#                 results.append(row)
            
#             images_buffer.clear()
#             paths_buffer.clear()
        
#         # Mini-batch processing
#         for img_name in sorted(all_images):
#             img_path = os.path.join(folder_path, img_name)
#             try:
#                 im = Image.open(img_path).convert('RGB')
#                 im_t = transform(im)
#                 images_buffer.append(im_t)
#                 paths_buffer.append(img_path)
#                 if len(images_buffer) >= batch_size:
#                     run_inference(images_buffer, paths_buffer)
#             except Exception as e:
#                 print(f"[WARNING] Could not read {img_path}: {str(e)}")
        
#         # leftover
#         if len(images_buffer)>0:
#             run_inference(images_buffer, paths_buffer)
        
#         # store CSV
#         df = pd.DataFrame(results)
#         csv_name = f"results_{true_emotion}.csv"
#         out_path = os.path.join(output_dir, csv_name)
#         df.to_csv(out_path, index=False)
        
#         # per-folder accuracy
#         if len(df)>0:
#             accuracy = 100.0*df['correct'].mean()
#             print(f"[INFO] Folder '{folder_name}' => {true_emotion}, num images={len(df)} => accuracy={accuracy:.2f}%")
#         else:
#             print(f"[WARNING] No results for folder: {folder_name}")


# def main():
#     # example usage
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # path to your model weights
#     weights_path = r"C:\myproject\models\my_efficientnet_b0_finetuned.pt"
    
#     # define class mapping
#     class_to_idx = {
#         'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
#         'Happiness':4,'Neutral':5,'Sadness':6,'Surprise':7
#     }
#     num_classes = len(class_to_idx)
    
#     # create / load your model
#     model = create_model(num_classes, weights_path, device=device)
    
#     # define the input images root
#     root_dir = r"C:\myproject\test"  # this has subfolders like "angry_6", "sad_2", etc.
    
#     # output dir
#     output_dir = r"C:\myproject\test_results"
#     # Add timestamp to output dir
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(r"C:\myproject\test_results", f"predictions_{timestamp}")
    
#     predict_and_save_results(
#         model=model,
#         root_dir=root_dir,
#         class_to_idx=class_to_idx,
#         output_dir=output_dir,
#         img_size=224,
#         batch_size=32,
#         device=device
#     )
#     print("[INFO] Done. Each folder's predictions are in output_dir as CSV.")

# if __name__ == "__main__":
#     main()





# #########################
# # 1) Utility Functions
# #########################

# def load_pretrained_efficientnet_b0(weights_path, num_classes, device):
#     model = timm.create_model('tf_efficientnet_b0', pretrained=False)
#     model.classifier = nn.Sequential(nn.Linear(1280, num_classes))

#     try:
#         checkpoint = torch.load(weights_path, map_location=device)
#         print(f"[INFO] Loaded checkpoint from {weights_path}")
#         if isinstance(checkpoint, collections.OrderedDict):
#             model.load_state_dict(checkpoint, strict=True)
#             print("[INFO] Loaded pure state_dict directly.")
#         else:
#             # Possibly a full model
#             if hasattr(checkpoint, 'state_dict'):
#                 model.load_state_dict(checkpoint.state_dict(), strict=True)
#                 print("[INFO] Extracted state_dict from entire model object.")
#             else:
#                 print("[WARNING] checkpoint not recognized, skipping load.")
#     except Exception as e:
#         print(f"[ERROR] Could not load weights: {str(e)}")

#     model = model.to(device)
#     model.eval()
#     return model

# def get_transform(img_size=224):
#     """
#     Parameters:
#     img_size : int
#         Image size for resizing (224 for B0, 260 for B2)
#     Returns:
#     torchvision.transforms.Compose
#         Transform pipeline for preprocessing images
#     """
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

# def compute_folder_metrics(
#     model, folder_path, class_to_idx, device='cuda', batch_size=32, img_size=224
# ):
#     """
#     Process one folder (which presumably corresponds to a single ground-truth emotion).
#     Returns a dict containing y_true, y_pred, y_probs, file_paths, the "inferred" true_emotion, etc.
#     """
#     transform = get_transform(img_size)
#     all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
#     if len(all_files) == 0:
#         print(f"[WARNING] No images found in {folder_path}")
#         return None

#     # Inferred emotion from folder name
#     folder_name = os.path.basename(folder_path)
#     # If folder is "angry_6", we parse out "angry"
#     import re
#     match = re.match(r'([a-zA-Z]+)_?\d*', folder_name)
#     raw_emotion = match.group(1).lower() if match else folder_name.lower()

#     # Convert that to standardized emotion name if needed
#     emotion_mapping = {
#         'angry':'Angry','anger':'Angry','contempt':'Contempt','disgust':'Disgust',
#         'fear':'Fear','happy':'Happiness','happiness':'Happiness','neutral':'Neutral',
#         'sad':'Sadness','sadness':'Sadness','surprise':'Surprise','surprised':'Surprise'
#     }
#     true_emotion = emotion_mapping.get(raw_emotion, raw_emotion)
    
#     # If not found in class_to_idx, we set -1 or skip
#     if true_emotion not in class_to_idx:
#         print(f"[WARNING] {true_emotion} not in class_to_idx => using -1 as true idx.")
#         true_class_idx = -1
#     else:
#         true_class_idx = class_to_idx[true_emotion]

#     # We'll store results for each image
#     y_true = []
#     y_pred = []
#     y_probs = []   # shape (N, num_classes)
#     file_paths = []

#     # Process in mini-batches
#     num_classes = len(class_to_idx)
#     idx_to_class = {v:k for k,v in class_to_idx.items()}
#     images_buffer = []
#     buffer_paths = []

#     def run_inference(images_buffer, buffer_paths):
#         # stack and run
#         batch = torch.stack(images_buffer, dim=0).to(device)
#         with torch.no_grad():
#             logits = model(batch)  # shape (B, num_classes)
#             # convert to probabilities
#             probs = F.softmax(logits, dim=1).cpu().numpy()
#         # store
#         for i in range(len(images_buffer)):
#             y_probs.append(probs[i])
#         images_buffer.clear()
#         buffer_paths.clear()

#     for fname in tqdm(all_files, desc=f"[{folder_name}]"):
#         path = os.path.join(folder_path, fname)
#         try:
#             img = Image.open(path).convert('RGB')
#             img_tensor = transform(img)
#             images_buffer.append(img_tensor)
#             buffer_paths.append(path)
#             y_true.append(true_class_idx)
            
#             if len(images_buffer) >= batch_size:
#                 run_inference(images_buffer, buffer_paths)
#         except Exception as e:
#             print(f"[WARNING] error reading {path}: {str(e)}")

#     # leftover
#     if len(images_buffer) > 0:
#         run_inference(images_buffer, buffer_paths)

#     # Now we have y_probs => shape (N, num_classes), y_true => list of class idx
#     y_probs = np.array(y_probs)
#     y_pred = np.argmax(y_probs, axis=1)
#     accuracy = 100.0 * np.mean((np.array(y_true) == y_pred))

#     # build results
#     results = []
#     for i in range(len(y_true)):
#         true_idx = y_true[i]
#         pred_idx = y_pred[i]
#         row = {
#             "file": all_files[i],
#             "path": os.path.join(folder_path, all_files[i]),
#             "true_emotion": true_emotion,
#             "true_idx": int(true_idx),
#             "predicted_emotion": idx_to_class[pred_idx] if pred_idx in idx_to_class else "Unknown",
#             "predicted_idx": int(pred_idx),
#             "correct": (true_idx == pred_idx),
#             "confidence": float(y_probs[i, pred_idx])
#         }
#         # add per-class probabilities
#         for c_idx, c_name in idx_to_class.items():
#             row[f"prob_{c_name}"] = float(y_probs[i,c_idx])
#         results.append(row)

#     metrics_info = {
#         "true_emotion": true_emotion,
#         "accuracy": accuracy,
#         "y_true": np.array(y_true),
#         "y_pred": y_pred,
#         "y_probs": y_probs,
#         "results": results
#     }
#     return metrics_info


# def classification_report_dict(y_true, y_pred, class_names):
#     # Force classification_report to handle all classes from 0..7, 
#     # ignoring any zero-division issues.
#     from sklearn.metrics import classification_report
#     report = classification_report(
#         y_true, y_pred, labels=range(len(class_names)),
#         target_names=class_names, zero_division=0, output_dict=True
#     )
#     return report

# def build_confusion_matrix_and_save(y_true, y_pred, class_names, out_path="combined_cm.png"):
#     cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
#     fig, ax = plt.subplots(figsize=(8,6))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()
#     print(f"[INFO] Saved combined confusion matrix to {out_path}")


# def main():
#     # 1) Basic config
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"[INFO] Using device: {device}")
    
#     # your weights
#     weights_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\my_efficientnet_b0_finetuned_test_cuda_10.pt"
    
#     # class mapping
#     class_to_idx_8 = {
#         'Angry': 0, 
#         'Contempt': 1, 
#         'Disgust': 2, 
#         'Fear': 3, 
#         'Happiness': 4, 
#         'Neutral': 5, 
#         'Sadness': 6, 
#         'Surprise': 7
#     }
#     class_names = list(class_to_idx_8.keys())
    
#     # load model
#     model = load_pretrained_efficientnet_b0(weights_path, num_classes=len(class_to_idx_8), device=device)

#     # your test root with subfolders, each subfolder=some emotion
#     test_root = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_test"
    
#     base_output_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0"
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     custom_dir_name = f"synth_finetuned_all_{timestamp}"
#     output_dir = os.path.join(base_output_dir, custom_dir_name)
#     os.makedirs(output_dir, exist_ok=True)
    
#     # For global analysis
#     global_y_true = []
#     global_y_pred = []
#     global_y_probs = []
#     global_results = []

#     # 2) Iterate subfolders
#     for folder_name in os.listdir(test_root):
#         folder_path = os.path.join(test_root, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
        
#         print("\n==================================================")
#         print(f"[INFO] Evaluating folder: {folder_path}")
#         # Evaluate
#         metrics_info = compute_folder_metrics(
#             model, folder_path, class_to_idx_8, device=device, batch_size=32, img_size=224
#         )
#         if metrics_info is None:
#             continue
        
#         # 3) Save per-folder results
#         # build subdir
#         subdir_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         result_subdir = os.path.join(output_dir, f"evaluation_{folder_name}_{subdir_ts}")
#         os.makedirs(result_subdir, exist_ok=True)

#         # CSV for that folder
#         df = pd.DataFrame(metrics_info["results"])
#         csv_path = os.path.join(result_subdir, f"results_{metrics_info['true_emotion']}.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"[INFO] Per-folder CSV => {csv_path}")

#         # Per-folder confusion matrix
#         y_true_folder = metrics_info["y_true"]
#         y_pred_folder = metrics_info["y_pred"]
#         y_probs_folder= metrics_info["y_probs"]  # shape (N,8)
#         cm_path = os.path.join(result_subdir, "confusion_matrix.png")
#         build_confusion_matrix_and_save(y_true_folder, y_pred_folder, class_names, cm_path)

#         # classification report
#         rep_dict = classification_report_dict(y_true_folder, y_pred_folder, class_names)
#         # add false positives if you want
#         # or just keep it as is
#         with open(os.path.join(result_subdir, "classification_report.json"), 'w') as f:
#             json.dump(rep_dict, f, indent=4)

#         # reliability diagram if you want, e.g.:
#         # y_probs_folder are probabilities
#         # let's do a standard reliability diagram
#         # but note that y_true_folder are class indices, shape (N)
#         # define function or do it quickly:
#         diag_path = os.path.join(result_subdir, "reliability_diagram.png")
#         # we can do the same approach as your existing function if you want:
#         # plot_reliability_diagram(y_true_folder, raw_logits?? => we only have prob => you can do a small hack
#         # for demonstration, let's skip or do a direct approach
#         # ...
        
#         # 4) Merge into global arrays
#         global_y_true.extend(list(y_true_folder))
#         global_y_pred.extend(list(y_pred_folder))
#         if len(global_y_probs)==0:
#             global_y_probs = y_probs_folder
#         else:
#             global_y_probs = np.concatenate((global_y_probs, y_probs_folder), axis=0)
#         global_results.extend(metrics_info["results"])

#         print(f"[INFO] Folder accuracy: {metrics_info['accuracy']:.2f}% => {metrics_info['true_emotion']} => {len(y_true_folder)} images")

#     # 5) Now do global analysis across all subfolders
#     global_y_true = np.array(global_y_true)
#     global_y_pred = np.array(global_y_pred)
#     # global_y_probs => shape (N,8)

#     # global CSV
#     all_df = pd.DataFrame(global_results)
#     all_csv_path = os.path.join(output_dir, "combined_results.csv")
#     all_df.to_csv(all_csv_path, index=False)
#     print(f"[INFO] Wrote combined CSV => {all_csv_path}")
#     # Global confusion matrix
#     combined_cm_path = os.path.join(output_dir, "combined_confusion_matrix.png")
#     build_confusion_matrix_and_save(global_y_true, global_y_pred, class_names, combined_cm_path)

#     # global classification report
#     global_report = classification_report_dict(global_y_true, global_y_pred, class_names)
#     with open(os.path.join(output_dir, "combined_classification_report.json"), 'w') as f:
#         json.dump(global_report, f, indent=4)
    
#     # optional: global reliability diagram, if you want
#     # we do the same approach => we can define a function or do it inline:
#     # e.g.
#     # see existing approach: we have global_y_probs => shape(N,8), global_y_true => shape(N)
#     # let's do a quick reliability function:

#     def plot_global_reliability(y_true, y_probs, out_path, n_bins=10):
#         # pick probability of correct class
#         correct_probs = []
#         correct_or_not = []
#         for i in range(len(y_true)):
#             gt = y_true[i]
#             p  = y_probs[i, gt]
#             correct_probs.append(p)
#             correct_or_not.append(1 if gt==np.argmax(y_probs[i]) else 0)
#         prob_true, prob_pred = calibration_curve(correct_or_not, correct_probs, n_bins=n_bins)
#         plt.figure()
#         plt.plot(prob_pred, prob_true, 'o-', label='Model')
#         plt.plot([0,1],[0,1],'--', color='gray', label='Perfect')
#         plt.title("Global Reliability Diagram")
#         plt.xlabel("Predicted Probability")
#         plt.ylabel("True Probability")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(out_path)
#         plt.close()

#     reliab_path = os.path.join(output_dir, "combined_reliability_diagram.png")
#     plot_global_reliability(global_y_true, global_y_probs, reliab_path, n_bins=10)
#     print(f"[INFO] Wrote global reliability diagram => {reliab_path}")

#     print("[INFO] Done all analysis.")


# if __name__=="__main__":
#     main()
