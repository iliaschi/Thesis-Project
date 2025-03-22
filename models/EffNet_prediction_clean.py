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

import torchvision.transforms as transforms



def create_efficientnet_b0(num_classes, device='cuda'):
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(1280, num_classes)
    )
    model.to(device)
    model.eval()
    return model

def load_weights(model, weight_path, device='cuda', strict=True):
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
        eps = 1e-12  # small constant to prevent log(0)
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
            # Compute the negative log likelihood for the true class:
            true_prob = max(probs[i, true_idx], eps)
            row['neg_log_likelihood'] = float(-np.log(true_prob))
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

    # 40 epochs finetuned on synthetic
    # C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\models
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_fine_20250319_040801\models"


    # 40 epochs trained on synthetic
    # C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models"


    # base model 40 epochs
    # C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model"

    # full finetuned gender splits at : C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_013917finetuned_real
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_013917finetuned_real\models"
    # full synthetic gender splits at : C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_122332synthetic_on_vggface2
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_20250316_122332synthetic_on_vggface2\models"

    # 2) Your test root that has subfolders for each emotion
    #    e.g. test_root/Angry_6, test_root/Disgust_12, etc.
    
    #### REAL rafdb C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test
    # test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test"

    # SYNTHETIC full test data C:\Users\ilias\Python\Thesis-Project\data\synthetic\test_splits\100M_100W
    # test_root = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\test_splits\100M_100W"

    # Synthetic full no folder
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_100_combined"

    # 3) A global output folder
    # global_out_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203"

    model_parent_dir = os.path.dirname(model_dir)
    print(model_parent_dir)
    global_out_dir = model_parent_dir
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    overall_dir = os.path.join(global_out_dir, f"predictions_on_synth_2_{timestamp}")
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


