import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import re

def create_efficientnet_b0(num_classes, device='cuda'):
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(1280, num_classes)
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

def predict_combined(model, folder_path, transform, class_to_idx, device='cuda', batch_size=32):
    """
    Predict on a unified folder where each image's filename encodes its true label.
    Expected filename format: e.g. "Angry_1.jpg", "Contempt_12.png", etc.
    The script will extract the leading alphabetical characters and map them to the proper label.
    
    Returns a DataFrame with columns:
      file, path, true_emotion, true_idx, predicted_emotion, predicted_idx,
      correct, confidence, neg_log_likelihood, and probability for each class.
    """
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if len(all_images) == 0:
        print(f"[WARNING] No images found in {folder_path}")
        return pd.DataFrame([])
    
    # Mapping to standardize emotion names.
    emotion_mapping = {
        'angry': 'Angry',
        'anger': 'Angry',
        'contempt': 'Contempt',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happiness',
        'happiness': 'Happiness',
        'neutral': 'Neutral',
        'sad': 'Sadness',
        'sadness': 'Sadness',
        'surprise': 'Surprise',
        'surprised': 'Surprise'
    }
    
    results = []
    buffer_imgs = []
    buffer_paths = []
    buffer_true_emotions = []
    buffer_true_idxs = []
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    eps = 1e-12  # For numerical stability
    
    def run_batch(buf_imgs, buf_paths, buf_true_emotions, buf_true_idxs):
        batch = torch.stack(buf_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        for i in range(len(buf_imgs)):
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
            for cix, cname in idx_to_class.items():
                row[f"prob_{cname}"] = float(probs[i, cix])
            results.append(row)
        buf_imgs.clear()
        buf_paths.clear()
        buf_true_emotions.clear()
        buf_true_idxs.clear()
    
    # Process each image in sorted order.
    for fname in sorted(all_images):
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            im_t = transform(img)
            buffer_imgs.append(im_t)
            buffer_paths.append(fpath)
            # Extract alphabetical part from filename.
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
        except Exception as e:
            print(f"[WARNING] Could not read {fpath}: {str(e)}")
    if len(buffer_imgs) > 0:
        run_batch(buffer_imgs, buffer_paths, buffer_true_emotions, buffer_true_idxs)
    
    df = pd.DataFrame(results)
    return df

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set model weights directory (update as needed).
    
    # Affect Net : C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model"
    
    # Fine drop 0.3
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_22_1652do03\models"
    # Fine drop 0.5
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_22_1831do05\models"
    # Fine drop 0.1
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_1912_dr01\models"
    # Fine drop 0 new classifier
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2031\models"
    # Fine drop 0 full 40
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230\models"
    
    # Synthetic model
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models"

    # Gender splits
    # synth
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_0316_1223synthetic_on_vggface2_gender\models"
    #real
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_0316_013917fine_real_gender\models"

    # Fraction 0.25
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0324_0201frac025\models"
    # Fraction 0.5
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0324_0154_frac05\models"
    # Fraction 0.75
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0324_0131_fract0.75\models"

    
    
    # Set the combined test folder (output from your combine script).
    # test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_100_combined"
    
    # RAFDB real data test merged
    test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_combined"
    
    # Global output folder for predictions.
    # global_out_dir = r"C:\Users\ilias\Python\Thesis-Project\results"
    model_parent_dir = os.path.dirname(model_dir)
    print(model_parent_dir)
    global_out_dir = model_parent_dir

    timestamp = datetime.now().strftime("%m%d_%H%M")
    overall_dir = os.path.join(global_out_dir, f"pred_comb_sy_on_re_2_{timestamp}")
    os.makedirs(overall_dir, exist_ok=True)
    
    # Define the class mapping.
    class_to_idx = {
        'Angry': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3,
        'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7
    }
    
    # Loop over each model weight file in model_dir.
    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(('.pth', '.pt')):
            continue
        
        weight_path = os.path.join(model_dir, fname)
        model_name = os.path.splitext(fname)[0]
        
        # Create a subfolder for this model's results.
        model_out_dir = os.path.join(overall_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        print(f"\n============================")
        print(f"[INFO] Evaluating model: {fname} on test folder => {test_folder}")
        print(f"[INFO] Results subfolder => {model_out_dir}")
        
        num_classes = len(class_to_idx)
        model = create_efficientnet_b0(num_classes, device=device)
        try:
            load_weights(model, weight_path, device=device, strict=False)
            print(f"[INFO] Loaded checkpoint from {fname}")
        except Exception as e:
            print(f"[ERROR] Could not load {fname}: {str(e)}")
            continue
        
        transform = get_transform(224)
        
        # Run predictions on the combined folder.
        df = predict_combined(model, test_folder, transform, class_to_idx, device=device, batch_size=32)
        if len(df) > 0:
            combined_csv = os.path.join(model_out_dir, "combined_results.csv")
            df.to_csv(combined_csv, index=False)
            print(f"[INFO] Wrote combined CSV => {combined_csv}")
        else:
            print("[WARNING] No predictions were made.")
    
    print("[INFO] All done.")

if __name__ == "__main__":
    main()



# import os
# import torch
# import torch.nn as nn
# import timm
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from datetime import datetime
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torchvision import transforms

# def create_efficientnet_b0(num_classes, device='cuda'):
#     model = timm.create_model('tf_efficientnet_b0', pretrained=False)
#     model.classifier = nn.Sequential(
#         nn.Linear(1280, num_classes)
#     )
#     model.to(device)
#     model.eval()
#     return model

# def load_weights(model, weight_path, device='cuda', strict=True):
#     checkpoint = torch.load(weight_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=strict)
#     return model

# def get_transform(img_size=224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std =[0.229, 0.224, 0.225]
#         )
#     ])

# def predict_combined(model, folder_path, transform, class_to_idx, device='cuda', batch_size=32):
#     """
#     Predict on a unified folder where each image's filename encodes its true label.
#     Expected filename format: "Angry_1.jpg", "Contempt_12.png", etc.
    
#     Returns a DataFrame with columns:
#       file, path, true_emotion, true_idx, predicted_emotion, predicted_idx,
#       correct, confidence, neg_log_likelihood, and probability for each class.
#     """
#     all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
#     if len(all_images) == 0:
#         print(f"[WARNING] No images found in {folder_path}")
#         return pd.DataFrame([])
    
#     results = []
#     buffer_imgs = []
#     buffer_paths = []
#     buffer_true_emotions = []
#     buffer_true_idxs = []
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     eps = 1e-12  # for numerical stability
    
#     def run_batch(buf_imgs, buf_paths, buf_true_emotions, buf_true_idxs):
#         batch = torch.stack(buf_imgs, dim=0).to(device)
#         with torch.no_grad():
#             logits = model(batch)
#             probs = F.softmax(logits, dim=1).cpu().numpy()
#         for i in range(len(buf_imgs)):
#             row = {}
#             row['file'] = os.path.basename(buf_paths[i])
#             row['path'] = buf_paths[i]
#             row['true_emotion'] = buf_true_emotions[i]
#             row['true_idx'] = int(buf_true_idxs[i])
#             pred_idx = int(np.argmax(probs[i]))
#             row['predicted_idx'] = pred_idx
#             row['predicted_emotion'] = idx_to_class.get(pred_idx, "Unknown")
#             row['correct'] = (pred_idx == buf_true_idxs[i])
#             row['confidence'] = float(probs[i, pred_idx])
#             # Compute negative log likelihood for the true class:
#             true_prob = max(probs[i, buf_true_idxs[i]], eps)
#             row['neg_log_likelihood'] = float(-np.log(true_prob))
#             # Store all class probabilities.
#             for cix, cname in idx_to_class.items():
#                 row[f"prob_{cname}"] = float(probs[i, cix])
#             results.append(row)
#         buf_imgs.clear()
#         buf_paths.clear()
#         buf_true_emotions.clear()
#         buf_true_idxs.clear()
    
#     # Process each image in sorted order.
#     for fname in sorted(all_images):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             img = Image.open(fpath).convert('RGB')
#             im_t = transform(img)
#             buffer_imgs.append(im_t)
#             buffer_paths.append(fpath)
#             # Extract true label from the filename.
#             # Assumes the filename is like "Angry_1.jpg": split on '_' and take the first part.
#             base = os.path.basename(fname)
#             true_emotion = base.split('_')[0]
#             buffer_true_emotions.append(true_emotion)
#             true_idx = class_to_idx.get(true_emotion, -1)
#             buffer_true_idxs.append(true_idx)
#             if len(buffer_imgs) >= batch_size:
#                 run_batch(buffer_imgs, buffer_paths, buffer_true_emotions, buffer_true_idxs)
#         except Exception as e:
#             print(f"[WARNING] Could not read {fpath}: {str(e)}")
#     if len(buffer_imgs) > 0:
#         run_batch(buffer_imgs, buffer_paths, buffer_true_emotions, buffer_true_idxs)
    
#     df = pd.DataFrame(results)
#     return df

# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Set model weights directory (update as needed).
#     # Affect Net : C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model
#     model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base\model"
    
#     # Set the combined test folder (output from your combine script).
#     test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\synth_100_combined"
    
#     # Global output folder for predictions.
#     # global_out_dir = r"C:\Users\ilias\Python\Thesis-Project\results"
#     model_parent_dir = os.path.dirname(model_dir)
#     print(model_parent_dir)
#     global_out_dir = model_parent_dir
    
#     timestamp = datetime.now().strftime("%m%d_%H%M%S")
#     overall_dir = os.path.join(global_out_dir, f"predictions_combined_synth_{timestamp}")
#     os.makedirs(overall_dir, exist_ok=True)
    
#     # Define the class mapping.
#     class_to_idx = {
#         'Angry': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3,
#         'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7
#     }
    
#     # Loop over each model weight file in model_dir.
#     for fname in os.listdir(model_dir):
#         if not fname.lower().endswith(('.pth', '.pt')):
#             continue
        
#         weight_path = os.path.join(model_dir, fname)
#         model_name = os.path.splitext(fname)[0]
        
#         # Create a subfolder for this model's results.
#         model_out_dir = os.path.join(overall_dir, model_name)
#         os.makedirs(model_out_dir, exist_ok=True)
        
#         print(f"\n============================")
#         print(f"[INFO] Evaluating model: {fname} on test folder => {test_folder}")
#         print(f"[INFO] Results subfolder => {model_out_dir}")
        
#         num_classes = len(class_to_idx)
#         model = create_efficientnet_b0(num_classes, device=device)
#         try:
#             load_weights(model, weight_path, device=device, strict=False)
#             print(f"[INFO] Loaded checkpoint from {fname}")
#         except Exception as e:
#             print(f"[ERROR] Could not load {fname}: {str(e)}")
#             continue
        
#         transform = get_transform(224)
        
#         # Run predictions on the combined folder.
#         df = predict_combined(model, test_folder, transform, class_to_idx, device=device, batch_size=32)
#         if len(df) > 0:
#             combined_csv = os.path.join(model_out_dir, "combined_results.csv")
#             df.to_csv(combined_csv, index=False)
#             print(f"[INFO] Wrote combined CSV => {combined_csv}")
#         else:
#             print("[WARNING] No predictions were made.")
    
#     print("[INFO] All done.")

# if __name__ == "__main__":
#     main()
