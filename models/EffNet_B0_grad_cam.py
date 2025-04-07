# import os
# import re
# import json
# import collections
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from datetime import datetime

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
# from torchvision import transforms

# #######################################
# # 1) CREATE MODEL & LOAD WEIGHTS
# #######################################
# def create_efficientnet_b0(num_classes, device='cuda'):
#     """
#     Create and return an EfficientNet-B0 with a custom classifier layer for 'num_classes'.
#     """
#     model = timm.create_model('tf_efficientnet_b0', pretrained=False)
#     model.classifier = nn.Sequential(
#         nn.Linear(1280, num_classes)
#     )
#     model.to(device)
#     model.eval()
#     return model

# def load_weights(model, weight_path, device='cuda', strict=True):
#     checkpoint = torch.load(weight_path, map_location=device)
#     if isinstance(checkpoint, collections.OrderedDict):
#         model.load_state_dict(checkpoint, strict=strict)
#     elif hasattr(checkpoint, "state_dict"):
#         model.load_state_dict(checkpoint.state_dict(), strict=strict)
#     else:
#         print(f"[WARNING] checkpoint not recognized at {weight_path}")
#     return model

# #######################################
# # 2) TRANSFORMS
# #######################################
# def get_transform(img_size=224):
#     return transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

# #######################################
# # 3) PREDICT FOLDER
# #######################################
# def predict_folder(model, folder_path, true_emotion, transform, class_to_idx, device='cuda', batch_size=32):
#     """
#     Predict on a single subfolder that presumably belongs to 'true_emotion'.
#     Returns a DataFrame with columns:
#       file, path, true_emotion, true_idx, predicted_emotion, predicted_idx,
#       correct, confidence, prob_..., ...
#     """
#     idx_to_class = {v:k for k,v in class_to_idx.items()}
#     true_idx = class_to_idx.get(true_emotion, -1)

#     all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
#     if len(all_images)==0:
#         print(f"[WARNING] No images found in {folder_path}")
#         return pd.DataFrame([])

#     results = []
#     buffer_imgs = []
#     buffer_paths= []

#     def run_batch(buf_imgs, buf_paths):
#         batch = torch.stack(buf_imgs, dim=0).to(device)
#         with torch.no_grad():
#             logits = model(batch)
#             probs  = F.softmax(logits, dim=1).cpu().numpy()
#         eps = 1e-12  # small constant to prevent log(0)
#         for i in range(len(buf_imgs)):
#             row = {}
#             row['file'] = os.path.basename(buf_paths[i])
#             row['path'] = buf_paths[i]
#             row['true_emotion'] = true_emotion
#             row['true_idx'] = int(true_idx)
#             pred_idx = np.argmax(probs[i])
#             row['predicted_idx'] = int(pred_idx)
#             row['predicted_emotion'] = idx_to_class[pred_idx] if pred_idx in idx_to_class else "Unknown"
#             row['correct'] = (pred_idx==true_idx)
#             row['confidence'] = float(probs[i,pred_idx])
#             # Compute negative log-likelihood for the true class:
#             true_prob = max(probs[i, true_idx], eps)
#             row['neg_log_likelihood'] = float(-np.log(true_prob))
#             # store all class probabilities
#             for cix, cname in idx_to_class.items():
#                 row[f"prob_{cname}"] = float(probs[i,cix])
#             results.append(row)

#     for fname in sorted(all_images):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             img = Image.open(fpath).convert('RGB')
#             im_t= transform(img)
#             buffer_imgs.append(im_t)
#             buffer_paths.append(fpath)
#             if len(buffer_imgs)>=batch_size:
#                 run_batch(buffer_imgs, buffer_paths)
#                 buffer_imgs.clear()
#                 buffer_paths.clear()
#         except Exception as e:
#             print(f"[WARNING] could not read {fpath}: {str(e)}")

#     # leftover
#     if len(buffer_imgs)>0:
#         run_batch(buffer_imgs, buffer_paths)
#         buffer_imgs.clear()
#         buffer_paths.clear()

#     df = pd.DataFrame(results)
#     return df

# #######################################
# # 4) GRAD-CAM HOOKS & COMPUTATION
# #######################################
# class GradCamHook:
#     """
#     Helper class to store feature maps & gradients from the final conv layer
#     """
#     def __init__(self, module):
#         self.module = module
#         self.fmap = None
#         self.grad = None
#         self.fwd_handle = module.register_forward_hook(self.forward_hook)
#         self.bwd_handle = module.register_backward_hook(self.backward_hook)

#     def forward_hook(self, module, input_, output_):
#         self.fmap = output_.detach()

#     def backward_hook(self, module, grad_in, grad_out):
#         self.grad = grad_out[0].detach()

#     def remove(self):
#         self.fwd_handle.remove()
#         self.bwd_handle.remove()

# def compute_gradcam(model, gradcam_hook, img_tensor, target_class, device='cuda'):
#     """
#     Compute Grad-CAM for a single image & target_class.
#     Returns a (H,W) heatmap in [0..1].
#     """
#     model.eval()
#     img_tensor = img_tensor.unsqueeze(0).to(device)

#     logits = model(img_tensor)
#     loss = logits[0, target_class]

#     model.zero_grad()
#     loss.backward()

#     # feature maps => shape (1, C, Hf, Wf) => hook stores [0] => (C,Hf,Wf)
#     fmap = gradcam_hook.fmap[0]
#     grad = gradcam_hook.grad[0]

#     # global average pool the gradient => per-channel weight
#     alpha = grad.view(grad.shape[0], -1).mean(dim=1)  # shape (C,)
#     cam_map = torch.zeros((fmap.shape[1], fmap.shape[2]), dtype=fmap.dtype, device=fmap.device)
#     for c in range(fmap.shape[0]):
#         cam_map += alpha[c] * fmap[c]

#     cam_map = F.relu(cam_map)
#     cam_map -= cam_map.min()
#     if cam_map.max() > 0:
#         cam_map /= cam_map.max()
#     cam_map = cam_map.detach().cpu().numpy()

#     # upsample to original image size
#     H, W = img_tensor.shape[2], img_tensor.shape[3]
#     cam_map = cv2.resize(cam_map, (W,H), interpolation=cv2.INTER_LINEAR)
#     cam_map = np.maximum(cam_map, 0)
#     if cam_map.max() > 0:
#         cam_map /= cam_map.max()

#     return cam_map  # shape (H,W)

# def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5):
#     """
#     overlay the Grad-CAM heatmap onto original RGB image => returns BGR for cv2 saving
#     """
#     if isinstance(orig_img, Image.Image):
#         orig_img = np.array(orig_img)  # shape (H,W,3) in RGB
#     H,W,_ = orig_img.shape
#     if heatmap.shape[:2] != (H,W):
#         heatmap = cv2.resize(heatmap, (W,H))

#     heat_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
#     heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
#     blend = (alpha * heat_color + (1 - alpha) * orig_img).astype(np.uint8)
#     blend_bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
#     return blend_bgr

# #######################################
# # 5) MAIN
# #######################################
# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # ----- Weight Path -----
#     # 1) Where your model weights are stored
#     # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\...."

#     # --- Part A ---
#     # Affect Net
#     model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\model"

#     # Best Finetuned classification layer only dropout 0
#     # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230_drop0_OK\models_best"

#     # Best Synthetic
#     # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models_best"
    
    
#     # 2) Test root => subfolders "Angry","Disgust", etc.
#     test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_merged"
    



#     model_parent_dir = os.path.dirname(model_dir)
#     timestamp = datetime.now().strftime("%m%d_%H%M%S")
#     overall_dir = os.path.join(model_parent_dir, f"gradcam_pred_{timestamp}")
#     os.makedirs(overall_dir, exist_ok=True)


#     # 3) Class mapping
#     class_to_idx = {
#         'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
#         'Happiness':4,'Neutral':5,'Sadness':6,'Surprise':7
#     }
#     idx_to_class = {v:k for k,v in class_to_idx.items()}

#     # 4) transform
#     tfm = get_transform(224)

#     # 5) Loop over each .pth or .pt in model_dir
#     for fname in os.listdir(model_dir):
#         if not fname.lower().endswith(('.pth','.pt')):
#             continue

#         weight_path = os.path.join(model_dir, fname)
#         model_name = os.path.splitext(fname)[0]
#         model_out_dir = os.path.join(overall_dir, model_name)
#         os.makedirs(model_out_dir, exist_ok=True)

#         print(f"\n============================")
#         print(f"[INFO] Evaluating (GradCAM) model: {fname}")
#         print(f"[INFO] Output => {model_out_dir}")

#         # create model
#         num_classes = len(class_to_idx)
#         model = create_efficientnet_b0(num_classes, device=device)
#         load_weights(model, weight_path, device=device, strict=False)
#         model.eval()

#         # Identify final conv layer => e.g. model.conv_head
#         last_conv = model.conv_head
#         gradcam_hook = GradCamHook(last_conv)

#         # We'll unify all predictions
#         all_dfs = []
#         subfolders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
#         for subf in subfolders:
#             subf_path = os.path.join(test_root, subf)
#             true_emotion = subf
#             df = predict_folder(model, subf_path, true_emotion, tfm, class_to_idx, device=device, batch_size=32)
#             if len(df) > 0:
#                 out_csv = os.path.join(model_out_dir, f"results_{true_emotion}.csv")
#                 df.to_csv(out_csv, index=False)
#                 print(f"[INFO] Wrote {len(df)} rows => {out_csv}")
#                 all_dfs.append(df)

#         if len(all_dfs) > 0:
#             combined = pd.concat(all_dfs, ignore_index=True)
#             combined_csv = os.path.join(model_out_dir, "combined_results.csv")
#             combined.to_csv(combined_csv, index=False)
#             print(f"[INFO] Wrote combined CSV => {combined_csv}")

#             # 6) Grad-CAM for only correct predictions
#             corr_df = combined[combined['correct']==True]
#             print(f"[INFO] Found {len(corr_df)} correct predictions => generating GradCAM overlays.")

#             gradcam_dir = os.path.join(model_out_dir, "gradcam")
#             os.makedirs(gradcam_dir, exist_ok=True)

#             # optional limit => e.g. 200 images max
#             # corr_df = corr_df.sample(n=200, random_state=123)  # if desired

#             for i, row in tqdm(corr_df.iterrows(), total=len(corr_df), desc="GradCAM for correct samples"):
#                 fpath = row['path']
#                 pred_idx = row['predicted_idx']
#                 pred_class_name = row['predicted_emotion']

#                 # load image
#                 try:
#                     img_pil = Image.open(fpath).convert('RGB')
#                 except:
#                     continue

#                 # compute gradcam
#                 img_t = tfm(img_pil)
#                 heatmap = compute_gradcam(model, gradcam_hook, img_t, pred_idx, device=device)
#                 overlay_bgr = overlay_heatmap_on_image(img_pil, heatmap, alpha=0.4)

#                 # save => e.g. gradcam/{pred_class_name}/original_filename.png
#                 class_subdir = os.path.join(gradcam_dir, pred_class_name)
#                 os.makedirs(class_subdir, exist_ok=True)
#                 base_fname = os.path.basename(fpath)
#                 out_img_path = os.path.join(class_subdir, base_fname)
#                 cv2.imwrite(out_img_path, overlay_bgr)

#             gradcam_hook.remove()
#             print(f"[INFO] GradCAM overlays => {gradcam_dir}")
#         else:
#             print("[WARNING] No data from subfolders? Skipping GradCAM for this model.")

#     print("[INFO] All done with GradCAM across all models.")

# ################################
# # ENTRY POINT
# ################################
# if __name__=="__main__":
#     main()


import os
import re
import collections
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm

########################
# 1) MODEL + LOADER
########################
def create_efficientnet_b0(num_classes, device='cuda'):
    """
    Create an EfficientNet-B0 from timm, with a custom classifier for 'num_classes'.
    """
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(1280, num_classes)
    )
    model.to(device)
    model.eval()
    return model

def load_weights(model, weight_path, device='cuda', strict=True):
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, collections.OrderedDict):
        model.load_state_dict(checkpoint, strict=strict)
    elif hasattr(checkpoint, "state_dict"):
        model.load_state_dict(checkpoint.state_dict(), strict=strict)
    else:
        print(f"[WARNING] checkpoint not recognized: {weight_path}")
    return model

########################
# 2) TRANSFORM
########################
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

########################
# 3) PREDICT FOLDER
########################
def predict_folder(model, folder_path, true_emotion, transform, class_to_idx, device='cuda', batch_size=32):
    """
    Predict on a single subfolder for 'true_emotion'.
    Return a DataFrame with columns:
      file, path, true_emotion, true_idx, predicted_emotion, predicted_idx, correct, confidence, etc.
    """
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    true_idx = class_to_idx.get(true_emotion, -1)

    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    if len(all_images)==0:
        return pd.DataFrame([])

    results = []
    buffer_imgs = []
    buffer_paths= []

    def run_batch(buf_imgs, buf_paths):
        batch = torch.stack(buf_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        eps = 1e-12
        for i in range(len(buf_imgs)):
            row = {}
            row['file'] = os.path.basename(buf_paths[i])
            row['path'] = buf_paths[i]
            row['true_emotion'] = true_emotion
            row['true_idx'] = int(true_idx)
            pred_idx = np.argmax(probs[i])
            row['predicted_idx'] = int(pred_idx)
            row['predicted_emotion'] = idx_to_class[pred_idx] if pred_idx in idx_to_class else "Unknown"
            row['correct'] = (pred_idx==true_idx)
            row['confidence'] = float(probs[i,pred_idx])
            results.append(row)

    for fname in sorted(all_images):
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            im_t= transform(img)
            buffer_imgs.append(im_t)
            buffer_paths.append(fpath)
            if len(buffer_imgs)>=batch_size:
                run_batch(buffer_imgs, buffer_paths)
                buffer_imgs.clear()
                buffer_paths.clear()
        except:
            pass

    if len(buffer_imgs)>0:
        run_batch(buffer_imgs, buffer_paths)
        buffer_imgs.clear()

    df = pd.DataFrame(results)
    return df

########################
# 4) GRAD-CAM HOOKS
########################
class GradCamHook:
    def __init__(self, module):
        self.module = module
        self.fmap = None
        self.grad = None
        self.fwd_handle = module.register_forward_hook(self.forward_hook)
        self.bwd_handle = module.register_backward_hook(self.backward_hook)
    def forward_hook(self, m, i, o):
        self.fmap = o.detach()
    def backward_hook(self, m, grad_in, grad_out):
        self.grad = grad_out[0].detach()
    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

def compute_gradcam(model, hook, img_tensor, target_class, device='cuda'):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    logits = model(img_tensor)
    loss = logits[0, target_class]

    model.zero_grad()
    loss.backward()

    fmap = hook.fmap[0]  # shape (C,Hf,Wf)
    grad = hook.grad[0]  # shape (C,Hf,Wf)

    alpha = grad.view(grad.shape[0], -1).mean(dim=1)  # shape (C,)
    cam_map = torch.zeros((fmap.shape[1], fmap.shape[2]), dtype=fmap.dtype, device=fmap.device)
    for c in range(fmap.shape[0]):
        cam_map += alpha[c] * fmap[c]

    cam_map = F.relu(cam_map)
    cam_map -= cam_map.min()
    if cam_map.max() > 0:
        cam_map /= cam_map.max()

    cam_map = cam_map.detach().cpu().numpy()
    # upsample to 224x224
    cam_map = cv2.resize(cam_map, (224,224), interpolation=cv2.INTER_LINEAR)
    cam_map = np.maximum(cam_map, 0)
    if cam_map.max() > 0:
        cam_map /= cam_map.max()

    return cam_map  # shape (224,224)

def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5):
    """
    Blend heatmap with orig_img (H,W,3 in RGB).
    Return BGR for saving via cv2.
    """
    if isinstance(orig_img, Image.Image):
        orig_img = np.array(orig_img)
    H, W, _ = orig_img.shape
    # if needed, resize heatmap => (W,H)
    if heatmap.shape[:2] != (H,W):
        heatmap = cv2.resize(heatmap, (W,H))
    heat_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blend = (alpha * heat_color + (1 - alpha)*orig_img).astype(np.uint8)
    bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
    return bgr

#######################################
# 5) MAIN
#######################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Weight Path -----
    # 1) Where your model weights are stored
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\...."

    # --- Part A ---
    # Affect Net
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\model"

    # Best Finetuned classification layer only dropout 0
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230_drop0_OK\models_best"

    # Best Synthetic
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models_best"
    
    
    # 2) Test root => subfolders "Angry","Disgust", etc.
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_merged"
    



    model_parent_dir = os.path.dirname(model_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    overall_dir = os.path.join(model_parent_dir, f"gradcam_pred_{timestamp}")
    os.makedirs(overall_dir, exist_ok=True)


    # 3) Class mapping
    class_to_idx = {
        'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
        'Happy':4,'Neutral':5,'Sad':6,'Surprise':7
    }
    idx_to_class = {v:k for k,v in class_to_idx.items()}

########################
# 5) MAIN
########################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    # ----- Weight Path -----
    # 1) Where your model weights are stored
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\...."

    # --- Part A ---
    # Affect Net
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\model"

    # Best Finetuned classification layer only dropout 0
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230_drop0_OK\models_best"

    # Best Synthetic
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models_best"
    
    
    # 2) Test root => subfolders "Angry","Disgust", etc.
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_merged"



    model_parent = os.path.dirname(model_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(model_parent, f"gradcam_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # test root => e.g. "Angry","Disgust","Fear" subfolders
    # test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test"

    # classes
    class_to_idx = {
        'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
        'Happy':4,'Neutral':5,'Sad':6,'Surprise':7
    }
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    tfm = get_transform(224)

    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(('.pth','.pt')):
            continue

        weight_path = os.path.join(model_dir, fname)
        model_name = os.path.splitext(fname)[0]
        model_out = os.path.join(out_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        print(f"\n===============================")
        print(f"[INFO] GradCAM for model: {fname}")
        print(f"[INFO] => {model_out}")

        # create & load
        model = create_efficientnet_b0(num_classes, device=device)
        load_weights(model, weight_path, device=device, strict=True)
        model.eval()

        # final conv
        last_conv = model.conv_head
        hook = GradCamHook(last_conv)

        # We'll gather up to 100 correct images per predicted class
        max_per_class = 50
        count_per_class = {c:0 for c in range(num_classes)}

        # We'll store an average heatmap for each class => shape (224,224)
        accum_heatmaps = {c:[] for c in range(num_classes)}

        # subfolders
        subfolders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root,d))]

        for subf in subfolders:
            subf_path = os.path.join(test_root, subf)
            # parse subf as emotion
            true_emotion = subf
            df = predict_folder(model, subf_path, true_emotion, tfm, class_to_idx, device=device)
            if len(df)==0:
                continue
            # for each row => if correct => do gradcam if we haven't reached 100 for that predicted class
            for i, row in df.iterrows():
                if not row['correct']:
                    continue
                pred_idx = int(row['predicted_idx'])
                if count_per_class[pred_idx] >= max_per_class:
                    continue  # skip if already have enough
                # gradcam
                try:
                    img_pil = Image.open(row['path']).convert('RGB')
                except:
                    continue
                img_t = tfm(img_pil)
                cam_map = compute_gradcam(model, hook, img_t, pred_idx, device=device)
                accum_heatmaps[pred_idx].append(cam_map)

                # overlay
                overlay_bgr = overlay_heatmap_on_image(img_pil.resize((224,224)), cam_map, alpha=0.4)

                # save => model_out/gradcam/{class_name}/img.png
                cls_name = idx_to_class[pred_idx]
                cls_dir = os.path.join(model_out, "gradcam", cls_name)
                os.makedirs(cls_dir, exist_ok=True)
                basef = os.path.basename(row['path'])
                out_img = os.path.join(cls_dir, basef)
                cv2.imwrite(out_img, overlay_bgr)
                count_per_class[pred_idx]+=1

        # remove hook
        hook.remove()
        print(f"[INFO] GradCAM done => saved overlays in {model_out}/gradcam")

        # build a combined average heatmap for each class
    #     combined_dir = os.path.join(model_out, "combined_heatmaps")
    #     os.makedirs(combined_dir, exist_ok=True)

    #     # heatmap per class
    #     for c_idx in range(num_classes):
    #         heat_list = accum_heatmaps[c_idx]
    #         if len(heat_list)==0:
    #             continue
    #         # average them
    #         # shape => (N,224,224)
    #         stack = np.stack(heat_list, axis=0)  # shape (N,224,224)
    #         avg_hm = stack.mean(axis=0)         # shape (224,224)
    #         # normalize again 0..1
    #         avg_hm = np.clip(avg_hm, 0, 1)

    #         # optionally save as color
    #         color_map = cv2.applyColorMap((avg_hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    #         # convert to BGR => no need, applyColorMap is BGR, for saving is fine
    #         # but let's do it in standard:
    #         # color_map is shape (224,224,3) in BGR
    #         outpath = os.path.join(combined_dir, f"combined_{idx_to_class[c_idx]}.png")
    #         cv2.imwrite(outpath, color_map)
    #         print(f"[INFO] Saved combined heatmap for class {idx_to_class[c_idx]} => {outpath}")

    # print("\n[INFO] All done with GradCAM for all models.")
    
        # build a combined average heatmap for each class
        combined_dir = os.path.join(model_out, "combined_heatmaps")
        os.makedirs(combined_dir, exist_ok=True)

        for c_idx in range(num_classes):
            heat_list = accum_heatmaps[c_idx]
            if len(heat_list) == 0:
                continue
            
            # Average the collected Grad-CAM maps: shape (N,224,224)
            stack = np.stack(heat_list, axis=0)  # shape (N,224,224)
            avg_hm = stack.mean(axis=0)         # shape (224,224)
            
            # 1) Ensure in range [0,1]
            avg_hm = np.clip(avg_hm, 0, 1)

            # 2) Keep only the top 300 highest-intensity pixels
            flat = avg_hm.ravel()
            num_pixels = flat.size
            if num_pixels > 1200:
                # Sort ascending
                sorted_vals = np.sort(flat)
                # The 300th-largest is at index [-300]
                threshold = sorted_vals[-300]
                # Zero everything below threshold
                avg_hm[avg_hm < threshold] = 0.0
            
            # 3) Renormalize so top region is in [0..1]
            m = avg_hm.max()
            if m > 0:
                avg_hm /= m

            # 4) Apply color map and save
            color_map = cv2.applyColorMap((avg_hm*255).astype(np.uint8), cv2.COLORMAP_JET)
            outpath = os.path.join(combined_dir, f"combined_{idx_to_class[c_idx]}.png")
            cv2.imwrite(outpath, color_map)
            print(f"[INFO] Saved top-300 thresholded heatmap for class {idx_to_class[c_idx]} => {outpath}")



if __name__=="__main__":
    main()
