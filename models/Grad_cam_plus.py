import os
import re
import collections
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

import cv2

##############################
# 1) CREATE MODEL & LOAD WEIGHTS
##############################

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

def load_weights(model, weight_path, device='cuda', strict=False):
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, collections.OrderedDict):
        model.load_state_dict(checkpoint, strict=strict)
    elif hasattr(checkpoint, "state_dict"):
        model.load_state_dict(checkpoint.state_dict(), strict=strict)
    else:
        print(f"[WARNING] checkpoint not recognized: {weight_path}")
    return model

##############################
# 2) TRANSFORMS
##############################

def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        )
    ])

##############################
# 3) PREDICT FOLDER
##############################

def predict_folder(model, folder_path, true_emotion, transform, class_to_idx, device='cuda', batch_size=32):
    """
    Predict on a single subfolder that presumably all belongs to 'true_emotion'.
    Return a DataFrame with columns:
      file, path, true_emotion, true_idx, predicted_emotion, predicted_idx, correct, confidence.
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
        for i in range(len(buf_imgs)):
            row = {}
            row['file'] = os.path.basename(buf_paths[i])
            row['path'] = buf_paths[i]
            row['true_emotion'] = true_emotion
            row['true_idx'] = int(true_idx)
            pred_idx = int(np.argmax(probs[i]))
            row['predicted_idx'] = pred_idx
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

##############################
# 4) GRAD-CAM++ IMPLEMENTATION
##############################

class GradCamHook:
    """
    We store forward feature maps + gradients from final conv layer
    """
    def __init__(self, module):
        self.module = module
        self.fmap = None
        self.grad = None
        self.fwd_handle = module.register_forward_hook(self.forward_hook)
        self.bwd_handle = module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.fmap = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

def compute_gradcam_pp(model, hook, img_tensor, target_class, device='cuda'):
    """
    Compute Grad-CAM++ for a single image, single class.
    Returns a (224,224) heatmap in [0..1].
    """
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    logits = model(img_tensor)
    loss = logits[0, target_class]

    model.zero_grad()
    loss.backward()

    # shape => (1,C,Hf,Wf)
    fmap = hook.fmap      # shape (1, C, Hf, Wf)
    grad = hook.grad      # shape (1, C, Hf, Wf)
    fmap = fmap[0]        # => (C,Hf,Wf)
    grad = grad[0]        # => (C,Hf,Wf)

    # approximate second & third derivatives
    second_deriv = grad * grad      # partial^2
    third_deriv  = second_deriv * grad  # partial^3

    # sum over spatial dims
    alpha_num = second_deriv.sum(dim=(1,2))  # shape (C,)
    denom = 2.0 * second_deriv.sum(dim=(1,2)) + (fmap * third_deriv).sum(dim=(1,2))
    # avoid zero
    denom = torch.where(denom==0, torch.ones_like(denom), denom)
    alpha = alpha_num / denom  # shape (C,)

    # Grad-CAM++ formula => ReLU(dY/dA)
    relu_dYdA = F.relu(grad)
    gc_map = torch.zeros_like(fmap[0])
    for c in range(fmap.shape[0]):
        gc_map += alpha[c] * relu_dYdA[c] * fmap[c]

    gc_map = F.relu(gc_map)
    # normalize
    gc_map -= gc_map.min()
    if gc_map.max()>0:
        gc_map /= gc_map.max()

    # upsample => (224,224)
    gc_map = gc_map.detach().cpu().numpy()
    gc_map = cv2.resize(gc_map, (224,224), interpolation=cv2.INTER_LINEAR)
    gc_map = np.clip(gc_map, 0, 1)

    return gc_map

def apply_top_pixels(heatmap, top_fraction=0.01):
    """
    Keep only the top 'top_fraction' fraction of pixels in the heatmap, zero out the rest.
    Then renormalize to [0..1].
    e.g. top_fraction=0.01 => top 1% of pixels kept.
    """
    flat = heatmap.ravel()
    num_pixels = flat.size
    top_count = int(num_pixels * top_fraction)
    if top_count < 1:
        return heatmap  # no change

    # sort ascending
    sorted_vals = np.sort(flat)
    threshold = sorted_vals[-top_count]  # 1% => -top_count
    heatmap[heatmap < threshold] = 0.0

    m = heatmap.max()
    if m>0:
        heatmap /= m
    return heatmap

def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5):
    """
    Overlay heatmap onto orig_img, returns BGR for cv2 saving.
    """
    if isinstance(orig_img, Image.Image):
        orig_img = np.array(orig_img)
    H,W,_ = orig_img.shape
    # resize heatmap => match (W,H)
    if heatmap.shape[:2] != (H,W):
        heatmap = cv2.resize(heatmap, (W,H))

    color_map = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    blend = (alpha * color_map + (1 - alpha) * orig_img).astype(np.uint8)
    bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
    return bgr


def build_class_consensus_heatmap(heatmaps, pixel_thresh=0.5, fraction=0.3):
    """
    Build a 'consensus' Grad-CAM map from multiple 224x224 heatmaps for one class.

    Steps:
      1) For each heatmap, create a binary mask where pixel > pixel_thresh => 1 else 0.
      2) Accumulate how many times each pixel is '1' across all heatmaps.
      3) If count >= fraction * len(heatmaps), set that pixel to 1 in the final consensus, else 0.
      4) Return a float array in [0..1], shape (224,224).

    :param heatmaps: list of arrays, each shape (224,224), in [0..1].
    :param pixel_thresh: e.g. 0.5 => we consider a pixel 'active' if heatmap[pix] >= 0.5.
    :param fraction: e.g. 0.3 => must appear active in at least 30% of the images.
    :return: consensus_map, shape (224,224) in [0..1].
    """
    if not heatmaps:
        return None

    # 1) Create a boolean mask for each heatmap => active where > pixel_thresh
    masks = []
    for hm in heatmaps:
        active = (hm >= pixel_thresh).astype(np.uint8)
        masks.append(active)

    # 2) Stack and sum
    stack = np.stack(masks, axis=0)  # shape (N,224,224)
    accum = stack.sum(axis=0)       # shape (224,224), each pixel is how many images had 'active=1'

    # 3) Build consensus mask
    num_images = len(heatmaps)
    required_count = int(np.ceil(fraction * num_images))  # e.g. 30% => must appear in >= that many images

    consensus_mask = (accum >= required_count).astype(np.float32)

    # Optionally we can keep it as a binary mask in [0..1],
    # or do a partial fraction approach. For demonstration, we'll keep it binary:
    return consensus_mask

def colorize_and_save_consensus_map(consensus_mask, out_path):
    """
    Convert a (224,224) array in [0..1] to a color map (COLORMAP_JET) and save via cv2.
    """
    if consensus_mask is None:
        print(f"[WARNING] No consensus map to save => {out_path}")
        return

    # 1) If we only have 0/1, we can just keep it that way or do a color map
    # Scale 0..1 => 0..255
    color_map = cv2.applyColorMap((consensus_mask*255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(out_path, color_map)
    print(f"[INFO] Saved consensus map => {out_path}")



##############################
# 5) MAIN
##############################

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


#     # This fraction is user-adjustable => top 1% of pixels
#     top_fraction = 0.1



#     model_parent = os.path.dirname(model_dir)
#     timestamp = datetime.now().strftime("%m%d_%H%M%S")
#     out_dir = os.path.join(model_parent, f"gradcam_pp_{timestamp}")
#     os.makedirs(out_dir, exist_ok=True)

#     # B) Test root with subfolders => "Angry","Disgust","Fear", etc.

#     class_to_idx = {
#         'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
#         'Happy':4,'Neutral':5,'Sad':6,'Surprise':7
#     }
#     idx_to_class = {v:k for k,v in class_to_idx.items()}

#     tfm = get_transform(224)



#     # Loop over each model
#     for fname in os.listdir(model_dir):
#         if not fname.lower().endswith(('.pth','.pt')):
#             continue
#         weight_path = os.path.join(model_dir, fname)
#         model_name = os.path.splitext(fname)[0]
#         model_out = os.path.join(out_dir, model_name)
#         os.makedirs(model_out, exist_ok=True)

#         print(f"\n=========================")
#         print(f"[INFO] GradCAM++ for model: {fname}")
#         print(f"[INFO] => {model_out}")

#         # build model
#         num_classes = len(class_to_idx)
#         model = create_efficientnet_b0(num_classes, device=device)
#         load_weights(model, weight_path, device=device, strict=False)
#         model.eval()

#         # final conv
#         last_conv = model.conv_head  # adjust if needed
#         hook = GradCamHook(last_conv)

#         # gather predictions
#         subfolders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root,d))]
#         all_dfs = []
#         for sf in subfolders:
#             sf_path = os.path.join(test_root, sf)
#             df = predict_folder(model, sf_path, sf, tfm, class_to_idx, device=device)
#             if len(df)>0:
#                 # optionally save CSV if you want
#                 out_csv = os.path.join(model_out, f"results_{sf}.csv")
#                 df.to_csv(out_csv, index=False)
#                 all_dfs.append(df)

#         if len(all_dfs) == 0:
#             print("[WARNING] No subfolders had data => skipping.")
#             hook.remove()
#             continue

#         combined = pd.concat(all_dfs, ignore_index=True)
#         # optional => combined.to_csv(os.path.join(model_out, "combined_results.csv"), index=False)

#         # filter correct predictions
#         corr_df = combined[combined['correct']==True]
#         print(f"[INFO] Found {len(corr_df)} correct predictions total => generating GradCAM++ overlays.")

#         # create gradcam folder
#         gradcam_dir = os.path.join(model_out, "gradcam_pp")
#         os.makedirs(gradcam_dir, exist_ok=True)

#         for i, row in tqdm(corr_df.iterrows(), total=len(corr_df),
#                            desc="Grad-CAM++ on correct images"):
#             fpath = row['path']
#             pred_idx = row['predicted_idx']
#             pred_cls_name = idx_to_class[pred_idx]

#             # load
#             try:
#                 pil_img = Image.open(fpath).convert('RGB')
#             except:
#                 continue
#             img_t = tfm(pil_img)

#             # compute gradcam++
#             gc_map = compute_gradcam_pp(model, hook, img_t, pred_idx, device)
#             # keep only top fraction => e.g. top 1%
#             gc_map = apply_top_pixels(gc_map, top_fraction=top_fraction)

#             # overlay
#             # note: the original image might be bigger than 224 => up to you
#             # let's do overlay on the 224x224 version
#             small_pil = pil_img.resize((224,224))
#             overlay_bgr = overlay_heatmap_on_image(small_pil, gc_map, alpha=0.4)

#             # save => gradcam_pp/{pred_cls_name}/filename
#             cls_dir = os.path.join(gradcam_dir, pred_cls_name)
#             os.makedirs(cls_dir, exist_ok=True)
#             base_fname = os.path.basename(fpath)
#             out_img_path = os.path.join(cls_dir, base_fname)
#             cv2.imwrite(out_img_path, overlay_bgr)

#         hook.remove()
#         print(f"[INFO] Done => {model_out}/gradcam_pp")

#     print("\n[INFO] All done with Grad-CAM++ for all models.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


#     # ----- Weight Path -----
#     # 1) Where your model weights are stored
#     # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\...."

#     # --- Part A ---
#     # Affect Net
    # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Affect_Net_base_ok\model"

#     # Best Finetuned classification layer only dropout 0
    model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\train_exp_3_fine_frz_0322_2230_drop0_OK\best_model_only"

#     # Best Synthetic
#     # model_dir = r"C:\Users\ilias\Python\Thesis-Project\results\training_experiment_2_synthetic_20250319_023203\models_best"
    
     # ----- Test Folders -----
    # The RAFDB real data test combined test folder
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test_merged"

    # The SYNTH real data test combined test folder.
    # test_root = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\test_splits\100M_100W"
    


#     # This fraction is user-adjustable => top 33% of pixels
    top_fraction = 0.33


    model_parent = os.path.dirname(model_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_dir = os.path.join(model_parent, f"gradcam_pp_base_synth_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Classes
    class_to_idx = {
        'Angry':0,'Contempt':1,'Disgust':2,'Fear':3,
        'Happy':4,'Neutral':5,'Sad':6,'Surprise':7
    }
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Transform
    tfm = get_transform(224)

    # For each model in model_dir
    for fname in os.listdir(model_dir):
        if not fname.lower().endswith(('.pth','.pt')):
            continue

        weight_path = os.path.join(model_dir, fname)
        model_name = os.path.splitext(fname)[0]
        model_out = os.path.join(out_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        print(f"\n=========================")
        print(f"[INFO] GradCAM++ for model: {fname}")
        print(f"[INFO] => {model_out}")

        # 1) Create + load model
        model = create_efficientnet_b0(num_classes, device=device)
        load_weights(model, weight_path, device=device, strict=False)
        model.eval()

        # 2) Attach Grad-CAM++ hook to final conv
        last_conv = model.conv_head  # adjust if needed
        hook = GradCamHook(last_conv)

        # 3) Predict on test subfolders
        subfolders = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root,d))]
        all_dfs = []
        for sf in subfolders:
            sf_path = os.path.join(test_root, sf)
            df = predict_folder(model, sf_path, sf, tfm, class_to_idx, device=device)
            if len(df)>0:
                out_csv = os.path.join(model_out, f"results_{sf}.csv")
                df.to_csv(out_csv, index=False)
                all_dfs.append(df)

        if len(all_dfs) == 0:
            print("[WARNING] No data in subfolders => skipping.")
            hook.remove()
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        # optional => combined.to_csv(os.path.join(model_out, "combined_results.csv"), index=False)

        # filter correct
        corr_df = combined[combined['correct'] == True]
        print(f"[INFO] Found {len(corr_df)} correct predictions => generating GradCAM++ overlays.")

        # 4) Grad-CAM++ per correct image
        gradcam_dir = os.path.join(model_out, "gradcam_pp")
        os.makedirs(gradcam_dir, exist_ok=True)

        # We'll store the 224x224 Grad-CAM++ arrays for each class
        # so we can build a combined map afterward
        class_heatmaps = {c: [] for c in range(num_classes)}

        for i, row in tqdm(corr_df.iterrows(), total=len(corr_df), desc="Grad-CAM++ on correct images"):
            fpath = row['path']
            pred_idx = row['predicted_idx']
            pred_cls_name = idx_to_class[pred_idx]

            try:
                pil_img = Image.open(fpath).convert('RGB')
            except:
                continue
            img_t = tfm(pil_img)

            # compute gradcam++
            gc_map = compute_gradcam_pp(model, hook, img_t, pred_idx, device)
            # keep only top fraction => e.g. top 10%
            gc_map = apply_top_pixels(gc_map, top_fraction=top_fraction)

            # store for consensus
            class_heatmaps[pred_idx].append(gc_map)

            # overlay
            small_pil = pil_img.resize((224,224))
            overlay_bgr = overlay_heatmap_on_image(small_pil, gc_map, alpha=0.4)

            cls_dir = os.path.join(gradcam_dir, pred_cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            base_fname = os.path.basename(fpath)
            out_img_path = os.path.join(cls_dir, base_fname)
            cv2.imwrite(out_img_path, overlay_bgr)

        hook.remove()
        print(f"[INFO] Done => {model_out}/gradcam_pp")

        # 5) Build a combined “consensus” map for each class
        consensus_dir = os.path.join(model_out, "consensus_maps")
        os.makedirs(consensus_dir, exist_ok=True)

        # e.g. do a pixel-based fraction approach:
        # "pixel is active if above 0.5 in at least 30% of images"
        for c_idx in range(num_classes):
            heat_list = class_heatmaps[c_idx]
            if len(heat_list) == 0:
                continue
            c_name = idx_to_class[c_idx]
            cons_map = build_class_consensus_heatmap(heat_list,
                                                     pixel_thresh=0.5,
                                                     fraction=0.3)
            out_path = os.path.join(consensus_dir, f"{c_name}_consensus.png")
            colorize_and_save_consensus_map(cons_map, out_path)

    print("\n[INFO] All done with Grad-CAM++ for all models.")


if __name__=="__main__":
    main()
