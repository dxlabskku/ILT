
# --------------------------------------------------
# 0. Imports & Global constants
# --------------------------------------------------
import os, sys, glob, cv2
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import ticker
from sklearn.metrics import r2_score

from lithobench.dataset import loadersLitho, loadersILT, loadersAllLitho   
import lithobench.evaluate as evaluate                                     

# Default low‑resolution size for visualisation
global_size: Tuple[int, int] = (128, 128)


# --------------------------------------------------
# 1. Data‑set helpers
# --------------------------------------------------
def filesMaskOpt(folder: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (glp, target, pixelILT) file lists with identical basenames."""
    folderGLP    = os.path.join(folder, "glp")
    folderTarget = os.path.join(folder, "target")
    folderPixel  = os.path.join(folder, "pixelILT")

    filesGLP    = glob.glob(folderGLP + "/*.glp")
    filesTarget = glob.glob(folderTarget + "/*.png")
    filesPixel  = glob.glob(folderPixel + "/*.png")

    base = lambda p: os.path.basename(p)[:-4]      # strip extension
    common = (
        set(map(base, filesGLP))
        & set(map(base, filesTarget))
        & set(map(base, filesPixel))
    )

    filesGLP    = sorted([f for f in filesGLP    if base(f) in common], key=base)
    filesTarget = sorted([f for f in filesTarget if base(f) in common], key=base)
    filesPixel  = sorted([f for f in filesPixel  if base(f) in common], key=base)

    return filesGLP, filesTarget, filesPixel


def safe_linreg(x, y):
    """Robust 1‑D linear regression → (R², (slope, intercept))."""
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12:
        return 0.0, (0.0, 0.0)
    slope, intercept = np.polyfit(x.astype(np.float64), y.astype(np.float64), deg=1)
    r2 = r2_score(y, slope * x + intercept)
    return r2, (slope, intercept)


def binarize(tensor, threshold=0.49):
    """Numpy binary mask (float32)."""
    return (tensor > threshold).astype(np.float32)


def binarize_and_resize(tensor, threshold=0.5, out_size=global_size):
    """Tensor → binary → bilinear resize → numpy."""
    tensor = (tensor > threshold).float().squeeze().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=out_size, mode="bilinear", align_corners=False)
    return tensor.detach().cpu().squeeze().numpy()


def resize(tensor, out_size=global_size):
    """Tensor → bilinear resize → numpy."""
    tensor = tensor.squeeze().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=out_size, mode="bilinear", align_corners=False)
    return tensor.detach().cpu().squeeze().numpy()


# --------------------------------------------------
# 2. Grad‑CAM / Score‑CAM / Adaptive‑CAM
# --------------------------------------------------
activation = None       # forward hook buffer
gradient   = None       # backward hook buffer


def forward_hook(_, __, out):
    global activation
    activation = out                                 # shape [B,C,H,W]


def backward_hook(_, __, grad_out):
    global gradient
    gradient = grad_out[0]                           # shape [B,C,H,W]


# Layer slicing utility (unchanged from original)
def get_remainder_model(model, target_layer):
    """Return nn.Module covering layers AFTER `target_layer`."""
    class RemainderModel(nn.Module):
        def __init__(self, model, target):
            super().__init__()
            seq = []
            if target == "conv_head":
                seq = [
                    model.conv0, model.conv1, model.conv2, model.conv3, model.conv4,
                    model.res0, model.res1, model.res2, model.res3,
                    model.res4, model.res5, model.res6, model.res7, model.res8,
                    model.deconv4, model.deconv3, model.deconv2,
                    model.deconv1, model.deconv0, model.conv_tail,
                ]
            elif target == "conv4":
                seq = [
                    model.res0, model.res1, model.res2, model.res3,
                    model.res4, model.res5, model.res6, model.res7, model.res8,
                    model.deconv4, model.deconv3, model.deconv2,
                    model.deconv1, model.deconv0, model.conv_tail,
                ]
            elif target == "res0":
                seq = [
                    model.res1, model.res2, model.res3, model.res4,
                    model.res5, model.res6, model.res7, model.res8,
                    model.deconv4, model.deconv3, model.deconv2,
                    model.deconv1, model.deconv0, model.conv_tail,
                ]
            elif target == "res8":
                seq = [model.deconv4, model.deconv3, model.deconv2,
                       model.deconv1, model.deconv0, model.conv_tail]
            elif target == "deconv4":
                seq = [model.deconv3, model.deconv2, model.deconv1,
                       model.deconv0, model.conv_tail]
            elif target == "deconv0":
                seq = [model.conv_tail]
            elif target == "conv_tail":
                seq = [nn.Identity()]
            else:
                raise NotImplementedError(f"{target_layer} not implemented.")
            self.layers = nn.Sequential(*seq)

        def forward(self, x):
            return self.layers(x)
    return RemainderModel(model, target_layer)


def normalize_cam(cam):
    """Min‑max [0,1] normalisation."""
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)


def get_cam_maps_conv_tail(model, input_tensor, label_tensor, target_layer_):
    """
    Compute Grad‑CAM / Score‑CAM / Adaptive‑CAM for `target_layer_`.
    Returns the same tuple structure as before.
    """
    global activation, gradient
    activation = gradient = None

    # Attach hooks
    target_layer = getattr(model.netG.module, target_layer_, None)
    if target_layer is None:
        raise RuntimeError(f"{target_layer_} not found in model.netG.module")
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    # Unified 1024×1024 forward pass
    input_tensor = F.interpolate(input_tensor, size=(1024, 1024),
                                 mode="bilinear", align_corners=False)
    label_tensor = F.interpolate(label_tensor, size=(1024, 1024),
                                 mode="bilinear", align_corners=False)
    pred = model.run_for_gradcam(input_tensor)                        # [1,1,1024,1024]

    # Lithography simulation
    printedNom, _, _, printedNom_NS, _, _   = model.simLitho(pred.squeeze(1))
    printedNom2, _, _, printedNom_NS2, _, _ = model.simLitho(input_tensor.squeeze(1))
    printedNom3, _, _, printedNom_NS3, _, _ = model.simLitho(label_tensor.squeeze(1))

    # Loss & backward
    loss = F.l1_loss(printedNom.unsqueeze(1), input_tensor) \
         + F.mse_loss(pred, label_tensor)
    model.netG.zero_grad()
    loss.backward()

    # Remove hooks
    fh.remove()
    bh.remove()

    if activation is None or gradient is None:
        raise RuntimeError("Hook buffers empty")

    A, dA = activation, gradient                        # [1,C,H,W]
    B, C, _, _ = A.shape
    if B != 1:
        raise RuntimeError("Only batch size 1 supported")

    # ── 1. Grad‑CAM
    w = dA.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=global_size, mode="bilinear", align_corners=False)
    gradcam_map = cam[0, 0].detach().cpu().numpy()

    # ── 2. Score‑CAM
    score_list = []b
    with torch.no_grad():
        A_up = F.interpolate(A, size=(1024, 1024), mode="bilinear", align_corners=False)
        for ch in range(C):
            act = A_up[:, ch:ch+1]
            mask = (act - act.min()) / (act.max() - act.min() + 1e-8)
            out = model.run_for_gradcam(input_tensor * mask)
            score_list.append((out * pred).mean().item())
    w_sc = torch.tensor(score_list, device=A.device).view(1, C, 1, 1)
    sc_map = (A_up * w_sc).sum(dim=1, keepdim=True)
    sc_map = F.interpolate(sc_map, size=global_size, mode="bilinear", align_corners=False)
    scorecam_map = F.relu(sc_map)[0, 0].detach().cpu().numpy()

    # ── 3. Adaptive‑CAM
    remainder_model = get_remainder_model(model.netG.module, target_layer_)
    with torch.no_grad():
        A_up = F.interpolate(A, size=(1024, 1024), mode="bilinear", align_corners=False)
        base_score = remainder_model(torch.zeros_like(A))
        dce = []
        for ch in range(C):
            act = A_up[:, ch:ch+1]
            mask = (act - act.min()) / (act.max() - act.min() + 1e-8)
            mask_orig = F.interpolate(mask, size=A.shape[-2:], mode="bilinear", align_corners=False)
            A_masked = torch.zeros_like(A)
            A_masked[:, ch:ch+1] = A[:, ch:ch+1] * mask_orig
            dce.append((base_score - remainder_model(A_masked)).abs().mean())
        dce_w = torch.tensor(dce, device=A.device).view(1, C, 1, 1)
    ad_cam = (A * dce_w).sum(dim=1, keepdim=True)
    ad_cam = F.relu(ad_cam)
    ad_cam = F.interpolate(ad_cam, size=global_size, mode="bilinear", align_corners=False)
    ad_cam = ad_cam[0, 0].detach().cpu().numpy()  
    ad_cam = (ad_cam - ad_cam.min()) / (ad_cam.max() + 1e-8)



    return (gradcam_map, scorecam_map,
            printedNom2, printedNom, pred,
            printedNom_NS, printedNom_NS2,
            ad_cam, printedNom3, printedNom_NS3)


# --------------------------------------------------
# 3. 10 ICCAD-13 sample test & visualisation
# --------------------------------------------------
def testSingleMaskOpt(folder, model):
    """Run CAM visualisation on all samples in `folder`."""
    filesGLP, filesTarget, filesPixel = filesMaskOpt(folder)
    if not filesGLP:
        print("[ERROR] No matching files in glp/target/pixelILT.")
        return

    target_layers = ["conv4", "res8", "deconv0"]
    for target_layer in target_layers:
        print(f"\n===== Target Layer: {target_layer} =====")
        for glp, target_png, pixel_png in zip(filesGLP, filesTarget, filesPixel):

            # Load target / label images (grayscale)
            img_cv   = cv2.imread(target_png, cv2.IMREAD_GRAYSCALE)
            label_cv = cv2.imread(pixel_png,  cv2.IMREAD_GRAYSCALE)
            if img_cv is None or label_cv is None:
                print("Failed to read images."); continue

            pil_img   = Image.fromarray(img_cv)
            label_img = Image.fromarray(label_cv)
            to_tensor = transforms.ToTensor()       # (C,H,W) float32 ∈ [0,1]

            img_tensor   = to_tensor(pil_img).unsqueeze(0).cuda()
            label_tensor = to_tensor(label_img).unsqueeze(0).cuda()

            (gradcam_map, _, scorecam_map,
             printedNom2, printedNom, pred,
             printedNom_NS, printedNom_NS2,
             ad_cam, printedNom3, printedNom_NS3) = \
                get_cam_maps_conv_tail(model, img_tensor, label_tensor, target_layer)

            # --- Post‑processing & masks
            target_np = resize(img_tensor)
            label_np  = binarize_and_resize(label_tensor)
            pred_np   = binarize_and_resize(pred)
            printedNom2  = binarize_and_resize(printedNom2)
            printedNom   = binarize_and_resize(printedNom)
            printedNom3  = binarize_and_resize(printedNom3)
            printedNom_NS  = resize(printedNom_NS)
            printedNom_NS2 = resize(printedNom_NS2)
            printedNom_NS3 = resize(printedNom_NS3)

            ground_truth_difference = (
                (printedNom_NS3 - printedNom_NS2)
                * binarize((printedNom3 + printedNom2) / 2)
            )
            mask = binarize(np.abs(printedNom3 - printedNom2))
            gradcam_map  = normalize_cam(gradcam_map  * mask)
            scorecam_map = normalize_cam(scorecam_map * mask)
            ad_cam       = normalize_cam(ad_cam       * mask)
            ground_truth_difference = normalize_cam(ground_truth_difference)
            gray_img = normalize_cam(
                np.abs((printedNom_NS3 * printedNom3 - printedNom_NS2 * printedNom2) * mask)
            )

            # --- Fig: first row (5 images)
            plt.figure(figsize=(20, 6))
            titles = ["Target", "GT‑RWmask sim.", "Grad‑CAM", "Score‑CAM", "Adaptive‑CAM"]
            images = [target_np, ground_truth_difference, gradcam_map, scorecam_map, ad_cam]
            for i, (img, ttl) in enumerate(zip(images, titles), start=1):
                plt.subplot(2, 5, i)
                if "CAM" in ttl:
                    plt.imshow(target_np, cmap="gray")
                    plt.imshow(img, cmap="jet", alpha=0.8)
                else:
                    plt.imshow(img, cmap="gray")
                plt.title(ttl); plt.axis("off")

            # --- Fig: second row (3 correlations + Pred / GT Mask)
            corr_pairs = [("Grad", gradcam_map), ("Score", scorecam_map), ("Adaptive", ad_cam)]
            for j, (name, cam) in enumerate(corr_pairs, start=6):
                plt.subplot(2, 5, j)
                cam_flat, gray_flat = cam.flatten(), gray_img.flatten()
                plt.scatter(cam_flat, gray_flat, s=25, alpha=0.7,
                            c="blue", edgecolors="black")
                r2, (slope, inter) = safe_linreg(cam_flat, gray_flat)
                if r2:
                    x_ = np.linspace(cam_flat.min(), cam_flat.max(), 50)
                    plt.plot(x_, slope * x_ + inter, "r--")
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                plt.text(0.05, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                         va="top", ha="left", fontsize=8,
                         bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
                plt.xlabel(name); plt.xlim(-0.5, 1.2); plt.ylim(-0.5, 1.2)
                plt.grid(True, linestyle="--", alpha=0.5); plt.gca().set_aspect("equal")

            # Pred / GT mask
            plt.subplot(2, 5, 9); plt.title("Prediction"); plt.imshow(pred_np, cmap="gray"); plt.axis("off")
            plt.subplot(2, 5, 10); plt.title("GT Mask");  plt.imshow(label_np, cmap="gray"); plt.axis("off")

            plt.tight_layout(); plt.show(); plt.close()


# --------------------------------------------------
# 4. Main entry
# --------------------------------------------------
if __name__ == "__main__":
    from lithobench.ilt.damoilt import DAMOILT
    import torch

    # 1) load a model
    model = DAMOILT(size=(1024, 1024))
    netG_path = "your path"
    netD_path = "your path"
    model.load([netG_path, netD_path])

    # 2) move to GPU for netG
    if torch.cuda.is_available():
        model.netG = model.netG.cuda()
    model.netG.eval()   # inference mode

    # 3) test folder
    folder_test = "your path"

    # 4) execute
    testSingleMaskOpt(folder_test, model)

