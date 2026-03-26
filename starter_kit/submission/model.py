# =============================================================================
# IMPORTANT — WEIGHTS FILE NOT INCLUDED (GitHub file size restriction)
# =============================================================================
# This model requires the pretrained weights file:
#
#   ConvNeXt-Tiny_Y1toY2_head_ft_50ep.pth
#
# This file could NOT be uploaded to GitHub because it exceeds the 100 MB
# file size limit enforced by GitHub.
#
# To use this model you must download the weights file separately and place
# it alongside this model.py file before running or submitting.
#
# When submitting to CodaBench, include the .pth file in your ZIP:
#
#   zip submission.zip model.py ConvNeXt-Tiny_Y1toY2_head_ft_50ep.pth
#
# Download the complete starter kit from the CodaBench competition page —
# it includes the weights file and everything else you need to get started.
# =============================================================================

import subprocess
import sys

def _install(package, index_url=None):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", package]
    if index_url:
        cmd += ["--index-url", index_url]
    subprocess.check_call(cmd)

_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

for _pkg, _idx in [
    ("numpy",       None),
    ("torch",       _CPU_INDEX),
    ("torchvision", _CPU_INDEX),
]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[*] Installing {_pkg} ...")
        _install(_pkg, _idx)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------
# Dataset
# ----------------------------------------
_CH_SCALE    = [1567.0, 8316.0, 18126.0]
_IMGNET_MEAN = [0.485,  0.456,  0.406]
_IMGNET_STD  = [0.229,  0.224,  0.225]


class GrainDataset(Dataset):
    def __init__(self, X, crop_size=176):
        self.X  = X
        self.cs = crop_size
        self.scale = torch.tensor(_CH_SCALE).view(3, 1, 1)
        self.mean  = torch.tensor(_IMGNET_MEAN).view(3, 1, 1)
        self.std   = torch.tensor(_IMGNET_STD).view(3, 1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        x = (x / self.scale).clamp_(0.0, 1.0)
        _, H, W = x.shape
        s = self.cs
        if H > s and W > s:
            top  = (H - s) // 2
            left = (W - s) // 2
            x    = x[:, top:top + s, left:left + s]
        x = (x - self.mean) / self.std
        return x


# ----------------------------------------
# ConvNeXt-Tiny Architecture (no torchvision)
# ----------------------------------------
class _DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                           dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


class _LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias   = nn.Parameter(torch.zeros(c))
        self.eps    = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class _CNBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, ls_init=1e-6):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.ln  = _LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.ls  = nn.Parameter(ls_init * torch.ones(1, dim, 1, 1))
        self.dp  = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        h = self.dw(x)
        h = self.ln(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        return x + self.dp(self.ls * h)


class _CNDown(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.ln = _LayerNorm2d(ci)
        self.cv = nn.Conv2d(ci, co, 2, stride=2)

    def forward(self, x):
        return self.cv(self.ln(x))


_DIMS   = [96, 192, 384, 768]
_DEPTHS = [ 3,   3,   9,   3]


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, _DIMS[0], 4, stride=4), _LayerNorm2d(_DIMS[0]))
        total = sum(_DEPTHS)
        dp    = [0.0] * total          # drop_path=0 at inference
        bi    = 0
        self.stages = nn.ModuleList()
        self.downs  = nn.ModuleList()
        for si in range(4):
            stage = nn.Sequential(
                *[_CNBlock(_DIMS[si], dp[bi + j]) for j in range(_DEPTHS[si])])
            self.stages.append(stage)
            bi += _DEPTHS[si]
            if si < 3:
                self.downs.append(_CNDown(_DIMS[si], _DIMS[si + 1]))
        self.norm    = nn.LayerNorm(_DIMS[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=0.0)   # disabled at inference
        self.head    = nn.Linear(_DIMS[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, st in enumerate(self.stages):
            x = st(x)
            if i < 3:
                x = self.downs[i](x)
        x = self.norm(x.mean([2, 3]))
        return self.head(self.dropout(x))


# ----------------------------------------
# Weight loading
# ----------------------------------------
def _load_weights():
    WEIGHT_FILE = "ConvNeXt-Tiny_Y1toY2_head_ft_50ep.pth"
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        ".",
        os.getcwd(),
        "/app/submission",
        "/app/output",
        "/app/program",
        "/tmp",
    ]
    for d in search_dirs:
        p = os.path.join(d, WEIGHT_FILE)
        if os.path.isfile(p):
            print("[*] Loading weights:", p,
                  "(%d MB)" % (os.path.getsize(p) // 1_000_000))
            return torch.load(p, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        "ConvNeXt-Tiny_Y1toY2_head_ft_50ep.pth not found. "
        "Place the .pth file alongside model.py in the submission ZIP."
    )


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        Load ConvNeXt-Tiny with pre-trained Y1→Y2 weights.
        The network is ready for inference immediately after __init__.
        """
        print("[*] ConvNeXt-Tiny | inference only | Y1→Y2 weights")

        self.nc        = 8
        self.label_off = 1    # labels are 1-8, model indices 0-7
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp   = self.device.type == "cuda"
        print("[*] Device:", self.device)

        state = _load_weights()

        net = ConvNeXtTiny(self.nc).to(self.device)
        net.load_state_dict(state, strict=False)
        net.eval()

        self.net = net
        print("[*] Model ready")

    def fit(self, train_data):
        """
        No-op — weights are pre-loaded from the .pth file.
        The platform calls this method but no training is performed.
        """
        print("[*] fit() called — using pre-trained weights, no training performed")

    def predict(self, test_data):
        """
        Predict grain variety labels using multi-crop × D4 TTA.

        4 crop sizes × 8 D4 augmentations = 32 views per sample.

        Parameters
        ----------
        test_data : dict
            'X': numpy array (n_samples, H, W, C)

        Returns
        -------
        y : 1D numpy array of integer labels in [1, 8]
        """
        print("[*] - Predicting test set using ConvNeXt-Tiny")

        X     = test_data['X']
        n     = X.shape[0]
        crops = [160, 172, 184, 196]
        probs = torch.zeros(n, self.nc)

        for cs in crops:
            ds  = GrainDataset(X, crop_size=cs)
            ld  = DataLoader(ds, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=False)
            off = 0
            with torch.no_grad():
                for xb in ld:
                    xb  = xb.to(self.device, non_blocking=True)
                    bs_ = xb.size(0)
                    acc = torch.zeros(bs_, self.nc, device=self.device)
                    for hf in (False, True):              # flip / no flip
                        for rot in range(4):              # 0°, 90°, 180°, 270°
                            xa = xb.flip(3) if hf else xb
                            if rot:
                                xa = torch.rot90(xa, rot, [2, 3])
                            try:
                                with torch.amp.autocast("cuda", enabled=self.use_amp):
                                    acc += F.softmax(self.net(xa), dim=1)
                            except Exception:
                                acc += F.softmax(self.net(xa), dim=1)
                    probs[off:off + bs_] += acc.cpu()
                    off += bs_

        predictions = probs.argmax(1).numpy().astype(np.int64) + self.label_off
        print(f"[*] - Predicted {len(predictions)} samples")
        return predictions
