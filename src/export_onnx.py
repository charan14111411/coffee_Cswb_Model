# ===== CWSB: convert contender_B.pt -> ONNX (run in Colab; GPU not needed) =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "onnx", "onnxruntime"], check=False)

import os
import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn

PT_PATH = "contender_B.pt"      # change if the file is elsewhere, e.g. /content/drive/MyDrive/cwsb/v1/contender_B.pt
ONNX_PATH = "cwsb_v1.onnx"

if not os.path.exists(PT_PATH):                 # not found -> ask for an upload (Colab only)
    from google.colab import files
    PT_PATH = next(iter(files.upload()))

# 1. rebuild the network and load the trained weights
ck = torch.load(PT_PATH, map_location="cpu", weights_only=False)
IMG = ck.get("img", 384)
net = timm.create_model(ck["backbone"], pretrained=False, num_classes=2)
net.load_state_dict(ck["state_dict"])
net.eval()


class WithScore(nn.Module):
    """Input: photo batch (N,3,384,384), already resized/cropped/normalised.
    Output: score (N,) = probability of 'infected', 0..1 - the number the thresholds use."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)[:, 1]


model = WithScore(net).eval()
dummy = torch.randn(1, 3, IMG, IMG)

# 2. export
kw = dict(input_names=["image"], output_names=["score"], opset_version=17,
          dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}}, do_constant_folding=True)
try:
    torch.onnx.export(model, dummy, ONNX_PATH, dynamo=False, **kw)
except TypeError:                               # older PyTorch without the dynamo argument
    torch.onnx.export(model, dummy, ONNX_PATH, **kw)
onnx.checker.check_model(onnx.load(ONNX_PATH))

# 3. parity check: the ONNX file must give the same scores as PyTorch
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
torch.manual_seed(42)
x = torch.randn(4, 3, IMG, IMG)
with torch.no_grad():
    want = model(x).numpy()
got = sess.run(["score"], {"image": x.numpy()})[0]
diff = float(np.abs(want - got).max())
print(f"backbone      : {ck['backbone']}  input {IMG}x{IMG}")
print(f"PyTorch scores: {np.round(want, 4)}")
print(f"ONNX scores   : {np.round(got, 4)}")
print(f"largest difference: {diff:.2e}")
assert diff < 1e-4, "ONNX does not match PyTorch - do not use this file"
print(f"OK -> {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1e6:.0f} MB)")

try:                                            # download it (Colab only)
    from google.colab import files
    files.download(ONNX_PATH)
except ImportError:
    pass
