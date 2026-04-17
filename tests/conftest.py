import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

torch.set_float32_matmul_precision("highest")
torch.manual_seed(0)
