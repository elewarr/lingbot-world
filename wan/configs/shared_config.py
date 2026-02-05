import sys
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#

# Cross-platform dtype selection
# MPS doesn't support bfloat16 and has issues with mixed precision float16
# Use float32 for MPS to avoid Metal kernel issues, bfloat16 for CUDA
IS_MACOS = sys.platform == 'darwin'
HAS_MPS = IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()

# MPS has strict requirements - accumulator dtype must match destination
# float16 fails with "Destination NDArray and Accumulator NDArray cannot have different datatype"
# Use float32 for MPS, bfloat16 for CUDA
DEFAULT_DTYPE = torch.float32 if HAS_MPS else torch.bfloat16

wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = DEFAULT_DTYPE
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = DEFAULT_DTYPE

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.frame_num = 81
