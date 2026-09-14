# DP-RGMI: Differential privacy representation geometry for medical image analysis


This is the official repository of the paper **Differential privacy representation geometry for medical image analysis**.

Preprint version: [**URL**](https://arxiv.org/abs/2603.01098).



## Environment setup

Training and evaluation were performed strictly in FP32. The implementation was based on Python 3.10 using PyTorch 2.5 and torchvision 0.20. Core scientific computing libraries included NumPy 2.2, SciPy 1.15, scikit-learn 1.7, pandas 2.3, Opacus 1.5, and OpenCV 4.12. Hugging Face tooling comprised transformers 4.56, huggingface-hub 0.34, accelerate 1.10, tokenizers 0.21, and safetensors 0.6.

### Prerequisites

The codebase was originally developed with earlier library versions; however, the configuration below provides a fully compatible and CUDA-enabled environment validated on modern NVIDIA GPUs.  
PyTorch is installed via official wheels with CUDA support to avoid dependency conflicts, and all remaining packages are installed through `pip`.  
No system-wide CUDA toolkit installation is required, as the PyTorch wheels bundle the necessary CUDA runtime.


```
$ conda create -n NAME python=3.11 -y
$ conda activate NAME
$ python -m pip install --upgrade pip
```

```
$ python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

```
$ python -m pip install \
  accelerate \
  transformers \
  tokenizers \
  safetensors \
  huggingface-hub \
  matplotlib \
  pandas \
  timm \
  tensorboardX \
  tqdm \
  jupyter \
  scikit-learn \
  opencv-python \
  opacus
```


## Model initializations used


All pretrained initialization weights were obtained from official public repositories hosted on Hugging Face. 


**ImageNet (supervised):**
- ConvNeXt-Small : [https://huggingface.co/facebook/convnext-small-224](https://huggingface.co/facebook/convnext-small-224)  

**DINOv3 (self-supervised):**
- ConvNeXt-Small: [https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)



## Code structure

- `main_representationDP.py` — single entry point for training/evaluation.  
- `configs/config.yaml` — edit data paths, preprocessing, model/backbone, initialization.  
- `data/` — dataset I/O, preprocessing, augmentation.  
- `Train_Valid_representationDP.py` — training / validation loops.  
- `Prediction_representationDP.py` — inference & metrics.


## Quickstart

1) Prepare datasets following the paths and splits in `configs/config.yaml`.  
2) Choose an `experiment` name; the script will create a folder with checkpoints, metrics, TensorBoard logs, and a copy of the effective config.  
3) Launch training/evaluation from the project root, e.g.

```
python main_representationDP.py --config ./configs/config.yaml --experiment name
```

## In case you use this repository, please cite the original paper:

S. Tayebi Arasteh, M. Mohammadi, S. Nebelung, D. Truhn. *Differential privacy representation geometry for medical image analysis*. MICCAI 2026. 

### BibTex

    @article {dprgmi,
      author = {Tayebi Arasteh, Soroosh and Mohammadi, Marziyeh and Nebelung, Sven and Truhn, Daniel},
      title = {Differential privacy representation geometry for medical image analysis},
      year = {2026},
      journal = {MICCAI},
    }
