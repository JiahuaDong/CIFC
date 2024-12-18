<div align="center">
  
## How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?

---

Official implementation of **[How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?](https://arxiv.org/abs/2410.17594)**.
</div>


### **Framework**

<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/framework.png" width=980>
</div>


### 🌠  **Key Features:**
CIDM can can resolve catastrophic forgetting and concept neglect to learn new customization tasks in a concept-incremental
manner. Our work mainly has two parts: 
1. We propose a new practical Concept-Incremental Flexible Customization (CIFC) problem, where the main challenges are catastrophic forgetting and concept neglect. To address the challenges in the CIFC problem, we develop a novel Concept-Incremental text-to-image Diffusion Model (CIDM),
which can learn new personalized concepts continuously for versatile concept customization.
2. We devise a concept consolidation loss and an elastic weight aggregation module to mitigate the catastrophic forgetting of old personalized concepts, by exploring task-specific/task-shared knowledge and aggregating all low-rank weights of old concepts based on their contributions in the CIFC.
3. We develop a context-controllable synthesis strategy to tackle the concept neglect. It can control the contexts of synthesized image according to user-provided conditions, by enhancing expressive ability of region features with layer-wise textual embeddings and incorporating region noise estimation.



## 🔥 **Examples**

### Concept-incremental learning tasks
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/datasets.png" width=800>
</div>

### Single-concept customization
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/single-concept.png" width=800>
</div>

### Multi-concept customization
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/multi-concept.png" width=800>
</div>

### Custom style transfer
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/style_transfer.png" width=700>
</div>

### Custom image editing

<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/editing.png" width=450>
</div>

## :wrench: **Dependencies and Installation**

```bash
# Create and activate conda environment
conda create -n cidm python=3.8 -y
conda activate cidm

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y  # UPDATE ME!

pip install -r requirements.txt
```
## :computer: **CIDM Training**

### Step 1: Pretrained Model and Data Preparation

Modify the configurations under the path './options/cidm'. If you wish to train CIDM on your own data, you need to edit 'data_dir', 'caption_dir', and 'replace_mapping' in the configuration file. Additionally，you should specify the model path of **Stable diffusion 1.5** in './scripts/train.sh', referring to [**diffusers**](https://github.com/huggingface/diffusers).

```yaml
datasets:
  train:
    name: LoraDataset
    data_dir: ./datasets/images/dog
    caption_dir: ./datasets/caption/dog
    use_caption: true
    use_mask: false
    instance_transform:
      - { type: HumanResizeCropFinalV3, size: 512, crop_p: 0.5 }
      - { type: ToTensor }
      - { type: Normalize, mean: [ 0.5 ], std: [ 0.5 ] }
      - { type: ShuffleCaption, keep_token_num: 1 }
      - { type: EnhanceText, enhance_type: object }
    replace_mapping:
      dog: <dog1> <dog2>
    batch_size_per_gpu: 1
    dataset_enlarge_ratio: 300

  val_vis:
    name: PromptDataset
    prompts: ./datasets/validation_prompts/test_dog.txt
    num_samples_per_prompt: 4
    latent_size: [ 4,64,64 ]
    replace_mapping:
      dog: <dog1> <dog2>
    batch_size_per_gpu: 4
```

### Step 2: Start Training

We train our model on two GPUs, each with at least 10GB of memory. You can edit your GPU settings as follows:

```bash
accelerate config

```

Next, modify './scripts/train.sh' and run the following:
```bash
sh scripts/train.sh

```

### Step 3: Inference and Evaluation

Specify the model path of **Stable diffusion 1.5** and replace_prompt in './scripts/inference.sh', and run the following:

```bash
sh scripts/inference.sh

```
Compute text-alignment and image-alignment scores referring to [**Custom Diffusion**](https://github.com/adobe-research/custom-diffusion):

```bash
sh scripts/evaluate.sh
```

## 🚩 **TODO/Updates**
- [x] Quantitative Results of CIDM.
- [x] CIL Datasets used in our paper.
- [x] Source code of CIDM with SD 1.5 version.
- [ ] Source code of CIDM with SD XL version.
- [ ] Source code of versatile customization.
---

## 📜 Acknowledgement

We establish CIDM based on the following work, and we thank them for their open-source contributions：

[**diffusers**](https://github.com/huggingface/diffusers)

[**Dreambooth**](https://arxiv.org/abs/2208.12242)

[**Custom Diffusion**](https://github.com/adobe-research/custom-diffusion)

[**Mix-of-Show**](https://github.com/TencentARC/Mix-of-Show)


## 📧 Contact
If you have any questions, you are very welcome to email dongjiahua1995@gmail.com.



## 🌏 BibTeX
If you find CIDM useful for your research and applications, please cite using this BibTeX:

```BibTeX
@inproceedings{NEURIPS2024Dong,
author = {Dong, Jiahua and Liang, Wenqi and Li, Hongliu and Zhang, Duzhen and Cao, Meng and Ding, Henghui and Khan, Salman and Khan, Fahad},
title = {How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?},
booktitle = {Advances in Neural Information Processing Systems},
year = {2024},
}
