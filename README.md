<div align="center">
  
## How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?

---

Official implementation of **[How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?](https://arxiv.org/abs/2405.01434)**.
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
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/datasets.png" width=880>
</div>

### Single-concept customization
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/single-concept.png" width=880>
</div>

### Multi-concept customization
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/multi-concept.png" width=880>
</div>

### Custom style transfer
<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/style_transfer.png" width=720>
</div>

### Custom image editing

<div align="center">
<img src="https://github.com/JiahuaDong/CIFC/blob/main/Figs/editing.png" width=490>
</div>

## 🚩 **TODO/Updates**
- [x] Quantitative Results of CIDM.
- [x] CIL Datasets used in our paper.
- [ ] Source code of CIDM.
---


## Contact
If you have any questions, you are very welcome to email dongjiahua1995@gmail.com.

   



# BibTeX
If you find CIDM useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{dong2024how,
  title={How to Continually Adapt Text-to-Image Diffusion Models for Flexible Customization?},
  author={Jiahua Dong, Wenqi Liang, Hongliu Li, Duzhen Zhang, Meng Cao, Henghui Ding, Salman Khan, Fahad Khan},
  journal={NeurIPS 2024},
  year={2024}
}
