# Under Construction!! the code will be available soon!

<h2 align="center"> <a href="https://github.com/nazmul-karim170/FreeEditor-Text-to-3D-Scene-Editing">Free-Editor: Zero-shot Text-driven 3D Scene Editing</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![webpage](https://img.shields.io/badge/Webpage-blue)](https://free-editor.github.io/)
[![arXiv](https://img.shields.io/badge/Arxiv-2312.09313-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.13663)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nazmul-karim170/FreeEditor-Text-to-3D-Scene-Editing/blob/main/LICENSE) 


</h5>

## [Project page](https://free-editor.github.io/) | [Paper](https://arxiv.org/abs/2312.13663) 


<img src="assets/Top_teaser.png"/>

## 😮 Highlights

Free-Editor allows you to edit your 3D scenes by **editing only a single view** of that scene. The editing is **training-free** and can be done in a matter of **3 minutes!** instead of **70 minutes!** in SOTA. 



### 💡 Training-free, View Consistent, High-quality, and Fast-speed
- Stable Diffusion (SD) for image generation   -->   high-quality
- Single View editing  --> higher chance of view-consistent editing as it is hard to obtain consistent editing effects in multiple views with SD
- The editing process is training-free as we use a generalized NeRF model -->   fast high-quality 3D content reconstruction.



## 🚩 **Updates**

Welcome to **watch** 👀 this repository for the latest updates.

✅ **[2023.12.21]** : We have released our paper, Free-Editor on [arXiv](https://arxiv.org/abs/2312.13663).

✅ **[2023.12.18]** : Release [project page](https://free-editor.github.io/).
- [ ] Code release.

## 🛠️ Methodology

<img src="assets/Main_teaser.png"/>
Overview of our proposed method. We train a generalized NeRF (G(.)) that takes a single edited starting view and M source views to render a novel target view. Here, ”Edited Target View” is not the input to the model rather will be rendered and works as the ground truth for the prediction of G(.). In G(.) we employ a special Edit Transformer that utilizes: cross-attention to produce style-informed source feature maps that will be aggregated through an Epipolar Transformer. At inference, we can synthesize novel edited views in a zero-shot manner. To edit a scene, we take only a single image as the starting view and edit it using a Text-to-Image (T2I) diffusion model. Based on this starting view, we can render novel edited target views.

## 🚀 3D-Editing Results

### Qualitative comparison

<img src="assets/Comparison.png"/>

### Quantitative comparison

<img src="assets/quant.png"/>
Quantitative evaluation of scene edits in terms of Edit PSNR, CLIP Text-Image Directional Similarity (CTDS) and CLIP directional consistency (CDS).

## 👍 **Acknowledgement**
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
* [Instruct-NeRF2NeRF](https://github.com/ayaanzhaque/instruct-nerf2nerf)
* [Diffusers](https://github.com/huggingface/diffusers)

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and a citation :pencil:.

```BibTeX
@misc{karim2023freeeditor,
      title={Free-Editor: Zero-shot Text-driven 3D Scene Editing}, 
      author={Nazmul Karim and Umar Khalid and Hasan Iqbal and Jing Hua and Chen Chen},
      year={2023},
      eprint={2312.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<!---->
