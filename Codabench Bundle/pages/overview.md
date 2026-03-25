# Overview

<img src="https://raw.githubusercontent.com/moujar/Grain-Challenge-M1-AI/main/assets/paris-saclay.png" alt="Université Paris-Saclay" width="300" /> &nbsp;&nbsp; <img src="https://raw.githubusercontent.com/moujar/Grain-Challenge-M1-AI/main/assets/LISN-LOGO.png" alt="LISN" width="350" style="vertical-align: middle;" /> &nbsp;&nbsp; <img src="https://raw.githubusercontent.com/moujar/Grain-Challenge-M1-AI/main/assets/inrae.png" alt="INRAE" width="200" />

*Organized by **Université Paris-Saclay**, **LISN**, & **INRAE** — Grain Classification*

---

##  Challenge Description

Welcome to the **Grain Variety Classification Challenge**!

In real agricultural settings, different grain varieties are often **sown together** in the same field. After harvest, the final proportions of each variety may change due to differences in growth, disease resistance, or adaptation to the local environment. To study these effects and ensure quality control, it is essential to **accurately identify which grain belongs to which variety** after harvest.

To address this, **hyperspectral imaging** is used. Hyperspectral cameras capture rich spectral information far beyond what the human eye can see, making it possible to distinguish between visually similar grain varieties. However, this data is complex and high-dimensional — requiring **machine learning models** to analyze it effectively.

**Your goal:** Build a model that classifies individual grain images into the correct variety.

<img src="https://raw.githubusercontent.com/moujar/Grain-Challenge-M1-AI/main/assets/grain-gallery.png" alt="Sample grain images across 8 varieties" width="500" />

*Sample grain images across the 8 varieties (3 per class). Each 252×252 image is captured at spectral bands [22, 53, 89].*

---

##  What Is the Task?

- **Input:** A single 252 × 252 × 3 spectral image of one grain
- **Output:** Predicted variety label (1–8)
- **Problem type:** Multi-class image classification
- **Classes:** 8 grain varieties
- **Training images:** ~10,000+ (filtered by year)
- **Evaluation metric:** Classification **Accuracy**
- **Time constraint:** ≤ 20 minutes on Codabench GPU

This is structured as a **yearly benchmark**: models are trained and evaluated on **2019** and **2020** data independently to assess generalization across crop years.

---

## Who Can Participate?

This challenge is designed for:

- **Students** in machine learning, computer vision, or data science
- **Researchers** interested in agricultural AI or hyperspectral imaging
- **Anyone** who wants to practice image classification on real-world scientific data

**No prior experience with hyperspectral data is required.** The starter kit provides a complete working baseline.

---

##  How to Enter the Challenge

1. **Create an account** on [Codabench](https://www.codabench.org/) and register for this competition.
2. **Download the Starting Kit** from the **Files** tab — it includes the dataset, baseline code, and documentation.
3. **Explore the data** using the provided Jupyter notebook.
4. **Train your model** — you can start from the ConvNeXt baseline and then build your own.
5. **Submit** your `model.py` as a ZIP through the **My Submissions** tab.

>  **Tip:** Open the starter notebook directly in Google Colab for a zero-setup experience:
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moujar/Grain-Challenge-M1-AI/blob/main/starter_kit/README.ipynb)

---

##  What Will Be Evaluated?

Submissions are evaluated on **classification accuracy** — the percentage of test images correctly classified.

> `Balanced Accuracy = (Recall₁ + Recall₂ + ... + Recallₙ) / n`

Full details are on the **Evaluation** page. The leaderboard ranks participants by accuracy (higher is better).

---

## Baselines Provided

A baseline is included in the starting kit:

- **ConvNeXt** — End-to-end convolutional neural network trained with data augmentation. This provides a strong modern baseline for image classification. The implementation is available in the Starter Notebook.

Participants are encouraged to **beat these baselines** using the same or other architectures (EfficientNet, ViT, …), better augmentation, ensembling, or any other technique.

---

## Credits & Acknowledgments

- **Academic institution:** [Université Paris-Saclay](https://www.universite-paris-saclay.fr/)
- **Data provider:** [INRAE](https://www.inrae.fr/) (French National Research Institute for Agriculture, Food & Environment)
- **Platform:** [Codabench](https://www.codabench.org/)

**Challenge organizers:**
Oudoum Ali Houmed · Abderrahmane Moujar · Daryl Okou · Olutola Oloruntobi Paul · Ran Lu · Cristian-Ioan Bratu

We thank **INRAE** for making this dataset available for educational and research purposes, and **Université Paris-Saclay** for supporting this initiative.

---

## Contact

For questions about the challenge, dataset, or rules, please contact the organizers via the **Codabench platform** or open an issue on [GitHub](https://github.com/moujar/Grain-Challenge-M1-AI).