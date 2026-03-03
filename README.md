# Grain Challenge 2: Yearly Benchmark
<a href="https://colab.research.google.com/github/moujar/Grain-Challenge-M1-AI/blob/fix/src/README.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### ResNet-18 Baseline for Grain Classification
***

Welcome to the Starting Kit for the **Grain Variety Classification Challenge**! 

This notebook serves as a complete entry point for challenge participants. It provides an end-to-end pipeline covering exploratory data analysis, domain-specific dataset optimizations, the implementation of a baseline deep learning architecture (**ResNet-18**), and instructions on how to package your model for submission. Participants are encouraged to understand the baseline provided here and improve upon it.

---

## Challenge Overview & Problem Setting

The task is **multi-class classification of all 8 grain varieties from spectral RGB images**, specifically structured as a yearly benchmark evaluating models independently on the **2019** and **2020** datasets. Each sample is a 252×252 image captured at specific spectral bands [22, 53, 89], and the primary goal is to accurately map grains into **8 distinct variety classes**. Recognizing grain varieties quickly and accurately is crucial for agricultural quality control and logistics.

| Aspect | Detail |
|---|---|
| **Objective** | Classify grain variety from a single seed image |
| **Classes** | 8 varieties (labels 1–8) |
| **Training set** | 10,888 images (Filtered to Year 1 only) |
| **Test set** | 2,723 images (hidden during Codabench evaluation) |
| **Image size** | 252 × 252 × 3 (int16) |
| **Evaluation metric** | **Classification Accuracy** (primary metric for ranking)|
| **Constraint** | ≤ 20 minutes on Codabench GPU |

---

## Organizers & Credits

| Member |
|---|
| Oudoum Ali Houmed |
| Abderrahmane Moujar |
| Daryl Okou |
| Olutola Paul |
| Ran Lu |
| Cristian-Ioan Bratu |

---