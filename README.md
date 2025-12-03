# 🧠 Brain-Llama3.1-Encoding: A Preliminary Replication
## Encoding Human Brain fMRI Signals via LLaMA-3.1

> **Base Paper:** Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain. Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0 

This project is an empirical study based on **Computational Neurolinguistics**. Inspired by the work of Gao et al. (2025) published in *Nature Computational Science*, I constructed a simplified neural encoding pipeline based on **Meta-LLaMA-3.1-8B (base & instruct)** to attempt to verify some of the fundamental findings of **Gao et al. (2025)**.

The core objective of this project is to attempt to verify the distribution of the model's predictive ability across different layers within limited computational resources, and to verify whether **Instruction Tuning**  significantly alters the large language model's underlying semantic representations and their alignment with the human brain's language network.

---

## 1. Key Findings

By comparing the encoding performance of **LLaMA-3.1-Base** and **LLaMA-3.1-Instruct** on the **Le Petit Prince (LPP)** fMRI dataset, we reached the following preliminary conclusions:

###  1.1 The "Inverted-U" Trend (Middle Layer Effect)
We successfully replicated the classic finding in the field of neurolinguistics: the model's predictive ability regarding the brain exhibits an "Inverted-U" distribution.
<img width="1000" height="600" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/2dab9dd7-928d-4631-83fc-3756c6f9b425" />
* **Layer 0 (Embedding):** After removing the sentence length confound, its predictive capability is significantly lower than that of the middle layers.
* **Middle Layers (L16-L24):** **Predictive capability reaches its peak** (Max Pearson's $r \approx 0.60$, Top-5% $\approx 0.30$). This indicates that LLaMA's intermediate layer representations are closest to the regions of the human brain responsible for syntactic analysis and semantic integration.
* **Late Layers:** As the model focuses on Next-token Prediction, its alignment with the brain's general understanding mechanisms declines slightly.

<img width="4200" height="1800" alt="plot_B_heatmap" src="https://github.com/user-attachments/assets/718a3b92-a534-4533-9afb-ba2a231ee6a8" />
*(Figure: Layer-wise performance heatmap of the Base model across different Runs, showing high-response zones in middle layers)*

### 1.2 Impact of Fine-tuning
The experiment shows that the layer-wise alignment curves of the Base and Instruct versions are **highly overlapping**.
* This verifies the conclusion of Gao et al. (2025): While Instruction Tuning (RLHF/SFT) improves task performance, it does not significantly enhance the alignment between the model and human brain activity.
* As shown in the figure below, most layer points fall near the $y=x$ line, indicating that the underlying semantic representation mechanisms of both are fundamentally consistent.

<img width="2400" height="2400" alt="plot_C_scatter" src="https://github.com/user-attachments/assets/77da05c5-6b10-45be-bc13-8b8bab180979" />

<img width="3600" height="1800" alt="plot_A_difference (1)" src="https://github.com/user-attachments/assets/598ce0a4-9026-4bf9-b9c1-2aea3a5d0fcc" />

---

## 2. Methodology
### 2.1 Data & Environment
* **Dataset:** *Le Petit Prince* (LPP) fMRI Dataset (Sub-EN057, ~1.5h Audio) (Due to computational constraints, we only verified full fMRI data for one English subject).
* **Hardware:** AutoDL Cloud Server / **NVIDIA RTX 4090 (24GB VRAM)**.
* **Model Loading:** LLaMA-3.1-8B (BF16 / 4-bit Quantization via `bitsandbytes`).

### 2.2 Core Pipeline
1.  **Feature Extraction:** Extracted Hidden States for Layers 0-32, employing **Sentence-level Mean Pooling** to adapt to the low signal-to-noise ratio of the auditory task.
2.  **Hemodynamic Modeling (HRF Alignment):** Used `nilearn` for HRF convolution and performed a search for the Best Delay within the 4s-10s range.
3.  **Dimensionality Reduction & Regression (Encoding):** Used PCA (n=15) for dimensionality reduction, combined with Ridge Regression (5-Fold CV) for voxel-wise prediction.

---

## 3. Challenges & Methodology Evolution

The replication of this project was not smooth. To approximate the logic of the original paper, I went through three critical methodological iterations.

### 1: Engineering Adaptation & Feature Alignment
* **Challenge:** LLaMA-3's Tokenizer mechanism differs from older models, and the BF16 format is incompatible with some legacy environments.
* **Solution:**
    * Implemented **Token-to-Word Merging**: Merged Subword Token attention/hidden states into Word-level matrices, resolving token granularity mismatches.
    * **Lower-Triangle Flattening**: When processing Attention matrices, extracted and flattened the lower triangle to preserve geometric structural information.
    * **BOS Fix:** Corrected alignment offsets caused by LLaMA-3's specific Begin-Of-Sentence tokens.

### 2: Correction of Time Alignment Strategy
* **Trial & Error:** Initially attempted to align features to fMRI acquisition points every 2 seconds (TR).
* **Failure Cause:** In the auditory task lacking high-precision eye-tracking data, TR-level micro-alignment was overwhelmed by HRF delay and noise, resulting in $r \approx 0$.
* **Correction:** Pivoted to **Sentence-Level Analysis**. Referencing the supplementary materials of the paper, we treated the "whole sentence" as the unit of analysis, calculating the Mean BOLD signal within that time duration. This change significantly improved the Signal-to-Noise Ratio (SNR) and successfully captured correlation signals.

### 3: Confounding Variables
* **Phenomenon:** In mid-term testing, we found that the prediction score for Layer 0 (Embedding) was abnormally high ($r \approx 0.30$), and the scores across all layers appeared as a flat line with no layer-wise differentiation.
* **Diagnosis:** This is a typical **Sentence Length Effect**.
    * *Physiological Fact:* Longer sentences $\rightarrow$ Longer activation of the auditory cortex $\rightarrow$ Stronger BOLD signal.
    * *Model Feature:* Longer sentences $\rightarrow$ Greater cumulative vector energy/More non-zero Attention elements.
    * *Conclusion:* The model was effectively "counting words" to predict brain activity, rather than utilizing semantic information.
* **Final Solution (De-confounding):** Introduced control variables before regression analysis.
    1.  Calculated the duration of each sentence.
    2.  Performed **Residual Regression (Residualization)** on the fMRI signals to remove the parts explainable by duration.
    3.  Trained using the residual signals.
    * **Result:** The score for Layer 0 was effectively suppressed, and the semantic advantage of the middle layers was finally revealed, presenting the expected "Inverted-U" curve.

---

## 4. Limitations

As a simplified replication project under limited conditions, this project has the following objective limitations:
1.  **Sample Size:** Data from only a single subject (`sub-EN057`) was used, so results may be affected by individual variability.
2.  **Feature Granularity:** The original paper combined Eye-tracking data and Attention Matrices for fine-grained analysis, whereas this project mainly relied on Hidden States and audio timestamps.
3.  **Statistical Power:** Due to the small data volume, the current Base vs. Instruct difference analysis has not yet undergone large-scale permutation testing.

---

### References
1.  Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain.Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0
2.  Li, J., Bhattasali, S., Zhang, S. et al. Le Petit Prince multilingual naturalistic fMRI corpus. Sci Data 9, 530 (2022). https://doi.org/10.1038/s41597-022-01625-7

---
*Created by Haoyang Chai | Dec 2025*
