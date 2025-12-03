# Brain-Llama3.1-Encoding: A Preliminary Replication

> **Base Paper:** Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain.Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0

## Project Overview
This is a self-driven replication and extension of the "Brain-LLM Alignment" pipeline.

**Context (The Base Paper):**
Gao et al. (2025) established two key findings using LLaMA-2 and GPT series:
1.  **Scaling Law:** Larger models align better with human brain activity.
2.  **Instruction Tuning Gap:** Surprisingly, RLHF/Instruction tuning *does not* improve (and sometimes hurts) this alignment, suggesting "alignment with human intent" $\neq$ "alignment with neural mechanisms."

**My Goal:**
I wanted to verify if these findings hold for the newer **Meta-LLaMA-3.1-8B**. Using the **Le Petit Prince (LPP)** fMRI dataset, I built a simplified encoding pipeline from scratch to test the **Base vs. Instruct** hypothesis.

---

## 1. Key Findings

### 1: The "Inverted-U" Trend Persists
Replicating the classic neurolinguistic finding: The model's predictive power peaks at the **middle layers**, not the final output layers.

<img width="1000" height="600" alt="Layer Trend" src="https://github.com/user-attachments/assets/2dab9dd7-928d-4631-83fc-3756c6f9b425" />

* **Layer 0 (Embedding):** Low correlation (after removing duration confounds).
* **Middle Layers (L16-L24):** **Peak performance** (Max Pearson's $r \approx 0.60$). This suggests LLaMA-3.1's intermediate representations capture the syntactic/semantic integration processes similar to the brain's language network.
* **Late Layers:** Alignment drops as the model focuses on next-token probability distributions.

<img width="4200" height="1800" alt="Heatmap" src="https://github.com/user-attachments/assets/718a3b92-a534-4533-9afb-ba2a231ee6a8" />

### 2: Instruction Tuning Has Negligible Impact
Consistent with Gao et al., the alignment curves for **Base** and **Instruct** models are **highly overlapping**.

* Instruction tuning changes the output style but seemingly leaves the underlying semantic representation mechanism unchanged.
* As shown below, most layer-wise comparisons fall strictly on the $y=x$ line.

<img width="2400" height="2400" alt="Scatter Plot" src="https://github.com/user-attachments/assets/77da05c5-6b10-45be-bc13-8b8bab180979" />

---

## 2. Implementation Details

* **Dataset:** *Le Petit Prince* (LPP) fMRI Dataset (Sub-EN057, ~1.5h Audio).
* **Compute:** AutoDL / NVIDIA RTX 4090 (24GB).
* **Model:** LLaMA-3.1-8B (BF16 / 4-bit Quantization via `bitsandbytes`).
* **Encoding Pipeline:**
    * **Feature:** Hidden States (L0-L32) -> **Sentence-level Mean Pooling**.
    * **HRF:** Convolved with `nilearn` (Delay search: 4s-10s).
    * **Model:** PCA (n=15) + Ridge Regression (5-Fold CV).

---

## 3. Challenges & Fixes

Here are the main technical blockers I hit and how I solved them:

**1. The "Sentence Length" Confound (Critical)**
* *Issue:* Initial runs showed suspiciously high scores ($r \approx 0.30$) at Layer 0 (Embedding) and a flat trend across all layers.
* *Diagnosis:* The model was just "counting words." Longer sentences = stronger BOLD signal & higher vector energy. The regression was fitting **duration**, not **semantics**.
* *Fix:* Implemented **Confound Regression**. I calculated sentence duration, regressed it out of the fMRI signals, and trained the encoding model on the **residuals**. This successfully suppressed Layer 0 and revealed the true semantic "Inverted-U" curve.

**2. Alignment Strategy Pivot**
* *Issue:* Standard TR-level (2s) alignment yielded near-zero correlations due to the lack of precise eye-tracking timestamps for this specific auditory run.
* *Fix:* Pivoted to **Sentence-Level Analysis** (averaging BOLD signals per sentence duration). This significantly boosted the Signal-to-Noise Ratio (SNR) for the auditory task.

**3. LLaMA-3 Tokenizer Quirks**
* *Issue:* LLaMA-3's tokenizer handles special tokens differently than LLaMA-2, causing misalignment in the timeline.
* *Fix:* Wrote a custom `Token-to-Word` merger to aggregate subword attentions into word-level matrices before pooling.

---

## 4. Technical Constraints

To keep this replication feasible within a short timeframe, I made the following trade-offs compared to the original paper:

* **Hidden States vs. Attention Matrices:** The original paper used attention matrices (Head-wise analysis) to model specific cognitive functions (e.g., coreference). I used **Hidden States (Embeddings)**, which offer a more holistic but coarser view of semantic processing.
* **Pooling Strategy:** I used Mean Pooling. The original paper used Last-Token or specific alignments, but Mean Pooling proved more robust for the low-SNR auditory fMRI data in this specific setup.

---

### References
1.  Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain.Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0
2.  Li, J., Bhattasali, S., Zhang, S. et al. Le Petit Prince multilingual naturalistic fMRI corpus. Sci Data 9, 530 (2022). https://doi.org/10.1038/s41597-022-01625-7
