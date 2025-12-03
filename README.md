# 🧠 Brain-Llama3.1-Encoding: A Preliminary Replication
## 通过 LLaMA-3.1 编码人脑 fMRI 信号

> **Base Paper:** Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain. Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0

本项目是一个基于 **计算神经语言学 (Computational Neurolinguistics)** 的实证研究。受 Gao et al. (2025) 发表在 *Nature Computational Science* 上的工作启发，我在基于 **Meta-LLaMA-3.1-8B (base & instruct)** 构建了一套简化的神经编码pipeline，尝试验证文章的一些基本发现在Llama 3.1上的效果。

本项目的核心目标是在有限算力下，尝试验证 **指令微调 (Instruction Tuning)** 是否显著改变了大语言模型底层的语义表征及其与人脑语言网络的对齐度。

---

## 1. 核心发现

通过对比 **LLaMA-3.1-Base** 与 **LLaMA-3.1-Instruct** 在 **Le Petit Prince (LPP)** fMRI 数据集上的编码性能，我们得出了以下初步结论：

### 📈 1.1 中间层效应 (The "Inverted-U" Trend)
成功复现了神经语言学领域的经典发现：模型对大脑的预测能力呈现“倒U型”分布。
<img width="1000" height="600" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/2dab9dd7-928d-4631-83fc-3756c6f9b425" />
* **Layer 0 (Embedding):** 在去除了句子长度混淆后，其预测能力显著低于中间层。
* **Middle Layers (L16-L24):** **预测能力达到峰值** (Max Pearson's $r \approx 0.60$, Top-5% $\approx 0.30$)。这表明 LLaMA 的中间层表征最接近人类大脑进行句法分析和语义整合的区域。
* **Late Layers:** 随着模型专注于 Next-token Prediction，其与大脑通用理解机制的对齐度略有下降。

<img width="4200" height="1800" alt="plot_B_heatmap" src="https://github.com/user-attachments/assets/718a3b92-a534-4533-9afb-ba2a231ee6a8" />
*(图：Base 模型在不同 Run 上的层级性能热力图，显示中间层的高响应区)*

### 1.2 微调的影响
实验显示，Base 和 Instruct 版本的层级对齐曲线 **高度重叠**。
* 这验证了 Gao et al. (2025) 的结论：指令微调 (RLHF/SFT) 虽然提升了任务表现，但并没有显著增强模型与人类大脑活动的对齐程度。
* 如下图所示，大多数层级的点落在 $y=x$ 线附近，表明两者底层语义表征机制基本一致。

<img width="2400" height="2400" alt="plot_C_scatter" src="https://github.com/user-attachments/assets/77da05c5-6b10-45be-bc13-8b8bab180979" />

<img width="3600" height="1800" alt="plot_A_difference (1)" src="https://github.com/user-attachments/assets/598ce0a4-9026-4bf9-b9c1-2aea3a5d0fcc" />

---

## 2. 实验方法 
### 2.1 数据与环境
* **Dataset:** *Le Petit Prince* (LPP) fMRI Dataset (Sub-EN057, ~1.5h Audio) (受算力限制，我们只验证了一名英文被试的全程fMRI数据)。
* **Hardware:** AutoDL Cloud Server / **NVIDIA RTX 4090 (24GB VRAM)**.
* **Model Loading:** LLaMA-3.1-8B (BF16 / 4-bit Quantization via `bitsandbytes`).

### 2.2 核心Pipeline
1.  **特征提取 (Feature Extraction):** 提取 Layer 0-32 的 Hidden States，采用 **Sentence-level Mean Pooling** 以适应听力任务的低信噪比。
2.  **血流动力学建模 (HRF Alignment):** 使用 `nilearn` 进行 HRF 卷积，并在 4s-10s 范围内搜索最佳延迟 (Best Delay)。
3.  **降维与回归 (Encoding):** 使用 PCA (n=15) 降维，配合 Ridge Regression (5-Fold CV) 进行体素级预测。

---

## 3. 挑战与方法论演进

本项目的复现并非一帆风顺。为了逼近原论文的逻辑，我经历了三次关键的方法论迭代。

### 1: 工程适配与特征对齐
* **挑战:** LLaMA-3 的 Tokenizer 机制与旧版模型不同，且 BF16 格式在部分旧环境中不兼容。
* **方案:**
    * 实现了 **Token-to-Word Merging**：将 Subword Token 的注意力/隐藏层状态合并为单词级 (Word-level) 矩阵，解决了分词粒度不匹配问题。
    * **Lower-Triangle Flattening:** 在处理 Attention 矩阵时，提取下三角并展平，保留了几何结构信息。
    * **BOS Fix:** 修正了 LLaMA-3 特有的 Begin-Of-Sentence 标记导致的对齐偏移。

### 2: 时间对齐策略的修正
* **试错:** 初期尝试将特征对齐到每 2 秒 (TR) 的 fMRI 采集点。
* **失败原因:** 在缺乏高精度眼动数据 (Eye-tracking) 的听力任务中，TR 级的微观对齐被 HRF 延迟和噪声淹没，导致 $r \approx 0$。
* [cite_start]**修正:** 转向 **Sentence-Level Analysis**。参考论文补充材料，将“整句话”作为一个分析单位，计算该时间段内的 Mean BOLD 信号 [cite: 472, 473]。这一改变大幅提升了信噪比 (SNR)，成功捕捉到了相关性信号。

### 3: 混淆变量
* **现象:** 在中期测试中，我们发现 Layer 0 (Embedding) 的预测分数异常高 ($r \approx 0.30$)，且全层分数呈现平直线，无层级差异。
* **诊断:** 这是一个典型的 **句子长度效应 (Sentence Length Effect)**。
    * *生理事实:* 句子越长 $\rightarrow$ 听觉皮层激活越久 $\rightarrow$ BOLD 信号越强。
    * *模型特征:* 句子越长 $\rightarrow$ 向量累积能量/Attention 非零元素越多。
    * *结论:* 模型实际上是在“数单词个数”来预测大脑活动，而非利用语义信息。
* **最终方案 (De-confounding):** 在回归分析前引入控制变量。
    1. 计算每句话的时长 (Duration)。
    2. 对 fMRI 信号进行 **残差回归 (Residualization)**，剔除时长可解释的部分。
    3. 使用残差后的信号进行训练。
    * **结果:** Layer 0 的分数被有效抑制，中间层的语义优势终于显露出来，呈现出符合预期的“倒 U 型”曲线。

---

## 4. 局限性

作为一个有限条件下的简易复现项目，本项目存在以下客观局限：
1.  **样本量:** 仅使用了单被试 (`sub-EN057`) 数据，结果可能受个体差异影响 。
2.  **特征粒度:** 原论文结合了眼动数据 (Eye-tracking) 和 Attention Matrices 进行精细分析，本项目主要依赖 Hidden States 和音频时间戳。
3.  **统计效力:** 由于数据量较小，目前的 Base vs. Instruct 差异分析尚未通过大规模的置换检验。

---

## 🇬🇧 English Summary (Dev Log Included)

### Methodological Evolution & Challenges
This replication involved three critical iterations to align with the rigorous standards of Gao et al. (2025):

1.  [cite_start]**Feature Extraction Logic:** We implemented specific **Token-to-Word Merging** and **Lower-Triangle Flattening** to handle LLaMA-3's tokenizer and preserve the geometric structure of attention matrices[cite: 420, 454].
2.  **Time Alignment Pivot:** Initial attempts at TR-level (2s) alignment failed due to noise. We pivoted to **Sentence-Level Analysis** (Mean BOLD per sentence), which significantly improved the Signal-to-Noise Ratio (SNR) for the auditory task.
3.  **Solving the "Length Confound":**
    * *Issue:* Initial results showed suspiciously high performance at Layer 0 (Embedding) and a flat layer-wise trend.
    * *Diagnosis:* The model was predicting brain activity based on **Sentence Duration** (longer sentence = stronger BOLD = more feature energy), not semantics.
    * *Fix:* We implemented **Duration De-confounding** by regressing out sentence length from the fMRI signals before training. This successfully revealed the true semantic "Inverted-U" trend peaked at middle layers.

### Conclusion
This project successfully established a reproducible pipeline from **LLaMA-3.1 to fMRI**. [cite_start]While constrained by sample size, the results support the hypothesis that instruction tuning does not fundamentally alter the brain-like semantic representations in LLMs[cite: 12, 361].

---

### References
1.  Gao, C., Ma, Z., Chen, J. et al. Increasing alignment of large language models with language processing in the human brain. Nat Comput Sci 5, 1080–1090 (2025). https://doi.org/10.1038/s43588-025-00863-0
2.  Li, J., Bhattasali, S., Zhang, S. et al. Le Petit Prince multilingual naturalistic fMRI corpus. Sci Data 9, 530 (2022). https://doi.org/10.1038/s41597-022-01625-7

---
*Created by Haoyang Chai | Dec 2025*
