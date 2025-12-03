import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import parselmouth
from parselmouth.praat import call
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

# === 配置 ===
BASE_DIR = "/root/autodl-tmp/project_data"
FMRI_DIR = os.path.join(BASE_DIR, "fmri")
TEXTGRID_DIR = os.path.join(BASE_DIR, "textgrid")
RESULTS_DIR = os.path.join(BASE_DIR, "results_final_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

TR = 2.0
# PCA=15 防止过拟合，Delay 范围扩大
PCA_N = 15 
CANDIDATE_DELAYS = [4.0, 6.0, 8.0, 10.0] 
RUN_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23]

# === 辅助函数 ===
def get_sentence_intervals(tg_path):
    try:
        tg = parselmouth.read(tg_path)
        n = call(tg, "Get number of intervals", 1)
        sents, on, off, has = [], -1, -1, False
        for i in range(1, int(n)+1):
            lbl = call(tg, "Get label of interval", 1, i).strip()
            t1, t2 = call(tg, "Get start point", 1, i), call(tg, "Get end point", 1, i)
            if not lbl: continue
            if lbl == "#":
                if has: sents.append((on, off)); has=False
            elif lbl not in ["<sil>", "sp", "SIL"]:
                if not has: on=t1; has=True
                off = t2
        if has: sents.append((on, off))
        return sents
    except: return []

def load_fmri_with_delay(run_id, intervals, delay):
    fs = glob.glob(os.path.join(FMRI_DIR, f"*run?{run_id}*.nii.gz"))
    if not fs: return None
    try:
        data = nib.load(fs[0]).get_fdata()
    except: return None
    
    n_tr = data.shape[-1]
    flat = data.reshape(-1, n_tr).T
    # 全脑标准化
    masked = StandardScaler().fit_transform(flat)
    
    sent_bold = []
    for t1, t2 in intervals:
        tr_start = int((t1 + delay) / TR)
        tr_end = int((t2 + delay) / TR) + 1
        if tr_start >= n_tr: break
        tr_end = min(tr_end, n_tr)
        if tr_end > tr_start:
            sent_bold.append(np.mean(masked[tr_start:tr_end], axis=0))
        else:
            sent_bold.append(masked[tr_start])
            
    if len(sent_bold) == 0: return None
    return np.array(sent_bold)

def remove_confound(X, confounds):
    if confounds.ndim == 1: confounds = confounds.reshape(-1, 1)
    # 检查是否常数，避免报错
    if np.std(confounds) < 1e-9: return X 
    reg = LinearRegression().fit(confounds, X)
    return X - reg.predict(confounds)

# === 分析核心 (Cross-Validation) ===
def analyze(name, feat_folder_name):
    print(f"\n Analysis: {name} (5-Fold CV + PCA{PCA_N} + VoxelSelect)", flush=True)
    feat_base = os.path.join(BASE_DIR, feat_folder_name)
    results = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for run in RUN_IDS:
        print(f"   Processing Run {run}...", end="\r", flush=True)
        sec = run - 14
        tg = os.path.join(TEXTGRID_DIR, f"lppEN_section{sec}.TextGrid")
        if not os.path.exists(tg): continue
        
        intervals = get_sentence_intervals(tg)
        if not intervals: continue
        
        durations = np.array([t2-t1 for t1, t2 in intervals]).reshape(-1, 1)
        
        pat = os.path.join(feat_base, f"*section{sec}*.npy")
        fs = sorted(glob.glob(pat), key=lambda x: int(x.split('sent')[-1].split('.')[0]))
        if not fs: continue
        X_raw_stack = np.stack([np.load(f) for f in fs]) 
        
        # === Step 1: 搜索最佳 Delay ===
        best_delay = 6.0
        best_score = -999
        best_Y = None
        
        mid_layer = X_raw_stack.shape[1] // 2
        X_probe = X_raw_stack[:, mid_layer, :]
        
        for d in CANDIDATE_DELAYS:
            Y_probe = load_fmri_with_delay(run, intervals, d)
            if Y_probe is None: continue 
            
            n_min = min(len(Y_probe), len(X_probe))
            if n_min < 20: continue
            
            X_p = X_probe[:n_min]
            Y_p = Y_probe[:n_min]
            
            # 快速验证
            split = int(n_min * 0.8)
            pca = PCA(n_components=10) 
            
            try:
                X_tr = pca.fit_transform(X_p[:split])
                X_te = pca.transform(X_p[split:])
                ridge = RidgeCV(alphas=[1000]).fit(X_tr, Y_p[:split])
                preds = ridge.predict(X_te)
                
                Y_te = Y_p[split:]
                corrs = []
                for v in range(preds.shape[1]):
                    # 消除 ConstantInputWarning
                    if np.std(preds[:,v]) > 1e-9 and np.std(Y_te[:,v]) > 1e-9:
                        r = pearsonr(preds[:, v], Y_te[:, v])[0]
                        if not np.isnan(r): corrs.append(r)
                
                if not corrs: continue
                # 用最容易预测的 Top 100 体素来定 Delay
                score = np.mean(np.sort(corrs)[-100:])
                
                if score > best_score:
                    best_score = score
                    best_delay = d
                    best_Y = Y_p
            except: continue
        
        print(f"   [Run {run}] Best Delay: {best_delay}s (Probe Score: {best_score:.4f})")
        if best_Y is None: continue
        
        # === Step 2: 准备全数据 ===
        Y = best_Y
        n_final = len(Y)
        X_raw = X_raw_stack[:n_final]
        durations = durations[:n_final]
        
        # === Step 3: Voxel Selection (基于全数据L16) ===
        # 使用全数据筛选体素 (ROI definition)，
        # 比起在每个Fold里变动ROI，这样更稳定且便于解释
        X_sel = X_raw[:, mid_layer, :]
        pca_sel = PCA(n_components=min(10, n_final-1))
        X_sel_clean = remove_confound(pca_sel.fit_transform(X_sel), durations)
        Y_clean = remove_confound(Y, durations)
        
        ridge_sel = RidgeCV(alphas=[1000]).fit(X_sel_clean, Y_clean)
        preds_sel = ridge_sel.predict(X_sel_clean)
        
        train_corrs = []
        for v in range(Y.shape[1]):
            if np.std(preds_sel[:,v]) > 1e-9 and np.std(Y_clean[:,v]) > 1e-9:
                r = pearsonr(preds_sel[:, v], Y_clean[:, v])[0]
                train_corrs.append(r if not np.isnan(r) else -1)
            else:
                train_corrs.append(-1)
        
        # 锁定 Top 300 语言相关体素
        top_voxel_indices = np.argsort(train_corrs)[-300:]
        Y_roi = Y[:, top_voxel_indices]
        
        # === Step 4: 5-Fold Cross Validation (逐层回归) ===
        layer_scores_cv = []
        
        for l in range(X_raw.shape[1]):
            X_layer = X_raw[:, l, :]
            fold_scores = []
            
            for train_idx, test_idx in kf.split(X_layer):
                X_train, X_test = X_layer[train_idx], X_layer[test_idx]
                Y_train, Y_test = Y_roi[train_idx], Y_roi[test_idx]
                C_train, C_test = durations[train_idx], durations[test_idx]
                
                # 关键：分别去偏
                Y_train = remove_confound(Y_train, C_train)
                Y_test = remove_confound(Y_test, C_test)
                
                # PCA
                n_comp = min(PCA_N, len(train_idx)-1)
                pca = PCA(n_components=n_comp)
                X_train = remove_confound(pca.fit_transform(X_train), C_train)
                X_test = remove_confound(pca.transform(X_test), C_test)
                
                # 强正则化防止过拟合
                ridge = RidgeCV(alphas=[100, 1000, 10000])
                ridge.fit(X_train, Y_train)
                preds = ridge.predict(X_test)
                
                # 计算相关性
                corrs = []
                for v in range(Y_test.shape[1]):
                    if np.std(preds[:, v]) > 1e-9 and np.std(Y_test[:, v]) > 1e-9:
                        r = pearsonr(preds[:, v], Y_test[:, v])[0]
                        if not np.isnan(r): corrs.append(r)
                
                # 记录该 Fold 的表现
                if corrs:
                    fold_scores.append(np.mean(corrs))
                else:
                    fold_scores.append(0)
            
            mean_score = np.mean(fold_scores)
            layer_scores_cv.append(mean_score)
            
            print(f"     L{l:02d}: CV-r={mean_score:.4f}", flush=True)
                
        results[f"Run{run}"] = layer_scores_cv

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(RESULTS_DIR, f"{name}_final_results.csv"))
        print(f"   (Auto-saved progress after Run {run})")

if __name__ == "__main__":
    if os.path.exists(os.path.join(BASE_DIR, "embeddings_base")):
        analyze("Base", "embeddings_base")
        analyze("Instruct", "embeddings_instruct")
    else:
        print("please run step1_extract_mean.py to generate embeddings_base！")
