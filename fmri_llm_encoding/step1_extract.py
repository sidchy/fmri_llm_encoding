import os
import glob
import numpy as np
import torch
import parselmouth
from parselmouth.praat import call
from transformers import AutoModelForCausalLM, AutoTokenizer

# === ⚙️ 配置 ===
BASE_DIR = "/root/autodl-tmp/project_data"
TEXTGRID_DIR = os.path.join(BASE_DIR, "textgrid")
# ⚠️ 注意：这里输出到 embeddings_base (Mean Pooling 版)
MODEL_PATHS = {
    "Base": "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B",
    "Instruct": "/root/autodl-tmp/models_instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct"
}
TOKEN_BEGIN = "Ġ"

# === 🛠️ 辅助函数 ===
def parse_textgrid(tg_path):
    try:
        tg = parselmouth.read(tg_path)
        n = call(tg, "Get number of intervals", 1)
        sents, curr = [], []
        for i in range(1, int(n) + 1):
            lbl = call(tg, "Get label of interval", 1, i).strip()
            if not lbl: continue
            if lbl == "#":
                if curr: sents.append(" ".join(curr)); curr = []
            elif lbl not in ["<sil>", "sp", "SIL"]: curr.append(lbl)
        if curr: sents.append(" ".join(curr))
        return sents
    except: return []

def token_groups_robust(words, tokens):
    groups = []
    words_iter = iter(words)
    try:
        current_word = next(words_iter).lower()
    except StopIteration: return []
    text_buf, id_buf = '', []
    for i, token in enumerate(tokens):
        clean_tok = token.replace(TOKEN_BEGIN, '').lower().replace(' ', '')
        if text_buf != current_word:
            text_buf += clean_tok
            id_buf.append(i)
        else:
            groups.append(id_buf.copy())
            text_buf, id_buf = clean_tok, [i]
            try:
                current_word = next(words_iter).lower()
            except StopIteration: break
    if id_buf: groups.append(id_buf)
    return groups

# === 🧠 提取器 (Mean Pooling Mode) ===
class Extractor:
    def __init__(self, path):
        print(f"Loading {os.path.basename(path)}...", flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            output_hidden_states=True, 
            trust_remote_code=True
        )
        self.model.eval()

    def process(self, text):
        words = text.strip().split()
        if len(words) < 2: return None
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        has_bos = (raw_tokens and (raw_tokens[0] == '<|begin_of_text|>' or inputs.input_ids[0][0] == 128000))
        tokens_align = raw_tokens[1:] if has_bos else raw_tokens
        
        groups = token_groups_robust(words, tokens_align)
        if len(groups) != len(words): return None

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # [Layers, Seq, Dim]
        all_states = torch.stack(outputs.hidden_states).squeeze(1).float().cpu().numpy()
        if has_bos: all_states = all_states[:, 1:, :]
        
        layers_data = []
        for l in range(all_states.shape[0]):
            layer_embs = all_states[l]
            
            # 1. 单词级平均
            word_embs = []
            for group in groups:
                valid_idx = [i for i in group if i < layer_embs.shape[0]]
                if valid_idx:
                    word_embs.append(np.mean(layer_embs[valid_idx], axis=0))
            
            # 2. 句子级平均 (Mean Pooling)
            if word_embs:
                sent_emb = np.mean(np.vstack(word_embs), axis=0)
                layers_data.append(sent_emb)
            else:
                layers_data.append(layer_embs[-1]) # Fallback
                
        return np.array(layers_data)

def run(key):
    # 输出到 embeddings_base / embeddings_instruct
    out = os.path.join(BASE_DIR, f"embeddings_{key.lower()}")
    os.makedirs(out, exist_ok=True)
    
    ext = Extractor(MODEL_PATHS[key])
    files = sorted(glob.glob(os.path.join(TEXTGRID_DIR, "*.TextGrid")))
    cnt = 0
    
    print(f"Start processing {key} (Mean Pooling)...")
    for f in files:
        fname = os.path.basename(f)
        for i, sent in enumerate(parse_textgrid(f)):
            # 检查是否已存在，避免重复跑 (可选)
            # if os.path.exists(os.path.join(out, f"{fname}_sent{i}.npy")): continue
            
            res = ext.process(sent)
            if res is not None:
                np.save(os.path.join(out, f"{fname}_sent{i}.npy"), res)
                cnt += 1
    print(f"✅ {key} Done: {cnt} sentences.")

if __name__ == "__main__":
    run("Base")
    run("Instruct")