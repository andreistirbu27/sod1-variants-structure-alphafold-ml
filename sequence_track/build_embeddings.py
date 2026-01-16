import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# We upgrade to the 150M model for better biological signal
MODEL_NAME = "facebook/esm2_t30_150M_UR50D" 
INPUT_CSV = "data/variants.csv"
OUTPUT_NPZ = "sequence_track/features_sequence.npz"

# Canonical Human SOD1 Sequence (UniProt P00441)
SOD1_WT_SEQ = (
    "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTS"
    "AGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVV"
    "HEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ"
)

def determine_offset(df, sequence):
    """
    Bio-hack: Automatically checks if the CSV positions match 
    the sequence by 0-indexing, 1-indexing, or -1 (mature protein).
    """
    offsets = [0, -1, -2] # standard python, 1-based, or mature
    best_offset = 0
    best_match_count = -1

    print("Checking alignment...")
    for offset in offsets:
        matches = 0
        total = 0
        for _, row in df.iterrows():
            pos = row['pos']
            wt_csv = row['wt']
            
            # tentative index
            idx = pos + offset
            
            if 0 <= idx < len(sequence):
                total += 1
                if sequence[idx] == wt_csv:
                    matches += 1
        
        accuracy = matches / total if total > 0 else 0
        print(f"  Offset {offset}: {accuracy:.1%} match rate")
        
        if accuracy > best_match_count:
            best_match_count = accuracy
            best_offset = offset

    if best_match_count < 0.9:
        print("CRITICAL WARNING: Low alignment accuracy! Check your sequence vs CSV.")
    else:
        print(f"-> Using Offset {best_offset} (Accuracy: {best_match_count:.1%})")
        
    return best_offset

def get_residue_embedding(sequence, seq_idx, tokenizer, model, device):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
    # +1 for [CLS] token
    target_idx = seq_idx + 1
    
    if target_idx >= last_hidden_states.shape[1]:
        return None
        
    return last_hidden_states[0, target_idx, :].cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. Auto-Detect Alignment
    offset = determine_offset(df, SOD1_WT_SEQ)

    X_list = []
    y_list = []
    ids_list = []

    print("Generating embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        variant_id = row['variant_id']
        pos = row['pos']
        aa_mut = row['mut']
        label = row['label']
        
        # Apply detected offset
        seq_idx = pos + offset
        
        # Bounds check
        if seq_idx < 0 or seq_idx >= len(SOD1_WT_SEQ):
            continue

        # 2. Embed WT
        emb_wt = get_residue_embedding(SOD1_WT_SEQ, seq_idx, tokenizer, model, device)
        
        # 3. Embed Mut
        seq_list = list(SOD1_WT_SEQ)
        seq_list[seq_idx] = aa_mut
        mutant_seq = "".join(seq_list)
        emb_mut = get_residue_embedding(mutant_seq, seq_idx, tokenizer, model, device)
        
        if emb_wt is None or emb_mut is None:
            continue

        # 4. Feature Engineering: Concatenate [WT, Diff]
        # This gives the model "Starting Point" + "Direction of Change"
        diff = emb_mut - emb_wt
        final_feature = np.concatenate([emb_wt, diff]) 
        
        X_list.append(final_feature)
        y_list.append(label)
        ids_list.append(variant_id)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ, X=np.array(X_list), y=np.array(y_list), ids=np.array(ids_list))
    print(f"Saved {len(X_list)} variants. Feature shape: {np.array(X_list).shape}")

if __name__ == "__main__":
    main()