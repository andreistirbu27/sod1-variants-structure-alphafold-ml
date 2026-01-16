import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" 
INPUT_CSV = "data/variants.csv"
OUTPUT_NPZ = "sequence_track/features_sequence.npz"

SOD1_WT_SEQ = (
    "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTS"
    "AGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVV"
    "HEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ"
)

def get_residue_embedding(sequence, pos_0_indexed, tokenizer, model, device):
    """
    Extracts the embedding of the SPECIFIC residue at pos_0_indexed.
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
    # ESM-2 adds a [CLS] token at the start.
    # So, the 0-th residue of the protein is at index 1 of the tensor.
    target_idx = pos_0_indexed + 1
    
    # Check bounds (just in case)
    if target_idx >= last_hidden_states.shape[1]:
        # Fallback to mean if index is weird (shouldn't happen for SOD1)
        return last_hidden_states[0, 1:-1, :].mean(dim=0).cpu().numpy()

    # Extract specific vector (Shape: 320,)
    embedding = last_hidden_states[0, target_idx, :]
    
    return embedding.cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    df = pd.read_csv(INPUT_CSV)
    
    X_list = []
    y_list = []
    ids_list = []

    print("Generating residue-specific embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        variant_id = row['variant_id']
        pos = row['pos']       # 1-based
        aa_mut = row['mut']
        label = row['label']
        
        seq_idx = pos - 1      # 0-based index
        
        # 1. Embed WT at this position
        emb_wt = get_residue_embedding(SOD1_WT_SEQ, seq_idx, tokenizer, model, device)
        
        # 2. Create Mutant Sequence
        seq_list = list(SOD1_WT_SEQ)
        # Safety check
        if seq_idx < len(seq_list):
            seq_list[seq_idx] = aa_mut
            mutant_seq = "".join(seq_list)
            
            # 3. Embed Mutant at this position
            emb_mut = get_residue_embedding(mutant_seq, seq_idx, tokenizer, model, device)
            
            # 4. Feature: Difference
            diff_vector = emb_mut - emb_wt
            
            X_list.append(diff_vector)
            y_list.append(label)
            ids_list.append(variant_id)

    np.savez(OUTPUT_NPZ, X=np.array(X_list), y=np.array(y_list), ids=np.array(ids_list))
    print(f"Done. Features shape: {np.array(X_list).shape}")

if __name__ == "__main__":
    main()