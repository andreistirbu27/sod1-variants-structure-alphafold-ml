from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}

# Protein change formats we might see
RE_ONE_LETTER = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
RE_HGVS_3LETTER = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")

LABEL_MAP = {
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Pathogenic/Likely pathogenic": 1,
    "Benign": 0,
    "Likely benign": 0,
    "Benign/Likely benign": 0,
    "Uncertain significance": 0,
}


def parse_protein_change(row: pd.Series) -> tuple[int, str, str] | None:
    """
    Returns (pos, wt, mut) in 1-letter format, or None if not a simple missense.
    Priority:
      1) 'Protein change' column (often like G73S)
      2) parse from 'Name' column HGVS (like ... (p.Pro67Arg))
    """
    pc = str(row.get("Protein change", "") or "").strip()
    if pc:
        m = RE_ONE_LETTER.match(pc)
        if m:
            wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
            return pos, wt, mut

    name = str(row.get("Name", "") or "")
    m = RE_HGVS_3LETTER.search(name)
    if m:
        wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
        wt = AA3_TO_1.get(wt3)
        mut = AA3_TO_1.get(mut3)
        if wt and mut and wt != "*" and mut != "*":
            return pos, wt, mut

    return None

def gene_contains(gene_field: str, gene: str) -> bool:
    # Gene(s) can be "SOD1" or "SCAF4|SOD1" etc.
    parts = str(gene_field or "").split("|")
    return gene in parts

def simplify_clinsig(s: str) -> str:
    # Keep the main tag, drop extra annotations after semicolon/comma
    s = (s or "").strip()
    if not s:
        return ""
    return re.split(r"[;,]", s)[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to clinvar_result.txt (tab-delimited).")
    ap.add_argument("--out", dest="out_path", default="data/variants.csv", help="Output variants.csv path.")
    ap.add_argument("--gene", default="SOD1", help="Gene symbol to keep (default: SOD1).")
    ap.add_argument("--uniprot", default="P00441", help="UniProt ID to write (default: P00441 for SOD1).")
    ap.add_argument("--only_missense", action="store_true", help="Keep only rows whose Molecular consequence contains 'missense variant'.")
    ap.add_argument("--drop_unlabeled", action="store_true", help="Drop rows that aren't clearly benign/pathogenic.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t", dtype=str)
    # Some exports have an extra trailing empty column like 'Unnamed: 24'
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # Keep gene
    df = df[df["Gene(s)"].apply(lambda x: gene_contains(x, args.gene))].copy()

    if args.only_missense:
        df = df[df["Molecular consequence"].fillna("").str.contains("missense variant", na=False)].copy()

    # Parse pos/wt/mut
    parsed = df.apply(parse_protein_change, axis=1)
    df = df[parsed.notnull()].copy()
    df[["pos", "wt", "mut"]] = pd.DataFrame(parsed.dropna().tolist(), index=parsed.dropna().index)

    # Clinical significance -> label
    sig_raw = df["Germline classification"].fillna("")
    sig_simple = sig_raw.apply(simplify_clinsig)
    df["clinical_significance"] = sig_simple
    df["label"] = sig_simple.map(LABEL_MAP)  # becomes NaN for uncertain/conflicting/etc.

    if args.drop_unlabeled:
        df = df[df["label"].notnull()].copy()

    # Choose ClinVar identifier: Accession often looks like VCV...
    def pick_clinvar_id(row: pd.Series) -> str:
        acc = str(row.get("Accession", "") or "").strip()
        if acc.startswith("VCV"):
            return acc
        vid = str(row.get("VariationID", "") or "").strip()
        return vid

    df["clinvar_id"] = df.apply(pick_clinvar_id, axis=1)
    df["review_status"] = df.get("Germline review status", "")

    # Build variant_id
    df["variant_id"] = df.apply(lambda r: f"{args.gene}_{r['wt']}{int(r['pos'])}{r['mut']}", axis=1)

    out = df[[
        "variant_id",
        # keep these fixed so your downstream code doesn't change
        # (protein is the gene symbol in this project)
        # you can rename later if you want
    ]].copy()
    out["protein"] = args.gene
    out["uniprot_id"] = args.uniprot
    out["pos"] = df["pos"].astype(int)
    out["wt"] = df["wt"]
    out["mut"] = df["mut"]
    # label may be missing if Uncertain significance etc.
    out["label"] = df["label"].astype("Int64")
    out["clinvar_id"] = df["clinvar_id"]
    out["clinical_significance"] = df["clinical_significance"]
    out["review_status"] = df["review_status"].fillna("")

    out = out.drop_duplicates(subset=["variant_id"]).sort_values(["pos", "wt", "mut"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    # Print summary so you immediately see if you have enough benign/pathogenic
    print(f"Wrote: {out_path} ({len(out)} rows)")
    print("Label counts (0=benign, 1=pathogenic, <NA>=unlabeled):")
    print(out["label"].value_counts(dropna=False))

    if (out["label"] == 0).sum() == 0:
        print("\nWARNING: no benign missense variants found in this export.")
        print("You can still keep this file, but you'll need benign missense variants to train a classifier.")
        print("Tip: export more SOD1 missense variants from ClinVar without filtering to ALS only,")
        print("or specifically export Benign/Likely benign missense variants and merge.")

if __name__ == "__main__":
    main()
