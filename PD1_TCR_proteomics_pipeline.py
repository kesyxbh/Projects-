#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD1–TCR Proteomics Analysis Pipeline (Jurkat; aggregated/per-sample)

What this version guarantees (to match the supervisor’s requirements):
- Carries forward protein/gene annotations (e.g., Protein ID / Entry / Gene / Protein name)
  from the original Excel into ALL downstream outputs.
- In aggregated mode (current data): computes PD1 vs CTRL log2 fold-changes
  at each timepoint and saves annotated tables + a multi-sheet Excel workbook.
- Exports per-timepoint Top↑/↓ lists.
- Optionally performs pathway enrichment (GO/KEGG/Reactome via gseapy) if a Gene column exists.

Dependencies (install once):
    pip install -r requirements.txt
    # requirements.txt should include: pandas numpy scipy scikit-learn matplotlib openpyxl gseapy
"""


# ================================================================


from typing import List, Tuple, Optional
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

# ==============================================================================

# ======= USER COLUMN CONFIG (exact names from Excel) =======
USER_COL_CONFIG = {
    # Sheet to read from (set to the exact sheet tab name in workbook).
    "sheet_name": "LFQs_renamed_reorderd-2",


    # Map original annotation column names -> the standardized names
    # (the right-side names will appear in all outputs).
    "annotation_rename": {
        "Protein": "Protein",
        "Protein ID": "Protein ID",
        "Entry Name": "Entry Name",
        "Gene": "Gene",
        "Protein Length": "Protein Length",
        "Organism": "Organism",
        "Protein Existence": "Protein Existence",
        "Description": "Protein name",          # 我把 Description 统一显示为 Protein name
        "Protein Probability": "Protein Probability",
        "Top Peptide Probability": "Top Peptide Probability",
        # Add more if needed: "Original Col": "Output Col"
    },

    # EXACT mapping from intensity columns to (condition, time).
    # LEFT side MUST match Excel header text exactly.
    "condition_map": {
        "PD1 reporter unstimulated": ("PD1", "0 min"),
        "PD-1 reporter TCS Ctrl 5min": ("PD1", "5 min"),
        "PD-1 reporter TCS Ctrl 20 min": ("PD1", "20 min"),
        "PD-1 reporter TCS Ctrl 4h": ("PD1", "4 h"),
        "PD-1 TCS  PDL1 5 min": ("PD1", "5 min"),
        "PD-1 TCS PDL1 20 min": ("PD1", "20 min"),
        "PD-1 TCS PDL1 4h": ("PD1", "4 h"),

        "Reporter Control unstimulated": ("CTRL", "0 min"),
        "Reporter Control TCS Ctrl 5 min": ("CTRL", "5 min"),
        "Reporter Control TCS Ctrl 20 min": ("CTRL", "20 min"),
        "Reporter Control TCS Ctrl 4h": ("CTRL", "4 h"),
        "Reporter Control TCS PD-L1 5 min": ("CTRL", "5 min"),
        "Reporter Control TCS PD-L1 20 min": ("CTRL", "20 min"),
        "Reporter Control TCS PD-L1 4h": ("CTRL", "4 h"),
    },
}
# ==============================================================================


# -------------------------- I/O helpers --------------------------

def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Read metadata CSV and normalize column names to lowercase."""
    meta = pd.read_csv(meta_path)
    meta.columns = [c.strip().lower() for c in meta.columns]
    return meta

def load_proteomics_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    Load a sheet that has a 2-row header:
    - Row 0 holds most condition/intensity names (K..end).
    - Row 1 holds annotation headers for the first block (A..J).
    We coalesce header = row0 if non-empty, else row1.
    Then data start at row index 2.

    Also:
    - Strip/clean column names
    - Drop empty/'Unnamed' columns
    - Deduplicate headers if repeated
    - Rename annotation columns according to USER_COL_CONFIG['annotation_rename']
    """

    # 1) Pick sheet (respect config; fallback to first sheet)
    sheet = USER_COL_CONFIG.get("sheet_name")
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    if sheet and sheet in xls.sheet_names:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, engine="openpyxl")
        print(f"[loader] Using sheet: {sheet}")
    else:
        if sheet:
            print(f"[loader] Sheet '{sheet}' not found. Using first sheet: {xls.sheet_names[0]}")
        raw = pd.read_excel(xlsx_path, sheet_name=xls.sheet_names[0], header=None, engine="openpyxl")
        print(f"[loader] Using sheet: {xls.sheet_names[0]}")

    # 2) Build header by coalescing row0 and row1
    hdr0 = raw.iloc[0].astype(str).str.strip()
    hdr1 = raw.iloc[1].astype(str).str.strip()

    def pick(a: str, b: str) -> str:
        a_low = a.lower()
        if a and a_low not in ("nan", "none", ""):
            return a
        b_low = b.lower()
        if b and b_low not in ("nan", "none", ""):
            return b
        return ""

    new_cols = [pick(a, b) for a, b in zip(hdr0.tolist(), hdr1.tolist())]

    # 3) Data start at row 2
    df = raw.iloc[2:].copy()
    df.columns = new_cols

    # 4) Drop empty/unnamed columns, strip names, deduplicate
    keep = []
    for c in df.columns:
        low = str(c).strip().lower()
        drop = (low in ("", "nan", "none")) or low.startswith("unnamed")
        keep.append(not drop)
    df = df.loc[:, keep]
    df.columns = [str(c).strip() for c in df.columns]

    # Deduplicate headers conservatively (add suffixes if repeated)
    seen = {}
    fixed = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            fixed.append(c)
        else:
            seen[c] += 1
            fixed.append(f"{c}.{seen[c]}")
    df.columns = fixed

    # 5) Apply user renaming for annotation columns
    ann_rename = USER_COL_CONFIG.get("annotation_rename", {}) or {}
    for k in list(ann_rename.keys()):
        if k not in df.columns:
            print(f"[warn] annotation column not found in sheet: {k}")
    df = df.rename(columns=ann_rename)

    # 6) Reset index for safety
    return df.reset_index(drop=True)



# -------------------------- Column detection --------------------------

def get_exact_condition_columns(df: pd.DataFrame) -> List[str]:
    """
    Return only those condition/time columns that are explicitly listed
    in USER_COL_CONFIG['condition_map'] AND actually present in the sheet.
    """
    cond_map = USER_COL_CONFIG.get("condition_map", {}) or {}
    present = [c for c in cond_map.keys() if c in df.columns]
    missing = [c for c in cond_map.keys() if c not in df.columns]
    if missing:
        print("[warn] missing condition columns (check spelling/casing):", missing)
    return present

def condition_time_from_exact(colname: str) -> Tuple[str, str]:
    """
    Convert an intensity column header into (condition, time) using the exact mapping.
    """
    cond_map = USER_COL_CONFIG.get("condition_map", {}) or {}
    if colname not in cond_map:
        raise KeyError(f"Column not in USER_COL_CONFIG['condition_map']: {colname}")
    return cond_map[colname]


def guess_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer a real protein/gene identifier as the 'id' column for grouping.
    Falls back to the first column if nothing else is available.
    """
    preferred = [
        "Protein ID", "Protein_ID", "ProteinID",
        "Entry", "Entry Name",
        "Gene", "Gene name", "Gene names",
        "Accession", "UniProt", "Uniprot"
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in ["Gene","gene","Entry Name","Entry","EntryName","Protein ID","Protein","Accession","UniProt","uniprot"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) else None

def detect_sample_columns_from_meta(df: pd.DataFrame, meta: pd.DataFrame) -> List[str]:
    """Detect per-sample columns by matching df column names to metadata sample IDs."""
    sample_col = None
    for c in ["sample_id", "sample", "sampleid", "run", "file", "filename"]:
        if c in meta.columns:
            sample_col = c
            break
    if sample_col is None:
        return []
    meta_ids = set(meta[sample_col].astype(str))
    return [c for c in df.columns if str(c) in meta_ids]

def detect_condition_columns(df: pd.DataFrame) -> List[str]:
    """Heuristically detect condition/time aggregated columns."""
    keys = ["pd1", "pd-1", "pd-l1", "pdl1", "tcs", "ctrl", "control", "unstimulated", "min", "h", "reporter"]
    return [c for c in df.columns if any(k in str(c).lower() for k in keys)]

def detect_annotation_columns(df: pd.DataFrame) -> List[str]:
    """
    Return a list of annotation columns to carry through outputs.
    Excludes obvious intensity/condition columns.
    """
    keys = ["protein id","entry","entry name","gene","gene name","gene names",
            "accession","uniprot","protein name","description","protein"]
    bad = ["reporter", "ctrl", "control", "pd1", "pd-1", "pdl1", "min", "h", "tcs", "unstimulated"]
    ann = [c for c in df.columns if any(k in c.lower() for k in keys)]
    ann = [c for c in ann if not any(b in c.lower() for b in bad)]
    # keep unique, preserve order
    return list(dict.fromkeys(ann))

# -------------------------- Preprocessing & PCA --------------------------

def log2_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Log2(x + 1) transform for intensity-like data."""
    return np.log2(X.astype(float) + 1.0)

def median_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """Median-centering across columns to align distributions."""
    med = X.median(axis=0, skipna=True)
    return X.subtract(med, axis=1)

def pca_plot(X: pd.DataFrame, labels: Optional[pd.Series] = None, title: str = "PCA"):
    """
    Robust PCA (handles NaN/constant cols):
    - Coerce numeric, drop all-NaN columns
    - Median-impute remaining NaN (fallback 0)
    - Drop zero-variance columns
    """
    Xnum = X.apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.dropna(axis=1, how="all")
    col_medians = Xnum.median(axis=0, skipna=True).fillna(0.0)
    Xfilled = Xnum.fillna(col_medians)
    var = Xfilled.var(axis=0, skipna=True)
    Xfilled = Xfilled.loc[:, var > 0]
    if Xfilled.shape[0] < 2 or Xfilled.shape[1] < 2:
        print("PCA skipped: not enough non-NaN/non-constant data after filtering.")
        return
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xfilled)
    pca = PCA(n_components=2, random_state=0)
    comp = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(comp[:, 0], comp[:, 1])
    if labels is not None:
        lbl = pd.Series(labels).astype(str)
        lbl = lbl.iloc[:comp.shape[0]]
        for i, txt in enumerate(lbl.tolist()):
            ax.annotate(txt, (comp[i, 0], comp[i, 1]), fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title)
    plt.show()

# -------------------------- Aggregated-mode helpers --------------------------

def parse_condition_time_from_col(colname: str) -> Tuple[str, str]:
    """
    Parse 'condition' and 'time' from an aggregated column name.
    condition in {'PD1','CTRL'}, time in {'0 min','5 min','20 min','4 h','NA'}.
    """
    s = str(colname).lower()
    cond = "PD1" if any(k in s for k in ["pd-l1", "pdl1", "pd1", "pd-1"]) else "CTRL"
    if "unstimulated" in s or "0 min" in s or "0min" in s:
        t = "0 min"
    elif "5" in s and "min" in s:
        t = "5 min"
    elif "20" in s and "min" in s:
        t = "20 min"
    elif "4h" in s or "4 h" in s or "4hour" in s:
        t = "4 h"
    else:
        t = "NA"
    return cond, t

def aggregated_long(df: pd.DataFrame, id_col: str, cond_cols: List[str], ann_cols: List[str] = None) -> pd.DataFrame:
    """
    Build a long table from EXACTLY the condition columns specified by the user,
    carrying annotation columns along.
    Output columns: [id_col, <ann_cols...>, intensity, condition, time, sample_id]
    """
    ann_cols = ann_cols or []
    cols_keep = [id_col] + [c for c in ann_cols if c != id_col]

    parts = []
    for c in cond_cols:
        tmp = df.loc[:, cols_keep + [c]].copy()
        tmp["intensity"] = pd.to_numeric(tmp[c], errors="coerce")
        tmp.drop(columns=[c], inplace=True)
        cond, tm = condition_time_from_exact(c)  # mapping from USER_COL_CONFIG
        tmp["condition"] = cond
        tmp["time"] = tm
        tmp["sample_id"] = c
        parts.append(tmp)

    if parts:
        return pd.concat(parts, axis=0, ignore_index=True)
    else:
        return pd.DataFrame(columns=cols_keep + ["intensity", "condition", "time", "sample_id"])



def aggregated_fold_changes(df: pd.DataFrame, id_col: str, cond_cols: List[str], ann_cols: List[str] = None) -> pd.DataFrame:
    """
    Compute PD1 - CTRL log2 fold-changes per timepoint using EXACT column mapping,
    preserving annotation columns in the output.
    """
    long = aggregated_long(df, id_col, cond_cols, ann_cols=ann_cols)
    long["intensity"] = np.log2(long["intensity"].astype(float) + 1.0)

    cols_id = [id_col] + ([c for c in (ann_cols or []) if c != id_col])
    out = []
    for t, g in long.groupby("time"):
        if t == "NA":
            continue
        mean_pd1  = g[g["condition"] == "PD1"].groupby(cols_id)["intensity"].mean()
        mean_ctrl = g[g["condition"] == "CTRL"].groupby(cols_id)["intensity"].mean()
        joined = pd.concat([mean_pd1.rename("PD1"), mean_ctrl.rename("CTRL")], axis=1).reset_index()
        joined["log2FC_PD1_vs_CTRL"] = joined["PD1"] - joined["CTRL"]
        joined["time"] = t
        out.append(joined)

    if not out:
        return pd.DataFrame(columns=cols_id + ["PD1", "CTRL", "log2FC_PD1_vs_CTRL", "time"])
    return pd.concat(out, axis=0, ignore_index=True)



# -------------------------- Reporting helpers --------------------------

def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_top_tables(fc: pd.DataFrame, outdir: Path, k: int = 50):
    """Save Top↑/↓ tables per timepoint (annotated)."""
    outdir.mkdir(parents=True, exist_ok=True)
    fc = fc.copy()
    fc["log2FC_PD1_vs_CTRL"] = pd.to_numeric(fc["log2FC_PD1_vs_CTRL"], errors="coerce")
    for t, sub in fc.groupby("time"):
        sub = sub.dropna(subset=["log2FC_PD1_vs_CTRL"])
        up = sub.nlargest(k, "log2FC_PD1_vs_CTRL")
        down = sub.nsmallest(k, "log2FC_PD1_vs_CTRL")
        up.to_csv(outdir / f"top_up_{str(t).replace(' ','_')}.csv", index=False)
        down.to_csv(outdir / f"top_down_{str(t).replace(' ','_')}.csv", index=False)

def export_annotated_excel(fc: pd.DataFrame, out_xlsx: Path):
    """
    Export a multi-sheet Excel with annotations:
      - All (sorted by |log2FC|)
      - 0_min, 5_min, 20_min, 4_h (if present), each sheet sorted by |log2FC|
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df = fc.copy()
    df["absFC"] = df["log2FC_PD1_vs_CTRL"].abs()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.sort_values("absFC", ascending=False).drop(columns=["absFC"]).to_excel(writer, sheet_name="All", index=False)
        for t in ["0 min", "5 min", "20 min", "4 h"]:
            if t in df["time"].astype(str).unique():
                sub = df[df["time"].astype(str) == t].copy()
                sub["absFC"] = sub["log2FC_PD1_vs_CTRL"].abs()
                sub.sort_values("absFC", ascending=False).drop(columns=["absFC"]).to_excel(
                    writer, sheet_name=t.replace(" ", "_"), index=False
                )

def try_enrichment(fc: pd.DataFrame, outdir: Path):
    """
    Optional pathway enrichment using gseapy.Enrichr on top-k genes per direction/timepoint.
    Requires a 'Gene' or 'Gene names' column in `fc`.
    Skips silently if gseapy not installed or no gene column.
    """
    gene_cols_pref = ["Gene", "Gene names", "Gene name"]
    gene_col = next((c for c in gene_cols_pref if c in fc.columns), None)
    if gene_col is None:
        print("Enrichment skipped: no Gene column found.")
        return
    try:
        import gseapy as gp
    except Exception as e:
        print("Enrichment skipped: gseapy not installed. `pip install gseapy` to enable.")
        return

    outdir = Path(outdir) / "enrichment"
    outdir.mkdir(parents=True, exist_ok=True)
    libraries = ["GO_Biological_Process_2021", "KEGG_2021_Human", "Reactome_2016"]

    # For each timepoint & direction, take top 100 genes (unique)
    for t, sub in fc.groupby("time"):
        for direction, asc in [("UP", False), ("DOWN", True)]:
            sub2 = sub.dropna(subset=["log2FC_PD1_vs_CTRL"]).copy()
            sub2 = sub2.sort_values("log2FC_PD1_vs_CTRL", ascending=asc)
            genes = (
                sub2[gene_col]
                .astype(str)
                .str.split("[,; ]+")
                .explode()
                .str.strip()
                .replace({"": np.nan})
                .dropna()
                .drop_duplicates()
                .tolist()[:100]
            )
            if len(genes) < 5:
                continue
            for lib in libraries:
                try:
                    enr = gp.enrichr(
                        gene_list=genes,
                        gene_sets=lib,
                        outdir=None,
                        cutoff=0.5,  # results are filtered by adjp < 0.5 internally
                        verbose=False,
                    )
                    res = enr.results
                    if res is None or res.empty:
                        continue
                    # Save CSV
                    save_table(res, outdir / f"enrich_{t.replace(' ','_')}_{direction}_{lib}.csv")
                    # Quick barplot (top 10 by -log10(p))
                    try:
                        top = res.sort_values("Adjusted P-value", ascending=True).head(10)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(top["Term"], -np.log10(top["Adjusted P-value"]))
                        ax.set_xlabel("-log10(adj p)")
                        ax.set_title(f"{t} {direction} — {lib}")
                        plt.tight_layout()
                        fig.savefig(outdir / f"enrich_{t.replace(' ','_')}_{direction}_{lib}.png", dpi=150)
                        plt.close(fig)
                    except Exception:
                        pass
                except Exception as e:
                    # keep going even if one library fails
                    print(f"[enrich] skip {t} {direction} {lib}: {e}")

# -------------------------- Main driver --------------------------

def run_pipeline(prot_path: Path, meta_path: Path, outdir: Path, max_missing_frac: float = 0.5):
    """
    Main driver:
    - Load meta & proteomics table (supports 2-row headers via load_proteomics_excel).
    - Prefer aggregated-mode (condition_map) over per-sample.
    - Preserve annotation columns in all outputs.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load
    meta = load_metadata(meta_path)
    df   = load_proteomics_excel(prot_path)

    # Column inventory (for quick debugging)
    save_table(pd.DataFrame({"column": df.columns}), outdir / "proteomics_columns.csv")

    # ---- Detect columns
    # Prefer Protein ID -> Entry Name -> Gene -> fallback
    id_col = (
        "Protein ID"  if "Protein ID"  in df.columns else
        "Entry Name"  if "Entry Name"  in df.columns else
        "Gene"        if "Gene"        in df.columns else
        guess_id_column(df)
    )

    sample_cols = detect_sample_columns_from_meta(df, meta)  # may exist; aggregated has priority
    cond_cols   = get_exact_condition_columns(df)            # strictly the headers in USER_COL_CONFIG

    # Keep chosen id if it exists; only synthesize if it's missing
    if id_col not in df.columns:
        id_col = "__row_id__"
        df.insert(0, id_col, np.arange(1, len(df) + 1, dtype=int))

    # ---- Prefer aggregated mode if those columns exist
    if cond_cols:
        mode = "aggregated"

        # Numeric matrix (conditions x proteins), log2 -> median normalization
        X = df[cond_cols].apply(pd.to_numeric, errors="coerce")
        X = median_normalize(log2_transform(X))

        # Missingness by column (conditions)
        miss = (
            X.isna().mean()
             .rename("missing_fraction")
             .reset_index()
             .rename(columns={"index": "column"})
        )
        save_table(miss, outdir / "missingness_by_column.csv")

        # PCA across aggregated columns
        X_for_pca = X.T
        pca_plot(X_for_pca, labels=X_for_pca.index, title="PCA (aggregated columns)")

        # Explicit annotation columns to carry through (pick whichever exist)
        ann_cols = [c for c in [
            "Protein ID","Entry","Entry Name","Gene",
            "Protein","Protein name","Description",
            "Protein Length","Organism","Protein Existence",
            "Protein Probability","Top Peptide Probability"
        ] if c in df.columns]

        # FC with annotations
        fc = aggregated_fold_changes(df, id_col=id_col, cond_cols=cond_cols, ann_cols=ann_cols)
        save_table(fc, outdir / "aggregated_log2FC_by_time.csv")

        # Top lists + annotated workbook
        save_top_tables(fc, outdir=outdir, k=50)
        export_annotated_excel(fc, out_xlsx=outdir / "PD1_TCR_log2FC_annotated_FULL.xlsx")

        # Optional enrichment (requires 'Gene' and gseapy)
        try_enrichment(fc, outdir=outdir)

    # ---- fallback to per-sample mode (kept for completeness)
    elif sample_cols:
        mode = "per-sample"

        X = df[sample_cols].apply(pd.to_numeric, errors="coerce")
        X = median_normalize(log2_transform(X))
        X[id_col] = df[id_col].values

        keep = X.drop(columns=[id_col]).isna().mean(axis=1) <= max_missing_frac
        X = X.loc[keep]

        long = df[[id_col]].join(X.drop(columns=[id_col]))
        long = long.melt(id_vars=[id_col], var_name="sample_id", value_name="intensity")

        m = meta.copy()
        if "sample_id" not in m.columns:
            for c in ["sample", "sampleid", "run", "file", "filename"]:
                if c in m.columns:
                    m = m.rename(columns={c: "sample_id"})
                    break
        long = long.merge(m, on="sample_id", how="left")
        save_table(long.head(1000), outdir / "long_with_meta_preview.csv")

        pivot = long.pivot_table(index="sample_id", columns=id_col, values="intensity", aggfunc="mean")
        pca_plot(pivot, labels=long.drop_duplicates("sample_id")["time"], title="PCA (per-sample)")

    else:
        raise ValueError("Could not find aggregated or per-sample columns.")

    # ---- Run metadata
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "id_col": id_col,
            "prot_path": str(prot_path),
            "meta_path": str(meta_path),
            "n_columns": len(df.columns),
            "n_rows": int(df.shape[0]),
        }, f, ensure_ascii=False, indent=2)

    print(f"Pipeline finished in [{mode}] mode. Outputs written to: {outdir}")


# -------------------------- CLI --------------------------

if __name__ == "__main__":
    # Put data files in the project root or in the 'data/' folder.
    prot_path = Path("tp reordered (1).xlsx") if Path("tp reordered (1).xlsx").exists() else Path("data") / "tp reordered (1).xlsx"
    meta_path = Path("meta_data (1).csv")      if Path("meta_data (1).csv").exists()      else Path("data") / "meta_data (1).csv"
    outdir = Path("outputs")
    run_pipeline(prot_path, meta_path, outdir)
