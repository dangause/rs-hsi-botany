import pandas as pd
import numpy as np
from io import StringIO

def _normalize_band_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lower + strip
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    band_col   = next((c for c in df.columns if c.startswith("nr")), None)
    start_col  = next((c for c in df.columns if "start" in c and "wl" in c), None)
    middle_col = next((c for c in df.columns if ("middle" in c or "center" in c or "centre" in c) and "wl" in c), None)
    end_col    = next((c for c in df.columns if "end" in c and "wl" in c), None)
    sprg_col   = next((c for c in df.columns if c.startswith("sp.rg")), None)

    if not (band_col and middle_col and ((start_col and end_col) or sprg_col)):
        raise ValueError(f"Band meta columns not found. Got columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["BAND #"]  = pd.to_numeric(df[band_col], errors="coerce").astype("Int64")
    out["CW (nm)"] = pd.to_numeric(df[middle_col], errors="coerce")

    if start_col and end_col:
        start = pd.to_numeric(df[start_col], errors="coerce")
        end   = pd.to_numeric(df[end_col], errors="coerce")
        fwhm  = end - start
    else:
        fwhm = pd.to_numeric(df[sprg_col], errors="coerce")

    out["FWHM (nm)"] = fwhm
    out = out.dropna(subset=["BAND #","CW (nm)","FWHM (nm)"]).astype({"BAND #":"int"})
    return out.sort_values("BAND #").drop_duplicates("BAND #").reset_index(drop=True)

def load_enmap_band_meta_txt(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin1"]
    # Try pandas autodetect first (engine='python' lets sep be None -> sniff)
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            if df.shape[1] <= 1:
                raise ValueError("Single column read; wrong delimiter?")
            return _normalize_band_meta_columns(df)
        except Exception:
            pass

    # Try explicit common seps with utf-16 (Excel TSVs often are)
    for enc in encodings:
        for sep in ["\t", ",", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if df.shape[1] <= 1:
                    continue
                return _normalize_band_meta_columns(df)
            except Exception:
                continue

    # Try whitespace (multiple spaces)
    for enc in encodings:
        try:
            df = pd.read_csv(path, delim_whitespace=True, engine="python", encoding=enc)
            if df.shape[1] > 1:
                return _normalize_band_meta_columns(df)
        except Exception:
            pass

    # Last resort: fixed-width fields
    for enc in encodings:
        try:
            df = pd.read_fwf(path, encoding=enc)
            if df.shape[1] > 1:
                return _normalize_band_meta_columns(df)
        except Exception:
            pass

    # If we got here, show a helpful preview
    with open(path, "rb") as f:
        raw_bytes = f.read(1024)
    try:
        preview = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        preview = str(raw_bytes[:200])

    raise EmptyDataError(
        "Could not parse the band meta file. "
        "Check delimiter/encoding. First 1KB preview:\n---\n"
        + preview
        + "\n---"
    )
