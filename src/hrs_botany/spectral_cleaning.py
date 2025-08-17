# hrs_botany/spectral_masking.py
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Public constants (EnMAP L2A)
# -----------------------------
ENMAP_L2A_KEEP_RANGES: List[Tuple[float, float]] = [
    (430, 1330),   # VIS–NIR
    (1460, 1780),  # SWIR1
    (1960, 2450),  # SWIR2
]

# -----------------------------
# Config + core masking helpers
# -----------------------------
@dataclass
class SpectralCleanerConfig:
    wl_nm: List[float]                      # per-band center wavelengths (nm), ordered by band index
    keep_ranges_nm: List[Tuple[float, float]]  # wavelength keep windows
    stats_path: Optional[str] = None        # kept for API compatibility; unused in mask-only version

def wavelength_keep_mask(wl_nm: Iterable[float],
                         keep_ranges_nm: Iterable[Tuple[float, float]]) -> np.ndarray:
    """
    Center-based mask: keep band if its center wavelength ∈ any keep range.
    """
    wl = np.asarray(list(wl_nm), dtype=float)
    mask = np.zeros_like(wl, dtype=bool)
    for lo, hi in keep_ranges_nm:
        mask |= (wl >= lo) & (wl <= hi)
    return mask

def passband_keep_mask(centers_nm: np.ndarray,
                       fwhm_nm: np.ndarray,
                       keep_ranges_nm: Iterable[Tuple[float, float]]) -> np.ndarray:
    """
    Passband-aware mask: keep band if [center - FWHM/2, center + FWHM/2] overlaps any keep range.
    """
    centers_nm = np.asarray(centers_nm, dtype=float)
    fwhm_nm = np.asarray(fwhm_nm, dtype=float)
    lo = centers_nm - 0.5 * fwhm_nm
    hi = centers_nm + 0.5 * fwhm_nm
    mask = np.zeros_like(centers_nm, dtype=bool)
    for a, b in keep_ranges_nm:
        mask |= (hi >= a) & (lo <= b)  # interval overlap
    return mask

# -----------------------------
# Cleaner (masking columns only)
# -----------------------------
class SpectralCleaner:
    """
    Column-only masking:
      - preserves ALL rows,
      - keeps ONLY band_* columns whose wavelengths fall inside keep ranges,
      - keeps all non-band metadata columns.
    """
    def __init__(self, cfg: SpectralCleanerConfig):
        self.cfg = cfg
        self.band_cols: List[str] = []
        self.band_keep_mask: Optional[np.ndarray] = None   # can be injected (e.g., passband-aware)

    @staticmethod
    def _collect_band_cols(df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("band_")]
        cols.sort(key=lambda c: int(c.split("_")[1]))
        return cols

    def mask_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        # discover band columns
        if not self.band_cols:
            self.band_cols = self._collect_band_cols(df)

        # build default center-based mask if none provided
        if self.band_keep_mask is None:
            self.band_keep_mask = wavelength_keep_mask(self.cfg.wl_nm, self.cfg.keep_ranges_nm)

        # pick columns
        kept_band_cols = [c for c, keep in zip(self.band_cols, self.band_keep_mask) if keep]
        meta_cols = [c for c in df.columns if c not in self.band_cols]
        return df[meta_cols + kept_band_cols]

    # fit/transform API for convenience
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.mask_bands(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.mask_bands(df)


# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Optional, Sequence, Dict, Tuple, List

# import json
# import numpy as np
# import pandas as pd
# from scipy.signal import savgol_filter


# # ----------------------------
# # Config & small utilities
# # ----------------------------
# @dataclass
# class SpectralCleanConfig:
#     # Required
#     wl_nm: np.ndarray                         # shape (n_bands,), center wavelengths in nm

#     # Band curation (keep these wavelength ranges)
#     keep_ranges_nm: Sequence[tuple] = ((430, 1330), (1460, 1780), (1960, 2450))

#     # Physical reflectance limits
#     min_refl: float = 0.0
#     max_refl: float = 1.2

#     # Despiking (Hampel)
#     hampel_window: int = 5                    # odd integer ≥3
#     hampel_nsig: float = 4.0

#     # Smoothing (Savitzky–Golay); set sg_window=None to disable
#     sg_window: Optional[int] = 7              # odd integer ≥3 or None
#     sg_poly: int = 2

#     # Scene harmonization
#     scene_robust_scale: bool = True           # median/IQR per scene_id
#     groupby_col: str = "scene_id"             # group column for scaling

#     # Optional filters
#     min_coverage_keep: float = 0.0            # extra coverage filter in addition to your P/A logic
#     drop_all_band_nan_rows: bool = True       # drop rows where all kept bands are NaN

#     # Optional QA mask columns: drop row if any of these are True (bool) or 1
#     qa_drop_cols: Optional[Sequence[str]] = None

#     # Column pattern for spectral bands
#     band_prefix: str = "band_"                # band_1, band_2, ...
#     # Persist/IO
#     stats_path: Optional[Path] = None         # where to save/load per-scene stats (JSON)


# def _wavelength_keep_mask(wl_nm: np.ndarray, keep_ranges_nm: Sequence[tuple]) -> np.ndarray:
#     mask = np.zeros_like(wl_nm, dtype=bool)
#     for lo, hi in keep_ranges_nm:
#         mask |= (wl_nm >= lo) & (wl_nm <= hi)
#     return mask


# def _hampel_1d(x: np.ndarray, k: int = 5, nsig: float = 4.0) -> np.ndarray:
#     """Robustly replace spikes with local median using a Hampel filter."""
#     if x.ndim != 1:
#         raise ValueError("x must be 1D")
#     n = x.size
#     y = x.copy()
#     k = int(max(3, k))
#     half = k // 2
#     for i in range(n):
#         lo = max(0, i - half)
#         hi = min(n, i + half + 1)
#         w = x[lo:hi]
#         med = np.nanmedian(w)
#         mad = np.nanmedian(np.abs(w - med))
#         sigma = 1.4826 * mad
#         if np.isfinite(x[i]) and (sigma > 0) and (abs(x[i] - med) > nsig * sigma):
#             y[i] = med
#     return y


# def _robust_scale(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#     """Per-band robust scaling: (x - median) / IQR."""
#     med = np.nanmedian(X, axis=0)
#     q25 = np.nanpercentile(X, 25, axis=0)
#     q75 = np.nanpercentile(X, 75, axis=0)
#     iqr = q75 - q25
#     iqr[iqr == 0] = 1.0
#     Xs = (X - med) / iqr
#     return Xs, (med, iqr)


# # ----------------------------
# # Main cleaner
# # ----------------------------
# class SpectralCleaner:
#     """
#     Separate, reproducible spectral-cleaning step.

#     Workflow:
#       cleaner = SpectralCleaner(cfg)
#       df_clean = cleaner.fit_transform(df)       # training/building
#       df_eval  = cleaner.transform(df_eval)      # applies saved per-scene stats

#     Notes:
#       - Expects spectral bands as columns named like "band_1", "band_2", ...
#       - Requires cfg.wl_nm to match the band order.
#       - Keeps/loads per-scene robust-scaling stats (median/IQR) if cfg.stats_path is set.
#     """

#     def __init__(self, cfg: SpectralCleanConfig):
#         self.cfg = cfg
#         self.band_cols: List[str] = []
#         self.band_keep_mask: Optional[np.ndarray] = None
#         # per-group scaling: {group_value: (median, iqr)}
#         self.scalers_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

#     # ---------- IO for stats ----------
#     def save_stats(self, path: Optional[Path] = None):
#         """
#         Save config + per-group robust scaling stats to JSON.
#         Converts numpy arrays and Paths to JSON-friendly types.
#         """
#         path = Path(path or self.cfg.stats_path or "spectral_clean_stats.json")
#         path.parent.mkdir(parents=True, exist_ok=True)

#         # Make a JSON-safe copy of the config
#         cfg_dict = asdict(self.cfg)

#         # Convert ndarray fields to lists; Paths to str
#         for k, v in list(cfg_dict.items()):
#             if isinstance(v, np.ndarray):
#                 cfg_dict[k] = v.tolist()
#             elif isinstance(v, Path):
#                 cfg_dict[k] = str(v)
#             # also handle lists/tuples that might contain numpy types
#             elif isinstance(v, (list, tuple)):
#                 def _to_py(o):
#                     if isinstance(o, np.ndarray): return o.tolist()
#                     if isinstance(o, (np.floating, np.integer)): return o.item()
#                     if isinstance(o, Path): return str(o)
#                     return o
#                 cfg_dict[k] = [_to_py(o) for o in v]

#         payload = {
#             "config": cfg_dict,
#             "scalers": {
#                 str(k): {
#                     "median": v[0].tolist(),
#                     "iqr":    v[1].tolist(),
#                 }
#                 for k, v in self.scalers_.items()
#             },
#             "band_cols": list(self.band_cols),
#         }

#         with open(path, "w") as f:
#             json.dump(payload, f, indent=2)


#     def load_stats(self, path: Optional[Path] = None):
#         """
#         Load config + per-group robust scaling stats from JSON.
#         Restores numpy arrays and Path where appropriate.
#         """
#         path = Path(path or self.cfg.stats_path or "spectral_clean_stats.json")
#         with open(path) as f:
#             payload = json.load(f)

#         # Restore (or merge) config
#         cfg_loaded = payload.get("config", {})
#         # Put wl_nm back to np.array if present
#         if "wl_nm" in cfg_loaded and not isinstance(cfg_loaded["wl_nm"], np.ndarray):
#             cfg_loaded["wl_nm"] = np.array(cfg_loaded["wl_nm"], dtype=float)
#         # stats_path back to Path (or None)
#         if "stats_path" in cfg_loaded and cfg_loaded["stats_path"]:
#             cfg_loaded["stats_path"] = Path(cfg_loaded["stats_path"])

#         # Optionally, update current cfg with loaded values (keeps any runtime overrides)
#         # If you prefer to REPLACE entirely, do: self.cfg = SpectralCleanConfig(**cfg_loaded)
#         for k, v in cfg_loaded.items():
#             setattr(self.cfg, k, v)

#         # Restore scalers
#         self.scalers_ = {
#             k: (np.array(v["median"], dtype=float), np.array(v["iqr"], dtype=float))
#             for k, v in payload.get("scalers", {}).items()
#         }

#         # Restore band columns (keeps order)
#         self.band_cols = payload.get("band_cols", [])

#     def mask_bands(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Column-only masking:
#         - keep all non-band columns (metadata)
#         - keep only band_* columns whose wavelengths fall inside cfg.keep_ranges_nm
#         Does NOT drop rows or alter values.
#         """
#         # Discover band columns in order
#         if not self.band_cols:
#             self.band_cols = self._collect_band_cols(df)

#         # Ensure we have a wavelength keep mask
#         if self.band_keep_mask is None:
#             self.band_keep_mask = _wavelength_keep_mask(self.cfg.wl_nm, self.cfg.keep_ranges_nm)
#             # If you installed a passband-aware mask elsewhere, that will already be in self.band_keep_mask.

#         # Select columns
#         kept_band_cols = [c for c, keep in zip(self.band_cols, self.band_keep_mask) if keep]
#         meta_cols = [c for c in df.columns if c not in self.band_cols]

#         # Return metadata + kept bands (in that order)
#         return df[meta_cols + kept_band_cols]

#     def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Mask bands only. No row filtering, no scaling, no stats saved.
#         """
#         # Build band list and mask once; then just return masked columns.
#         return self.mask_bands(df)

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Mask bands only (same behavior as fit_transform).
#         """
#         return self.mask_bands(df)



#     # # # ---------- public API ----------
#     # # def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
#     # #     """Clean spectra and fit per-scene scalers (if enabled)."""
#     # #     out, per_group_stats = self._clean_core(df, fit_scalers=True)
#     # #     # store scalers
#     # #     if self.cfg.scene_robust_scale:
#     # #         self.scalers_ = per_group_stats
#     # #     if self.cfg.stats_path:
#     # #         self.save_stats(self.cfg.stats_path)
#     # #     return out

#     # # def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#     # #     """Clean spectra using already-fitted scalers (for eval/test)."""
#     # #     if self.cfg.scene_robust_scale and not self.scalers_:
#     # #         # try to lazy-load if path provided
#     # #         if self.cfg.stats_path and Path(self.cfg.stats_path).exists():
#     # #             self.load_stats(self.cfg.stats_path)
#     # #         else:
#     # #             raise RuntimeError("No scalers available. Call fit_transform() first or load_stats().")

#     # #     out, _ = self._clean_core(df, fit_scalers=False)
#     # #     return out

#     # # ---------- internals ----------
#     # def _collect_band_cols(self, df: pd.DataFrame) -> List[str]:
#     #     cols = [c for c in df.columns if isinstance(c, str) and c.startswith(self.cfg.band_prefix)]
#     #     if not cols:
#     #         raise ValueError(f"No spectral band columns found with prefix '{self.cfg.band_prefix}'")
#     #     # sort by integer suffix to ensure correct order
#     #     def bidx(c: str) -> int:
#     #         try:
#     #             return int(c.replace(self.cfg.band_prefix, ""))
#     #         except Exception:
#     #             return 10**9
#     #     cols = sorted(cols, key=bidx)
#     #     if len(cols) != len(self.cfg.wl_nm):
#     #         raise ValueError(
#     #             f"Band count mismatch: found {len(cols)} band columns, "
#     #             f"but cfg.wl_nm has {len(self.cfg.wl_nm)} wavelengths."
#     #         )
#     #     return cols

#     # def _apply_row_clean(self, s: np.ndarray) -> np.ndarray:
#     #     cfg = self.cfg
#     #     y = s.astype(float).copy()

#     #     # physical range
#     #     y[(y < cfg.min_refl) | (y > cfg.max_refl)] = np.nan
#     #     # drop bands outside keep mask
#     #     y[~self.band_keep_mask] = np.nan
#     #     # despike
#     #     if cfg.hampel_window and cfg.hampel_window >= 3:
#     #         y = _hampel_1d(y, k=cfg.hampel_window, nsig=cfg.hampel_nsig)
#     #     # smoothing with simple gap-filling for filter stability
#     #     if cfg.sg_window and cfg.sg_window >= 3:
#     #         nan_mask = ~np.isfinite(y)
#     #         if (~nan_mask).sum() >= cfg.sg_window:
#     #             x = np.arange(y.size)
#     #             y_fill = y.copy()
#     #             if nan_mask.any():
#     #                 y_fill[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])
#     #             y_smooth = savgol_filter(y_fill, window_length=cfg.sg_window, polyorder=cfg.sg_poly, mode="interp")
#     #             y = np.where(nan_mask, np.nan, y_smooth)
#     #     return y

#     # def _clean_core(self, df: pd.DataFrame, fit_scalers: bool) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
#     #     cfg = self.cfg
#     #     df = df.copy()

#     #     # --- discover band columns & wavelength mask once ---
#     #     if not self.band_cols:
#     #         self.band_cols = self._collect_band_cols(df)
#     #     if self.band_keep_mask is None:
#     #         self.band_keep_mask = _wavelength_keep_mask(cfg.wl_nm, cfg.keep_ranges_nm)

#     #     # --- optional QA-based row drops ---
#     #     if cfg.qa_drop_cols:
#     #         for col in cfg.qa_drop_cols:
#     #             if col in df.columns:
#     #                 before = len(df)
#     #                 bad = df[col].astype(float).fillna(0) != 0
#     #                 df = df[~bad]
#     #                 dropped = before - len(df)
#     #                 if dropped:
#     #                     print(f"▸ Dropped {dropped} rows by QA column '{col}'")

#     #     # --- optional coverage guard ---
#     #     if ("coverage_fraction" in df.columns) and (cfg.min_coverage_keep > 0):
#     #         before = len(df)
#     #         df = df[df["coverage_fraction"] >= float(cfg.min_coverage_keep)]
#     #         dropped = before - len(df)
#     #         if dropped:
#     #             print(f"▸ Dropped {dropped} rows by coverage ≥ {cfg.min_coverage_keep}")

#     #     # --- extract matrix (keeps current row order) ---
#     #     X = df[self.band_cols].to_numpy(dtype=float)

#     #     # --- clean spectra row-by-row ---
#     #     Xc = np.empty_like(X)
#     #     for i in range(X.shape[0]):
#     #         Xc[i] = self._apply_row_clean(X[i])

#     #     # --- optionally drop rows that are all-NaN on kept bands ---
#     #     if cfg.drop_all_band_nan_rows:
#     #         all_nan = np.isnan(Xc[:, self.band_keep_mask]).all(axis=1)
#     #         if all_nan.any():
#     #             keep_idx = ~all_nan
#     #             dropped = int(all_nan.sum())
#     #             df = df.loc[keep_idx].copy()
#     #             Xc = Xc[keep_idx]
#     #             print(f"▸ Dropped {dropped} rows with all NaNs on kept bands")

#     #     # >>> critical alignment fix: reset to positional RangeIndex <<<
#     #     df = df.reset_index(drop=True)
#     #     # Xc rows already correspond to df’s current order; no changes needed to Xc

#     #     # --- per-group robust scaling (median/IQR) ---
#     #     per_group_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
#     #     if cfg.scene_robust_scale:
#     #         if cfg.groupby_col not in df.columns:
#     #             raise ValueError(f"scene_robust_scale=True but groupby_col '{cfg.groupby_col}' not in DataFrame.")
#     #         assert len(df) == Xc.shape[0], f"Row mismatch: df={len(df)} vs Xc={Xc.shape[0]}"

#     #         Xs = np.empty_like(Xc)
#     #         # Iterate sub-dataframes; their .index now gives positional 0..N-1
#     #         for gval, sub in df.groupby(cfg.groupby_col, sort=False):
#     #             idx = sub.index.to_numpy()
#     #             Xi = Xc[idx, :]

#     #             if fit_scalers:
#     #                 Xi_s, stats = _robust_scale(Xi)
#     #                 per_group_stats[str(gval)] = stats
#     #             else:
#     #                 # use stored stats if available; else fall back to on-the-fly robust stats
#     #                 if str(gval) not in self.scalers_:
#     #                     med = np.nanmedian(Xi, axis=0)
#     #                     q25 = np.nanpercentile(Xi, 25, axis=0)
#     #                     q75 = np.nanpercentile(Xi, 75, axis=0)
#     #                     iqr = q75 - q25
#     #                     iqr[iqr == 0] = 1.0
#     #                     self.scalers_[str(gval)] = (med, iqr)
#     #                 med, iqr = self.scalers_[str(gval)]
#     #                 Xi_s = (Xi - med) / iqr

#     #             Xs[idx, :] = Xi_s
#     #     else:
#     #         Xs = Xc

#     #     # --- write back to DataFrame in-place ---
#     #     df_out = df.copy()
#     #     for j, col in enumerate(self.band_cols):
#     #         df_out[col] = Xs[:, j]

#     #     return df_out, per_group_stats

