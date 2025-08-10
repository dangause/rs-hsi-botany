from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List

import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ----------------------------
# Config & small utilities
# ----------------------------
@dataclass
class SpectralCleanConfig:
    # Required
    wl_nm: np.ndarray                         # shape (n_bands,), center wavelengths in nm

    # Band curation (keep these wavelength ranges)
    keep_ranges_nm: Sequence[tuple] = ((430, 1330), (1460, 1780), (1960, 2450))

    # Physical reflectance limits
    min_refl: float = 0.0
    max_refl: float = 1.2

    # Despiking (Hampel)
    hampel_window: int = 5                    # odd integer ≥3
    hampel_nsig: float = 4.0

    # Smoothing (Savitzky–Golay); set sg_window=None to disable
    sg_window: Optional[int] = 7              # odd integer ≥3 or None
    sg_poly: int = 2

    # Scene harmonization
    scene_robust_scale: bool = True           # median/IQR per scene_id
    groupby_col: str = "scene_id"             # group column for scaling

    # Optional filters
    min_coverage_keep: float = 0.0            # extra coverage filter in addition to your P/A logic
    drop_all_band_nan_rows: bool = True       # drop rows where all kept bands are NaN

    # Optional QA mask columns: drop row if any of these are True (bool) or 1
    qa_drop_cols: Optional[Sequence[str]] = None

    # Column pattern for spectral bands
    band_prefix: str = "band_"                # band_1, band_2, ...
    # Persist/IO
    stats_path: Optional[Path] = None         # where to save/load per-scene stats (JSON)


def _wavelength_keep_mask(wl_nm: np.ndarray, keep_ranges_nm: Sequence[tuple]) -> np.ndarray:
    mask = np.zeros_like(wl_nm, dtype=bool)
    for lo, hi in keep_ranges_nm:
        mask |= (wl_nm >= lo) & (wl_nm <= hi)
    return mask


def _hampel_1d(x: np.ndarray, k: int = 5, nsig: float = 4.0) -> np.ndarray:
    """Robustly replace spikes with local median using a Hampel filter."""
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.size
    y = x.copy()
    k = int(max(3, k))
    half = k // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        med = np.nanmedian(w)
        mad = np.nanmedian(np.abs(w - med))
        sigma = 1.4826 * mad
        if np.isfinite(x[i]) and (sigma > 0) and (abs(x[i] - med) > nsig * sigma):
            y[i] = med
    return y


def _robust_scale(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-band robust scaling: (x - median) / IQR."""
    med = np.nanmedian(X, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    Xs = (X - med) / iqr
    return Xs, (med, iqr)


# ----------------------------
# Main cleaner
# ----------------------------
class SpectralCleaner:
    """
    Separate, reproducible spectral-cleaning step.

    Workflow:
      cleaner = SpectralCleaner(cfg)
      df_clean = cleaner.fit_transform(df)       # training/building
      df_eval  = cleaner.transform(df_eval)      # applies saved per-scene stats

    Notes:
      - Expects spectral bands as columns named like "band_1", "band_2", ...
      - Requires cfg.wl_nm to match the band order.
      - Keeps/loads per-scene robust-scaling stats (median/IQR) if cfg.stats_path is set.
    """

    def __init__(self, cfg: SpectralCleanConfig):
        self.cfg = cfg
        self.band_cols: List[str] = []
        self.band_keep_mask: Optional[np.ndarray] = None
        # per-group scaling: {group_value: (median, iqr)}
        self.scalers_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ---------- IO for stats ----------
    def save_stats(self, path: Optional[Path] = None):
        path = Path(path or self.cfg.stats_path or "spectral_clean_stats.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "scalers": {
                str(k): {
                    "median": v[0].tolist(),
                    "iqr": v[1].tolist(),
                } for k, v in self.scalers_.items()
            },
            "band_cols": self.band_cols,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_stats(self, path: Optional[Path] = None):
        path = Path(path or self.cfg.stats_path or "spectral_clean_stats.json")
        with open(path) as f:
            payload = json.load(f)
        self.band_cols = payload["band_cols"]
        self.scalers_ = {
            k: (np.array(v["median"], dtype=float), np.array(v["iqr"], dtype=float))
            for k, v in payload["scalers"].items()
        }

    # ---------- public API ----------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean spectra and fit per-scene scalers (if enabled)."""
        out, per_group_stats = self._clean_core(df, fit_scalers=True)
        # store scalers
        if self.cfg.scene_robust_scale:
            self.scalers_ = per_group_stats
        if self.cfg.stats_path:
            self.save_stats(self.cfg.stats_path)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean spectra using already-fitted scalers (for eval/test)."""
        if self.cfg.scene_robust_scale and not self.scalers_:
            # try to lazy-load if path provided
            if self.cfg.stats_path and Path(self.cfg.stats_path).exists():
                self.load_stats(self.cfg.stats_path)
            else:
                raise RuntimeError("No scalers available. Call fit_transform() first or load_stats().")

        out, _ = self._clean_core(df, fit_scalers=False)
        return out

    # ---------- internals ----------
    def _collect_band_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(self.cfg.band_prefix)]
        if not cols:
            raise ValueError(f"No spectral band columns found with prefix '{self.cfg.band_prefix}'")
        # sort by integer suffix to ensure correct order
        def bidx(c: str) -> int:
            try:
                return int(c.replace(self.cfg.band_prefix, ""))
            except Exception:
                return 10**9
        cols = sorted(cols, key=bidx)
        if len(cols) != len(self.cfg.wl_nm):
            raise ValueError(
                f"Band count mismatch: found {len(cols)} band columns, "
                f"but cfg.wl_nm has {len(self.cfg.wl_nm)} wavelengths."
            )
        return cols

    def _apply_row_clean(self, s: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        y = s.astype(float).copy()

        # physical range
        y[(y < cfg.min_refl) | (y > cfg.max_refl)] = np.nan
        # drop bands outside keep mask
        y[~self.band_keep_mask] = np.nan
        # despike
        if cfg.hampel_window and cfg.hampel_window >= 3:
            y = _hampel_1d(y, k=cfg.hampel_window, nsig=cfg.hampel_nsig)
        # smoothing with simple gap-filling for filter stability
        if cfg.sg_window and cfg.sg_window >= 3:
            nan_mask = ~np.isfinite(y)
            if (~nan_mask).sum() >= cfg.sg_window:
                x = np.arange(y.size)
                y_fill = y.copy()
                if nan_mask.any():
                    y_fill[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])
                y_smooth = savgol_filter(y_fill, window_length=cfg.sg_window, polyorder=cfg.sg_poly, mode="interp")
                y = np.where(nan_mask, np.nan, y_smooth)
        return y

    def _clean_core(self, df: pd.DataFrame, fit_scalers: bool) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        cfg = self.cfg
        df = df.copy()

        # keep band columns & compute wavelength mask once
        if not self.band_cols:
            self.band_cols = self._collect_band_cols(df)
        if self.band_keep_mask is None:
            self.band_keep_mask = _wavelength_keep_mask(cfg.wl_nm, cfg.keep_ranges_nm)

        # optional QA drop
        if cfg.qa_drop_cols:
            for col in cfg.qa_drop_cols:
                if col in df.columns:
                    before = len(df)
                    mask_bad = df[col].astype(float).fillna(0) != 0
                    df = df[~mask_bad]
                    if before != len(df):
                        print(f"▸ Dropped {before - len(df)} rows by QA column '{col}'")

        # optional coverage guard
        if ("coverage_fraction" in df.columns) and (cfg.min_coverage_keep > 0):
            before = len(df)
            df = df[df["coverage_fraction"] >= float(cfg.min_coverage_keep)]
            if before != len(df):
                print(f"▸ Dropped {before - len(df)} rows by coverage ≥ {cfg.min_coverage_keep}")

        # extract matrix
        X = df[self.band_cols].to_numpy(dtype=float)

        # clean each spectrum
        Xc = np.empty_like(X)
        for i in range(X.shape[0]):
            Xc[i] = self._apply_row_clean(X[i])

        # drop rows with all-NaN (on kept bands) if requested
        if cfg.drop_all_band_nan_rows:
            all_nan = np.isnan(Xc[:, self.band_keep_mask]).all(axis=1)
            if all_nan.any():
                keep_idx = ~all_nan
                dropped = int(all_nan.sum())
                df = df.loc[keep_idx].copy()
                Xc = Xc[keep_idx]
                print(f"▸ Dropped {dropped} rows with all NaNs on kept bands")

        # per-group robust scaling
        per_group_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if cfg.scene_robust_scale:
            if cfg.groupby_col not in df.columns:
                raise ValueError(f"scene_robust_scale=True but groupby_col '{cfg.groupby_col}' not in DataFrame.")
            # iterate groups
            Xs = np.empty_like(Xc)
            for gval, idx in df.groupby(cfg.groupby_col).groups.items():
                idx = np.array(list(idx))
                Xi = Xc[idx, :]
                if fit_scalers:
                    Xi_s, stats = _robust_scale(Xi)
                    per_group_stats[str(gval)] = stats
                else:
                    if str(gval) not in self.scalers_:
                        # If a new group appears at transform time, fall back to identity
                        med = np.nanmedian(Xi, axis=0)
                        q25 = np.nanpercentile(Xi, 25, axis=0)
                        q75 = np.nanpercentile(Xi, 75, axis=0)
                        iqr = q75 - q25
                        iqr[iqr == 0] = 1.0
                        self.scalers_[str(gval)] = (med, iqr)
                    med, iqr = self.scalers_[str(gval)]
                    Xi_s = (Xi - med) / iqr
                Xs[idx, :] = Xi_s
        else:
            Xs = Xc

        # write back
        df_out = df.copy()
        for j, col in enumerate(self.band_cols):
            df_out[col] = Xs[:, j]

        # optional: attach a quick audit summary (counts per group) in memory
        return df_out, per_group_stats
