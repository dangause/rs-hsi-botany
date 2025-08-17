import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_avg_spectrum_with_band_importance(
    df: pd.DataFrame,
    species_code: str,
    band_df: pd.DataFrame,
    title_prefix: str = "Average Spectrum + Band Importances",
    derivative_order: int = 0,
    red_alpha_range=(0.02, 0.30),
    red_gamma: float = 0.7,
    respect_masks: bool = True,
    gap_alpha: float = 0.18,                # blue shading transparency
    gap_color: str = "#76a5ff",             # blue shading color
    drop_alpha: float = 0.35,               # dropped-band shading (gray)
    keep_ranges_nm=None,                    # usable ranges; default = EnMAP windows
):
    """
    Average spectra for classes 0/1 with vertical red bands (per-band
    importances). We *only* draw spectra inside `keep_ranges_nm` and
    shade everything outside those windows in light blue.
    """

    if keep_ranges_nm is None:
        # Typical EnMAP L2A usable ranges
        keep_ranges_nm = [(430, 1340), (1450, 1780), (1980, 2450)]

    # ---- X, y ----
    band_cols = [c for c in df.columns if c.startswith("band_")]
    if not band_cols:
        raise ValueError("No band_* columns found in df.")
    X = df[band_cols].to_numpy(dtype=float)

    if species_code not in df.columns:
        raise ValueError(f"Target '{species_code}' not found in df.")
    y = df[species_code].to_numpy(dtype=float)

    keep_rows = ~np.isnan(y)
    X, y = X[keep_rows], y[keep_rows].astype(int)
    n_bands = X.shape[1]

    # ---- Dropped bands via sentinel ----
    bad_band = np.any(X == -32768, axis=0)
    X[X == -32768] = np.nan

    # ---- Importances (median-impute) ----
    rf = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    col_med = np.nanmedian(X, axis=0)
    X_train = np.where(np.isnan(X), col_med, X)
    rf.fit(X_train, y)
    importances = rf.feature_importances_

    # ---- Wavelength centers & edges from band_df ----
    def get_series(name_variants):
        for nv in name_variants:
            for col in band_df.columns:
                if str(col).strip().lower() == nv:
                    return band_df[col].to_numpy()
        return None

    start = get_series(["start wl", "startwl", "start_wl"])
    end   = get_series(["end wl", "endwl", "end_wl"])
    mid   = get_series(["middle wl", "middlewl", "cw (nm)", "cw", "center wl", "centerwl"])
    width = get_series(["fwhm (nm)", "fwhm", "sp.rg.", "spectral range", "sp rg", "sp_rg"])

    if start is not None and end is not None:
        centers = 0.5 * (start + end)
        left_edges = start
        right_edges = end
    elif (mid is not None) and (width is not None):
        centers = mid
        left_edges = mid - width / 2.0
        right_edges = mid + width / 2.0
    elif mid is not None:
        centers = mid
        edges = np.zeros(len(centers) + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        first_gap = centers[1] - centers[0]
        last_gap = centers[-1] - centers[-2]
        edges[0] = centers[0] - first_gap / 2.0
        edges[-1] = centers[-1] + last_gap / 2.0
        left_edges = edges[:-1]
        right_edges = edges[1:]
    else:
        raise ValueError("band_df needs either Start+End WL, or Middle/CW (nm) (±width).")

    centers    = centers[:n_bands]
    left_edges = left_edges[:n_bands]
    right_edges= right_edges[:n_bands]

    # ---- Sort by wavelength ----
    order = np.argsort(centers)
    wl       = centers[order]
    wl_left  = left_edges[order]
    wl_right = right_edges[order]
    imp      = importances[order]
    bad_sorted = bad_band[order]

    # ---- Build 'in_keep' mask from keep_ranges_nm ----
    in_keep = np.zeros_like(wl, dtype=bool)
    for a, b in keep_ranges_nm:
        in_keep |= (wl >= a) & (wl <= b)

    # ---- Class means ----
    mean0 = np.nanmean(X[y == 0], axis=0)[order]
    mean1 = np.nanmean(X[y == 1], axis=0)[order]

    # also treat non-finite means as invalid
    invalid0 = bad_sorted | (~in_keep) | (~np.isfinite(mean0))
    invalid1 = bad_sorted | (~in_keep) | (~np.isfinite(mean1))

    m0 = np.ma.array(mean0, mask=invalid0)
    m1 = np.ma.array(mean1, mask=invalid1)

    # ---- Optional derivative ----
    def masked_derivative(vals_ma, x, n):
        if n <= 0:
            return vals_ma
        out = vals_ma.copy()
        for _ in range(n):
            arr = out.filled(np.nan)
            darr = np.gradient(arr, x)
            out = np.ma.masked_invalid(darr)
            out.mask = np.logical_or(out.mask, vals_ma.mask)
        return out

    m0 = masked_derivative(m0, wl, derivative_order)
    m1 = masked_derivative(m1, wl, derivative_order)

    # ---- Red alpha per band (zero outside keep or masked) ----
    imp_norm = imp / imp.max() if imp.max() > 0 else np.zeros_like(imp)
    a_min, a_max = red_alpha_range
    alpha_red = a_min + (a_max - a_min) * (imp_norm ** red_gamma)
    blocked = (~in_keep) | bad_sorted
    if respect_masks:
        alpha_red = np.where(blocked, 0.0, alpha_red)

    # ---- Helper to make spans from a boolean mask over bands ----
    def spans_from_bool(left, right, flag_true):
        spans = []
        i, n = 0, len(flag_true)
        while i < n:
            if flag_true[i]:
                j = i
                while j + 1 < n and flag_true[j + 1]:
                    j += 1
                spans.append((left[i], right[j]))
                i = j + 1
            else:
                i += 1
        return spans

    # Spans to shade: everything OUTSIDE the keep ranges
    outside_spans = spans_from_bool(wl_left, wl_right, ~in_keep)

    # --------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(12, 5.6))

    # 0) Shade outside-keep regions in light blue
    for L, R in outside_spans:
        ax.axvspan(L, R, color=gap_color, alpha=gap_alpha, zorder=0, linewidth=0)

    # 1) Red importance spans (only inside keep + not dropped)
    for L, R, a, ok in zip(wl_left, wl_right, alpha_red, ~blocked):
        if ok and a > 0:
            ax.axvspan(L, R, color="red", alpha=a, zorder=1, linewidth=0)

    # 2) Dropped/masked bands (gray)
    masked_idx = np.where(bad_sorted & in_keep)[0]
    if masked_idx.size:
        runs = np.split(masked_idx, np.where(np.diff(masked_idx) != 1)[0] + 1)
        for r in runs:
            L = wl_left[r[0]]
            R = wl_right[r[-1]]
            ax.axvspan(L, R, color="gray", alpha=drop_alpha, zorder=2, linewidth=0)

    # Pick consistent colors for all segments
    color0 = "C0"  # default matplotlib first color (blue)
    color1 = "C1"  # default matplotlib second color (orange)

    # 3) Plot spectra in contiguous segments strictly within keep ranges
    def plot_segmented(ax, x, y_ma, label=None, color=None, **kw):
        mask = np.asarray(y_ma.mask, dtype=bool)
        start = None
        first_line = None
        for i in range(len(x)):
            if mask[i]:
                if start is not None:
                    ln, = ax.plot(x[start:i], y_ma.data[start:i], color=color, **kw)
                    if first_line is None and label is not None:
                        ln.set_label(label)
                        first_line = ln
                    start = None
            else:
                if start is None:
                    start = i
        if start is not None:
            ln, = ax.plot(x[start:], y_ma.data[start:], color=color, **kw)
            if first_line is None and label is not None:
                ln.set_label(label)
                first_line = ln
        return first_line

    # Call with fixed colors
    line0 = plot_segmented(ax, wl, m0, label="class 0", color=color0, linewidth=1.4, zorder=3)
    line1 = plot_segmented(ax, wl, m1, label="class 1", color=color1, linewidth=1.4, zorder=3)


    # Y-lims ignoring masked
    vals = np.concatenate([m0.compressed(), m1.compressed()])
    if vals.size:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        pad = 0.03 * (vmax - vmin if vmax > vmin else 1.0)
        ax.set_ylim(vmin - pad, vmax + pad)

    # Labels / legend
    ylab = "Reflectance" if derivative_order == 0 else f"{derivative_order}ᵗʰ derivative"
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{title_prefix} — species {species_code} — derivative: {derivative_order}")
    ax.grid(True, alpha=0.25)

    from matplotlib.patches import Patch
    red_patch  = Patch(facecolor="red",  alpha=0.4, label="band importance (deeper red = higher)")
    gap_patch  = Patch(facecolor=gap_color, alpha=gap_alpha, label="instrument band gaps")
    drop_patch = Patch(facecolor="gray", alpha=drop_alpha, label="dropped bands")
    ax.legend(handles=[line0, line1, red_patch, gap_patch, drop_patch], loc="best")

    plt.tight_layout()
    plt.show()
