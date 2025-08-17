from pathlib import Path
from typing import Iterable, Optional, List, Union
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


class EnmapDatasetBuilder:
    """
    Build a modeling table by sampling EnMAP bands at pixel centroids
    from the GeoJSON tables produced by TreeEnmapGridder.

    Parameters
    ----------
    enmap_root : str | Path
        Root directory containing plot/scene folders with *SPECTRAL_IMAGE_COG.tiff + .json.
    pa_root : str | Path
        Root directory containing plot/scene folders with presence_absence.geojson.
    band_indices : Iterable[int] | None
        1-based band indices to extract. None = all bands in the TIFF.
    use_only_valid : bool
        If True, keep only rows where valid_pixel == True. Otherwise keep all.
    min_coverage : float
        Keep only rows with coverage_fraction >= this value (0–1). Ignored if column missing.
    mask_value : Optional[Union[int, float]]
        Any sampled band value equal to this will be replaced with NaN (e.g., -32768).
        Set to None to disable manual masking. May be overridden by TIFF nodata (see below).
    prefer_sample : bool
        If True, use rasterio.sample() (fast). If False, read 1×1 windows (also fast).
    verbose : bool
        Print progress and file lists.

    New convenience options
    -----------------------
    enforce_raster_crs : bool
        If True, reproject GeoJSON to the TIFF CRS when they differ.
    drop_all_species_nan : bool
        If True, drop pixels where *all* species columns are NaN (typical edge slivers).
    target_species : Optional[str]
        If set and present in the GeoJSON, drop rows where that species is NaN
        (i.e., remove partial+0 “thrown out” pixels for the chosen target).
    prefer_tiff_nodata : bool
        If True and the TIFF has a defined nodata value, use it to mask bands.
        If both TIFF nodata and mask_value are set, TIFF nodata takes precedence.
    """

    def __init__(
        self,
        enmap_root: Union[str, Path],
        pa_root: Union[str, Path],
        band_indices: Optional[Iterable[int]] = None,
        use_only_valid: bool = True,
        min_coverage: float = 0.0,
        mask_value: Optional[Union[int, float]] = -32768,
        prefer_sample: bool = True,
        verbose: bool = True,
        *,
        enforce_raster_crs: bool = True,
        drop_all_species_nan: bool = True,
        target_species: Optional[str] = None,
        prefer_tiff_nodata: bool = True,
    ):
        self.enmap_root = Path(enmap_root)
        self.pa_root = Path(pa_root)
        self.band_indices = None if band_indices is None else list(band_indices)
        self.use_only_valid = use_only_valid
        self.min_coverage = float(min_coverage)
        self.mask_value = mask_value
        self.prefer_sample = prefer_sample
        self.verbose = verbose

        self.enforce_raster_crs = enforce_raster_crs
        self.drop_all_species_nan = drop_all_species_nan
        self.target_species = target_species
        self.prefer_tiff_nodata = prefer_tiff_nodata

    # ---------- low-level sampling ----------
    def _sample_bands_at_point(
        self,
        src: rasterio.io.DatasetReader,
        x: float,
        y: float,
        band_indices: Optional[List[int]] = None,
        mask_value: Optional[Union[int, float]] = None,
    ) -> List[float]:
        """Sample selected bands at map coordinate (x,y). Returns list of length = len(band_indices)."""
        idxs = band_indices or list(range(1, src.count + 1))

        if self.prefer_sample:
            # rasterio.sample returns (n_points, n_bands)
            arr = list(src.sample([(x, y)], indexes=idxs))[0]
        else:
            # Convert to row/col, then read 1x1 windows per band
            row, col = src.index(x, y)
            vals = []
            w = Window(col, row, 1, 1)
            for i in idxs:
                v = src.read(i, window=w)[0, 0]
                vals.append(v)
            arr = np.array(vals, dtype=float)

        if mask_value is not None:
            arr = np.where(arr == mask_value, np.nan, arr)

        return arr.tolist()

    def _find_scene_files(self, scene_dir: Path):
        tiff_path = next(
            (f for f in scene_dir.iterdir()
             if "SPECTRAL_IMAGE_COG" in f.name and f.suffix.lower() == ".tiff"),
            None
        )
        json_path = next(
            (f for f in scene_dir.iterdir()
             if "SPECTRAL_IMAGE_COG" in f.name and f.suffix.lower() == ".json"),
            None
        )
        return tiff_path, json_path

    # ---------- main build ----------
    def build(
        self,
        output_path: Optional[Union[str, Path]] = None,
        append_every: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build the dataset. If output_path is provided:
          - If append_every is given, flush in chunks to Parquet (append mode).
          - Otherwise write once at the end.
        Returns the full DataFrame (also when writing to disk).
        """
        rows: List[dict] = []

        if self.verbose:
            print(f"Scanning EnMAP root: {self.enmap_root}")

        plot_dirs = [d for d in self.enmap_root.iterdir() if d.is_dir()]
        for plot_dir in tqdm(plot_dirs, desc="Plots"):
            plot_id = plot_dir.name
            if self.verbose:
                print(f"\nProcessing plot: {plot_id}")

            for scene_dir in sorted([d for d in plot_dir.iterdir() if d.is_dir()]):
                scene_id = scene_dir.name
                tiff_path, json_path = self._find_scene_files(scene_dir)

                if self.verbose:
                    try:
                        names = [f.name for f in scene_dir.iterdir()]
                    except Exception:
                        names = []
                    print(f"  Scene: {scene_id}")
                    print(f"    Files in scene dir: {names}")

                if not (tiff_path and json_path):
                    if self.verbose:
                        print(f"    ⚠️ Missing TIFF or JSON in {scene_dir}")
                    continue

                pa_geojson = self.pa_root / plot_id / scene_id / "presence_absence.geojson"
                if not pa_geojson.exists():
                    if self.verbose:
                        print(f"    ⚠️ Missing presence_absence.geojson: {pa_geojson}")
                    continue

                try:
                    # Load P/A grid
                    gdf = gpd.read_file(pa_geojson)

                    # Filter by valid/coverage if available
                    if self.use_only_valid and "valid_pixel" in gdf.columns:
                        gdf = gdf[gdf["valid_pixel"] == True]

                    if "coverage_fraction" in gdf.columns and self.min_coverage > 0:
                        gdf = gdf[gdf["coverage_fraction"] >= self.min_coverage]

                    if gdf.empty:
                        if self.verbose:
                            print("    ⚠️ No pixels after filtering.")
                        continue

                    # Precompute species columns once (string column names that are all digits)
                    species_cols = [c for c in gdf.columns if isinstance(c, str) and c.isdigit()]

                    # Optional: drop pixels where all species are NaN (edge slivers)
                    if self.drop_all_species_nan and species_cols:
                        before = len(gdf)
                        gdf = gdf[gdf[species_cols].notna().any(axis=1)]
                        if self.verbose and len(gdf) != before:
                            print(f"    ▸ Dropped {before - len(gdf)} all-NaN-species edge pixels")

                    # Optional: if modeling a single target species, remove rows where target is NaN
                    if self.target_species and self.target_species in gdf.columns:
                        before = len(gdf)
                        gdf = gdf[~gdf[self.target_species].isna()]
                        if self.verbose and len(gdf) != before:
                            print(f"    ▸ Dropped {before - len(gdf)} rows with NaN target={self.target_species}")

                    if gdf.empty:
                        if self.verbose:
                            print("    ⚠️ No pixels after species filters.")
                        continue

                    if self.verbose:
                        print(f"    ✅ Loaded {len(gdf)} pixels for sampling")

                    # Open raster once per scene
                    with rasterio.open(tiff_path) as src:
                        # CRS guard (reproject GeoJSON to raster CRS if needed)
                        if self.enforce_raster_crs and gdf.crs and src.crs and gdf.crs != src.crs:
                            gdf = gdf.to_crs(src.crs)
                            if self.verbose:
                                print(f"    ▸ Reprojected GeoJSON to {src.crs}")

                        # Decide band indices once
                        idxs = self.band_indices or list(range(1, src.count + 1))

                        # Decide mask value (prefer TIFF nodata if asked)
                        effective_mask_value = self.mask_value
                        if self.prefer_tiff_nodata and hasattr(src, "nodatavals"):
                            mv = next((v for v in src.nodatavals if v is not None), None)
                            if mv is not None:
                                effective_mask_value = mv
                                if self.verbose:
                                    print(f"    ▸ Using TIFF nodata value for masking: {mv}")

                        # Load meta if available (unused but handy to have)
                        try:
                            with open(json_path) as f:
                                meta = json.load(f)
                        except Exception:
                            meta = None

                        # -------- Option A: iterate with iterrows (safe for digit-named columns) --------
                        for _, r in gdf.iterrows():
                            # geometry is the polygon; centroid in raster CRS
                            c = r.geometry.centroid
                            band_vals = self._sample_bands_at_point(
                                src, c.x, c.y, idxs, mask_value=effective_mask_value
                            )

                            item = {
                                "plot_id": plot_id,
                                "scene_id": scene_id,
                                "pixel_id": r.get("pixel_id"),
                                "row": int(r.get("row")) if pd.notna(r.get("row")) else None,
                                "col": int(r.get("col")) if pd.notna(r.get("col")) else None,
                                # stored centroid from gridder if present (string/WKT or None)
                                "centroid": r.get("centroid"),
                            }

                            # Add bands
                            for i, v in enumerate(band_vals, 1):
                                item[f"band_{i}"] = v

                            # Add species columns (works directly with digit names)
                            for s in species_cols:
                                item[s] = r[s]

                            rows.append(item)

                            # Optional chunked append
                            if output_path and append_every and len(rows) >= append_every:
                                self._flush_rows(rows, output_path)
                                rows = []

                except Exception as e:
                    if self.verbose:
                        print(f"    ❌ Error processing {scene_id}: {e}")
                    continue

        # Finalize
        df = pd.DataFrame(rows)
        if self.verbose:
            print(f"\n✅ Total pixels added: {len(df)}")

        if output_path:
            if append_every:
                # flush any remainder
                if len(rows) > 0:
                    self._flush_rows(rows, output_path)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)

        return df

    def _flush_rows(self, rows: List[dict], output_path: Union[str, Path]):
        """Append a chunk to Parquet (create if not exists)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk = pd.DataFrame(rows)
        if output_path.exists():
            # Append by concatenation & overwrite file (Parquet doesn't append in place)
            existing = pd.read_parquet(output_path)
            pd.concat([existing, chunk], ignore_index=True).to_parquet(output_path, index=False)
        else:
            chunk.to_parquet(output_path, index=False)
        if self.verbose:
            print(f"    💾 Appended {len(chunk)} rows → {output_path}")




# from pathlib import Path
# from typing import Iterable, Optional, List, Union
# import json

# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import rasterio
# from rasterio.windows import Window
# from tqdm import tqdm


# class EnmapDatasetBuilder:
#     """
#     Build a modeling table by sampling EnMAP bands at pixel centroids
#     from the GeoJSON tables produced by TreeEnmapGridder.

#     Parameters
#     ----------
#     enmap_root : str | Path
#         Root directory containing plot/scene folders with *SPECTRAL_IMAGE_COG.tiff + .json.
#     pa_root : str | Path
#         Root directory containing plot/scene folders with presence_absence.geojson.
#     band_indices : Iterable[int] | None
#         1-based band indices to extract. None = all bands in the TIFF.
#     use_only_valid : bool
#         If True, keep only rows where valid_pixel == True. Otherwise keep all.
#     min_coverage : float
#         Keep only rows with coverage_fraction >= this value (0–1). Ignored if column missing.
#     mask_value : Optional[Union[int, float]]
#         Any sampled band value equal to this will be replaced with NaN (e.g., -32768).
#     prefer_sample : bool
#         If True, use rasterio.sample() (fast). If False, read 1×1 windows (also fast).
#     verbose : bool
#         Print progress and file lists.
#     """

#     def __init__(
#         self,
#         enmap_root: Union[str, Path],
#         pa_root: Union[str, Path],
#         band_indices: Optional[Iterable[int]] = None,
#         use_only_valid: bool = True,
#         min_coverage: float = 0.0,
#         mask_value: Optional[Union[int, float]] = -32768,
#         prefer_sample: bool = True,
#         verbose: bool = True,
#     ):
#         self.enmap_root = Path(enmap_root)
#         self.pa_root = Path(pa_root)
#         self.band_indices = None if band_indices is None else list(band_indices)
#         self.use_only_valid = use_only_valid
#         self.min_coverage = float(min_coverage)
#         self.mask_value = mask_value
#         self.prefer_sample = prefer_sample
#         self.verbose = verbose

#     # ---------- low-level sampling ----------
#     def _sample_bands_at_point(
#         self,
#         src: rasterio.io.DatasetReader,
#         x: float,
#         y: float,
#         band_indices: Optional[List[int]] = None,
#     ) -> List[float]:
#         """Sample selected bands at map coordinate (x,y). Returns list of length = len(band_indices)."""
#         # Decide which bands to read
#         idxs = band_indices or list(range(1, src.count + 1))

#         if self.prefer_sample:
#             # rasterio.sample returns (n_points, n_bands)
#             arr = list(src.sample([(x, y)], indexes=idxs))[0]
#         else:
#             # Convert to row/col, then read 1x1 windows per band (still efficient)
#             row, col = src.index(x, y)
#             vals = []
#             w = Window(col, row, 1, 1)
#             for i in idxs:
#                 v = src.read(i, window=w)[0, 0]
#                 vals.append(v)
#             arr = np.array(vals, dtype=float)

#         if self.mask_value is not None:
#             arr = np.where(arr == self.mask_value, np.nan, arr)

#         return arr.tolist()

#     def _find_scene_files(self, scene_dir: Path):
#         tiff_path = next(
#             (f for f in scene_dir.iterdir()
#              if "SPECTRAL_IMAGE_COG" in f.name and f.suffix.lower() == ".tiff"),
#             None
#         )
#         json_path = next(
#             (f for f in scene_dir.iterdir()
#              if "SPECTRAL_IMAGE_COG" in f.name and f.suffix.lower() == ".json"),
#             None
#         )
#         return tiff_path, json_path

#     # ---------- main build ----------
#     def build(
#         self,
#         output_path: Optional[Union[str, Path]] = None,
#         append_every: Optional[int] = None,
#     ) -> pd.DataFrame:
#         """
#         Build the dataset. If output_path is provided:
#           - If append_every is given, flush in chunks to Parquet (append mode).
#           - Otherwise write once at the end.
#         Returns the full DataFrame (also when writing to disk).
#         """
#         rows: List[dict] = []

#         if self.verbose:
#             print(f"Scanning EnMAP root: {self.enmap_root}")

#         plot_dirs = [d for d in self.enmap_root.iterdir() if d.is_dir()]
#         for plot_dir in tqdm(plot_dirs, desc="Plots"):
#             plot_id = plot_dir.name
#             if self.verbose:
#                 print(f"\nProcessing plot: {plot_id}")

#             for scene_dir in sorted([d for d in plot_dir.iterdir() if d.is_dir()]):
#                 scene_id = scene_dir.name
#                 tiff_path, json_path = self._find_scene_files(scene_dir)

#                 if self.verbose:
#                     try:
#                         names = [f.name for f in scene_dir.iterdir()]
#                     except Exception:
#                         names = []
#                     print(f"  Scene: {scene_id}")
#                     print(f"    Files in scene dir: {names}")

#                 if not (tiff_path and json_path):
#                     if self.verbose:
#                         print(f"    ⚠️ Missing TIFF or JSON in {scene_dir}")
#                     continue

#                 pa_geojson = self.pa_root / plot_id / scene_id / "presence_absence.geojson"
#                 if not pa_geojson.exists():
#                     if self.verbose:
#                         print(f"    ⚠️ Missing presence_absence.geojson: {pa_geojson}")
#                     continue

#                 try:
#                     gdf = gpd.read_file(pa_geojson)

#                     # Filter by valid/coverage if available
#                     if self.use_only_valid and "valid_pixel" in gdf.columns:
#                         gdf = gdf[gdf["valid_pixel"] == True]

#                     if "coverage_fraction" in gdf.columns and self.min_coverage > 0:
#                         gdf = gdf[gdf["coverage_fraction"] >= self.min_coverage]

#                     if gdf.empty:
#                         if self.verbose:
#                             print("    ⚠️ No pixels after filtering.")
#                         continue

#                     if self.verbose:
#                         print(f"    ✅ Loaded {len(gdf)} pixels for sampling")

#                     # Open raster once per scene
#                     with rasterio.open(tiff_path) as src:
#                         # Determine bands once
#                         idxs = self.band_indices or list(range(1, src.count + 1))

#                         # Load meta if you want to stash it (optional)
#                         try:
#                             with open(json_path) as f:
#                                 meta = json.load(f)
#                         except Exception:
#                             meta = None

#                         # Iterate pixels
#                         for _, r in gdf.iterrows():
#                             # Geometry centroid in the raster's CRS (GeoJSON is already in raster CRS per your pipeline)
#                             c = r.geometry.centroid
#                             band_vals = self._sample_bands_at_point(src, c.x, c.y, idxs)

#                             item = {
#                                 "plot_id": plot_id,
#                                 "scene_id": scene_id,
#                                 "pixel_id": r.get("pixel_id"),
#                                 "row": int(r.get("row")) if pd.notna(r.get("row")) else None,
#                                 "col": int(r.get("col")) if pd.notna(r.get("col")) else None,
#                                 "centroid": r.get("centroid"),  # keep as-is (string or geometry string)
#                             }

#                             # Add bands
#                             for i, v in enumerate(band_vals, 1):
#                                 item[f"band_{i}"] = v

#                             # Add species columns (numeric column names)
#                             species_cols = [c for c in gdf.columns if isinstance(c, str) and c.isdigit()]
#                             for s in species_cols:
#                                 item[s] = r[s]

#                             rows.append(item)

#                             # Optional chunked append
#                             if output_path and append_every and len(rows) >= append_every:
#                                 self._flush_rows(rows, output_path)
#                                 rows = []

#                 except Exception as e:
#                     if self.verbose:
#                         print(f"    ❌ Error processing {scene_id}: {e}")
#                     continue

#         # Finalize
#         df = pd.DataFrame(rows)
#         if self.verbose:
#             print(f"\n✅ Total pixels added: {len(df)}")

#         if output_path:
#             if append_every:
#                 # flush any remainder
#                 if len(rows) > 0:
#                     self._flush_rows(rows, output_path)
#                 # nothing else to do (file already written in chunks)
#             else:
#                 Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#                 df.to_parquet(output_path, index=False)

#         return df

#     def _flush_rows(self, rows: List[dict], output_path: Union[str, Path]):
#         """Append a chunk to Parquet (create if not exists)."""
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         chunk = pd.DataFrame(rows)
#         if output_path.exists():
#             # Append by concatenation & overwrite file (Parquet doesn't append in place)
#             existing = pd.read_parquet(output_path)
#             pd.concat([existing, chunk], ignore_index=True).to_parquet(output_path, index=False)
#         else:
#             chunk.to_parquet(output_path, index=False)
#         if self.verbose:
#             print(f"    💾 Appended {len(chunk)} rows → {output_path}")

