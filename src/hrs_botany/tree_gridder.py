import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box, Point
from rasterio.transform import rowcol

class TreeEnmapGridder:
    def __init__(self, trees_gdf, plots_gdf, enmap_root):
        self.trees = trees_gdf
        self.plots = plots_gdf
        self.enmap_root = Path(enmap_root)

    def _assign_pixels(self, tiff_path, plot_id):
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            tiff_crs = src.crs
            width, height = src.width, src.height

            plot_geom = self.plots[self.plots["plot_id"] == plot_id]
            if plot_geom.empty:
                raise ValueError(f"Plot ID '{plot_id}' not found.")

            trees_subset = self.trees[self.trees["plot_id"] == plot_id]
            if trees_subset.empty:
                return pd.DataFrame(), src

            trees_in_plot = gpd.clip(trees_subset, plot_geom)
            trees_proj = trees_in_plot.to_crs(tiff_crs)

            coords = [(geom.x, geom.y) for geom in trees_proj.geometry]
            rows_cols = [rowcol(transform, x, y) for x, y in coords]
            trees_proj["row"], trees_proj["col"] = zip(*rows_cols)
            trees_proj["pixel_id"] = trees_proj.apply(lambda r: f"{r.row}_{r.col}", axis=1)

            return trees_proj, src

    def _generate_pixel_grid(self, src):
        transform = src.transform
        width, height = src.width, src.height

        rows, cols, ids, geoms, centers = [], [], [], [], []
        for row in range(height):
            for col in range(width):
                x_min, y_max = transform * (col, row)
                x_max = x_min + transform.a
                y_min = y_max + transform.e
                geom = box(x_min, y_min, x_max, y_max)
                center = Point((x_min + x_max) / 2, (y_min + y_max) / 2)
                rows.append(row)
                cols.append(col)
                ids.append(f"{row}_{col}")
                geoms.append(geom)
                centers.append(center)

        return pd.DataFrame({
            "row": rows, "col": cols, "pixel_id": ids,
            "geometry": geoms, "centroid": centers
        })

    def _make_table(self, trees_proj, src, species_col="gbif_taxon_key", binary=False):
        pixel_grid = self._generate_pixel_grid(src)

        if trees_proj.empty:
            return gpd.GeoDataFrame(pixel_grid, geometry="geometry", crs=src.crs)

        grouped = trees_proj.groupby(["pixel_id", trees_proj[species_col]]).size()
        table = grouped.unstack(fill_value=0).reset_index()

        if binary:
            species_cols = table.columns.drop("pixel_id")
            table[species_cols] = (table[species_cols] > 0).astype(int)

        full = pixel_grid.merge(table, on="pixel_id", how="left")
        full = full.fillna(0)
        full.update(full.select_dtypes(include='number').astype(int))
        return gpd.GeoDataFrame(full, geometry="geometry", crs=src.crs)

    def process_all_scenes(self, output_root, mode="presence_absence", overwrite=False):
        output_root = Path(output_root)
        assert mode in ["presence_absence", "counts"]

        for plot_dir in self.enmap_root.iterdir():
            if not plot_dir.is_dir():
                continue
            plot_id = plot_dir.name
            for scene_dir in plot_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                tiffs = list(scene_dir.glob("*SPECTRAL_IMAGE_COG.tiff"))
                if not tiffs:
                    continue
                tiff_path = tiffs[0]
                out_dir = output_root / plot_id / scene_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{mode}.geojson"
                if out_file.exists() and not overwrite:
                    continue

                try:
                    trees_proj, src = self._assign_pixels(tiff_path, plot_id)
                    if mode == "presence_absence":
                        result = self._make_table(trees_proj, src, binary=True)
                    else:
                        result = self._make_table(trees_proj, src, binary=False)
                    result.to_file(out_file, driver="GeoJSON")
                except Exception as e:
                    print(f"Failed for {plot_id} / {scene_dir.name}: {e}")
    def plot_species_grid(
        self,
        plot_id,
        geojson_path,
        species,
        ax=None,
        cmap="Greens",
        legend=True,
        edgecolor="none",
        alpha_inside=1.0,
        alpha_outside=0.2,
        boundary_color="black",
        figsize=(8, 8),
    ):
        """
        Plot species grid for a given plot with overlayed plot boundary.

        Parameters:
            plot_id (str): Plot ID to match in self.plots
            geojson_path (str or Path): Path to presence/absence or count GeoJSON
            species (str): Column name of the species to plot
        """
        gdf = gpd.read_file(geojson_path)

        if species not in gdf.columns:
            raise ValueError(f"Species '{species}' not found in {geojson_path}")

        plot_geom = self.plots[self.plots["plot_id"] == plot_id]
        if plot_geom.empty:
            raise ValueError(f"Plot ID '{plot_id}' not found in self.plots")

        # Reproject plot to match grid
        plot_geom = plot_geom.to_crs(gdf.crs)

        # Identify inside/outside pixels
        gdf["inside_plot"] = gdf.centroid.within(plot_geom.iloc[0].geometry)

        inside = gdf[gdf["inside_plot"]]
        outside = gdf[~gdf["inside_plot"]]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot background pixels (grey)
        if not outside.empty:
            outside.plot(ax=ax, color="lightgrey", edgecolor=edgecolor, alpha=alpha_outside)

        # Plot species values inside the plot
        inside.plot(
            ax=ax,
            column=species,
            cmap=cmap,
            legend=legend,
            edgecolor=edgecolor,
            alpha=alpha_inside,
        )

        # Overlay the plot boundary
        plot_geom.boundary.plot(ax=ax, color=boundary_color, linewidth=2)

        ax.set_title(f"{species} — Plot {plot_id}")
        ax.set_axis_off()
        return ax