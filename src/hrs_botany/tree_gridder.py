from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box, Point
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import ScalarFormatter


class TreeEnmapGridder:
    def __init__(self, trees_gdf, plots_gdf, enmap_scene_root, output_table_root):
        self.trees = trees_gdf
        self.plots = plots_gdf
        self.enmap_scene_root = Path(enmap_scene_root)
        self.output_table_root = Path(output_table_root)

        # Create mapping dictionaries for species info
        self.gbif_to_common = self.trees.drop_duplicates(subset="gbif_taxon_key") \
            .set_index("gbif_taxon_key")["common_name"].to_dict()
        self.gbif_to_scientific = self.trees.drop_duplicates(subset="gbif_taxon_key") \
            .set_index("gbif_taxon_key")["scientific_name"].to_dict()

    def _assign_pixels(self, tiff_path, plot_id):
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            tiff_crs = src.crs
            plot_geom = self.plots[self.plots["plot_id"] == plot_id]
            if plot_geom.empty:
                raise ValueError(f"Plot ID '{plot_id}' not found.")
            trees_subset = self.trees[self.trees["plot_id"] == plot_id]
            if trees_subset.empty:
                return pd.DataFrame(), src
            trees_in_plot = gpd.clip(trees_subset, plot_geom).to_crs(tiff_crs)
            coords = [(geom.x, geom.y) for geom in trees_in_plot.geometry]
            rows_cols = [rowcol(transform, x, y) for x, y in coords]
            trees_in_plot["row"], trees_in_plot["col"] = zip(*rows_cols)
            trees_in_plot["pixel_id"] = trees_in_plot.apply(lambda r: f"{r.row}_{r.col}", axis=1)
            trees_in_plot["plot_id"] = plot_id
            return trees_in_plot, src

    def _generate_pixel_grid(self, src):
        transform = src.transform
        width, height = src.width, src.height
        records = []
        for row in range(height):
            for col in range(width):
                x_min, y_max = transform * (col, row)
                x_max = x_min + transform.a
                y_min = y_max + transform.e
                records.append({
                    "row": row,
                    "col": col,
                    "pixel_id": f"{row}_{col}",
                    "geometry": box(x_min, y_min, x_max, y_max),
                    "centroid": Point((x_min + x_max) / 2, (y_min + y_max) / 2)
                })
        return pd.DataFrame(records)

    def _make_table(self, trees_proj, src, species_col="gbif_taxon_key", binary=False):
        pixel_grid = self._generate_pixel_grid(src)
        pixel_gdf = gpd.GeoDataFrame(pixel_grid, geometry="geometry", crs=src.crs)

        if trees_proj.empty:
            pixel_gdf["valid_pixel"] = False
            return pixel_gdf

        grouped = trees_proj.groupby(["pixel_id", trees_proj[species_col]]).size()
        table = grouped.unstack(fill_value=0).reset_index()

        if binary:
            table.iloc[:, 1:] = (table.iloc[:, 1:] > 0).astype(int)

        full = pixel_gdf.merge(table, on="pixel_id", how="left").fillna(0)
        full.update(full.select_dtypes(include="number").astype(int))

        plot_geom = self.plots[self.plots["plot_id"] == trees_proj.iloc[0]["plot_id"]]
        if not plot_geom.empty:
            plot_geom_proj = plot_geom.to_crs(pixel_gdf.crs)
            fully_inside = full.geometry.apply(lambda g: plot_geom_proj.iloc[0].geometry.contains(g))
            full["valid_pixel"] = fully_inside

            species_cols = [col for col in full.columns if col not in {"row", "col", "pixel_id", "geometry", "centroid", "valid_pixel"}]
            for col in species_cols:
                full.loc[~fully_inside, col] = -1

        return gpd.GeoDataFrame(full, geometry="geometry", crs=src.crs)

    def process_all_scenes(self, mode="presence_absence", overwrite=False):
        assert mode in ["presence_absence", "counts"]
        for plot_dir in self.enmap_scene_root.iterdir():
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
                out_dir = self.output_table_root / plot_id / scene_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{mode}.geojson"
                if out_file.exists() and not overwrite:
                    continue
                try:
                    trees_proj, src = self._assign_pixels(tiff_path, plot_id)
                    result = self._make_table(
                        trees_proj, src,
                        species_col="gbif_taxon_key",
                        binary=(mode == "presence_absence")
                    )
                    result.to_file(out_file, driver="GeoJSON")
                except Exception as e:
                    print(f"Failed for {plot_id} / {scene_dir.name}: {e}")

    def get_species_options(self, plot_id):
        """Return DataFrame of species present in the specified plot."""
        trees_plot = self.trees[self.trees["plot_id"] == plot_id]
        if trees_plot.empty:
            raise ValueError(f"No trees found for plot ID '{plot_id}'")

        gbif_keys = sorted(trees_plot["gbif_taxon_key"].dropna().unique())
        data = []
        for gbif in gbif_keys:
            common = self.gbif_to_common.get(gbif, "Unknown")
            scientific = self.gbif_to_scientific.get(gbif, "Unknown")
            data.append({
                "gbif_taxon_key": gbif,
                "common_name": common,
                "scientific_name": scientific
            })
        return pd.DataFrame(data)

    def plot_species_grid(self, plot_id, species, mode="presence_absence",
                        rgb_image=None, extent=None, max_cols=3, buffer=0.001,
                        figsize=(6, 6), show_stem_points=True,
                        stem_point_color="lightgreen", stem_point_size=4,
                        stem_point_alpha=0.6):
        import matplotlib.ticker as mticker

        plot_dir = self.output_table_root / plot_id
        if not plot_dir.exists():
            raise FileNotFoundError(f"No output directory for plot {plot_id}")
        geojson_paths = sorted(plot_dir.glob(f"**/{mode}.geojson"))
        if not geojson_paths:
            raise FileNotFoundError(f"No {mode}.geojson files found for plot {plot_id}")
        plot_geom = self.plots[self.plots["plot_id"] == plot_id]
        if plot_geom.empty:
            raise ValueError(f"Plot ID '{plot_id}' not found in plots")

        n = len(geojson_paths)
        ncols = min(n, max_cols)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axes = axes.flat if n > 1 else [axes]

        for ax, path in zip(axes, geojson_paths):
            gdf = gpd.read_file(path)
            if species not in gdf.columns:
                print(f"Skipping {path.name} — species '{species}' not found")
                ax.set_axis_off()
                continue

            gdf = gdf.to_crs("EPSG:4326")
            plot_geom_proj = plot_geom.to_crs("EPSG:4326")

            if "valid_pixel" in gdf.columns:
                gdf = gdf[gdf["valid_pixel"]]

            # 🔧 Fix: Recompute centroids after file read
            gdf["centroid"] = gdf.geometry.centroid

            if rgb_image is not None and extent is not None:
                ax.imshow(rgb_image, extent=extent, origin="upper", alpha=1.0, zorder=1)

            if mode == "presence_absence":
                gdf["presence"] = (gdf[species] > 0).astype(int)
                color_map = ListedColormap(["lightgray", "green"])
                norm = Normalize(vmin=0, vmax=1)
                gdf.plot(ax=ax, column="presence", cmap=color_map, norm=norm,
                        edgecolor="gray", linewidth=0.2, alpha=0.8, zorder=2)

            elif mode == "counts":
                max_val = gdf[species].max()
                norm = Normalize(vmin=0, vmax=max(1, max_val))
                cmap = plt.cm.Greens
                gdf.plot(ax=ax, column=species, cmap=cmap, norm=norm,
                        edgecolor="gray", linewidth=0.2, alpha=0.8, zorder=2)

                # Annotate each pixel with count if > 0
                for _, row in gdf.iterrows():
                    val = row[species]
                    if val > 0:
                        x, y = row["centroid"].x, row["centroid"].y
                        ax.text(x, y, str(val), fontsize=6, ha='center', va='center', zorder=3)

            if show_stem_points:
                stem_gdf = self.trees[
                    (self.trees["plot_id"] == plot_id) &
                    (self.trees["gbif_taxon_key"] == species)
                ].copy().to_crs("EPSG:4326")
                stem_gdf.plot(ax=ax, color=stem_point_color,
                            markersize=stem_point_size,
                            alpha=stem_point_alpha, zorder=4)

            plot_geom_proj.boundary.plot(ax=ax, color="red", linewidth=2, zorder=5)

            xmin, ymin, xmax, ymax = plot_geom_proj.total_bounds
            ax.set_xlim(xmin - buffer, xmax + buffer)
            ax.set_ylim(ymin - buffer, ymax + buffer)

            ax.set_title(path.parent.name.replace("ENMAP01-____L2A-", ""), fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            xfmt = ScalarFormatter(useMathText=False, useOffset=False)
            yfmt = ScalarFormatter(useMathText=False, useOffset=False)
            xfmt.set_scientific(False)
            yfmt.set_scientific(False)
            ax.xaxis.set_major_formatter(xfmt)
            ax.yaxis.set_major_formatter(yfmt)

        for ax in axes[n:]:
            ax.set_axis_off()

        if mode == "presence_absence":
            zero_patch = mpatches.Patch(color="lightgray", label="0 stems (inside plot)")
            one_patch = mpatches.Patch(color="green", label=">0 stems (inside plot)")
            fig.legend(handles=[zero_patch, one_patch], title="Presence",
                    loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
        else:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, orientation='horizontal', fraction=0.03, pad=0.05,
                        label='Stem Count')

        fig.suptitle(f"{species} — Plot {plot_id}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        return fig
