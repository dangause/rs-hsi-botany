import os
import time
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape
from pystac_client import Client
from pystac_client.exceptions import APIError
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class EnMAPDownloader:
    def __init__(self, gdf: gpd.GeoDataFrame, limit=10, simplify_tolerance=0.0001):
        """
        Parameters:
        - gdf: GeoDataFrame with plot polygons and 'plot_id' column
        - limit: max number of STAC items to fetch per plot
        - simplify_tolerance: simplify tolerance in degrees to reduce geometry complexity
        """
        self.gdf = gdf.to_crs("EPSG:4326")
        self.limit = limit
        self.simplify_tolerance = simplify_tolerance
        self.catalog = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1/")
        self.collection = "ENMAP_HSI_L2A"
        self.results = {}

    def _get_items_with_retry(self, search, retries=3, delay=2):
        """Try fetching STAC items with retries on API errors."""
        for attempt in range(retries):
            try:
                return list(search.get_items())
            except APIError as e:
                print(f"[Retry {attempt+1}] APIError: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

    def query_images(self):
        """Query all EnMAP STAC items that intersect each plot."""
        print("Starting EnMAP query for all plots...")
        for idx, row in self.gdf.iterrows():
            plot_id = row["plot_id"]
            try:
                geom = mapping(row["geometry"].simplify(self.simplify_tolerance, preserve_topology=True))
                search = self.catalog.search(
                    collections=[self.collection],
                    intersects=geom,
                    limit=self.limit
                )
                items = self._get_items_with_retry(search)
                self.results[plot_id] = items
                print(f"✅ {plot_id}: found {len(items)} items")
            except Exception as e:
                print(f"❌ {plot_id}: Error - {e}")

    def inspect_metadata(self, plot_id):
        """Return list of STAC metadata dicts for a given plot ID."""
        return [item.to_dict() for item in self.results.get(plot_id, [])]

    def list_available_assets(self, plot_id):
        """List all available asset keys for a given plot ID."""
        items = self.results.get(plot_id, [])
        if not items:
            return []
        return list(items[0].assets.keys())

    def filter_results_by_properties(self, plot_id, filters):
        """
        Filters items for a plot by property conditions.

        Parameters:
        - plot_id: str, the plot ID.
        - filters: dict, e.g., {"eo:cloud_cover": lambda v: float(v) < 10}

        Returns:
        - List of filtered STAC Items.
        """
        items = self.results.get(plot_id, [])
        if not items:
            return []

        def item_passes(item):
            props = item.properties
            for key, test_func in filters.items():
                if key not in props:
                    return False
                try:
                    if not test_func(props[key]):
                        return False
                except:
                    return False
            return True

        return [item for item in items if item_passes(item)]

    def summarize_metadata(self, plot_ids=None):
        """
        Generate a summary DataFrame of EnMAP metadata for selected or all plots.

        Parameters:
        - plot_ids: optional list of plot IDs to include

        Returns:
        - pd.DataFrame with columns: plot_id, plot_datetime, item_id, hsi_datetime,
        cloud_cover, tileID, href
        """
        summary = []
        target_plots = plot_ids if plot_ids is not None else self.results.keys()

        for plot_id in target_plots:
            items = self.results.get(plot_id, [])
            # Get plot_datetime from self.gdf
            try:
                plot_row = self.gdf[self.gdf["plot_id"] == plot_id]
                plot_datetime = plot_row.iloc[0]["survey_date"] if not plot_row.empty else None
            except Exception:
                plot_datetime = None

            for item in items:
                props = item.properties
                assets = item.assets
                summary.append({
                    "plot_id": plot_id,
                    "plot_datetime": plot_datetime,
                    "item_id": item.id,
                    "hsi_datetime": props.get("datetime"),
                    "cloud_cover": props.get("eo:cloud_cover"),
                    "tileID": props.get("enmap:tileID"),
                    "href": assets["image"].href if "image" in assets else "N/A"
                })

        return pd.DataFrame(summary)

    def visualize_plot_and_enmap_bounds(self, plot_id, filtered_items=None, basemap=True):
        """
        Plot the OFO plot and overlapping EnMAP scenes with optional basemap.

        Parameters:
        - plot_id: str
        - filtered_items: optional list of filtered STAC items
        - basemap: bool, whether to include a basemap from contextily
        """
        # Select the plot
        plot_geom = self.gdf[self.gdf["plot_id"] == plot_id]
        if plot_geom.empty:
            print(f"No plot found with ID '{plot_id}'")
            return

        # Use filtered or unfiltered EnMAP scenes
        items = filtered_items if filtered_items is not None else self.results.get(plot_id, [])
        if not items:
            print(f"No EnMAP scenes found for plot {plot_id}")
            return

        # Convert STAC geometries to GeoDataFrame
        enmap_polys = [shape(item.geometry) for item in items]
        enmap_gdf = gpd.GeoDataFrame(geometry=enmap_polys, crs="EPSG:4326")

        # Reproject to EPSG:3857 for basemap compatibility
        plot_geom_3857 = plot_geom.to_crs(epsg=3857)
        enmap_gdf_3857 = enmap_gdf.to_crs(epsg=3857)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_geom_3857.plot(ax=ax, edgecolor='green', facecolor='none', linewidth=2)
        enmap_gdf_3857.boundary.plot(ax=ax, edgecolor='red', linewidth=1)

        # Add basemap
        if basemap:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=plot_geom_3857.crs)

        # Custom legend
        plot_patch = mpatches.Patch(facecolor='none', edgecolor='green', label='OFO Plot', linewidth=2)
        scene_patch = mpatches.Patch(facecolor='none', edgecolor='red', label='EnMAP Scenes', linewidth=1)
        ax.legend(handles=[plot_patch, scene_patch])

        plt.title(f"Plot '{plot_id}' and Overlapping EnMAP Scene Boundaries")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()

    def crop_to_plot_and_save(self, plot_id, tiff_path, out_dir="./cropped_tiles"):
        """
        Crop the TIFF to the geometry of the plot and save a new smaller file.

        Parameters:
        - plot_id: str, the plot ID
        - tiff_path: path to the full downloaded EnMAP TIFF
        - out_dir: where to save cropped image
        """
        os.makedirs(out_dir, exist_ok=True)
        plot_geom = self.gdf[self.gdf["plot_id"] == plot_id]
        if plot_geom.empty:
            print(f"❌ No geometry found for plot {plot_id}")
            return

        try:
            with rasterio.open(tiff_path) as src:
                plot_geom_proj = plot_geom.to_crs(src.crs)
                shapes = [mapping(geom) for geom in plot_geom_proj.geometry]
                out_image, out_transform = mask(src, shapes, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

            out_path = tiff_path  # overwrite original file


            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"✅ Cropped and saved: {out_path}")
            return out_path

        except Exception as e:
            print(f"❌ Error cropping {tiff_path} for {plot_id}: {e}")

    def download_item_product(self, plot_id, item, asset_key="image", out_dir="./downloads", username=None, password=None, headless=True, overwrite=False, crop=True):
        """
        Download an EnMAP asset and optionally crop to the plot geometry.

        Parameters:
        - plot_id: plot identifier
        - item: STAC item
        - asset_key: typically 'image'
        - out_dir: download destination
        - username/password: EO-SSO login
        - headless: whether to run Chrome headlessly
        - overwrite: force re-download
        - crop: whether to crop image after download
        """
        if username is None or password is None:
            print("❌ Username and password must be provided.")
            return

        plot_dir = os.path.join(out_dir, plot_id)
        os.makedirs(plot_dir, exist_ok=True)

        try:
            asset = item.assets[asset_key]
            href = asset.href

            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_experimental_option("prefs", {
                "download.default_directory": os.path.abspath(plot_dir),
                "download.prompt_for_download": False,
                "safebrowsing.enabled": True
            })

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            try:
                login_url = f"https://sso.eoc.dlr.de/eoc/auth/login?service={href}"
                driver.get(login_url)

                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "username")))
                driver.find_element(By.ID, "username").send_keys(username)
                password_input = driver.find_element(By.ID, "password")
                password_input.send_keys(password)
                password_input.send_keys(Keys.RETURN)

                print(f"⬇️ Downloading EnMAP image for plot {plot_id}...")

                # Wait for fully downloaded .tif or .tiff to appear
                max_wait = 1000
                waited = 0
                pattern_tif = os.path.join(plot_dir, f"*{item.id}*SPECTRAL_IMAGE_COG.tif")
                pattern_tiff = os.path.join(plot_dir, f"*{item.id}*SPECTRAL_IMAGE_COG.tiff")

                print(f"⏳ Waiting for TIFF file matching ID {item.id} to finish downloading...")

                while waited < max_wait:
                    matches = [
                        f for f in glob.glob(pattern_tif) + glob.glob(pattern_tiff)
                        if not f.endswith('.crdownload')
                    ]
                    if matches:
                        final_path = matches[0]
                        print(f"✅ Found downloaded file: {final_path}")
                        break
                    time.sleep(2)
                    waited += 2
                else:
                    print(f"❌ Timeout: no valid TIFF found for {item.id}")
                    return


                print(f"✅ Download complete: {final_path}")
                if crop:
                    self.crop_to_plot_and_save(plot_id, final_path)

            finally:
                driver.quit()
                print("✅ Selenium driver closed.")

        except Exception as e:
            print(f"❌ Download error for {plot_id}, item {item.id}: {e}")

    def batch_download(self, plot_ids=None, asset_key="image", out_dir="./downloads", username=None, password=None, headless=True, filters=None, max_per_plot=None, overwrite=False, crop=True):
        """
        Batch download EnMAP assets with filtering and smart organization.

        Parameters:
        - plot_ids: list of plot IDs to include (default: all in self.results)
        - asset_key: asset key to download (default: 'image')
        - out_dir: parent download directory
        - username/password: EO-SSO credentials
        - headless: Chrome headless mode
        - filters: e.g. {"eo:cloud_cover": lambda v: float(v) < 20}
        - max_per_plot: max downloads per plot
        - overwrite: whether to overwrite existing files
        - crop: whether to crop TIFF after download
        """
        if username is None or password is None:
            print("❌ Username and password must be provided.")
            return

        if plot_ids is None:
            plot_ids = list(self.results.keys())

        timestamp = time.strftime("%Y%m%d")
        root_dir = os.path.join(out_dir, f"enmap_downloads_{timestamp}")
        os.makedirs(root_dir, exist_ok=True)
        print(f"📁 All files will be saved under: {root_dir}")

        for plot_id in plot_ids:
            print(f"\n📦 Processing plot: {plot_id}")
            items = self.results.get(plot_id, [])
            if not items:
                print(f"⚠️ No items found for plot {plot_id}")
                continue

            if filters:
                items = self.filter_results_by_properties(plot_id, filters)
                print(f"🔍 Filtered to {len(items)} item(s) after applying filters")
            else:
                print(f"🔍 Found {len(items)} item(s)")

            if max_per_plot is not None:
                items = items[:max_per_plot]

            for item in items:
                self.download_item_product(
                    plot_id=plot_id,
                    item=item,
                    asset_key=asset_key,
                    out_dir=root_dir,
                    username=username,
                    password=password,
                    headless=headless,
                    overwrite=overwrite,
                    crop=crop
                )