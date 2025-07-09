import os
import time
import glob
import json
import tempfile
from pathlib import Path
from csv import DictWriter


import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
from pystac_client.exceptions import APIError
import rasterio
from rasterio.mask import mask
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
        self.gdf = gdf.to_crs("EPSG:4326")
        self.limit = limit
        self.simplify_tolerance = simplify_tolerance
        self.catalog = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1/")
        self.collection = "ENMAP_HSI_L2A"
        self.results = {}

    def _get_items_with_retry(self, search, retries=3, delay=2):
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

    def filter_results_by_properties(self, plot_id, filters):
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

    def crop_to_plot_and_save(self, plot_id, tiff_path):
        out_dir = os.path.dirname(tiff_path)
        plot_geom = self.gdf[self.gdf["plot_id"] == plot_id]
        if plot_geom.empty:
            print(f"❌ No geometry found for plot {plot_id}")
            return

        try:
            with rasterio.open(tiff_path) as src:
                plot_geom_proj = plot_geom.to_crs(src.crs)
                shapes = [mapping(geom) for geom in plot_geom_proj.geometry]
                out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

            with rasterio.open(tiff_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"✅ Cropped and saved: {tiff_path}")
            return tiff_path

        except Exception as e:
            print(f"❌ Error cropping {tiff_path} for {plot_id}: {e}")

    def download_item_product(
        self,
        plot_id,
        item,
        asset_key="image",
        out_dir="./downloads",
        username=None,
        password=None,
        headless=True,
        overwrite=False,
        crop=True,
    ):
        if username is None or password is None:
            print("❌ Username and password must be provided.")
            return

        item_dir = Path(out_dir) / plot_id / item.id
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            asset = item.assets[asset_key]
            href = asset.href

            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless=new")

            temp_profile = tempfile.mkdtemp()
            chrome_options.add_argument(f"--user-data-dir={temp_profile}")

            chrome_options.add_experimental_option("prefs", {
                "download.default_directory": str(item_dir.resolve()),
                "download.prompt_for_download": False,
                "safebrowsing.enabled": True,
                "profile.default_content_settings.popups": 0,
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

                print(f"⬇️ Downloading EnMAP image for plot {plot_id} item {item.id}...")

                max_wait = 1000
                waited = 0
                final_path = None
                pattern = str(item_dir / "*SPECTRAL_IMAGE_COG.tif*")

                while waited < max_wait:
                    matches = [
                        f for f in glob.glob(pattern)
                        if not f.endswith('.crdownload')
                    ]
                    if matches:
                        final_path = matches[0]
                        print(f"✅ Found downloaded file: {final_path}")
                        break
                    time.sleep(2)
                    waited += 2
                else:
                    print(f"❌ Timeout: No valid TIFF found for item {item.id}")
                    return

                if crop:
                    self.crop_to_plot_and_save(plot_id, final_path)

                sidecar_path = os.path.splitext(final_path)[0] + ".json"
                with open(sidecar_path, "w") as f:
                    json.dump({
                        "plot_id": plot_id,
                        "item_id": item.id,
                        "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "cloud_cover": item.properties.get("eo:cloud_cover"),
                        "datetime": item.properties.get("datetime"),
                        "href": href,
                        "output_tiff": os.path.basename(final_path),
                        "cropped": crop
                    }, f, indent=2)

            finally:
                driver.quit()
                print("✅ Selenium driver closed.")

        except Exception as e:
            print(f"❌ Download error for {plot_id}, item {item.id}: {e}")

    def batch_download(self, plot_ids=None, asset_key="image", out_dir="./downloads", username=None, password=None, headless=True, filters=None, max_per_plot=None, overwrite=False, crop=True):
        if username is None or password is None:
            print("❌ Username and password must be provided.")
            return

        if plot_ids is None:
            plot_ids = list(self.results.keys())

        timestamp = time.strftime("%Y%m%d")
        root_dir = Path(out_dir) / f"enmap_downloads_{timestamp}"
        root_dir.mkdir(parents=True, exist_ok=True)
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
                    out_dir=str(root_dir),
                    username=username,
                    password=password,
                    headless=headless,
                    crop=crop,
                    overwrite=overwrite
                )

    def generate_summary_csv(self, download_dir, output_csv="enmap_download_summary.csv"):
        summary_rows = []
        download_dir = Path(download_dir)

        for plot_dir in download_dir.iterdir():
            if not plot_dir.is_dir():
                continue
            for item_dir in plot_dir.iterdir():
                if not item_dir.is_dir():
                    continue
                for json_file in item_dir.glob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            metadata = json.load(f)
                            metadata["sidecar_path"] = str(json_file)
                            summary_rows.append(metadata)
                    except Exception as e:
                        print(f"⚠️ Failed to read {json_file}: {e}")

        if not summary_rows:
            print("⚠️ No sidecar metadata found.")
            return

        output_path = Path(download_dir) / output_csv
        with open(output_path, "w", newline="") as f:
            writer = DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"✅ Summary CSV written to: {output_path}")