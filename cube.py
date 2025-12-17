
import os
import gc
import re
import time
import glob
import math
import sys
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from dask.diagnostics import ProgressBar

import stackstac
from pystac_client import Client
import planetary_computer as pc  

from rasterio.enums import Resampling as RResampling
import argparse

from cube_helper import(process_tile)





os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "8")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.json,.xml"
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
ProgressBar().register()


stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
train_data  = "..geojson_from_shp.. "



def parameters():
    result = {
        "min_items_cube": 5,
        "max_items": 50,
        "bad_granule_limit": 3,
        "assets_drop_limit": 8,
        "max_drop_tol": 0.5,
        "rasterize_px": 1024,
        "bbox_i": 0.01,
        "resolution_c": 20,
        "train_year": 2017,
        "cloud_cover": 80,
        "epsg_t": 3857,
        "scl_req": (4, 5, 6),
        "target_column": "class_name",
        "assets_req": ("B02","B03","B04","B05","B06","B07","B08","B11","B12","SCL"),
        "bands_r": ("B02","B03","B04","B05","B06","B07","B08","B11","B12"),
        "incides_compute": ("NDVI",),
        "raster_polygon_touch": False,
    }
    return result



params = parameters()
min_items_cube = params["min_items_cube"]
max_items = params["max_items"]
bad_granule_limit = params["bad_granule_limit"]
assets_drop_limit = params["assets_drop_limit"]
max_drop_tol = params["max_drop_tol"]
rasterize_px = params["rasterize_px"]
bbox_i = params["bbox_i"]
resolution_c = params["resolution_c"]
train_year = params["train_year"]
cloud_cover = params["cloud_cover"]
epsg_t = params["epsg_t"]
scl_req = params["scl_req"]
target_column = params["target_column"]
assets_req = params["assets_req"]
bands_r = params["bands_r"]
incides_compute = params["incides_compute"]
raster_polygon_touch = params["raster_polygon_touch"]












def data_load_bbox(data_vec, source_crs_fallback = 2154, target_crs = 4326):
    print("load data")
    vect = gpd.read_file(data_vec)
    if vect.crs is None:
        vect = vect.set_crs(source_crs_fallback)

    vect_crs = vect.to_crs(target_crs)
    bbox = vect_crs.total_bounds
    bbox_list = bbox.tolist()

    return bbox_list, vect





def add_fid(data):
    data = data.copy()
    if "fid" not in data.columns:
        data_length = len(data)
        data["fid"] = range(1, data_length + 1)

    if data["fid"].duplicated().any():
        duplicated_mask = data["fid"].duplicated(keep=False)
        duplicate_rows = data[duplicated_mask]
        duplicate_sorted = duplicate_rows.sort_values("fid")
        number_duplicated = len(duplicate_sorted)
        error_msg = ("Duplizierte 'fid' in Vektordaten gefunden (n=)"+ str(number_duplicated))

        raise ValueError(error_msg)
    return data



def _pad_bbox(bbox, pad=bbox_i):
    xmin, ymin, xmax, ymax = bbox
    return [xmin - pad, ymin - pad, xmax + pad, ymax + pad]





#stac

def stac_build_ref_cube(items, resolution = resolution_c, epsg = epsg_t, bounds_latlon = None):
    bands_r_list = list(bands_r)
    stac_dict = dict(
        assets = bands_r_list,
        resolution = resolution,
        epsg = epsg,
        dtype = np.float64, 
        rescale = False,
        resampling = RResampling.bilinear,
    )

    if bounds_latlon is not None:
        padded_bbox = _pad_bbox(bounds_latlon, pad=bbox_i)
        stac_dict["bounds_latlon"] = tuple(padded_bbox)
    
    reference_cube = (
        stackstac
        .stack(items, **stac_dict)
        .rename(band = "band", x = "longitude", y = "latitude", time = "time")
    )
    list_ref_cube_= list(reference_cube.coords)
    for coord in list_ref_cube_:
        if coord not in {"band", "time", "latitude", "longitude"}:
            reference_cube = reference_cube.drop_vars(coord)

    reference_cube = reference_cube.astype("float32")
    reference_cube = reference_cube.chunk({"time": 1, "latitude": 2048, "longitude": 2048})
    reference_cube = reference_cube.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    return reference_cube





def stac_scl(items, resolution = resolution_c, epsg = epsg_t, bounds_latlon = None):
    stac_dict = dict(
        assets = ["SCL"],
        resolution = resolution,
        epsg = epsg,
        dtype = np.uint8,
        rescale = False,
        resampling = RResampling.nearest,
        fill_value = np.uint8(255),  
    )

    if bounds_latlon is not None:
        padded_bbox = _pad_bbox(bounds_latlon, pad=bbox_i)
        stac_dict["bounds_latlon"] = tuple(padded_bbox)

    scl_cube = (
        stackstac
        .stack(items, **stac_dict)
        .rename(band="band", x="longitude", y="latitude", time="time")
    )
    coord_scl = list(scl_cube.coords)

    for coord in coord_scl:
        if coord not in {"band", "time", "latitude", "longitude"}:
            scl_cube = scl_cube.drop_vars(coord)

    scl_cube = scl_cube.chunk({"time": 1, "latitude": 2048, "longitude": 2048})
    scl_cube = scl_cube.rio.write_crs(f"EPSG:{epsg}", inplace=True)

    return scl_cube



def stac_it(items, bounds_latlon=None):

    reference_cube = stac_build_ref_cube(items, bounds_latlon=bounds_latlon)
    ref_scl_cube  = stac_scl(items, bounds_latlon=bounds_latlon)

    full_cube = xr.concat([reference_cube, ref_scl_cube], dim="band")
    return full_cube


######


def _is_valid(a, valid_classes):
    return np.isin(a, valid_classes)


def apply_mask_and_ndvi(cube, indices = incides_compute):
    scl = cube.sel(band="SCL").astype("uint8")

    valid_classes = np.array(scl_req, dtype=np.uint8)


    valid = xr.apply_ufunc(_is_valid, scl, dask = "parallelized", output_dtypes = [bool], kwargs = {"valid_classes": valid_classes})
    refl = cube.sel(band=list(bands_r)).where(valid)

    i_arr = []
    if "NDVI" in indices:
        b08 = refl.sel(band="B08")
        b04 = refl.sel(band="B04")

        ndvi_numerator = b08 - b04
        ndvi_denominator = b08 + b04

        eps = 1e-6
        ndvi = xr.where(np.abs(ndvi_denominator) >= eps, ndvi_numerator / ndvi_denominator, np.nan)

        ndvi = ndvi.assign_coords(band = "NDVI")
        ndvi = ndvi.expand_dims("band")
        i_arr.append(ndvi)


    output_ = xr.concat([refl] + i_arr, dim="band") 
    return output_


def get_band_names(cube_with_band_dim):
    if "band" not in cube_with_band_dim.dims:
        raise ValueError("band missing")
    
    band_values = cube_with_band_dim["band"].values
    band_list = band_values.tolist()
    band_names = [str(b) for b in band_list]
    return band_names







def extract_geom_tiled_bincount(cube, vector_data, fid_col, tile_size, all_touched):
   
    if "time" not in cube.dims: 
        raise ValueError("Time dont exist")


    time_dimension = cube["time"].values[0]
    time_step = pd.Timestamp(time_dimension)
    time_vali = pd.to_datetime(time_step)
    
    cube1 = cube.isel(time=0).transpose("band", "latitude", "longitude")



    vector_data = vector_data.to_crs(cube.rio.crs)
    if fid_col not in vector_data.columns:
        raise ValueError(f"'{fid_col}' fehlt in GeoDataFrame")
    
   

    height = int(cube1.sizes["latitude"])
    width = int(cube1.sizes["longitude"])
    transform = cube1.rio.transform()

    pixel_sum_acc  = defaultdict(float)  
    pixel_count_acc  = defaultdict(int)   

    bands = get_band_names(cube1)   

    n_tiles_total = math.ceil(height / tile_size) * math.ceil(width / tile_size)
    tile_counter = 0
    log_every = max(1, n_tiles_total // 20)  

    print("Height" + str(height) + " W=" + str(width) + " tile_px=" + str(tile_size) + " tiles≈" + str(n_tiles_total))


    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)

        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)

            tile_counter += 1
            if (tile_counter % log_every == 0) or (tile_counter == n_tiles_total):
                print(" tiles " + str(tile_counter) + "/" + str(n_tiles_total))

            process_tile(cube1 = cube1, vector_data = vector_data, fid_col = fid_col, bands = bands, y0 = y0, y1 = y1, x0 = x0, x1 = x1, transform = transform, all_touched = all_touched, sum_acc = pixel_sum_acc, count_acc = pixel_count_acc)




    if not pixel_sum_acc:
        print("extract: keine gültigen Pixel innerhalb der Polygone gefunden.")
        return pd.DataFrame({"fid": [], "time": []})

    per_fid = defaultdict(dict)
    for (fid, band), s in pixel_sum_acc.items():
        pixel_count = pixel_count_acc[(fid, band)]
        per_fid[fid][band] = s / pixel_count

    records = []
    for fid, band_map in per_fid.items():
        row = {"fid": int(fid), "time": time_vali}
        row.update(band_map)
        records.append(row)

    result_df = pd.DataFrame.from_records(records)
    ordered = ["fid", "time"]

    for b in bands:
        if b in result_df.columns:
            ordered.append(b)

    print(" extract: result shape=" + str(result_df.shape))
    return result_df[ordered]



error_token_pattern = re.compile(r'(T\d{2}[A-Z]{3}_\d{8}T\d{6})')  # z.B. T30UVU_20170425T112121

def toke_err(err_msg):
    token_match = error_token_pattern.search(err_msg)
    if token_match is not None: 
        return token_match.group(1)
    else:
        return None



def drop_token_seq(items, token):

    keep = []
    drop = []

    for it in items:
        hrefs = []
        for a in it.assets.values():
            if a.href:
                hrefs.append(a.href)
            else:
                hrefs.append("")

        found = False
        for i in hrefs:
            if token in i:
                found = True
                break

        if found:
            drop.append(it)
        else:
            keep.append(it)

    length_drop = len(drop)
    length_keep = len(keep)        
    print("drop_items: token = '" + str(token)+ "'  removed = " + str(length_drop)+ "  keep = " + str(length_keep))
    return keep



def item_date(item):
    date_ = getattr(item, "datetime", None)
    if date_ is not None:
        date_p = pd.to_datetime(date_)
        return date_p
    else:
        return None



def item_date_token(item, token):
    for it in item:
        for i in it.assets.values():
            href = (i.href or "")
            if token in href:
                date_ = item_date(it)
                if date_ is None: 
                    return None
                else: 
                    return date_.date()
      
    return None




def drop_items_date(items, bad_date):
    keep = []
    drop = []
    for i in items:
        date_ = item_date(i)
        if date_ is not None and date_.date() == bad_date:
            drop.append(i)
        else:
            keep.append(i)

    length_drop = len(drop)
    length_keep = len(keep) 
    print("drop_items: date = " + str(bad_date) + "  removed = " + str(length_drop) + "  keep = " + str(length_keep))
    return keep





def cloud_cover_(item):
    properties = getattr(item, "properties", {}) or {}
    value = properties.get("eo:cloud_cover")
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    return 100.0




def item_id(it):
    if hasattr(it, "id") and it.id is not None:
        return str(it.id)
    
    try:
        assets_v = it.assets.values()
        assets_iter = iter(assets_v)
        first_a = next(assets_iter)
        href = first_a.href
        return href
    except Exception:
        return repr(it)



def _ok_date(it, banned_dates):
    date_ = item_date(it)
    if date_ is None:
        return True
    else:
        day = date_.date()
        is_banned = day in banned_dates
        return not is_banned

    




def select_top_cloud_cover(items, min_first=min_items_cube, month_cap=max_items, banned_dates=None):
    banned_dates = banned_dates or set()

    filtered_it = []
    for it in items:
        if _ok_date(it, banned_dates):
            filtered_it.append(it)

    filtered_it.sort(
        key = lambda it: (
            cloud_cover_(it),
            item_date(it) or pd.Timestamp("1970-01-01")
        )
    )

    if not filtered_it:
        print("selection error ")
        return [], []
    

    length_filtered_it = len(filtered_it)
    base_count = min_first
    available = length_filtered_it
    take_n = min(base_count, available)

    base = filtered_it[:take_n]

    length_base = len(base)

    rest = filtered_it[length_base:]
    need = max(0, month_cap - len(base))
    extra = rest[:need]
    length_extra = len(extra)

    selected = base + extra
    pool = rest[need:] 
    length_selected = len(selected)
    length_pool = len(pool)

    print("selection: total=" + str(len(items)) + " filtered=" + str(length_filtered_it) + " take_base=" + str(length_base)
            + " min_first=" + str(min_first) + " +extra=" + str(length_extra) + " → selected=" + str(length_selected) + " pool=" + str(length_pool)
            + " cap=" + str(month_cap))


    return selected, pool






def fill_selection_from_pool(items_c, pool, month_limit, banned_dates):
    banned_dates = banned_dates or set()
    used_item_ids = set()

    for it in items_c:
        item_id_u = item_id(it)
        used_item_ids.add(item_id_u)


    new_current = list(items_c)
    new_pool = []
    added = 0

    for it in pool:
        date_items = item_date(it)
        if date_items is not None and date_items.date() in banned_dates:
            continue
        item_id_u = item_id(it)
        if item_id_u in used_item_ids:
            continue
        if len(new_current) < month_limit:
            new_current.append(it)
            used_item_ids.add(item_id_u)
            added += 1
        else:
            new_pool.append(it)

    if len(new_current) < month_limit:
        for it in pool:
            item_id_u = item_id(it)
            if item_id_u in used_item_ids:
                continue

            date_items = item_date(it)
            if date_items is not None and date_items.date() in banned_dates:
                continue

            new_pool.append(it)

    new_pool_length = len(new_pool)
    new_current_length = len(new_current)
    print(" topup: added = " + str(added) + " now = " + str(new_current_length) + "/" + str(month_limit) + " pool_left = " + str(new_pool_length))

    return new_current, new_pool




def search_items(client, bbox4326, year, month, date_time_r):

    search = client.search(
        collections = ["sentinel-2-l2a"],
        bbox = bbox4326,
        datetime = date_time_r,
        query={"s2:product_type": {"eq": "S2MSI2A"},
               "eo:cloud_cover": {"lt": cloud_cover}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
    )

    
    items = list(search.item_collection())

    assets_item_a = []
    items_month_a = []

    for i in items:
        if set(assets_req).issubset(set(i.assets.keys())):
            assets_item_a.append(i)
    items = assets_item_a

    for i in items:
        date_time = getattr(i, "datetime", None)
        if date_time and date_time.month == month:
            items_month_a.append(i)

    items = items_month_a

    return items




def monthly_median_build(items, bbox4326):

    cube = stac_it(items, bounds_latlon=bbox4326)
    cube = apply_mask_and_ndvi(cube, indices=incides_compute)

    monthly_median = cube.median(dim="time", skipna=True)
    return monthly_median


def handle_persist_403_error(error_msg, attempt, persist_retries, bbox4326, year, month, data_time_range, banned_dates):
    if ("HTTP response code: 403" in error_msg
        or "403" in error_msg
        or "not recognized as being in a supported file format" in error_msg
        or "Read failed" in error_msg):
        if attempt <= persist_retries + 1:
            print(" Re-sign STAC & retry...", flush=True)
            gc.collect()
            client = Client.open(stac_url, modifier=pc.sign_inplace)
            all_items = search_items(client, bbox4326, year, month, data_time_range)
            items, pool = select_top_cloud_cover(all_items, min_first = min_items_cube, month_cap = max_items, banned_dates=banned_dates)
            time.sleep(3)
            return client, items, pool
    return None, None, None




def run_single_month(bbox4326, data_set, year, month, out_dir, persist_retries = 2):
    os.makedirs(out_dir, exist_ok=True)

    month_start = pd.Timestamp(year=year, month=month, day=1)

    next_month  = (month_start + pd.offsets.MonthBegin(1))

    start_str = month_start.strftime('%Y-%m-%d')
    end_str = (next_month - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    data_time_range = start_str + "/" + end_str




    month_str = "{:02d}".format(month)

    print("\n  monat " + month_str + "  date: " + data_time_range + " ")

    client = Client.open(stac_url, modifier=pc.sign_inplace)
    all_items = search_items(client, bbox4326, year, month, data_time_range)
    length_all_items = len(all_items)

    print(" monat " + month_str + ": " + str(length_all_items) + " Items ")

    out_path_month = os.path.join(out_dir, "df_" + str(year) + "_" + month_str + ".parquet")



    if not all_items:
        empty_frame = pd.DataFrame({"fid": [], "time": []})
        empty_frame.to_parquet(out_path_month, index=False, compression="snappy")
        print("Safe empty: " + str(out_path_month), flush=True)
        return out_path_month
    


    banned_dates: set
    banned_dates = set()

    items, pool = select_top_cloud_cover(
        all_items,
        min_first=min_items_cube,
        month_cap=max_items,
        banned_dates=banned_dates
    )




    t0 = time.perf_counter()

    initial_items_n = len(items)
    drops = 0
    attempt = 0
    drops_by_date = defaultdict(int)



    while True:
        attempt += 1
        monthly_median = monthly_median_build(items, bbox4326)

        print("materialize (persist)... ")
        try:
            with ProgressBar():
                monthly_median = monthly_median.persist()
            print(" materialize done.")
            break


        except Exception as e:
            error_message = str(e)
            print("persist failed (attempt " + str(attempt) + "): " + str(type(e).__name__) + ": " + str(error_message))
            client_new, items_new, pool_new = handle_persist_403_error(error_message, attempt, persist_retries , bbox4326, year, month, data_time_range, banned_dates)

            if client_new is not None:
                client = client_new
                items, pool = items_new, pool_new
                continue



            token = toke_err(error_message)


            if token:
                lenght_items = len(items)
                new_items = drop_token_seq(items, token)
                length_new_items = len(new_items)
                dropped_now = lenght_items - length_new_items

                if dropped_now <= 0:
                    raise 

                drops += dropped_now
                items = new_items

                bad_date = item_date_token(items + pool, token)

                if bad_date is not None:
                    drops_by_date[bad_date] += 1
                    if drops_by_date[bad_date] >= bad_granule_limit:
                        banned_dates.add(bad_date)
                        items = drop_items_date(items, bad_date)
                        pool = drop_items_date(pool,  bad_date)

                items, pool = fill_selection_from_pool(items, pool, max_items, banned_dates=banned_dates)

                max_initial_items = max(1, initial_items_n)
                drop_rat = drops / max_initial_items

                too_many = False
                if drops > assets_drop_limit and drop_rat > max_drop_tol:
                    too_many = True
                lenght_items = len(items)
                not_enough_left = (lenght_items < min_items_cube)

                if too_many and not_enough_left:
                    error_message = (
                        "Too many bad assets (" + str(drops) + "/" + str(initial_items_n) + "). Remaining = " + str(lenght_items)
                        + " < min_items_cube = " + str(min_items_cube) + "."
                    )
                    raise RuntimeError(error_message) from e
                length_bannend_dates = len(banned_dates)
                print(
                    "[WARN] continuing with " + str(lenght_items) + "/" + str(initial_items_n) + " selected items " + "(drops total=" + str(drops)
                    + ", banned_days=" + str(length_bannend_dates) + ")")

                time.sleep(1)
                continue

            raise


    time_coord = pd.Timestamp(year=year, month=month, day=1)
    monthly_median = monthly_median.expand_dims(time=[time_coord])

    print("extract (tiled)...")
    with ProgressBar():
        extracted_data_month = extract_geom_tiled_bincount(monthly_median, data_set, fid_col = "fid", tile_size = rasterize_px, all_touched = raster_polygon_touch)

    extracted_data_month.to_parquet(out_path_month, index=False, compression="snappy")
    t1 = time.perf_counter()
    time_t = t1-t0
    print("Safe: " + str(out_path_month) + " time: " + str(round(time_t, 2)) + " s", flush=True)


    del monthly_median, extracted_data_month, items, pool
    gc.collect()
    return out_path_month


def interp_series(value_series, maxgap_fill):
    if value_series.notna().sum() < 2:
        return value_series
    interpolated_series = value_series.interpolate(
        method = "linear",
        limit = maxgap_fill,
        limit_direction = "both",
        limit_area = "inside"
    )
    interpolated_series = interpolated_series.ffill(limit = maxgap_fill)
    interpolated_series = interpolated_series.bfill(limit = maxgap_fill)
    return interpolated_series





def convert_to_wide_format(data_long, target, id_col="fid", include_ndvi = True,
                           band_pattern = r"^(B0?\d{2}|NDVI)$",
                           zero_to_na = True,
                           maxgap_fill = 2, max_total_missing = 4, max_missing_streak = 2,
                           drop_incomplete = True,
                           static_feature_cols = None,
                           band_order = None):
    

    if static_feature_cols is None:
        static_feature_cols = []

    data_long_copy = data_long.copy()
    all_col = list(data_long_copy.columns)

    spectral_band_cols = []
    for i in all_col:
        if re.match(band_pattern, str(i)):
            spectral_band_cols.append(i)


    if include_ndvi and "NDVI" in all_col and "NDVI" not in spectral_band_cols:
        spectral_band_cols.append("NDVI")


    if not spectral_band_cols:
        raise ValueError("No bands found")


    unique_band_cols = []
    visited_bands = set()
    for i in spectral_band_cols:
        if i not in visited_bands:
            visited_bands.add(i)
            unique_band_cols.append(i)

    spectral_band_cols = unique_band_cols

    standard_band_order = list(bands_r) + ["NDVI"]  

    if band_order is None:
        band_order = []
        for band_name in standard_band_order:
            if band_name in spectral_band_cols:
                band_order.append(band_name)

        for band_name in spectral_band_cols:
            if band_name not in band_order:
                band_order.append(band_name)
    else:
        filtered_band_order = []
        for band_name in band_order:
            if band_name in spectral_band_cols:
                filtered_band_order.append(band_name)
        band_order = filtered_band_order

    ordered_band_cols = []
    for band_name in band_order:
        if band_name in spectral_band_cols:
            ordered_band_cols.append(band_name)
    spectral_band_cols = ordered_band_cols

    data_long_copy["time"] = pd.to_datetime(data_long_copy["time"])

    times = sorted(data_long_copy["time"].dropna().unique())

    if zero_to_na:
        data_long_copy[spectral_band_cols] = data_long_copy[spectral_band_cols].replace(0.0, np.nan)

    quality_list = []


    data_sorted = data_long_copy.sort_values(["time"])
    grouped_by_id = data_sorted.groupby(id_col, sort=False)

    for fid, fid_data in grouped_by_id:
        bands_only = fid_data[spectral_band_cols]
        missing_mask = bands_only.isna()
        missing_any_row = missing_mask.any(axis=1)
        missing_aligned = missing_any_row.reindex(fid_data.index)
        miss = missing_aligned.values
        total_missing = int(miss.sum())
        max_missing_run = 0
        current_run_length = 0
        for i in miss:
            if i:
                current_run_length += 1
                max_missing_run = max(max_missing_run, current_run_length)
            else:
                current_run_length = 0
        quality_list.append((fid, total_missing, max_missing_run))

    help = [id_col, "total_missing", "max_missing_run"]
    quality_stats = pd.DataFrame(quality_list, columns= help)
    mask_total_missing = quality_stats["total_missing"] <= max_total_missing
    mask_streak = quality_stats["max_missing_run"] <= max_missing_streak
    mask_keep_fid = mask_total_missing & mask_streak
    quality_stats_filtered = quality_stats[mask_keep_fid]
    keep = quality_stats_filtered[id_col]

    data_long_copy = data_long_copy[data_long_copy[id_col].isin(keep)]

   
    fid_uni = data_long_copy[id_col].unique()
    complete_index = pd.MultiIndex.from_product([fid_uni, times], names=[id_col, "time"])
    data_long_copy = data_long_copy.set_index([id_col, "time"])
    data_long_copy = data_long_copy.reindex(complete_index)
    data_long_copy = data_long_copy.reset_index()


    data_long_copy = data_long_copy.sort_values([id_col, "time"])
    grouped_by_id = data_long_copy.groupby(id_col, sort=False)

    for band_name in spectral_band_cols:
        band_grouped_id = grouped_by_id[band_name]
        interpolated_band = band_grouped_id.transform(
            lambda s: interp_series(s, maxgap_fill)
        )
        data_long_copy[band_name] = interpolated_band


    time_step_map = {}
    for i, current_time in enumerate(times):
        time_step_label = "T" + str(i + 1)
        time_step_map[current_time] = time_step_label


    data_long_copy["time_step"] = data_long_copy["time"].map(time_step_map)

    parts = []
    for band in spectral_band_cols:
        band_wide = data_long_copy.pivot(index=id_col, columns="time_step", values = band)

        new_column_names = []
        for time_step_name in band_wide.columns:
            new_name = band + "_" + str(time_step_name)
            new_column_names.append(new_name)
        band_wide.columns = new_column_names
        parts.append(band_wide)


    wide_data = pd.concat(parts, axis=1).reset_index()

    time_steps = []
    time_number = len(times)
    for index in range(time_number):
        step_number = index + 1
        step_label = "T" + str(step_number)
        time_steps.append(step_label)
    
    feature_order = []
    for current_time_step in time_steps:
        for current_band in spectral_band_cols:
            column_name = current_band + "_" + str(current_time_step)
            feature_order.append(column_name)


    available_feature_columns = []
    for column_name in feature_order:
        if column_name in wide_data.columns:
            available_feature_columns.append(column_name)

    wide_data = wide_data[[id_col] + available_feature_columns]

    for static_feature in static_feature_cols:
        if static_feature in data_long.columns:
            static_feature_per_id = data_long[[id_col, static_feature]].drop_duplicates(subset=id_col)

            if static_feature_per_id.groupby(id_col)[static_feature].nunique().max() > 1:
                subset = data_long[[id_col, static_feature]]
                subset_no_na = subset.dropna()
                grouped_by_id = subset_no_na.groupby(id_col, as_index=False)
                static_feature_per_id = grouped_by_id.agg({static_feature: "first"})

            id_values = static_feature_per_id[id_col]
            static_values = static_feature_per_id[static_feature]
            id_to_static_value = dict(zip(id_values, static_values))
            wide_data[static_feature] = wide_data[id_col].map(id_to_static_value)



    if target in data_long.columns:
        target_id = data_long[[id_col, target]].drop_duplicates(subset=id_col)
        wide_data[target] = wide_data[id_col].map(dict(zip(target_id[id_col], target_id[target])))
    else:
        print("target not recognized")



    meta_col = [id_col, target]
    for static_feature in static_feature_cols:
        if static_feature in wide_data.columns:
            meta_col.append(static_feature)
        
    if drop_incomplete:
        feature_only_columns = []
        for column_name in wide_data.columns:
            if column_name not in meta_col:
                feature_only_columns.append(column_name)
        wide_data = wide_data.dropna(axis=0, subset=feature_only_columns, how="any")

    print("Wide-Format:", wide_data.shape, flush=True)
    return wide_data






def finalize_year(out_dir, data_set, year, target_column, keep_monthly=False):

    paths = sorted(glob.glob(os.path.join(out_dir, "df_" + str(year) + "_*.parquet")))
    

    data_month = []
    for p in paths:
        data_month.append(pd.read_parquet(p))

    data_long = pd.concat(data_month, ignore_index=True)


    all_unique_fid = data_set["fid"].unique()

    times_full = pd.date_range(f"{year}-01-01", periods=12, freq="MS")

    full_fid_time_index = pd.MultiIndex.from_product([all_unique_fid, times_full], names=["fid", "time"])

    if "time" in data_long.columns:
        data_long["time"] = pd.to_datetime(data_long["time"], errors="coerce")

    data_long = data_long.set_index(["fid", "time"])
    data_long = data_long.reindex(full_fid_time_index)
    data_long = data_long.reset_index()

    merge_column = ["fid", target_column]
    if "layer" in data_set.columns:
        merge_column = merge_column + ["layer"]

    #merge
    data_long = data_long.merge(data_set[merge_column], on="fid", how="left")

    
    band_order = ["B02","B03","B04","B05","B06","B07","B08","B11","B12","NDVI"]

    #was necessary at the beginning for some comparisons
    if "layer" in data_set.columns:
        extra_static_cols = ["layer"]
    else:
        extra_static_cols = []

    wide_data = convert_to_wide_format(
        data_long, target=target_column, id_col = "fid",
        include_ndvi=True, zero_to_na = True,
        maxgap_fill = 2, max_total_missing = 4, max_missing_streak = 2,
        drop_incomplete = True,
        static_feature_cols = extra_static_cols,
        band_order=band_order
    )



    feature_patter = re.compile(r'^(B0?\d{2}|NDVI)_T\d+$')


    time_series_feature_cols = []
    for c in wide_data.columns:
        if feature_patter.match(c):
            time_series_feature_cols.append(c)



    wide_data[time_series_feature_cols] = wide_data[time_series_feature_cols].astype("float32")

    if target_column in wide_data.columns:
        wide_data[target_column] = wide_data[target_column].astype("category")


    if "layer" in wide_data.columns:
        wide_data["layer"] = wide_data["layer"].astype("category")

    output_final_data = os.path.join(out_dir, "df_full_train_data_test_" + str(year) + ".parquet")
    wide_data.to_parquet(output_final_data, index=False, compression="snappy")
    print("Final gespeichert: " + str(output_final_data), flush=True)

    if not keep_monthly:
        for p in paths:
            try:
                os.remove(p)
            except Exception as e:
                print("Can´t " + str(p) + " delete: " + str(e))




def run_all_month(vector_path, year, out_dir, keep_monthly=False):

    os.makedirs(out_dir, exist_ok=True)
    bbox4326, data_src = data_load_bbox(vector_path)
    data3857 = add_fid(data_src.to_crs(epsg=epsg_t))

    for m in range(1, 13):
        run_single_month(bbox4326, data3857, year, m, out_dir)

    finalize_year(out_dir, data3857, year, target_column, keep_monthly=keep_monthly)





if __name__ == "__main__":

    epsg_t = parameters()["epsg_t"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default=train_data)
    ap.add_argument("--year", type=int, default=train_year)
    ap.add_argument("--out", default=".")
    ap.add_argument("--keep-monthly", action="store_true")

    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    run_all_month(a.vector, a.year, a.out, keep_monthly = a.keep_monthly)
