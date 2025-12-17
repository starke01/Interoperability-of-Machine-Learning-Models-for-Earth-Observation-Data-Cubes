import numpy as np

from rasterio.enums import Resampling 
from rasterio.features import rasterize
import rasterio.windows as rio_windows
from rasterio.errors import RasterioIOError


def process_tile(cube1, vector_data, fid_col, bands, y0, y1, x0, x1, transform, all_touched, sum_acc, count_acc):
    width = x1 - x0
    height = y1 - y0
    window = rio_windows.Window(col_off = x0, row_off = y0, width = width, height = height)

    tile_transform = rio_windows.transform(window, transform)
    left, bottom, right, top = rio_windows.bounds(window, transform)

    tile_bounds = (left, bottom, right, top)

    sindex = vector_data.sindex
    sindex_hits = sindex.intersection(tile_bounds)
    sindex_list = list(sindex_hits)

    if not sindex_list:
        return

    candidate_features = vector_data.iloc[sindex_list]

    fid_values = candidate_features[fid_col].values
    geom_values = candidate_features.geometry.values

    pair_geom = zip(fid_values, geom_values)
    shapes = ((geom, int(fid)) for fid, geom in pair_geom)

    feature_id_raster = rasterize(shapes = shapes, out_shape = (height, width), transform = tile_transform, fill = 0, dtype = "int32", all_touched = all_touched)

    if feature_id_raster.max() == 0:
        return

    tiles = cube1.isel(
        latitude=slice(y0, y1),
        longitude=slice(x0, x1),
    ).compute()

    array_tiles = np.asarray(tiles.data)  

    mask = feature_id_raster > 0
    if not np.any(mask):
        return

    masked_feature_ids = feature_id_raster[mask].astype(np.int64)
    unique_fids = np.unique(masked_feature_ids)          
    fid_bin_index = np.searchsorted(unique_fids, masked_feature_ids)
    n_fids = len(unique_fids)

    # per band: Sum and count per fid
    for band_id, bname in enumerate(bands):
        masked_band_values = array_tiles[band_id, :, :][mask]

        valid = ~np.isnan(masked_band_values)
        if not np.any(valid):
            continue

        sum_per_fid = np.bincount(fid_bin_index[valid],weights = masked_band_values[valid],minlength = n_fids)
        count = np.bincount(fid_bin_index[valid],minlength = n_fids)

        for k in range(n_fids):
            if count[k] == 0:
                continue
            band__fid_key = (int(unique_fids[k]), bname)
            sum_acc[band__fid_key] += float(sum_per_fid[k])
            count_acc[band__fid_key] += int(count[k])


