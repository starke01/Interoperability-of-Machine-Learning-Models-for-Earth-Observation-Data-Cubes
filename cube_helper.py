

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
    idx_gen = sindex.intersection(tile_bounds)
    sindex_list = list(idx_gen)

    if not sindex_list:
        return

    sub = vector_data.iloc[sindex_list]

    fid_v = sub[fid_col].values

    geom_values = sub.geometry.values

    pair_geom = zip(fid_v, geom_values)

    shapes = ((geom, int(fid)) for fid, geom in pair_geom)



    labels = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=tile_transform,
        fill=0,
        dtype="int32",
        all_touched=all_touched,
    )

    if labels.max() == 0:
        return

    tiles = cube1.isel(
        latitude=slice(y0, y1),
        longitude=slice(x0, x1),
    ).compute()

    array_tiles = np.asarray(tiles.data)  

    mask = labels > 0
    if not np.any(mask):
        return

    lab = labels[mask].astype(np.int64)

    unique_lab = np.unique(lab)
    unique_lab.sort()
    dense_idx = np.searchsorted(unique_lab, lab)
    length_u_lab = len(unique_lab)

    # pro Band: Summe und Count je fid aufsummieren
    for band_id, bname in enumerate(bands):
        vals = array_tiles[band_id, :, :][mask]

        nn = ~np.isnan(vals)
        if not np.any(nn):
            continue

        sum_vals = np.bincount(
            dense_idx[nn],
            weights = vals[nn],
            minlength = length_u_lab,
        )
        count = np.bincount(
            dense_idx[nn],
            minlength = length_u_lab,
        )

        for k in range(length_u_lab):
            if count[k] == 0:
                continue
            key = (int(unique_lab[k]), bname)
            sum_acc[key] += float(sum_vals[k])
            count_acc[key] += int(count[k])


