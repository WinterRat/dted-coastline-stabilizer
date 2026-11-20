from contextlib import contextmanager
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
from osgeo import gdal
import shapely

from .inputs import footprint, raster, reproject
from .workers import tile_results


WGS84 = "EPSG:4326"
DTED_NODATA = -32767
WARP_NODATA = np.iinfo(np.int32).min
INTERVALS = (120, 1200, 3600)


@contextmanager
def result_folder(parent, prefix):
    parent = Path(parent)
    final = parent / f"{prefix}-{uuid4().hex[:8]}"
    with TemporaryDirectory(prefix=f".{prefix}-", dir=parent) as temporary:
        pending = Path(temporary) / final.name
        pending.mkdir()
        yield pending, final
        pending.rename(final)


def covered_tiles(dem):
    gt = dem.GetGeoTransform()
    inset = 0.5 if dem.GetMetadataItem("AREA_OR_POINT") == "Point" else 0
    area = reproject(footprint(gt, inset, inset, dem.RasterXSize - 2 * inset, dem.RasterYSize - 2 * inset),
                     dem.GetProjection(), WGS84)
    west, south, east, north = (round(value, 9) for value in area.bounds)
    if east - west > 180:
        raise ValueError("Export rasters on each side of the antimeridian in separate runs.")
    return [(lon, lat) for lat in range(max(-90, floor(south)), min(90, ceil(north)))
            for lon in range(max(-180, floor(west)), min(180, ceil(east)))
            if area.intersection(shapely.box(lon, lat, lon + 1, lat + 1)).area > 1e-14]


def tile_grid(lon, lat, level):
    zone = min(abs(lat), abs(lat + 1))
    divisor = 1 if zone < 50 else 2 if zone < 70 else 3 if zone < 75 else 4 if zone < 80 else 6
    rows = INTERVALS[level]
    columns = rows // divisor
    dx, dy = 1 / columns, 1 / rows
    return columns + 1, rows + 1, (lon - dx / 2, dx, 0, lat + 1 + dy / 2, 0, -dy)


def warp_tile(dem, grid, cancel, check, threads=1):
    width, height, gt = grid
    left, dx, _, top, _, negative_dy = gt
    source_nodata = dem.GetRasterBand(1).GetNoDataValue()
    try:
        # With nearest-neighbor sampling we can check the numeric NoData value after
        # warping and let GDAL honor any independent source validity mask.
        warped = gdal.Warp("", dem, format="MEM", dstSRS=WGS84, width=width, height=height,
                           outputBounds=(left, top + height * negative_dy, left + width * dx, top),
                           resampleAlg="near", errorThreshold=0, srcNodata="None",
                           dstNodata=float(WARP_NODATA), outputType=gdal.GDT_Int32,
                           warpOptions=[f"NUM_THREADS={threads}"],
                           callback=lambda *_: not cancel())
    except RuntimeError:
        check()
        raise
    check()
    tile = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Int16)
    tile.SetSpatialRef(warped.GetSpatialRef())
    tile.SetGeoTransform(gt)
    tile.SetMetadataItem("AREA_OR_POINT", "Point")
    band = tile.GetRasterBand(1)
    band.SetNoDataValue(DTED_NODATA)
    with warped:
        for y in range(0, height, 128):
            check()
            values = warped.ReadAsArray(0, y, width, min(128, height - y))
            valid = values != WARP_NODATA
            if source_nodata is not None:
                valid &= values != source_nodata
            if np.any(valid & (values <= DTED_NODATA)):
                raise ValueError("DTED reserves −32767 m for NoData and cannot store −32768 m as an elevation.")
            values[~valid] = DTED_NODATA
            band.WriteArray(values.astype(np.int16), 0, y)
    return tile


def write_tile(driver, tile, path, cancel, check):
    try:
        with driver.CreateCopy(str(path), tile, strict=1, callback=lambda *_: not cancel()):
            pass
    except RuntimeError:
        check()
        raise
    width, height = tile.RasterXSize, tile.RasterYSize
    with raster(path) as saved:
        if (saved.GetDriver().ShortName != "DTED" or (saved.RasterXSize, saved.RasterYSize) != (width, height) or
                not np.allclose(saved.GetGeoTransform(), tile.GetGeoTransform(), rtol=0, atol=1e-9)):
            raise ValueError(f"The exported DTED grid could not be verified for {path.name}.")
        for y in range(0, height, 128):
            check()
            window = (0, y, width, min(128, height - y))
            if not np.array_equal(saved.ReadAsArray(*window), tile.ReadAsArray(*window)):
                raise ValueError(f"The exported elevation samples could not be verified for {path.name}.")


def write_dted_tiles(source, directory, level, cancel, check, progress, workers=1):
    """Resample the validated elevation output to the standard DTED post grid."""
    driver = gdal.GetDriverByName("DTED")
    if driver is None:
        raise ValueError("This QGIS installation has no GDAL DTED driver.")
    with raster(source) as dem:
        tiles = covered_tiles(dem)
    if not tiles:
        raise ValueError("No standard DTED tiles overlap the elevation raster.")
    # Each Level 2 warp can hold a full 3601 by 3601 raster in memory.
    concurrent_tiles = min(len(tiles), workers, 2 if level == 2 else workers)
    warp_threads = max(1, workers // concurrent_tiles)

    def process(coordinates):
        check()
        lon, lat = coordinates
        name = f"{'n' if lat >= 0 else 's'}{abs(lat):02d}_{'e' if lon >= 0 else 'w'}{abs(lon):03d}.dt{level}"
        with raster(source) as dem:
            with warp_tile(dem, tile_grid(lon, lat, level), cancel, check, warp_threads) as tile:
                write_tile(driver, tile, directory / name, cancel, check)
        return name

    names = []
    with tile_results(tiles, process, check, concurrent_tiles) as results:
        for index, (_, name) in enumerate(results, 1):
            names.append(name)
            progress(80 + 19 * index / len(tiles))
    return names
