"""Generate an editable sea area from RGB satellite imagery."""

from math import ceil, floor, sqrt
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import binary_dilation, label, uniform_filter
import shapely

from .coast import parts
from .dted import result_folder
from .engine import checked, vector_layer
from .inputs import footprint, imagery_vrt, raster, reference, reproject, rgb_bands


MAX_ANALYSIS_PIXELS = 4_000_000
MAX_TILE_PIXELS = 4 * MAX_ANALYSIS_PIXELS
REVIEW_MIXED = "Possible tidal flat or estuary: low contrast"
REVIEW_GAP = "Imagery gap near the coastline"


def open_imagery(sources, folder, cancel, progress):
    checked(cancel)
    folder.mkdir(parents=True, exist_ok=True)
    mosaic, _ = imagery_vrt(sources, folder / "mosaic.vrt", prefer_native=True)
    checked(cancel)
    progress(100)
    return mosaic


def pixel_window(ds, area, pad):
    inverse = gdal.InvGeoTransform(ds.GetGeoTransform())
    pixels = shapely.affinity.affine_transform(area, [inverse[1], inverse[2], inverse[4],
                                                      inverse[5], inverse[0], inverse[3]])
    left, top, right, bottom = pixels.bounds
    x0, y0 = max(0, floor(left) - pad), max(0, floor(top) - pad)
    x1 = min(ds.RasterXSize, ceil(right) + pad)
    y1 = min(ds.RasterYSize, ceil(bottom) + pad)
    return x0, y0, x1 - x0, y1 - y0


def window_grid(ds, x, y):
    gt = ds.GetGeoTransform()
    left, top = gdal.ApplyGeoTransform(gt, x, y)
    return (left, gt[1], gt[2], top, gt[4], gt[5])


def fit_grid(gt, width, height, max_pixels):
    if width * height <= max_pixels:
        return gt, width, height
    scale = sqrt(width * height / max_pixels)
    cols = max(1, min(max_pixels, floor(width / scale)))
    rows = max(1, min(max_pixels // cols, floor(height / scale)))
    x_scale, y_scale = width / cols, height / rows
    # Keep all four corners fixed, including for rotated source imagery.
    grid = (gt[0], gt[1] * x_scale, gt[2] * y_scale, gt[3], gt[4] * x_scale, gt[5] * y_scale)
    return grid, cols, rows


def memory_raster(width, height, count, kind, gt, crs):
    ds = gdal.GetDriverByName("MEM").Create("", width, height, count, kind)
    ds.SetGeoTransform(gt)
    ds.SetProjection(crs)
    return ds


def paint_mask(area, gt, crs, width, height):
    with (memory_raster(width, height, 1, gdal.GDT_Byte, gt, crs) as mask,
          ogr.GetDriverByName("MEM").CreateDataSource("") as vectors):
        vector_layer(vectors, reference(crs), "area", ogr.wkbPolygon, [],
                     ((polygon, ()) for polygon in parts(area, {"Polygon"})))
        gdal.RasterizeLayer(mask, [1], vectors.GetLayer(), burn_values=[1])
        return mask.ReadAsArray().astype(bool)


def valid_tile(ds, window, cancel):
    x, y, width, height = window
    gt, width, height = fit_grid(window_grid(ds, x, y), width, height, MAX_TILE_PIXELS)
    read = dict(buf_xsize=width, buf_ysize=height, callback=lambda *_: not cancel())
    indices = rgb_bands(ds)
    bands = [ds.GetRasterBand(index) for index in indices]
    # A single read lets tiled RGB GeoTIFFs decode all channels together.
    if len({band.DataType for band in bands}) == 1:
        rgb = ds.ReadAsArray(*window, band_list=indices, **read)
    else:
        rgb = [band.ReadAsArray(*window, **read) for band in bands]
    checked(cancel)
    valid = np.ones((height, width), dtype=bool)
    for band, values in zip(bands, rgb):
        valid &= np.isfinite(values)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= values != nodata
        if not (band.GetMaskFlags() & gdal.GMF_ALL_VALID):
            valid &= band.GetMaskBand().ReadAsArray(*window, **read) > 0
    for index in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(index)
        if band.GetColorInterpretation() == gdal.GCI_AlphaBand:
            valid &= band.ReadAsArray(*window, **read) > 0
    checked(cancel)
    tile = memory_raster(width, height, 4, gdal.GDT_Float32, gt, ds.GetProjection())
    for index, values in enumerate(rgb, 1):
        tile.GetRasterBand(index).WriteArray(np.where(valid, values, 0))
    tile.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    tile.GetRasterBand(4).WriteArray(valid.astype(np.uint8) * 255)
    return tile


def mosaic_window(sources, gt, crs, width, height, cancel, progress):
    extent = footprint(gt, 0, 0, width, height)
    bounds = extent.bounds
    rgb = np.zeros((3, height, width), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    for index, source in enumerate(sources):
        checked(cancel)
        with raster(source) as ds:
            # Transform the padded analysis extent back to each source before reading pixels.
            local = reproject(extent, crs, ds.GetProjection())
            window = pixel_window(ds, local, 2)
            if min(window[2:]) <= 0:
                continue
            try:
                with valid_tile(ds, window, cancel) as tile, gdal.Warp(
                    "", tile, format="MEM", outputBounds=bounds, width=width, height=height, dstSRS=crs,
                    srcAlpha=True, dstAlpha=True, resampleAlg="near", callback=lambda *_: not cancel(),
                ) as projected:
                    data = projected.ReadAsArray()
                    valid = data[3] > 0
                    # In an overlap, only valid pixels in a later tile can replace earlier pixels.
                    rgb[:, valid] = data[:3, valid]
                    covered |= valid
            except RuntimeError:
                checked(cancel)
                raise
        progress(10 + 20 * (index + 1) / len(sources))
    return rgb, covered


def image_features(rgb, valid):
    red, green, blue = rgb
    brightness = rgb.mean(axis=0)
    total = np.maximum(np.abs(rgb).sum(axis=0), 1)
    weight = np.maximum(uniform_filter(valid.astype(np.float32), size=5), 1e-6)
    light = np.where(valid, brightness, 0)
    mean = uniform_filter(light, size=5) / weight
    variance = uniform_filter(light * light, size=5) / weight - mean * mean
    return np.stack(((red - green) / total, (blue - green) / total, brightness,
                     np.sqrt(np.maximum(variance, 0))))


def sample_pixels(points, gt, valid, name):
    inverse = gdal.InvGeoTransform(gt)
    height, width = valid.shape
    positions = []
    for x, y in points:
        col, row = map(floor, gdal.ApplyGeoTransform(inverse, x, y))
        if not (0 <= row < height and 0 <= col < width and valid[row, col]):
            raise ValueError(f"Place {name} samples on valid imagery inside the area of interest.")
        positions.append((row, col))
    return positions


def classify_rgb(features, valid, land, offshore, cancel):
    if not land or not offshore or set(land).intersection(offshore):
        raise ValueError("Place land and offshore samples at different locations.")
    height, width = valid.shape
    prototypes = []
    sample_groups = []
    for samples in (land, offshore):
        group = []
        for row, col in samples:
            rows = slice(max(0, row - 1), min(height, row + 2))
            cols = slice(max(0, col - 1), min(width, col + 2))
            group.append(np.median(features[:, rows, cols][:, valid[rows, cols]], axis=1))
        sample_groups.append(group)
        prototypes.extend(group)
    centers = np.asarray(prototypes)
    subsample = valid[::4, ::4]
    scene = features[:, ::4, ::4][:, subsample] if subsample.any() else features[:, valid]
    low, high = np.percentile(scene, [10, 90], axis=1)
    scale = np.maximum.reduce((high - low, np.ptp(centers, axis=0),
                               np.array([0.025, 0.025, 1.0, 1.0]))).astype(np.float32)
    if np.max(np.ptp(centers, axis=0) / scale) < 0.05:
        raise ValueError("The land and offshore samples have similar colors and textures. Choose more distinct samples.")
    features /= scale[:, None, None]
    distances = []
    for samples in sample_groups:
        nearest = np.full(valid.shape, np.inf, dtype=np.float32)
        for sample in samples:
            checked(cancel)
            distance = np.zeros_like(nearest)
            for channel, value in zip(features, sample / scale):
                distance += (channel - value) ** 2
            np.minimum(nearest, distance, out=nearest)
        distances.append(nearest)
    dry, wet = distances
    water = (wet < dry) & valid
    for row, col in land:
        water[row, col] = False
    for row, col in offshore:
        water[row, col] = True
    # Four-neighbor connectivity requires a shared pixel edge.
    components, _ = label(water)
    seeds = {int(components[row, col]) for row, col in offshore}
    sea = np.isin(components, list(seeds))
    confidence = np.abs(dry - wet) / np.maximum(dry + wet, 1e-6)
    return sea, water, confidence


def polygon_mask(values, gt, crs, cancel):
    height, width = values.shape
    with (memory_raster(width, height, 1, gdal.GDT_Byte, gt, crs) as ds,
          ogr.GetDriverByName("MEM").CreateDataSource("") as vectors):
        band = ds.GetRasterBand(1)
        band.WriteArray(values.astype(np.uint8))
        layer = vectors.CreateLayer("areas", reference(crs), ogr.wkbPolygon)
        try:
            gdal.Polygonize(band, band, layer, -1, callback=lambda *_: not cancel())
        except RuntimeError:
            checked(cancel)
            raise
        polygons = []
        for feature in layer:
            checked(cancel)
            geometry = shapely.from_wkb(bytes(feature.GetGeometryRef().ExportToWkb()))
            polygons.extend(parts(shapely.make_valid(geometry), {"Polygon"}))
        return shapely.union_all(polygons)


def coastlines(sea, coverage):
    if sea.is_empty:
        return []
    # Every remaining edge separates sea from valid pixels on the land side.
    shared = sea.boundary.difference(coverage.boundary)
    return list(parts(shapely.line_merge(shared), {"LineString"}))


def generate_shoreline(sources, mosaic, area, land, offshore, draft, cancel, progress):
    checked(cancel)
    with raster(mosaic) as image:
        x, y, width, height = pixel_window(image, area, 3)
        if min(width, height) <= 0:
            raise ValueError("Draw the area of interest over the imagery.")
        gt, width, height = fit_grid(window_grid(image, x, y), width, height, MAX_ANALYSIS_PIXELS)
        crs = image.GetProjection()
    progress(8)
    rgb, covered = mosaic_window(sources, gt, crs, width, height, cancel, progress)
    checked(cancel)
    in_area = paint_mask(area, gt, crs, width, height)
    valid = covered & in_area
    land = sample_pixels(land, gt, valid, "land")
    offshore = sample_pixels(offshore, gt, valid, "offshore")
    sea, water, confidence = classify_rgb(image_features(rgb, covered), valid, land, offshore, cancel)
    if not np.any(binary_dilation(sea) & valid & ~water):
        raise ValueError("No coastline found. Choose an area and samples that include both land and offshore water.")
    progress(60)
    coverage = polygon_mask(valid, gt, crs, cancel).intersection(area)
    sea_area = polygon_mask(sea, gt, crs, cancel).intersection(coverage)
    if not coastlines(sea_area, coverage):
        raise ValueError("No sea and land boundary was found within the valid imagery.")
    near_sea = binary_dilation(sea, iterations=8)
    ambiguous = valid & near_sea & (confidence <= 0.35)
    neighborhoods, count = label(binary_dilation(ambiguous, iterations=2) & valid & near_sea)
    sizes = np.bincount(neighborhoods[ambiguous], minlength=count + 1)
    reviews = []
    for mask, reason in ((sizes[neighborhoods] >= 5, REVIEW_MIXED),
                         (in_area & ~covered & near_sea, REVIEW_GAP)):
        polygons = polygon_mask(mask, gt, crs, cancel).intersection(area)
        reviews.extend((polygon, (reason,)) for polygon in parts(polygons, {"Polygon"}))
    draft.parent.mkdir(parents=True, exist_ok=True)
    srs = reference(crs)
    with ogr.GetDriverByName("GPKG").CreateDataSource(str(draft)) as ds:
        for name, geometry in (("sea", sea_area), ("coverage", coverage)):
            vector_layer(ds, srs, name, ogr.wkbPolygon, [], ((p, ()) for p in parts(geometry, {"Polygon"})))
        vector_layer(ds, srs, "review", ogr.wkbPolygon, [("reason", ogr.OFTString)], reviews)
    checked(cancel)
    progress(100)
    return draft


def edited_sea(polygons, coverage, cancel):
    if not polygons or any(not geom.is_valid or geom.is_empty for geom in polygons):
        raise ValueError("The sea area contains empty or self-intersecting polygons. Fix the vertices and save again.")
    checked(cancel)
    return shapely.union_all(polygons).intersection(coverage)


def save_shapefiles(parent, crs, polygons, coverage, cancel, progress):
    sea = edited_sea(polygons, coverage, cancel)
    lines = coastlines(sea, coverage)
    if not lines:
        raise ValueError("There is no coastline to save. Edit the sea area so it meets land within the imagery.")
    srs = reference(crs)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    with result_folder(parent, "satellite-coast") as (pending, final):
        for name, kind, features in (("coastline", ogr.wkbLineString, lines),
                                     ("sea_area", ogr.wkbPolygon, parts(sea, {"Polygon"}))):
            checked(cancel)
            with driver.CreateDataSource(str(pending / f"{name}.shp")) as ds:
                layer = ds.CreateLayer(name, srs, kind, options=["ENCODING=UTF-8"])
                layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
                for number, geometry in enumerate(features, 1):
                    checked(cancel)
                    feature = ogr.Feature(layer.GetLayerDefn())
                    feature.SetField("id", number)
                    feature.SetGeometry(ogr.CreateGeometryFromWkb(bytes(geometry.wkb)))
                    layer.CreateFeature(feature)
        for name in ("coastline", "sea_area"):
            if not all((pending / f"{name}.{suffix}").is_file() for suffix in ("shp", "shx", "dbf", "prj")):
                raise OSError(f"Some of the required files for {name} could not be saved.")
        checked(cancel)
    progress(100)
    return final
