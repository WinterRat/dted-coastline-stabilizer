from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import ctypes
import json
import math
import os
from pathlib import Path
import tempfile
from threading import local
from time import perf_counter
from uuid import uuid4

import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import maximum_filter
import shapely
from shapely.strtree import STRtree

from .coast import (
    KEEP, LAND, NODATA, SEA, UNKNOWN, RasterSampler, classify, coast_domain, coast_network, parts, trusted_polygons,
)
from .dted import write_dted_tiles
from .inputs import DEFAULT_WORK_CRS, describe, footprint, raster, reference, reproject, transform
from .inputs import reference_range, vector, work_reference
from .workers import tile_results, worker_count


MASK = "sea-mask.tif"
OUTPUT = "elevation.tif"
CHANGES = "changed-pixels.tif"
PACKAGE = "coast.gpkg"
STATISTICS = "statistics.json"
OPTIONS = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]


class Cancelled(Exception):
    pass


def peak_memory():
    if os.name != "nt":
        import resource
        scale = 1 if os.uname().sysname == "Darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    class ProcessCounters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    *[(field, ctypes.c_size_t) for field in ("peak", "working", "peak_paged",
                      "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]

    counters = ProcessCounters()
    counters.size = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    if ctypes.WinDLL("psapi").GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                                  ctypes.byref(counters), counters.size):
        return counters.peak
    return None


def checked(cancel):
    if cancel():
        raise Cancelled()


def windows(ds, size):
    width = height = size
    block_width, _ = ds.GetRasterBand(1).GetBlockSize()
    if block_width == ds.RasterXSize and block_width >= 4 * size:
        # Read distinct scanlines so workers don't each pull the same wide TIFF strips.
        width = block_width
        height = min(size, max(1, 2_000_000 // width))
    for y in range(0, ds.RasterYSize, height):
        for x in range(0, ds.RasterXSize, width):
            yield x, y, min(width, ds.RasterXSize - x), min(height, ds.RasterYSize - y)


def read_byte(band, window):
    return band.ReadAsArray(*window, buf_obj=np.empty((window[3], window[2]), dtype=np.uint8))


def explicit_mask(band):
    return band.GetMaskBand() if band.GetMaskFlags() & (gdal.GMF_ALL_VALID | gdal.GMF_NODATA) == 0 else None


def create_like(path, ds, data_type, nodata, workers):
    out = gdal.GetDriverByName("GTiff").Create(str(path), ds.RasterXSize, ds.RasterYSize, 1, data_type,
                                               options=[*OPTIONS, f"NUM_THREADS={min(workers, 4)}"])
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    pixel = ds.GetMetadataItem("AREA_OR_POINT")
    if pixel:
        out.SetMetadataItem("AREA_OR_POINT", pixel)
    if nodata is not None:
        out.GetRasterBand(1).SetNoDataValue(nodata)
    return out


def vector_layer(ds, srs, name, geometry_type, fields, records):
    layer = ds.CreateLayer(name, srs, geometry_type)
    for field, kind in fields:
        layer.CreateField(ogr.FieldDefn(field, kind))
    layer.StartTransaction()
    for geom, attributes in records:
        feature = ogr.Feature(layer.GetLayerDefn())
        if geom is not None:
            feature.SetGeometry(ogr.CreateGeometryFromWkb(bytes(geom.wkb)))
        for index, value in enumerate(attributes):
            feature.SetField(index, value)
        layer.CreateFeature(feature)
    layer.CommitTransaction()


def prepare_coast_edit(config, info, folder, cancel, progress):
    def check():
        checked(cancel)

    check()
    progress(10)
    with raster(config["dem"]) as dem:
        if describe(dem) != info["dem"]:
            raise ValueError("The source DTED metadata changed. Check the inputs again.")
        base, domain = coast_domain(info, dem, config)
    lines, _ = coast_network(info, domain, config["join"], check)
    if not any(line.intersects(base) for line in lines):
        raise ValueError("The DTED extent and shoreline do not overlap. Check the input files and shoreline CRS.")
    check()
    progress(75)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as staging:
        path = Path(staging) / PACKAGE

        def records():
            for line in lines:
                check()
                yield line, ()

        with ogr.GetDriverByName("GPKG").CreateDataSource(str(path)) as ds:
            vector_layer(ds, reference(info["work"]), "edited", ogr.wkbLineString, [], records())
        check()
        destination = folder / PACKAGE
        path.rename(destination)
    progress(100)
    return str(folder)


def save_vectors(path, work, automatic, active, review, decisions, edits, shore):
    ints, string = ogr.OFTInteger, ogr.OFTString
    srs = reference(work)
    with ogr.GetDriverByName("GPKG").CreateDataSource(str(path)) as ds:
        for name, lines in (("automatic", automatic), ("edited", active), ("confirmed_shore", shore)):
            vector_layer(ds, srs, name, ogr.wkbLineString, [], ((line, ()) for line in lines))
        vector_layer(ds, srs, "review", ogr.wkbLineString, [("reason", string)],
                     ((line, (reason,)) for geom, reason in review
                      for line in parts(geom, {"LineString", "LinearRing"})))
        vector_layer(ds, srs, "faces", ogr.wkbPolygon, [("class", ints), ("reason", string)],
                     ((p, (label, reason)) for p, label, reason in decisions))
        vector_layer(ds, srs, "manual", ogr.wkbPolygon, [("class", ints), ("sequence", ints)],
                     ((p, (label, sequence)) for geom, label, sequence in edits.get("overrides", [])
                      for p in parts(shapely.make_valid(geom), {"Polygon"})))
        vector_layer(ds, srs, "seeds", ogr.wkbPoint, [("class", ints)],
                     ((point, (label,)) for point, label in edits.get("seeds", [])))
        vector_layer(ds, None, "history", ogr.wkbNone,
                     [("time", string), ("layer", string), ("command", string)],
                     ((None, tuple(entry)) for entry in edits.get("history", [])))


def tile_classes(ds, window, geometries, labels, tree, background):
    x, y, w, h = window
    gt = ds.GetGeoTransform()
    area = footprint(gt, *window, samples=1)
    candidates = sorted(tree.query(area, predicate="intersects"))
    # A polygon covering the whole tile supersedes all earlier classifications.
    for order in range(len(candidates) - 1, -1, -1):
        index = candidates[order]
        if geometries[index].covers(area):
            background = labels[index]
            candidates = candidates[order + 1:]
            break
    if not candidates:
        values = np.full((h, w), background, dtype=np.uint8)
    else:
        left, top = gdal.ApplyGeoTransform(gt, x, y)
        with (gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte) as holder,
              ogr.GetDriverByName("MEM").CreateDataSource("") as mem):
            holder.SetGeoTransform((left, gt[1], gt[2], top, gt[4], gt[5]))
            holder.SetProjection(ds.GetProjection())
            holder.GetRasterBand(1).Fill(background)
            values = rasterize_classes(holder, mem, ds.GetProjection(), candidates, geometries, labels)
    return values


def rasterize_classes(holder, mem, projection, candidates, geometries, labels):
    layer = mem.CreateLayer("classes", reference(projection), ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("label", ogr.OFTInteger))
    for index in candidates:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("label", int(labels[index]))
        feature.SetGeometry(ogr.CreateGeometryFromWkb(geometries[index].wkb))
        layer.CreateFeature(feature)
    gdal.RasterizeLayer(holder, [1], layer, options=["ATTRIBUTE=label", "ALL_TOUCHED=FALSE"])
    return holder.ReadAsArray()


def masks_and_shore(base, active):
    groups = {LAND: [], SEA: []}
    # Walk newest to oldest for ordered overrides. Base faces have disjoint interiors.
    overrides = [geom for geom, label, reason in base[1]]
    covered = shapely.GeometryCollection()
    for geom, label, _ in reversed(base[1]):
        if label in groups:
            groups[label].append(geom.difference(covered))
        covered = covered.union(geom)
    for geom, label, _ in base[0]:
        if label in groups:
            groups[label].append(geom.difference(covered) if overrides else geom)
    sea, land = (shapely.union_all(groups[k]) for k in (SEA, LAND))
    confirmed = sea.boundary.intersection(land.boundary) if not sea.is_empty and not land.is_empty else None
    if confirmed is None:
        return []
    # Only actual shoreline and manual class boundaries contribute to the transition.
    allowed = shapely.union_all([*active, *(geom.boundary for geom in overrides)])
    return list(parts(confirmed.intersection(allowed), {"LineString"}))


def input_baseline(setting, samples):
    if setting == "off":
        return None
    if setting in ("0", "-50"):
        return int(setting)
    if len(samples) < 3:
        return None
    counts = Counter(samples)
    choices = [value for value in (0, -50) if counts[value] / len(samples) >= 0.8]
    return choices[0] if len(choices) == 1 else None


def metric_area(ds, window, project):
    x, y, w, h = window
    gt = ds.GetGeoTransform()
    center_x, center_y = gdal.ApplyGeoTransform(gt, x + w / 2, y + h / 2)
    x, y = project(np.array([center_x, center_x + gt[1], center_x + gt[2]]),
                   np.array([center_y, center_y + gt[4], center_y + gt[5]]))
    return abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))


def verified_shore(path, polygons, domain, work, check):
    # Satellite export saves the real coastline separately from imagery and AOI edges.
    area = Path(path)
    companion = area.with_name("coastline.shp")
    if area.name.lower() == "sea_area.shp" and companion.is_file():
        lines = []
        with vector(companion) as layer:
            crs = layer.GetSpatialRef()
            if crs:
                for feature in layer:
                    check()
                    geometry = feature.GetGeometryRef()
                    if geometry is not None:
                        line = reproject(shapely.from_wkb(bytes(geometry.ExportToWkb())), crs.ExportToWkt(), work)
                        lines.extend(parts(line.intersection(domain), {"LineString"}))
                return lines
    boundary = shapely.union_all(polygons).boundary.intersection(domain).difference(domain.boundary)
    return list(parts(boundary, {"LineString"}))


def make_mask(dem, directory, info, config, edits, cancel, progress, workers):
    dem_info = info["dem"]
    source = dem_info["wkt"]
    def check():
        checked(cancel)

    progress(8)
    limited = config.get("limit_water", False)
    domain = reproject(footprint(dem.GetGeoTransform(), 0, 0, dem.RasterXSize, dem.RasterYSize), source, info["work"])
    if limited:
        auto, active, decisions, review, samples, movement = [], [], [], [], [], []
        water = []
        for polygon in trusted_polygons(config["verified_water"], info["work"]):
            check()
            clipped = list(parts(polygon.intersection(domain), {"Polygon"}))
            if clipped:
                water.append(polygon)
                decisions.extend((p, SEA, "Verified sea area") for p in clipped)
        if not decisions:
            raise ValueError("The verified sea areas do not overlap the source DTED.")
        shore = verified_shore(config["verified_water"], water, domain, info["work"], check)
    elif info["imagery"]:
        with raster(info["imagery"]) as image:
            sampler = RasterSampler(image, info["work"], [1, 2, 3], 4)
            auto, active, decisions, review, samples, movement = classify(info, dem, config, edits, sampler, check)
    else:
        auto, active, decisions, review, samples, movement = classify(info, dem, config, edits, None, check)
    overrides_count = 0 if limited else sum(
        len(list(parts(shapely.make_valid(item[0]), {"Polygon"}))) for item in edits.get("overrides", []))
    manual = decisions[-overrides_count:] if overrides_count else []
    automatic_faces = decisions[:-overrides_count] if overrides_count else decisions
    if not limited:
        shore = masks_and_shore((automatic_faces, manual), active)
    baseline = input_baseline(config["input_value"], samples)
    polygons = [reproject(item[0], info["work"], source) for item in decisions]
    shapely.prepare(polygons)
    labels = [item[1] for item in decisions]
    tree = STRtree(polygons)
    thread = local()
    band = dem.GetRasterBand(1)
    source_valid = explicit_mask(band)
    nodata = band.GetNoDataValue()
    counts = Counter()
    area = 0.0
    potential_reclaimed = 0
    tiles = list(windows(dem, config.get("tile", 512)))
    def mask_tile(window):
        check()
        values = tile_classes(dem, window, polygons, labels, tree, KEEP if limited else UNKNOWN)
        data = band.ReadAsArray(*window) if nodata is not None or baseline is not None else None
        if nodata is not None:
            values[data == nodata] = NODATA
        if source_valid is not None:
            values[read_byte(source_valid, window) == 0] = NODATA
        totals = np.bincount(values.ravel(), minlength=256)
        unknown_area = 0
        if totals[UNKNOWN]:
            if not hasattr(thread, "project"):
                thread.project = transform(source, info["work"])
            unknown_area = int(totals[UNKNOWN]) * metric_area(dem, window, thread.project)
        reclaimed = int(np.count_nonzero((values == LAND) & (data == baseline))) if baseline is not None else 0
        return values, totals, unknown_area, reclaimed

    with (create_like(directory / MASK, dem, gdal.GDT_Byte, NODATA, workers) as mask,
          tile_results(tiles, mask_tile, check, workers) as results):
        out = mask.GetRasterBand(1)
        for index, (window, (values, totals, unknown_area, reclaimed)) in enumerate(results, 1):
            out.WriteArray(values, *window[:2])
            counts.update({int(label): int(totals[label]) for label in np.flatnonzero(totals)})
            area += unknown_area
            potential_reclaimed += reclaimed
            if index % 20 == 0 or index == len(tiles):
                progress(35 + 24 * index / len(tiles))
    reason_area = Counter()
    covered = shapely.union_all([geom for geom, _, _ in manual])
    for geom, label, reason in automatic_faces:
        if label == UNKNOWN:
            reason_area[reason] += geom.intersection(domain).difference(covered).area
    if baseline is not None and edits.get("previous_mask"):
        with raster(edits["previous_mask"]) as previous, raster(directory / MASK) as current:
            if same_grid(dem, previous):
                for window in tiles:
                    check()
                    prior = previous.ReadAsArray(*window)
                    now = current.ReadAsArray(*window)
                    suspect = (prior == SEA) & (now == LAND) & (band.ReadAsArray(*window) == baseline)
                    if np.any(suspect):
                        array = gdal.GetDriverByName("MEM").Create("", window[2], window[3], 1, gdal.GDT_Byte)
                        gt = dem.GetGeoTransform()
                        x, y = gdal.ApplyGeoTransform(gt, *window[:2])
                        array.SetGeoTransform((x, gt[1], gt[2], y, gt[4], gt[5]))
                        array.GetRasterBand(1).WriteArray(suspect.astype(np.uint8))
                        with ogr.GetDriverByName("MEM").CreateDataSource("") as mem:
                            layer = mem.CreateLayer("suspect")
                            gdal.Polygonize(array.GetRasterBand(1), array.GetRasterBand(1), layer, -1)
                            for feature in layer:
                                geom = shapely.from_wkb(bytes(feature.GetGeometryRef().ExportToWkb()))
                                review.append((reproject(geom.boundary, source, info["work"]), "Possible former sea area. Verify the land elevation."))
    save_vectors(directory / PACKAGE, info["work"], auto, active, review, decisions, edits, shore)
    report = {
        "class_pixels": {str(k): v for k, v in counts.items()}, "unknown_area_m2": area,
        "unknown_reason_geometry_area_m2": dict(reason_area), "input_sea_value": baseline,
        "baseline_samples": len(samples), "review_reasons": dict(Counter(reason for _, reason in review)),
        "baseline_land_pixels_unchanged": potential_reclaimed,
        "image_displacement_m": {"median": float(np.median(movement)), "p95": float(np.percentile(movement, 95))}
        if movement else None,
    }
    return report, shore


def same_grid(a, b):
    return (a.RasterXSize == b.RasterXSize and a.RasterYSize == b.RasterYSize and
            a.GetGeoTransform() == b.GetGeoTransform() and
            bool(a.GetSpatialRef().IsSame(b.GetSpatialRef())) and
            a.GetMetadataItem("AREA_OR_POINT") == b.GetMetadataItem("AREA_OR_POINT"))


def shore_from(path):
    with ogr.Open(str(path), 0) as ds:
        layer = ds.GetLayerByName("confirmed_shore")
        return [shapely.from_wkb(bytes(f.GetGeometryRef().ExportToWkb())) for f in layer]


def transition(dem, mask, water, window, project, tree, width, curve, narrow):
    x, y, w, h = window
    gt = dem.GetGeoTransform()
    pixel = max(0.01, math.sqrt(metric_area(dem, window, project)))
    radius = max(1, int(math.ceil(width / pixel)))
    halo = radius if narrow else 0
    x0, y0 = max(0, x - halo), max(0, y - halo)
    x1, y1 = min(dem.RasterXSize, x + w + halo), min(dem.RasterYSize, y + h + halo)
    tile = footprint(gt, x0, y0, x1 - x0, y1 - y0, samples=4)
    region = shapely.transform(tile, project, interleaved=False).envelope
    indices = tree.query(region, predicate="dwithin", distance=width)
    if not len(indices):
        return None, 0
    nearby = tree.geometries.take(indices)
    west, south, east, north = shapely.total_bounds(nearby)
    sea = read_byte(mask, (x0, y0, x1 - x0, y1 - y0)) == SEA if halo else water
    cy, cx = np.nonzero(sea)
    distance = np.full(sea.shape, -np.inf)
    if len(cx):
        col, row = cx + x0 + 0.5, cy + y0 + 0.5
        mx, my = project(gt[0] + col * gt[1] + row * gt[2], gt[3] + col * gt[4] + row * gt[5])
        distance[cy, cx] = width
        close = np.flatnonzero((mx >= west - width) & (mx <= east + width) &
                               (my >= south - width) & (my <= north + width))
        if len(close):
            near_tree = STRtree(nearby)
            pairs, d = near_tree.query_nearest(shapely.points(mx[close], my[close]), max_distance=width,
                                               return_distance=True, all_matches=False)
            indices = close[pairs[0]]
            distance[cy[indices], cx[indices]] = d
    core = np.s_[y - y0:y - y0 + h, x - x0:x - x0 + w]
    short = np.zeros((h, w), dtype=bool)
    if narrow:
        local_width = maximum_filter(distance, size=radius * 2 + 1, mode="constant", cval=-np.inf)[core]
        short = (distance[core] >= 0) & (local_width < width)
        scale = np.where(short, np.maximum(local_width, pixel / 2), width)
    else:
        scale = width
    fraction = np.clip(distance[core] / scale, 0, 1)
    if curve == "smoothstep":
        fraction = fraction * fraction * (3 - 2 * fraction)
    return np.rint(-50 * fraction).astype(np.int16), int(np.count_nonzero(short))


def export_dem(dem, directory, work, shore, config, cancel, progress, workers):
    sea = int(config["output_value"])
    width = float(config["transition"]) if sea == -50 else 0
    band = dem.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if sea not in (0, -50) or sea == nodata:
        raise ValueError("Choose 0 m or −50 m for the output sea elevation. The value must differ from NoData.")
    if width and shore and nodata is not None and -50 <= nodata <= 0:
        raise ValueError("The transition elevation range includes the DEM's NoData value.")
    # Build short segments so nearest-neighbor queries examine a local part of a long coast.
    coords = [shapely.get_coordinates(shapely.segmentize(line, 250)) for line in shore] if width else []
    tree = STRtree(np.concatenate([shapely.linestrings(np.stack((xy[:-1], xy[1:]), axis=1)) for xy in coords])
                   if coords else [])
    thread = local()
    source = dem.GetProjection()
    tile_size = config.get("tile", 512)
    tiles = list(windows(dem, tile_size))
    count = shortened = total = total_squared = valid_count = 0
    low, high = None, None
    source_valid = explicit_mask(band)

    def check():
        checked(cancel)

    def elevation_tile(window):
        check()
        values = band.ReadAsArray(*window)
        valid = read_byte(source_valid, window) if source_valid is not None else None
        invalid = values == nodata
        if valid is not None:
            invalid |= valid == 0
        water = (read_byte(mask_band, window) == SEA) & ~invalid
        result = values.copy()
        adjusted = 0
        if water.any():
            result[water] = sea
            if len(tree):
                if not hasattr(thread, "project"):
                    thread.project = transform(source, work)
                x, y, w, h = window
                for start in range(0, w, tile_size):
                    columns = slice(start, min(start + tile_size, w))
                    local_water = water[:, columns]
                    if not local_water.any():
                        continue
                    piece = (x + start, y, columns.stop - start, h)
                    profile, count = transition(dem, mask_band, local_water, piece, thread.project, tree, width,
                                                config["curve"], config["narrow"])
                    if profile is not None:
                        result[:, columns][local_water] = profile[local_water]
                    adjusted += count
        difference = (result != values).astype(np.uint8)
        changed_count = int(difference.sum())
        difference[invalid] = NODATA
        samples = result[~invalid] if invalid.any() else result
        stats = (int(samples.min()), int(samples.max()), int(samples.sum()),
                 int(np.square(samples, dtype=np.int64).sum()), samples.size) if samples.size else None
        return result, difference, valid, changed_count, adjusted, stats

    with (raster(directory / MASK, thread_safe=workers > 1) as mask,
          create_like(directory / OUTPUT, dem, band.DataType, nodata, workers) as out,
          create_like(directory / CHANGES, dem, gdal.GDT_Byte, NODATA, workers) as changed,
          tile_results(tiles, elevation_tile, check, workers) as results):
        if not same_grid(dem, mask):
            raise ValueError("The saved sea mask grid differs from the source DEM grid.")
        mask_band = mask.GetRasterBand(1)
        written = out.GetRasterBand(1)
        written.SetUnitType(band.GetUnitType())
        if band.GetScale() is not None:
            written.SetScale(band.GetScale())
        if band.GetOffset() is not None:
            written.SetOffset(band.GetOffset())
        if source_valid is not None:
            with gdal.config_option("GDAL_TIFF_INTERNAL_MASK", "YES", thread_local=True):
                out.CreateMaskBand(gdal.GMF_PER_DATASET)
        change_band = changed.GetRasterBand(1)
        validity_band = explicit_mask(written)
        for index, (window, (result, difference, valid, changed_count, adjusted, stats)) in enumerate(results, 1):
            count += changed_count
            shortened += adjusted
            written.WriteArray(result, *window[:2])
            change_band.WriteArray(difference, *window[:2])
            if validity_band is not None:
                validity_band.WriteArray(valid, *window[:2])
            if stats:
                minimum, maximum, subtotal, squares, size = stats
                low = minimum if low is None else min(low, minimum)
                high = maximum if high is None else max(high, maximum)
                total += subtotal
                total_squared += squares
                valid_count += size
            if index % 20 == 0 or index == len(tiles):
                progress(60 + 27 * index / len(tiles))
        if valid_count:
            mean = total / valid_count
            deviation = math.sqrt(max(0, total_squared / valid_count - mean**2))
            written.SetStatistics(low, high, mean, deviation)
        written.SetMetadataItem("STATISTICS_VALID_PERCENT", f"{100 * valid_count / (dem.RasterXSize * dem.RasterYSize):.12g}")
    return {"output_sea_value": sea, "transition_m": width, "narrow_adjusted_pixels": shortened,
            "changed_pixels": count, "elevation_range_m": [low, high] if valid_count else None}


def validate(dem, directory, config, cancel, progress, workers):
    def check():
        checked(cancel)

    threaded = workers > 1
    with (raster(directory / OUTPUT, thread_safe=threaded) as result,
          raster(directory / MASK, thread_safe=threaded) as mask,
          raster(directory / CHANGES, thread_safe=threaded) as changed):
        a, b = dem.GetRasterBand(1), result.GetRasterBand(1)
        source_valid, result_valid = explicit_mask(a), explicit_mask(b)
        mask_band, changed_band = mask.GetRasterBand(1), changed.GetRasterBand(1)
        nodata = a.GetNoDataValue()
        if (not all(same_grid(dem, other) for other in (result, mask, changed)) or
                a.DataType != b.DataType or a.GetNoDataValue() != b.GetNoDataValue() or
                a.GetUnitType() != b.GetUnitType() or a.GetScale() != b.GetScale() or a.GetOffset() != b.GetOffset() or
                (source_valid is None) != (result_valid is None)):
            raise ValueError("The output grid or band metadata did not match after reopening.")
        tiles = list(windows(dem, config.get("tile", 512)))
        def validate_tile(window):
            check()
            before, after = a.ReadAsArray(*window), b.ReadAsArray(*window)
            classes = read_byte(mask_band, window)
            invalid = before == nodata
            if source_valid is not None:
                valid = read_byte(source_valid, window)
                if not np.array_equal(valid, read_byte(result_valid, window)):
                    raise ValueError("The source and output validity masks differ.")
                invalid |= valid == 0
            changed_pixels = read_byte(changed_band, window)
            if (np.any((classes == NODATA) != invalid) or np.any((before != after) & ((classes != SEA) | invalid))
                    or np.any(changed_pixels[~invalid] != (before != after)[~invalid])):
                raise ValueError("Land, unclassified, or NoData values changed, or the changed-pixel mask failed validation.")

        with tile_results(tiles, validate_tile, check, workers) as results:
            for index, _ in enumerate(results, 1):
                if index % 20 == 0 or index == len(tiles):
                    progress(87 + 12 * index / len(tiles))


def output_bundle(config, info, edits, parent, reuse=None, cancel=lambda: False, progress=lambda value: None,
                  *, dted_level=None):
    started = perf_counter()
    workers = worker_count(config.get("workers"))
    if config.get("limit_water") and info["work"] == info["dem"]["wkt"]:
        # Earlier projects stored the source CRS for verified polygons.
        info = dict(info, work=work_reference(config.get("work_crs") or DEFAULT_WORK_CRS))
    processing_progress = progress if dted_level is None else lambda value: progress(value * 0.8)
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / ("coast-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    with ExitStack() as stack:
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".coast-", dir=parent)))
        dem = stack.enter_context(raster(config["dem"], thread_safe=workers > 1))
        if describe(dem) != info["dem"]:
            raise ValueError("The source DEM metadata changed. Check the inputs again.")
        if reuse:
            reuse = Path(reuse)
            for name in (MASK, PACKAGE):
                with open(reuse / name, "rb") as source, open(directory / name, "wb") as target:
                    while chunk := source.read(4 * 1024 * 1024):
                        checked(cancel)
                        target.write(chunk)
            report = json.loads((reuse / STATISTICS).read_text(encoding="utf8"))
            shore = shore_from(directory / PACKAGE)
        else:
            report, shore = make_mask(dem, directory, info, config, edits, cancel, processing_progress, workers)
        report.update(export_dem(dem, directory, info["work"], shore, config, cancel, processing_progress, workers))
        validate(dem, directory, config, cancel, processing_progress, workers)
        ref = info.get("reference") or {}
        ref_range = ref.get("display_range")
        if ref and "display_range" not in ref and config.get("reference"):
            with raster(config["reference"]) as target:
                ref_range = reference_range(target)
        report["display_range_m"] = ref_range or report["elevation_range_m"]
        if dted_level is not None:
            report["dted_tiles"] = write_dted_tiles(directory / OUTPUT, directory, dted_level, cancel,
                                                    lambda: checked(cancel), progress, workers)
        else:
            report.pop("dted_tiles", None)
        report.update(seconds=perf_counter() - started, peak_process_bytes=peak_memory(), workers=workers,
                      reused_mask=bool(reuse), dem_grid={k: v for k, v in info["dem"].items() if k != "wkt"})
        (directory / STATISTICS).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        (directory / "settings.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
        checked(cancel)
        directory.rename(final)
    progress(100)
    return str(final), report
