from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile

import numpy as np
from osgeo import gdal, ogr, osr
import shapely
from shapely.geometry import Polygon


gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

COAST_CRS = "EPSG:5186"
DEFAULT_WORK_CRS = "EPSG:32652"


def reference(value):
    if not value:
        raise ValueError("Check the coordinate reference system (CRS).")
    srs = osr.SpatialReference()
    srs.SetFromUserInput(value)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def transform(source, target):
    operation = osr.CoordinateTransformation(reference(source), reference(target))

    def project(x, y):
        if not len(x):
            return x, y
        xy = np.asarray(operation.TransformPoints(np.column_stack((x, y))))[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError("Check the extent used for the coordinate transformation.")
        return xy[:, 0], xy[:, 1]

    return project


def reproject(geom, source, target):
    return shapely.transform(geom, transform(source, target), interleaved=False)


def raster(path, *, thread_safe=False):
    flags = gdal.OF_RASTER | gdal.OF_READONLY
    if thread_safe:
        flags |= gdal.OF_THREAD_SAFE
    return gdal.OpenEx(str(path), flags)


def describe(ds):
    srs = ds.GetSpatialRef()
    if srs is None or ds.GetGeoTransform(can_return_null=True) is None:
        raise ValueError("The raster must have a CRS and a grid transform.")
    band = ds.GetRasterBand(1)
    return {
        "crs": srs.GetName(),
        "authority": ":".join(filter(None, (srs.GetAuthorityName(None), srs.GetAuthorityCode(None)))),
        "wkt": ds.GetProjection(),
        "affine": list(ds.GetGeoTransform()),
        "size": [ds.RasterXSize, ds.RasterYSize],
        "pixel": ds.GetMetadataItem("AREA_OR_POINT"),
        "type": gdal.GetDataTypeName(band.DataType),
        "unit": band.GetUnitType(),
        "nodata": band.GetNoDataValue(),
        "bands": ds.RasterCount,
    }


def footprint(gt, x, y, width, height, samples=12):
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    points = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        for fraction in np.linspace(0, 1, samples, endpoint=False):
            col, row = np.asarray(start) + fraction * (np.asarray(end) - start)
            points.append(gdal.ApplyGeoTransform(gt, col, row))
    return Polygon(points)


def work_reference(value):
    srs = reference(value)
    if not srs.IsProjected() or abs(srs.GetLinearUnits() - 1) > 1e-8:
        raise ValueError("Enter a projected CRS that uses meters for distance calculations.")
    return srs.ExportToWkt()


def reference_range(ds):
    band = ds.GetRasterBand(1)
    if (ds.RasterCount != 1 or band.GetScale() not in (None, 1) or band.GetOffset() not in (None, 0)
            or band.GetUnitType().lower() not in ("", "m", "meter", "metre", "meters", "metres")):
        return None
    low, high = band.GetStatistics(True, False)[:2]
    if low >= high:
        try:
            low, high = band.ComputeRasterMinMax(True)
        except RuntimeError:
            return None
    if np.isfinite([low, high]).all() and low < high:
        return [low, high]
    return None


def cached_shape(source, cache):
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Select a shoreline shapefile.")
    siblings = {p.suffix.lower(): p for p in source.parent.iterdir() if p.stem.lower() == source.stem.lower()}
    if ".shx" in siblings:
        return str(source)
    members = [siblings[suffix] for suffix in (".shp", ".dbf", ".prj", ".cpg") if suffix in siblings]
    stamp = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in members]
    cache = Path(cache).resolve()
    if cache == source.parent or source.parent in cache.parents:
        raise ValueError("Place the shoreline working cache outside the source folder.")
    cache.mkdir(parents=True, exist_ok=True)
    folder = cache / sha256(repr(stamp).encode()).hexdigest()[:20]
    if not (folder / source.with_suffix(".shx").name).is_file():
        with tempfile.TemporaryDirectory(dir=cache) as staging:
            staging = Path(staging)
            for path in members:
                shutil.copyfile(path, staging / (source.stem + path.suffix.lower()))
            with gdal.config_option("SHAPE_RESTORE_SHX", "YES", thread_local=True):
                with gdal.OpenEx(str(staging / source.name), gdal.OF_VECTOR | gdal.OF_READONLY):
                    pass
            try:
                staging.rename(folder)
            except OSError:
                if not folder.is_dir():
                    raise
    return str(folder / source.name)


@contextmanager
def vector(path):
    with gdal.OpenEx(str(path), gdal.OF_VECTOR | gdal.OF_READONLY) as ds:
        yield ds.GetLayer()


def rgb_bands(ds):
    channels = {ds.GetRasterBand(i).GetColorInterpretation(): i for i in range(1, ds.RasterCount + 1)}
    rgb = [channels.get(channel, i) for i, channel in enumerate(
           (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand), 1)]
    return rgb if len(set(rgb)) == 3 else [1, 2, 3]


def imagery_vrt(source, destination, *, prefer_native=False):
    if not source:
        return None, []
    if isinstance(source, (list, tuple)):
        files = [Path(file).resolve() for file in source]
    else:
        source = Path(source).resolve()
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".tif", ".tiff"}) if source.is_dir() else [source]
    if not files:
        raise ValueError("No GeoTIFF files were found in the imagery folder.")
    metadata = []
    mosaics = []
    vrt_crs = None
    tile_bands = []
    native_tiles = prefer_native
    for file in files:
        with raster(file) as ds:
            info = describe(ds)
            if info["bands"] < 3:
                raise ValueError(f"The file {file.name} must have RGB bands.")
            tile_bands.append(rgb_bands(ds))
            vrt_crs = vrt_crs or info["wkt"]
            metadata.append(dict(info, name=file.name))
            if native_tiles:
                _, x_size, x_rotation, _, y_rotation, y_size = info["affine"]
                rgb = (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)
                native_tiles = (
                    info["wkt"] == vrt_crs and info["bands"] == len(rgb)
                    and x_size > 0 and y_size < 0 and x_rotation == y_rotation == 0
                    and all(band.GetColorInterpretation() == color and band.GetMaskFlags() == gdal.GMF_ALL_VALID
                            for band, color in zip(ds, rgb))
                )
    if native_tiles:
        # A warped VRT reads at full resolution even when QGIS asks for a zoomed-out view.
        if len(files) == 1:
            return str(files[0]), metadata
        with gdal.BuildVRT(str(destination), [str(file) for file in files], addAlpha=True, strict=True):
            pass
        return str(destination), metadata

    # Give each image its own alpha mask before mosaicking so gaps remain explicit.
    for i, (file, bands) in enumerate(zip(files, tile_bands)):
        tile = Path(destination).with_name(f"imagery-source-{i}.vrt")
        with gdal.Warp(str(tile), str(file), format="VRT", dstSRS=vrt_crs, srcBands=bands, dstAlpha=True):
            pass
        mosaics.append(str(tile))
    with gdal.BuildVRT(str(destination), mosaics, bandList=[1, 2, 3, 4], strict=True):
        pass
    return str(destination), metadata


def inspect(config, cache, allow_partial=False):
    limited = config.get("limit_water", False)
    if limited and (not config.get("dem") or not config.get("verified_water")):
        raise ValueError("Select a source DTED and a verified sea-area polygon file.")
    if limited:
        with vector(config["verified_water"]) as layer:
            if (ogr.GT_Flatten(layer.GetGeomType()) not in (ogr.wkbPolygon, ogr.wkbMultiPolygon)
                    or not layer.GetSpatialRef() or layer.GetFeatureCount() < 1):
                raise ValueError("Verified sea areas must contain polygons and have a CRS.")
    if not limited and not allow_partial and (not config.get("dem") or not config.get("shape")):
        raise ValueError("Select a source DEM and a shoreline shapefile.")
    if allow_partial and not any(config.get(key) for key in ("dem", "shape", "imagery")):
        raise ValueError("Choose a source DTED, a shoreline shapefile, or satellite imagery.")
    dem = None
    if config.get("dem"):
        with raster(config["dem"]) as ds:
            dem = describe(ds)
            if ds.RasterCount != 1 or ds.GetRasterBand(1).DataType != gdal.GDT_Int16:
                raise ValueError("A single-band Int16 DEM is required.")
            b = ds.GetRasterBand(1)
            if (b.GetScale() not in (None, 1) or b.GetOffset() not in (None, 0)
                    or b.GetUnitType().lower() not in ("", "m", "meter", "metre", "meters", "metres")):
                raise ValueError("The DEM must use meters with a scale of 1 and an offset of 0.")
    work = work_reference(config.get("work_crs") or DEFAULT_WORK_CRS)
    shape_info = None
    if config.get("shape") and not limited:
        shape = cached_shape(config["shape"], cache)
        with vector(shape) as layer:
            srs = layer.GetSpatialRef()
            supplied = config.get("shape_crs", "").strip()
            if supplied:
                srs = reference(supplied)
            crs = srs.ExportToWkt() if srs is not None else None
            geometry = ogr.GT_Flatten(layer.GetGeomType())
            if geometry not in (ogr.wkbLineString, ogr.wkbMultiLineString):
                raise ValueError("The shoreline shapefile must contain line geometry.")
            shape_info = {"count": layer.GetFeatureCount(), "bounds": list(layer.GetExtent()), "crs": crs,
                          "confirmed": bool(crs), "path": shape}
    ref = None
    if config.get("reference"):
        with raster(config["reference"]) as ds:
            ref = describe(ds)
            ref["display_range"] = reference_range(ds)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    imagery, images = imagery_vrt(config.get("imagery") if not limited else None, cache / "imagery.vrt")
    return {"dem": dem, "shape": shape_info, "reference": ref, "work": work,
            "imagery": imagery, "images": images}
