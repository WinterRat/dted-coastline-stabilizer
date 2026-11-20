from pathlib import Path
import json
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal, ogr
import shapely
from shapely.geometry import LineString, Point, Polygon

from dted_coastline_stabilizer.coast import KEEP, LAND, SEA, UNKNOWN
from dted_coastline_stabilizer.engine import MASK, OUTPUT, PACKAGE, Cancelled, output_bundle
from dted_coastline_stabilizer.inputs import inspect, raster, reference


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "example"
X, Y = 322000, 4150000
W, H = 352, 272
CRS = "EPSG:32652"
NODATA = -32767


def shore_coords():
    return [(X + 270, Y + 350), (X + 270, Y - 80), (X + 220, Y - 95),
            (X + 220, Y - 165), (X + 270, Y - 180), (X + 272, Y - 255),
            (X + 335, Y - 255), (X + 335, Y - 267), (X + 272, Y - 267),
            (X + 272, Y - 380), (X + 247, Y - 392), (X + 240, Y - 425),
            (X + 267, Y - 450), (X + 267, Y - 900)]


def geom_fixtures():
    coords = shore_coords()
    mainland = Polygon([*coords, (X + 1050, Y - 900), (X + 1050, Y + 350)])
    island = Point(X + 125, Y - 185).buffer(30)
    lake = Point(X + 500, Y - 210).buffer(26)
    dangle = LineString([(X + 437, Y - 469), (X + 451, Y - 481), (X + 449, Y - 494)])
    return mainland, island, lake, dangle


def write_tiff(path, data, transform, pixel="Area", invalid=None):
    values = data if data.ndim == 3 else data[np.newaxis]
    bands, rows, cols = values.shape
    kind = gdal.GDT_Byte if data.dtype == np.uint8 else gdal.GDT_Int16
    with gdal.GetDriverByName("GTiff").Create(str(path), cols, rows, bands, kind,
           ["TILED=YES", "COMPRESS=DEFLATE"]) as ds:
        ds.SetSpatialRef(reference(CRS))
        ds.SetGeoTransform(transform)
        ds.SetMetadataItem("AREA_OR_POINT", pixel)
        for index, array in enumerate(values, 1):
            band = ds.GetRasterBand(index)
            band.WriteArray(array)
            if bands == 1:
                band.SetNoDataValue(NODATA)
                band.SetUnitType("metre")
            else:
                band.SetColorInterpretation((gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)[index-1])
        if invalid is not None:
            ds.CreateMaskBand(gdal.GMF_PER_DATASET)
            ds.GetRasterBand(1).GetMaskBand().WriteArray(np.where(invalid, 0, 255).astype(np.uint8))


def make_example(data=DATA):
    data.mkdir(exist_ok=True)
    (data / "imagery").mkdir(exist_ok=True)
    mainland, island, lake, dangle = geom_fixtures()
    geometry = mainland.union(island)
    dem_x, dem_y = np.meshgrid(X + np.arange(W) * 2 + 1, Y - np.arange(H) * 2 - 1)
    dry = shapely.contains_xy(geometry, dem_x, dem_y)
    fresh = shapely.contains_xy(lake, dem_x, dem_y)
    hills = (4 + ((dem_x - X) / 33 + (Y - dem_y) / 65).astype(np.int16))
    for base, pixel in [(0, "Area"), (-50, "Point")]:
        dem = np.where(dry, hills, base).astype(np.int16)
        dem[fresh] = base
        dem[58:75, 232:246] = 0
        dem[20:35, 200:230] = -50
        dem[187:195, 222:241] = -1
        dem[5:8, 5:10] = NODATA
        invalid = np.zeros((H, W), dtype=bool)
        invalid[5:8, 5:10] = True
        invalid[40:43, 330:335] = True
        write_tiff(data / f"dem-{base}.tif", dem, (X, 2, 0, Y, 0, -2), pixel, invalid)
    ix, iy = np.meshgrid(X + np.arange(W * 2) + 0.5, Y - np.arange(H * 2) - 0.5)
    real_land = shapely.affinity.translate(mainland, xoff=4).union(island)
    dry_image = shapely.contains_xy(real_land, ix, iy) & ~shapely.contains_xy(lake, ix, iy)
    texture = (12 * np.sin(ix / 19) * np.cos(iy / 24) + 3 * np.sin(iy / 3)).astype(int)
    rgb = np.asarray([np.where(dry_image, a, b) + texture for a, b in [(136, 25), (142, 89), (67, 127)]])
    tidal = (iy < Y - 300) & (iy > Y - 356) & (abs(ix - X - 272) < 29)
    for n, base in enumerate((92, 102, 103)):
        rgb[n, tidal] = base + texture[tidal] // 4
    gap = (iy < Y - 408) & (iy > Y - 448) & (ix < X + 311)
    write_tiff(data / "imagery" / "synthetic-rgb.tif", rgb.astype(np.uint8), (X, 1, 0, Y, 0, -1), invalid=gap)
    shp = data / "shoreline.shp"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    with (ogr.Open(str(shp), 1) if shp.exists() else driver.CreateDataSource(str(shp))) as ds:
        existing = ds.GetLayer()
        if existing is None:
            layer = ds.CreateLayer("shoreline", reference(CRS), ogr.wkbLineString)
            for line in (LineString(shore_coords()), island.boundary, lake.boundary, dangle):
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(ogr.CreateGeometryFromWkb(line.wkb))
                layer.CreateFeature(feature)
    return data


def config(base=0, data=DATA):
    return dict(dem=str(data / f"dem-{base}.tif"), shape=str(data / "shoreline.shp"),
                shape_crs="", imagery=str(data / "imagery"), work_crs=CRS, reference="",
                verified_water="", input_value="auto", output_value=-50, search=10,
                join=1, transition=0, curve="linear", narrow=True, refine=True, tile=96)


def seeds():
    return {"seeds": [(Point(X + 72, Y - 230), SEA), (Point(X + 440, Y - 90), LAND),
                      (Point(X + 125, Y - 185), LAND)], "history": []}


def matrix(cache):
    data = make_example(cache / "fixtures")
    runs = []
    isolated = cache / "unreferenced"
    isolated.mkdir()
    bare_shape = isolated / "shoreline.shp"
    shutil.copyfile(data / "shoreline.shp", bare_shape)
    without_prj = dict(config(data=data), shape=str(bare_shape), imagery="")
    assert not inspect(without_prj, cache / "shape-cache")["shape"]["confirmed"]
    fixed = inspect(dict(without_prj, shape_crs=CRS), cache / "shape-cache")["shape"]
    assert fixed["confirmed"] and Path(fixed["path"]).with_suffix(".shx").is_file()
    assert list(isolated.iterdir()) == [bare_shape], "source sidecars must stay untouched"
    for baseline in (0, -50):
        options = config(baseline, data)
        info = inspect(options, cache / f"cache-{baseline}")
        first = None
        for value in (0, -50):
            options["output_value"] = value
            output, report = output_bundle(options, info, seeds(), cache, reuse=first)
            out = Path(output)
            with (raster(out / MASK) as m, raster(out / OUTPUT) as result, raster(options["dem"]) as dem):
                mask, before, after = m.ReadAsArray(), dem.ReadAsArray(), result.ReadAsArray()
                assert np.array_equal(before[mask != SEA], after[mask != SEA])
                assert np.all(after[mask == SEA] == value)
                assert dem.GetMetadataItem("AREA_OR_POINT") == result.GetMetadataItem("AREA_OR_POINT")
                assert np.array_equal(dem.GetRasterBand(1).GetMaskBand().ReadAsArray(),
                                      result.GetRasterBand(1).GetMaskBand().ReadAsArray())
                assert mask[94, 62] == LAND, "island"
                assert mask[105, 250] == UNKNOWN, "unseeded inland lake"
                assert mask[60, 240] == LAND and before[60, 240] == 0
                assert mask[26, 215] == LAND and before[26, 215] == -50
                assert mask[42, 333] == 255 and before[42, 333] != NODATA
                assert report["input_sea_value"] == baseline
                actual = after[mask != 255]
                minimum, maximum, mean, stddev = result.GetRasterBand(1).GetStatistics(False, False)
                assert (minimum, maximum) == (actual.min(), actual.max())
                assert np.allclose([mean, stddev], [actual.mean(), actual.std()])
                if first:
                    assert (out / MASK).read_bytes() == (Path(first) / MASK).read_bytes()
            runs.append({"input": baseline, "output": value, **report})
            first = first or output
        options["transition"] = 12
        options["curve"] = "smoothstep"
        one, one_report = output_bundle(options, info, seeds(), cache, reuse=first)
        options["tile"] = 259
        two, two_report = output_bundle(dict(options, workers=1), info, seeds(), cache)
        with (raster(Path(one) / OUTPUT) as a, raster(Path(two) / OUTPUT) as b,
              raster(Path(one) / MASK) as am, raster(Path(two) / MASK) as bm):
            assert np.array_equal(am.ReadAsArray(), bm.ReadAsArray()), "mask seam"
            assert np.array_equal(a.ReadAsArray(), b.ReadAsArray()), "transition seam"
            assert np.any((a.ReadAsArray() > -50) & (a.ReadAsArray() < 0) & (am.ReadAsArray() == SEA)), "transition"
        assert one_report["elevation_range_m"] == two_report["elevation_range_m"]
        assert one_report["narrow_adjusted_pixels"] > 0, "narrow channel"
        assert one_report["review_reasons"].get("Imagery gap or edge", 0) > 0
        assert one_report["review_reasons"].get("Low imagery contrast or possible tidal flat", 0) > 0
        coarse, coarse_report = output_bundle(dict(options, refine=False), info, seeds(), cache)
        edit = seeds()
        polygon = shapely.box(X + 100, Y - 325, X + 180, Y - 275)
        edit["overrides"] = [(polygon, LAND, 1), (polygon.buffer(-8), KEEP, 2)]
        edit["previous_mask"] = str(Path(first) / MASK)
        edited, edited_report = output_bundle(options, info, edit, cache)
        with raster(Path(edited) / MASK) as edited_mask:
            classes = edited_mask.ReadAsArray()
            assert classes[149, 64] == KEEP and classes[139, 52] == LAND
        runs.append({"input": baseline, "transition": one_report, "unrefined": coarse_report,
                     "manual_edit": edited_report, "tile_seams_equal": True})
    before = set(cache.iterdir())
    try:
        output_bundle(options, info, seeds(), cache, cancel=lambda: True)
        raise AssertionError("cancellation did not stop")
    except Cancelled:
        assert set(cache.iterdir()) == before
    (ROOT / "synthetic-results.json").write_text(json.dumps({"runs": runs, "cancelled_cleanly": True,
                                                  "unreferenced_shape_guard": True},
                                                   indent=2, ensure_ascii=False), encoding="utf8")
    print("Synthetic matrix, manual priority, imagery review, tile seams and cancellation passed.")


if __name__ == "__main__":
    if "--fixtures" in sys.argv:
        print(make_example())
    else:
        with TemporaryDirectory(dir=os.environ.get("DTED_TEST_TEMP")) as cache:
            matrix(Path(cache))
