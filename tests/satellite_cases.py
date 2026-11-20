"""Run with the QGIS Python shell from app/."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal
import shapely

from dted_coastline_stabilizer.imagery import REVIEW_GAP, REVIEW_MIXED, coastlines, generate_shoreline, open_imagery
from dted_coastline_stabilizer.inputs import raster, reference


X, Y = 322000, 4150000
FIXTURE = Path(__file__).resolve().parents[1] / "example" / "imagery" / "synthetic-rgb.tif"


def xy(x, y):
    return X + x, Y - y


def generate(files, folder):
    check, progress = lambda: False, lambda _: None
    mosaic = open_imagery(files, folder, check, progress)
    area = shapely.Polygon([xy(-16, -8), xy(558, 9), xy(560, 467), xy(-12, 468)])
    package = generate_shoreline(files, mosaic, area, [xy(420, 90)], [xy(72, 230)],
                                 folder / "draft.gpkg", check, progress)
    return read_draft(package)


def read_draft(package):
    with gdal.OpenEx(str(package), gdal.OF_VECTOR) as ds:
        sea, coverage = [shapely.union_all(
            [shapely.from_wkb(bytes(f.GetGeometryRef().ExportToWkb())) for f in ds.GetLayerByName(name)])
            for name in ("sea", "coverage")]
        reasons = {f["reason"] for f in ds.GetLayerByName("review")}
    coast = shapely.union_all(coastlines(sea, coverage))
    assert sea.is_valid and coast.length > 0
    assert coast.intersection(coverage.boundary).length < 1e-7
    assert coast.difference(coverage.buffer(1e-7)).is_empty
    return sea, coverage, coast, reasons


def verify_large_area(folder):
    folder.mkdir()
    left, right = folder / "left.tif", folder / "right.tif"
    driver = gdal.GetDriverByName("GTiff")
    for path, offset, width in ((left, 0, 4400), (right, 4000, 2000)):
        with driver.Create(str(path), width, 4000, 3, gdal.GDT_Byte,
                           options=["TILED=YES", "COMPRESS=DEFLATE"]) as ds:
            ds.SetGeoTransform((X + offset, 1, 0, Y, 0, -1))
            ds.SetProjection(reference("EPSG:32652").ExportToWkt())
            for band, ground, water in zip(ds, (167, 151, 115), (22, 55, 94)):
                band.Fill(ground)
                band.WriteArray(np.full((2000, width), water, dtype=np.uint8), 0, 2000)
            if path == right:
                ds.CreateMaskBand(gdal.GMF_PER_DATASET)
                mask = ds.GetRasterBand(1).GetMaskBand()
                mask.Fill(255)
                for col in (100, 1500):
                    mask.WriteArray(np.zeros((60, 150), dtype=np.uint8), col, 1970)

    files = [left, right]
    check, progress = lambda: False, lambda _: None
    mosaic = open_imagery(files, folder, check, progress)
    area = shapely.box(*xy(-10, 4010), *xy(6010, -10))
    draft = generate_shoreline(files, mosaic, area, [xy(800, 700)], [xy(1000, 3500)],
                               folder / "draft.gpkg", check, progress)
    sea, coverage, coast, reviews = read_draft(draft)
    assert coverage.bounds == shapely.box(*xy(0, 4000), *xy(6000, 0)).bounds
    assert sea.covers(shapely.Point(xy(5950, 3900)))  # Connected across both tiles
    assert not sea.covers(shapely.Point(xy(5950, 100)))
    assert coverage.covers(shapely.Point(xy(4175, 2000)))  # Valid overlap in the earlier tile
    assert not coverage.covers(shapely.Point(xy(5575, 2000)))
    assert REVIEW_GAP in reviews
    for col in (100, 4000, 4400, 5900):
        assert coast.distance(shapely.Point(xy(col, 2000))) <= 5


def verify(folder):
    sea, coverage, coast, reviews = generate([FIXTURE], folder / "single")
    assert sea.covers(shapely.Point(xy(72, 230)))  # Offshore
    assert not sea.covers(shapely.Point(xy(125, 185)))  # Island
    assert not sea.covers(shapely.Point(xy(500, 210)))  # Inland lake
    assert sea.covers(shapely.Point(xy(300, 261)))  # Connected estuary
    assert not coverage.covers(shapely.Point(xy(80, 427)))  # Image gap
    assert not coverage.covers(shapely.Point(xy(-1, 230)))  # Outside image
    assert {REVIEW_MIXED, REVIEW_GAP} <= reviews
    assert any(line.is_ring and shapely.Polygon(line).covers(shapely.Point(xy(125, 185)))
               for line in shapely.get_parts(coast))

    # Split across the island and the mainland coast, and mask pixels in the overlap of the later tile.
    left, right = folder / "left.tif", folder / "right.tif"
    with gdal.Translate(str(left), str(FIXTURE), srcWin=(0, 0, 305, 544)):
        pass
    with gdal.Translate(str(right), str(FIXTURE), srcWin=(125, 0, 579, 544)) as ds:
        ds.GetRasterBand(1).GetMaskBand().WriteArray(np.zeros((40, 155), dtype=np.uint8), 0, 175)
    split_sea, split_coverage, split_coast, _ = generate([left, right], folder / "split")
    assert split_sea.equals(sea) and split_coverage.equals(coverage) and split_coast.equals(coast)

    # A NoData value in even one RGB band must invalidate the pixel, including with a dataset mask.
    missing = folder / "band-nodata.tif"
    with gdal.Translate(str(missing), str(FIXTURE)) as ds:
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.WriteArray(np.zeros((24, 18), dtype=np.uint8), 55, 185)
    nodata_sea, nodata_coverage, nodata_coast, _ = generate([missing], folder / "nodata")
    notch = shapely.box(*xy(55, 209), *xy(73, 185))
    assert coverage.covers(notch) and sea.covers(notch)
    assert nodata_coverage.intersection(notch).area == nodata_sea.intersection(notch).area == 0
    assert nodata_coast.intersection(notch.boundary).length == 0
    with raster(FIXTURE) as ds:
        assert ds.GetRasterBand(1).GetNoDataValue() is None


if __name__ == "__main__":
    with TemporaryDirectory(prefix="satellite-cases-") as tmp:
        verify(Path(tmp))
        verify_large_area(Path(tmp) / "large")
    print("Island, lake, estuary, tidal review, image and NoData edges, and overlapping tile seams passed.")
    print("Large-area analysis across 24 million native pixels passed.")
