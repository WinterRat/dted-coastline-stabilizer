# DTED Coastline Stabilizer

DTED Coastline Stabilizer is a QGIS plugin for generating editable coastlines and sea polygons from satellite imagery and applying sea elevations to DTED data. The current version is **1.1.4** and requires **QGIS 4.2.2 or later**.

QGIS에서 **플러그인 → 플러그인 관리 및 설치 → ZIP에서 설치**를 열고 [dted_coastline_stabilizer.zip](dted_coastline_stabilizer.zip)을 선택하세요. 설치 후 플러그인 패널 상단의 **한글 사용법**을 누르면 사용 안내가 열립니다. 소스와 설치 파일은 [WinterRat의 플러그인 저장소](https://github.com/WinterRat/dted-coastline-stabilizer)에서 확인할 수 있습니다.

## Changes in 1.1.4

- Improved the shoreline from imagery and added a smooth 12 m seaward transition by default for new jobs that export at −50 m.
- Applied the validation reference's display range to the preview and exported QGIS style while keeping the actual elevation statistics in the GeoTIFF.
- Parallelized preview, raster export, validation, and native DTED tile generation.

## Install the plugin in QGIS 4.2.2

1. To use QGIS in English, open **Settings → Options → General**, enable **Override System Locale**, set **User interface translation** to **American English**, and restart QGIS.
2. Open **Plugins → Manage and Install Plugins… → Install from ZIP**.
3. Select [dted_coastline_stabilizer.zip](dted_coastline_stabilizer.zip) and click **Install Plugin**. QGIS reads the ZIP directly.
4. If you close the panel, reopen it from **Plugins → DTED Coastline Stabilizer → DTED Coastline Stabilizer**.

See the [QGIS language settings](https://docs.qgis.org/testing/en/docs/user_manual/introduction/qgis_configuration.html#general-settings) and the [Install from ZIP guide](https://docs.qgis.org/testing/en/docs/user_manual/plugins/plugins.html#the-install-from-zip-tab).

## Load the example

On the **Inputs** tab, choose `dem-0.tif` as **Source DTED**, the `imagery` folder as **RGB Imagery Folder**, and `shoreline.shp` as **Shoreline Shapefile**. All three are in the `example` folder. Click **Check Inputs and Load Map**. To try an input with a sea elevation of −50 m, select `dem--50.tif`.

## Generate a coastline from satellite imagery

1. Open **Satellite** and click **Load RGB GeoTIFFs…**. To try it with the example, open `example/imagery/synthetic-rgb.tif`. This workflow can start with imagery alone.
2. Use **Draw Area of Interest** to outline an area that includes land and offshore water. Right-click to finish. Add representative points with **Add Land Sample** and **Add Offshore Sample**, then click **Generate Sea Area and Coastline**. The plugin automatically analyzes larger areas at a coarser resolution, so small shoreline features may be less detailed.
3. Select a **Review Areas** entry to zoom to a possible tidal flat, estuary, or imagery gap. Use **Edit Coastline Vertices** to adjust the sea boundary. The coastline follows those edits and omits edges at imagery gaps and the area of interest.
4. Click **Save Coastline and Sea Shapefiles…** to save `coastline.shp` and `sea_area.shp` with their `.shx`, `.dbf`, and `.prj` files. Drag either shapefile into QGIS to reopen it. To use the saved sea polygons for elevation processing, click **Approve Sea Area for DTED**, choose **Source DTED**, and click **Check Inputs and Load Map**.

## Edit the shoreline over the DTED

1. Open **Map Editing** and click **Start DTED Editing**. Click **Inspect Elevation**, then click a point on the map to see its source elevation. After previewing edits, the same tool also shows the output elevation and mask class.
2. Use **Edit Vertices** to move, add, or delete vertices with the QGIS Vertex Tool. **Replace Segment** draws a new path between two points on the same shoreline. **Add Shoreline** draws a new line. Left-click to add points and right-click to finish. Enable project snapping in QGIS when needed. **Undo** and **Redo** apply to the active edit layer.
3. Click **Save Edited Shoreline…** to export the edited lines to a GeoPackage in the source shoreline CRS. Save the QGIS project as a `.qgz` to resume editing later. Keep the input and working files available at the paths referenced by the project.

For your own data, choose the source elevation raster, RGB imagery folder, and shoreline shapefile. If the shapefile has no `.prj`, enter its verified CRS in **Shoreline CRS**, click **Confirm Shoreline CRS**, then click **Check Inputs and Load Map** again. The default **Working CRS (meters)** is EPSG:32652. You can also load a separate elevation raster as a validation reference.

## Export sea elevations

1. In **Analysis and Export**, choose the **Input Sea Elevation** and **Output Sea Elevation**. The plugin reports the detected input value when the available evidence supports it. For new projects, −50 m output has a smooth 12 m seaward transition. Set **Transition Width** to 0 m for a sharp boundary.
2. Click **Preview DTED Elevation**. When you load a **Validation Reference**, the plugin uses the same display range for both layers and records the actual output range separately. For shoreline-based classification, select an item in **Review Areas** to zoom to it. In the example, place one seed in the western sea and one land seed each on the eastern mainland and the offshore island. For gaps and ambiguous areas, edit the shoreline or draw polygons with **Force Land**, **Force Sea**, or **Keep Original**. The most recently applied polygon wins where polygons overlap.
3. Choose **Export Format** and click **Validate and Export…** to save a new results folder. Preview and export process raster tiles in parallel. **GeoTIFF** keeps the source grid. The DTED options also produce standard one-degree WGS 84 tiles at the selected level using nearest-neighbor resampling.

| Output | Contents |
| --- | --- |
| `elevation.tif` | Elevation GeoTIFF with the edits applied |
| `elevation.qml` | QGIS display style when the reference range differs from the output statistics |
| `*.dt0`, `*.dt1`, or `*.dt2` | Native DTED tiles when a DTED level is selected |
| `sea-mask.tif` | Land 1, sea 2, unclassified 3, keep original 4, NoData 255 |
| `changed-pixels.tif` | Pixels whose elevation changed |
| `coast.gpkg` | Automatic and edited shorelines, review areas, classified polygons, seeds, manual overrides, and edit history |
| `settings.json`, `statistics.json` | Run settings, pixel counts, unclassified area, and processing statistics |

## Build and verify

Run these commands from the repository folder. Building the install ZIP requires Python 3.12 or later. Run the validation commands with the Python environment supplied by QGIS so GDAL, NumPy, Shapely, and SciPy are available.

```text
python build_plugin.py
python -c "import runpy; runpy.run_path('tests/synthetic.py', run_name='__main__')"
python -c "import runpy; runpy.run_path('tests/satellite_cases.py', run_name='__main__')"
```

The example files are synthetic and can be used to try the plugin without supplying a local dataset.
