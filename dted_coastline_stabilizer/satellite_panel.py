"""QGIS map workflow for the satellite coastline tab."""

from contextlib import suppress
import json
from pathlib import Path
from uuid import uuid4

import shapely
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem
from qgis.core import Qgis, QgsFeature, QgsFillSymbol, QgsGeometry, QgsPointXY, QgsRasterLayer, QgsVectorLayer
from qgis.gui import QgsRubberBand

from .coast import geometry_signature
from .imagery import coastlines, edited_sea, generate_shoreline, open_imagery, save_shapefiles


SATELLITE_TITLE = "Satellite Coastline"


class SatellitePanel:
    def __init__(self, plugin):
        self.plugin = plugin
        self.project = plugin.project
        self.canvas = plugin.canvas
        self.session = uuid4().hex
        self.sources = []
        self.area = []
        self.land = []
        self.offshore = []
        self.saved = None
        self.coverage = None
        self.edit_connections = []
        self.redraw = QTimer(plugin.dock)
        self.redraw.setSingleShot(True)
        self.redraw.setInterval(150)
        plugin.connect(self.redraw.timeout, self.refresh_coast)

        box = plugin.tab("Satellite")
        self.page = plugin.tabs.widget(plugin.tabs.count() - 1)
        title = QLabel(SATELLITE_TITLE)
        title.setStyleSheet("font-weight: bold")
        box.addWidget(title)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        box.addWidget(plugin.button("Load RGB GeoTIFFs…", self.choose_images))
        box.addWidget(self.source_label)
        draw = plugin.button("Draw Area of Interest", self.draw_area)
        draw.setToolTip("Larger areas are analyzed at a coarser resolution so they can be processed together.")
        box.addWidget(draw)
        row = QHBoxLayout()
        row.addWidget(plugin.button("Add Land Sample", lambda: self.place_sample("land")))
        row.addWidget(plugin.button("Add Offshore Sample", lambda: self.place_sample("offshore")))
        box.addLayout(row)
        row = QHBoxLayout()
        self.sample_label = QLabel()
        row.addWidget(self.sample_label, 1)
        row.addWidget(plugin.button("Clear Samples", self.clear_samples))
        box.addLayout(row)
        box.addWidget(plugin.button("Generate Sea Area and Coastline", self.generate))
        box.addWidget(plugin.button("Edit Coastline Vertices", self.edit_vertices))
        box.addWidget(QLabel("Review Areas (select to zoom)"))
        self.reviews = QListWidget()
        self.reviews.setMinimumHeight(65)
        self.reviews.setWordWrap(True)
        plugin.connect(self.reviews.currentItemChanged, self.zoom_review)
        box.addWidget(self.reviews, 1)
        box.addWidget(plugin.button("Save Coastline and Sea Shapefiles…", self.choose_export))
        box.addWidget(plugin.button("Approve Sea Area for DTED", self.approve))
        self.markers = {}
        for key, kind, color in (("area", Qgis.GeometryType.Polygon, "#ffe184"),
                                 ("land", Qgis.GeometryType.Point, "#ff8e59"),
                                 ("offshore", Qgis.GeometryType.Point, "#47dfff")):
            band = QgsRubberBand(self.canvas, kind)
            band.setStrokeColor(QColor(color))
            band.setFillColor(QColor(0, 0, 0, 0))
            band.setWidth(2)
            band.setIconSize(13)
            self.markers[key] = (band, kind)
        plugin.connect(self.canvas.destinationCrsChanged, self.refresh_samples)
        plugin.connect(plugin.tabs.currentChanged, self.refresh_samples)
        self.refresh_samples()

    def role(self, role):
        return self.plugin.role(role, self.session)

    def add_layer(self, layer, role, visible=True):
        return self.plugin.add_layer(layer, role, visible, session=self.session, group_name=SATELLITE_TITLE, top=True)

    def choose_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self.plugin.dock, "Select RGB GeoTIFFs",
                                                str(Path(self.sources[0]).parent) if self.sources else "",
                                                "GeoTIFF (*.tif *.tiff)")
        if paths:
            self.load_images(paths)

    def replaceable(self):
        sea = self.role("sea")
        if sea and sea.isModified():
            self.plugin.say("Save or roll back the current sea layer edits in QGIS, then try again.")
            return False
        return True

    def load_images(self, paths):
        cache = self.plugin.cache() / "satellite" / uuid4().hex
        paths = sorted(set(paths))

        def loaded(path):
            image = QgsRasterLayer(path, "RGB Satellite Imagery", "gdal")
            if not image.isValid():
                raise ValueError("QGIS could not open the satellite imagery.")
            self.plugin.end_tool()
            self.plugin.clear_vertex()
            self.disconnect_edit()
            self.session = uuid4().hex
            self.sources = paths
            self.area, self.land, self.offshore = [], [], []
            self.saved = None
            self.coverage = None
            if not self.project.mapLayers():
                self.project.setCrs(image.crs())
            self.add_layer(image, "satellite_image")
            self.refresh_samples()
            self.fill_reviews()
            self.canvas.setExtent(self.canvas.mapSettings().layerExtentToOutputExtent(image, image.extent()))
            self.canvas.refresh()
            self.plugin.say("Draw an area of interest and mark representative land and offshore samples.")

        self.plugin.launch("Loading satellite imagery", lambda cancel, progress: open_imagery(paths, cache, cancel, progress), loaded)

    def refresh_samples(self):
        image = self.role("satellite_image")
        self.source_label.setText(", ".join(Path(p).name for p in self.sources) if image else "Select RGB GeoTIFFs to start.")
        self.sample_label.setText(f"Land {len(self.land)} · Offshore {len(self.offshore)}")
        for key, (band, kind) in self.markers.items():
            band.reset(kind)
            points = [QgsPointXY(*xy) for xy in getattr(self, key)]
            if not image or not points or self.plugin.tabs.currentWidget() != self.page:
                continue
            geometry = (QgsGeometry.fromPolygonXY([points + [points[0]]]) if key == "area"
                        else QgsGeometry.fromMultiPointXY(points))
            band.setToGeometry(geometry, image.crs())
            band.show()

    def draw_area(self):
        image = self.role("satellite_image")
        if not image:
            self.plugin.say("Load RGB GeoTIFFs first.")
            return

        def set_area(_layer, points):
            vertices = [(pt.x(), pt.y()) for pt in points]
            polygon = shapely.Polygon(vertices)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError("Draw an area of interest whose edges do not cross.")
            self.area = vertices
            self.refresh_samples()
            self.plugin.say("Area of interest set. Add land and offshore samples within it.")

        self.plugin.start_sketch(image, "polygon", "Left-click to add points and right-click to finish.",
                                 set_area, snap=False)

    def place_sample(self, key):
        image = self.role("satellite_image")
        if not image or not self.area:
            self.plugin.say("Draw an area of interest on the map first.")
            return
        name = "land" if key == "land" else "offshore"

        def add_sample(_layer, points):
            point = (points[0].x(), points[0].y())
            if not shapely.Polygon(self.area).contains(shapely.Point(point)):
                raise ValueError("Select a point inside the area of interest.")
            getattr(self, key).append(point)
            self.refresh_samples()
            self.plugin.say(f"Added a {name} sample.")

        self.plugin.start_sketch(image, "point", f"Click a representative {name} point away from the shoreline.",
                                 add_sample, snap=False)

    def clear_samples(self):
        self.plugin.end_tool()
        self.land.clear()
        self.offshore.clear()
        self.refresh_samples()

    def generate(self):
        image = self.role("satellite_image")
        if not image or not self.area or not self.land or not self.offshore:
            self.plugin.say("Load imagery, draw an area of interest, and add both land and offshore samples.")
            return
        if not self.replaceable():
            return
        self.plugin.end_tool()
        draft = self.plugin.cache() / "satellite" / uuid4().hex / "draft.gpkg"
        args = (list(self.sources), image.source(), shapely.Polygon(self.area), list(self.land), list(self.offshore), draft)
        self.plugin.launch("Classifying sea areas by color and texture",
                           lambda cancel, progress: generate_shoreline(*args, cancel, progress), self.show_draft)

    def show_draft(self, path):
        if not self.replaceable():
            return
        sea = QgsVectorLayer(f"{path}|layername=sea", "Satellite Sea Area (editable)", "ogr")
        review = QgsVectorLayer(f"{path}|layername=review", "Satellite Review Areas", "ogr")
        if not sea.isValid() or not review.isValid():
            raise ValueError("QGIS could not open the generated sea and review areas.")
        self.plugin.clear_vertex()
        self.disconnect_edit()
        self.saved = None
        sea.renderer().setSymbol(QgsFillSymbol.createSimple({"color": "19,173,230,78", "outline_style": "no"}))
        review.renderer().setSymbol(QgsFillSymbol.createSimple({"color": "255,174,52,65", "outline_color": "#ffc26e"}))
        review.setReadOnly(True)
        self.add_layer(sea, "sea")
        self.add_layer(review, "satellite_review")
        coast = self.role("satellite_coast")
        if coast:
            self.project.removeMapLayer(coast.id())
        self.attach_edit(sea)
        self.refresh_coast()
        self.fill_reviews()
        self.plugin.say("Sea area and coastline are ready. Check the review areas and edit the vertices where needed.")

    def polygons(self, layer):
        return [shapely.from_wkb(bytes(feature.geometry().asWkb())) for feature in layer.getFeatures()
                if feature.hasGeometry()]

    def attach_edit(self, sea):
        path = sea.source().split("|", 1)[0]
        coverage = QgsVectorLayer(f"{path}|layername=coverage", "Valid Imagery Coverage", "ogr")
        self.coverage = shapely.union_all(self.polygons(coverage)) if coverage.isValid() else None
        for signal in (sea.undoStack().indexChanged, sea.afterCommitChanges, sea.afterRollBack):
            callback = lambda *_: self.redraw.start()
            signal.connect(callback)
            self.edit_connections.append((signal, callback))

    def disconnect_edit(self):
        self.redraw.stop()
        for signal, callback in self.edit_connections:
            with suppress(RuntimeError, TypeError):
                signal.disconnect(callback)
        self.edit_connections.clear()

    def refresh_coast(self):
        sea = self.role("sea")
        if not sea or self.coverage is None:
            return
        coast = self.role("satellite_coast")
        if coast is None or coast.crs() != sea.crs():
            coast = QgsVectorLayer("LineString", "Satellite Coastline", "memory")
            coast.setCrs(sea.crs())
            coast.renderer().symbol().setColor(QColor("#69efff"))
            coast.renderer().symbol().setWidth(0.48)
            coast.setReadOnly(True)
            coast.setCustomProperty("skipMemoryLayersCheck", 1)
            self.add_layer(coast, "satellite_coast")
        try:
            area = edited_sea(self.polygons(sea), self.coverage, lambda: False)
            lines = coastlines(area, self.coverage)
        except ValueError as error:
            lines = []
            self.plugin.say(str(error))
        provider = coast.dataProvider()
        provider.truncate()
        features = []
        for line in lines:
            feature = QgsFeature(coast.fields())
            geometry = QgsGeometry()
            geometry.fromWkb(line.wkb)
            feature.setGeometry(geometry)
            features.append(feature)
        provider.addFeatures(features)
        coast.updateExtents()
        coast.triggerRepaint()

    def fill_reviews(self):
        self.reviews.clear()
        layer = self.role("satellite_review")
        if layer:
            for feature in layer.getFeatures():
                item = QListWidgetItem(str(feature["reason"]))
                item.setData(Qt.ItemDataRole.UserRole, int(feature.id()))
                self.reviews.addItem(item)

    def zoom_review(self, item, _previous):
        layer = self.role("satellite_review")
        if item is None or not layer:
            return
        feature = layer.getFeature(item.data(Qt.ItemDataRole.UserRole))
        if feature.hasGeometry():
            extent = feature.geometry().boundingBox()
            extent.scale(2.5)
            image = self.role("satellite_image")
            if image and image.crs() == layer.crs():
                distance = 20 * max(image.rasterUnitsPerPixelX(), image.rasterUnitsPerPixelY())
                extent.grow(distance)
            self.project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(True)
            self.canvas.setExtent(self.canvas.mapSettings().layerExtentToOutputExtent(layer, extent))
            self.canvas.refresh()

    def edit_vertices(self):
        sea = self.role("sea")
        if not sea:
            self.plugin.say("Generate the sea area and coastline first.")
            return
        self.plugin.start_vertex(sea, "Use the QGIS Vertex Tool to edit the sea boundary. The coastline updates as you edit.")

    def choose_export(self):
        if not self.role("sea"):
            self.plugin.say("Generate the sea area and coastline first.")
            return
        folder = QFileDialog.getExistingDirectory(self.plugin.dock, "Select a Folder for the Shapefiles")
        if folder:
            self.export(folder)

    def export(self, folder):
        layer = self.role("sea")
        if not layer or self.coverage is None:
            self.plugin.say("Generate the sea area and coastline first.")
            return
        polygons = self.polygons(layer)
        signature = geometry_signature(polygons)
        crs, coverage = layer.crs().toWkt(), self.coverage

        def saved(path):
            self.saved = str(path), signature
            self.plugin.say(f"Saved the coastline and sea shapefiles.\n{path}")

        self.plugin.launch("Saving the coastline and sea shapefiles",
                           lambda cancel, progress: save_shapefiles(folder, crs, polygons, coverage, cancel, progress),
                           saved)

    def approve(self):
        sea = self.role("sea")
        if not sea or not self.saved or self.saved[1] != geometry_signature(self.polygons(sea)):
            self.plugin.say("Review your edits and save the current sea area as a shapefile before approving it.")
            return
        path = Path(self.saved[0]) / "sea_area.shp"
        if not all(path.with_suffix(suffix).is_file() for suffix in (".shp", ".shx", ".dbf", ".prj")):
            self.plugin.say("The saved sea shapefile or a required sidecar is missing. Export the shapefiles again.")
            return
        self.plugin.fields["verified_water"].setText(str(path))
        self.plugin.limit_water.setChecked(True)
        self.plugin.tabs.setCurrentIndex(0)
        self.plugin.say("Sea area approved for DTED. Select Source DTED, check the inputs, and choose 0 m or −50 m.")

    def remember(self):
        state = dict(session=self.session, sources=self.sources, area=self.area, land=self.land,
                     offshore=self.offshore, saved=self.saved)
        self.project.writeEntry("DtedCoastline", "/satellite", json.dumps(state, ensure_ascii=False))

    def restore(self):
        self.disconnect_edit()
        state = json.loads(self.project.readEntry("DtedCoastline", "/satellite", "{}")[0])
        self.session = state.get("session") or uuid4().hex
        self.sources, self.area = state.get("sources", []), state.get("area", [])
        self.land, self.offshore = state.get("land", []), state.get("offshore", [])
        self.saved = state.get("saved")
        self.coverage = None
        sea = self.role("sea")
        if sea:
            self.attach_edit(sea)
            self.refresh_coast()
        self.refresh_samples()
        self.fill_reviews()

    def dispose(self):
        self.disconnect_edit()
        for band, _ in self.markers.values():
            band.scene().removeItem(band)
        self.markers.clear()
