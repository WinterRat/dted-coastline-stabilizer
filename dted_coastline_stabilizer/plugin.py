import json
from math import isfinite
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy
import scipy
import shapely
from shapely.ops import substring
from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication, Qt, QT_VERSION_STR, PYQT_VERSION_STR, QUrl
from qgis.PyQt.QtGui import QAction, QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QPushButton,
    QScrollArea, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)
from qgis.core import (
    Qgis, QgsApplication, QgsContrastEnhancement, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature,
    QgsGeometry, QgsMapLayer, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle, QgsTask,
    QgsSingleBandGrayRenderer, QgsVectorFileWriter, QgsVectorLayer,
)
from qgis.gui import QgsDockWidget, QgsMapToolEdit, QgsRubberBand

from .coast import KEEP, LAND, NODATA, SEA, UNKNOWN, geometry_signature
from .engine import Cancelled, CHANGES, MASK, OUTPUT, PACKAGE, output_bundle, prepare_coast_edit
from .inputs import COAST_CRS, DEFAULT_WORK_CRS, inspect
from .satellite_panel import SatellitePanel


TITLE = "DTED Coastline Stabilizer"
GUIDE_LABEL = "한글 사용법"
SCOPE = "DtedCoastline"
ROLES = "dted-coast-role"
SESSION = "dted-coast-session"
GENERATED_SHORE = "dted-coast-generated-shore"
SOURCE_FIELDS = ("dem", "imagery", "shape", "reference", "verified_water")
OUTPUT_FIELDS = ("output_value", "curve", "narrow", "transition")
DEFAULT_TRANSITION = 12
DEFAULT_CURVE = "smoothstep"


def host_versions():
    import sys
    return (f"QGIS {Qgis.QGIS_VERSION.split('-')[0]} · Qt {QT_VERSION_STR} · PyQt {PYQT_VERSION_STR}\n"
            f"Python {sys.version.split()[0]} · GDAL {gdal.VersionInfo('RELEASE_NAME')}\n"
            f"NumPy {numpy.__version__} · Shapely {shapely.__version__} · SciPy {scipy.__version__}")


def shoreline_visible(layer, color, width):
    if layer.renderer() and layer.renderer().symbol():
        symbol = layer.renderer().symbol()
        symbol.setColor(QColor(color))
        if hasattr(symbol, "setWidth"):
            symbol.setWidth(width)


def elevation_visible(layer, limits):
    if not limits or limits[0] >= limits[1] or not isinstance(layer.renderer(), QgsSingleBandGrayRenderer):
        return
    renderer = layer.renderer()
    contrast = QgsContrastEnhancement(layer.dataProvider().dataType(1))
    contrast.setContrastEnhancementAlgorithm(QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum,
                                               False)
    contrast.setMinimumValue(limits[0], False)
    contrast.setMaximumValue(limits[1])
    renderer.setContrastEnhancement(contrast)


class Job(QgsTask):
    def __init__(self, label, work, callback):
        super().__init__(label, QgsTask.Flag.CanCancel)
        self.work = work
        self.callback = callback
        self.result = None
        self.error = ""

    def run(self):
        try:
            self.result = self.work(self.isCanceled, self.setProgress)
            return not self.isCanceled()
        except Cancelled:
            return False
        except Exception as error:
            self.error = str(error)
            return False

    def finished(self, ok):
        callback = self.callback
        self.callback = None
        self.work = None
        if callback:
            callback(ok, self.result, self.error)


class SketchTool(QgsMapToolEdit):
    def __init__(self, canvas, kind, done, snap):
        super().__init__(canvas)
        self.kind = kind
        self.done = done
        self.snap = snap
        self.points = []
        self.band = QgsRubberBand(canvas, Qgis.GeometryType.Polygon if kind == "polygon" else Qgis.GeometryType.Line)
        self.band.setColor(QColor("#00bfa5"))
        self.band.setWidth(2)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.mapPoint()
            if self.snap:
                match = self.canvas().snappingUtils().snapToMap(event.pos())
                if match.isValid():
                    point = match.point()
            self.points.append(QgsPointXY(point))
            self.band.addPoint(point)
            if self.kind == "point":
                self.done(self.points[:])
        elif event.button() == Qt.MouseButton.RightButton:
            count = 3 if self.kind == "polygon" else 2
            if len(self.points) >= count:
                self.done(self.points[:])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.done([])

    def deactivate(self):
        self.points.clear()
        self.band.hide()
        super().deactivate()

    def dispose(self):
        self.band.scene().removeItem(self.band)
        self.done = None
        self.deleteLater()


class CoastPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.project = QgsProject.instance()
        self.canvas = iface.mapCanvas()
        self.task = None
        self.active = False
        self.connections = []
        self.edit_connections = {}
        self.tool = None
        self.previous_tool = None
        self.native_vertex = None
        self.guide = None
        self.epoch = 0
        self.reset_session()

    def reset_session(self):
        self.session = uuid4().hex
        self.loaded = None
        self.info = None
        self.preview = None
        self.preview_key = None
        self.edit_generation = 0
        self.history = []
        self.edit_layers = {}
        self.confirmed = None

    def connect(self, signal, slot):
        signal.connect(slot)
        self.connections.append((signal, slot))

    def initGui(self):
        self.active = True
        self.action = QAction(TITLE, self.iface.mainWindow())
        self.action.setObjectName("dtedCoastlineAction")
        self.iface.addPluginToMenu(TITLE, self.action)
        self.guide_action = QAction("Korean User Guide", self.iface.mainWindow())
        self.iface.addPluginToMenu(TITLE, self.guide_action)
        self.dock = QgsDockWidget(TITLE, self.iface.mainWindow())
        self.dock.setObjectName("dtedCoastlineDock")
        self.dock.setMinimumWidth(360)
        self.build_panel()
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.connect(self.action.triggered, self.dock.show)
        self.connect(self.guide_action.triggered, self.open_guide)
        self.connect(self.iface.projectRead, self.restore_project)
        self.connect(self.iface.newProjectCreated, self.restore_project)
        self.connect(self.project.writeProject, self.remember)
        self.restore_project()
        self.dock.show()

    def unload(self):
        self.active = False
        self.remember()
        self.end_tool()
        self.clear_vertex()
        self.cancel(detach=True)
        self.disconnect_edits()
        self.satellite.dispose()
        for signal, slot in reversed(self.connections):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.connections.clear()
        for action in (self.action, self.guide_action):
            self.iface.removePluginMenu(TITLE, action)
            action.deleteLater()
        if self.guide is not None:
            self.guide.close()
            self.guide.deleteLater()
            self.guide = None
        self.iface.removeDockWidget(self.dock)
        self.dock.deleteLater()

    def open_guide(self):
        if self.guide is None:
            path = Path(__file__).resolve().with_name("user-guide-ko.html")
            if not path.is_file():
                self.iface.messageBar().pushWarning(TITLE, "한글 사용법 파일을 찾을 수 없습니다.")
                return

            self.guide = QDialog(self.iface.mainWindow())
            self.guide.setWindowTitle(f"{TITLE} · {GUIDE_LABEL}")
            self.guide.resize(720, 660)
            layout = QVBoxLayout(self.guide)
            browser = QTextBrowser()
            browser.setSource(QUrl.fromLocalFile(str(path)))
            browser.document().setDocumentMargin(14)
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.button(QDialogButtonBox.StandardButton.Close).setText("닫기")
            buttons.rejected.connect(self.guide.close)
            layout.addWidget(buttons)

        self.guide.show()
        self.guide.raise_()
        self.guide.activateWindow()

    def button(self, text, slot):
        button = QPushButton(text)
        self.connect(button.clicked, slot)
        return button

    def build_panel(self):
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 12, 12, 12)
        heading = QLabel("Coastline Tools")
        font = heading.font()
        font.setPointSize(font.pointSize() + 5)
        font.setBold(True)
        heading.setFont(font)
        header = QHBoxLayout()
        header.addWidget(heading, 1)
        header.addWidget(self.button(GUIDE_LABEL, self.open_guide))
        layout.addLayout(header)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.inputs_tab()
        self.analysis_tab()
        self.review_tab()
        self.satellite = SatellitePanel(self)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.cancel_button = self.button("Cancel Task", self.cancel)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setFrameShape(QFrame.Shape.StyledPanel)
        self.status.setMargin(8)
        layout.addWidget(self.status)
        self.dock.setWidget(body)

    def tab(self, label):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        page = QWidget()
        box = QVBoxLayout(page)
        outer.setWidget(page)
        self.tabs.addTab(outer, label)
        return box

    def path_row(self, name, folder=False, pattern="GeoTIFF (*.tif *.tiff)"):
        field = QLineEdit()
        self.fields[name] = field
        if name == "shape":
            self.connect(field.textChanged, self.shape_changed)
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        def choose():
            previous = field.text()
            if folder:
                value = QFileDialog.getExistingDirectory(self.dock, "Select Folder", previous)
            else:
                value, _ = QFileDialog.getOpenFileName(self.dock, "Select File", previous, pattern)
            if value:
                field.setText(value)
        browse = self.button("Browse…", choose)
        browse.setAccessibleName("Browse for " + name.replace("_", " "))
        row.addWidget(browse)
        return wrapper

    def inputs_tab(self):
        box = self.tab("Inputs")
        self.fields = {}
        form = QFormLayout()
        form.addRow("Source DTED", self.path_row("dem", pattern="DTED / GeoTIFF (*.dt0 *.dt1 *.dt2 *.tif *.tiff)"))
        form.addRow("RGB Imagery Folder", self.path_row("imagery", folder=True))
        form.addRow("Shoreline Shapefile", self.path_row("shape", pattern="Shapefile (*.shp)"))
        form.addRow("Validation Reference", self.path_row("reference"))
        form.addRow("Verified Sea Areas", self.path_row("verified_water", pattern="Vector (*.gpkg *.shp *.geojson)"))
        self.limit_water = QCheckBox("Apply only within verified sea areas")
        self.limit_water.setToolTip("Only apply the output elevation inside the selected polygons. A shoreline file is optional.")
        form.addRow(self.limit_water)
        self.shape_crs = QLineEdit()
        self.shape_crs.setPlaceholderText(f"Provider-confirmed CRS (supplied shoreline uses {COAST_CRS})")
        self.connect(self.shape_crs.textChanged, self.shape_changed)
        form.addRow("Shoreline CRS", self.shape_crs)
        self.confirm_button = self.button("Confirm Shoreline CRS", self.confirm_crs)
        form.addRow("", self.confirm_button)
        self.work_crs = QLineEdit(DEFAULT_WORK_CRS)
        form.addRow("Working CRS (meters)", self.work_crs)
        box.addLayout(form)
        box.addWidget(self.button("Check Inputs and Load Map", self.load_inputs))
        self.input_details = QLabel(host_versions())
        self.input_details.setWordWrap(True)
        self.input_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.input_details)
        box.addStretch()

    def choice(self, values):
        widget = QComboBox()
        for label, value in values:
            widget.addItem(label, value)
        return widget

    def number(self, value, upper):
        widget = QDoubleSpinBox()
        widget.setRange(0, upper)
        widget.setDecimals(1)
        widget.setSuffix(" m")
        widget.setValue(value)
        return widget

    def analysis_tab(self):
        box = self.tab("Analysis and Export")
        form = QFormLayout()
        self.input_value = self.choice([("Auto Detect", "auto"), ("0 m", "0"), ("−50 m", "-50"), ("Disabled", "off")])
        self.output_value = self.choice([("−50 m", -50), ("0 m", 0)])
        self.refine = QCheckBox("Refine shoreline using imagery")
        self.refine.setChecked(True)
        self.search = self.number(10, 60)
        self.join = self.number(3, 20)
        self.transition = self.number(DEFAULT_TRANSITION, 100)
        self.transition.setToolTip("Width of the seaward slope from the coast to −50 m. Set 0 m for a sharp boundary.")
        self.curve = self.choice([("Linear", "linear"), ("Smoothstep", "smoothstep")])
        self.curve.setCurrentIndex(self.curve.findData(DEFAULT_CURVE))
        self.narrow = QCheckBox("Narrow the transition in confined waters")
        self.narrow.setChecked(True)
        for label, widget in (("Input Sea Elevation", self.input_value), ("Output Sea Elevation", self.output_value),
                              ("Imagery Search Radius", self.search), ("Endpoint Join Distance", self.join),
                              ("Transition Width", self.transition), ("Transition Curve", self.curve)):
            form.addRow(label, widget)
        form.addRow(self.refine)
        form.addRow(self.narrow)
        self.connect(self.output_value.currentIndexChanged, self.output_selected)
        box.addLayout(form)
        box.addWidget(self.button("Preview DTED Elevation", self.run_preview))
        self.dted_level = self.choice([("GeoTIFF", None), ("GeoTIFF + DTED Level 0 (.dt0)", 0),
                                       ("GeoTIFF + DTED Level 1 (.dt1)", 1), ("GeoTIFF + DTED Level 2 (.dt2)", 2)])
        export_form = QFormLayout()
        export_form.addRow("Export Format", self.dted_level)
        box.addLayout(export_form)
        box.addWidget(self.button("Validate and Export…", self.save_as))
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        box.addWidget(self.summary)
        self.histogram = QLabel("Mask codes\nLand 1 · Sea 2 · Unclassified 3\nKeep Original 4 · NoData 255")
        self.histogram.setWordWrap(True)
        box.addWidget(self.histogram)
        box.addStretch()

    def review_tab(self):
        box = self.tab("Map Editing")
        self.guidance = QLabel("Edit the shoreline over the DTED. Left-click to add points, right-click to finish, and press Esc to cancel.")
        self.guidance.setWordWrap(True)
        box.addWidget(self.guidance)
        row = QHBoxLayout()
        row.addWidget(self.button("Start DTED Editing", self.start_dted_edit))
        row.addWidget(self.button("Inspect Elevation", self.dted_elevation))
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button("Land Seed", lambda: self.seed(LAND)))
        row.addWidget(self.button("Sea Seed", lambda: self.seed(SEA)))
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button("Edit Vertices", self.vertex_tool))
        row.addWidget(self.button("Replace Segment", self.replace_tool))
        row.addWidget(self.button("Add Shoreline", self.add_shoreline))
        box.addLayout(row)
        row = QHBoxLayout()
        for text, value in (("Force Land", LAND), ("Force Sea", SEA), ("Keep Original", KEEP)):
            row.addWidget(self.button(text, lambda checked=False, v=value: self.polygon(v)))
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button("Undo", lambda: self.undo_redo(False)))
        row.addWidget(self.button("Redo", lambda: self.undo_redo(True)))
        box.addLayout(row)
        box.addWidget(self.button("Save Edited Shoreline…", self.save_shoreline))
        box.addWidget(QLabel("Review Areas (select to zoom)"))
        self.reviews = QListWidget()
        self.connect(self.reviews.currentItemChanged, self.zoom_review)
        box.addWidget(self.reviews, 1)

    def say(self, message):
        self.status.setText(message)

    def output_selected(self):
        enabled = self.output_value.currentData() == -50
        for widget in (self.transition, self.curve, self.narrow):
            widget.setEnabled(enabled)

    def shape_changed(self):
        self.confirmed = None
        self.confirm_button.setText("Confirm Shoreline CRS")

    def confirm_crs(self):
        candidate = self.shape_crs.text().strip()
        if not self.fields["shape"].text() or not QgsCoordinateReferenceSystem(candidate).isValid():
            self.say("Choose a shoreline file and enter a valid CRS.")
            return
        self.confirmed = (self.fields["shape"].text().strip(), candidate)
        self.confirm_button.setText("Shoreline CRS Confirmed")
        self.say(f"Shoreline CRS {candidate} confirmed. Click Check Inputs and Load Map to apply it.")

    def config(self):
        options = {k: field.text().strip() for k, field in self.fields.items()}
        options.update(shape_crs=self.confirmed[1] if self.confirmed and self.confirmed[0] == options["shape"] else "",
                       work_crs=self.work_crs.text().strip(), output_value=self.output_value.currentData(),
                       input_value=self.input_value.currentData(), refine=self.refine.isChecked(),
                       limit_water=self.limit_water.isChecked(),
                       search=self.search.value(), join=self.join.value(), transition=self.transition.value(),
                       curve=self.curve.currentData(), narrow=self.narrow.isChecked())
        return options

    def intended_sources(self, options):
        return tuple(options[k] for k in (*SOURCE_FIELDS, "shape_crs", "work_crs", "limit_water"))

    def cache(self):
        return Path(QgsApplication.qgisSettingsDirPath()) / "dted-coastline-cache" / self.session

    def launch(self, label, worker, done):
        if self.task:
            self.say("Wait for the current task to finish or cancel it before trying again.")
            return
        epoch = self.epoch
        self.say(label)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        def finish(ok, result, error):
            if not self.active or epoch != self.epoch:
                return
            self.task = None
            self.progress.hide()
            self.cancel_button.hide()
            if ok:
                try:
                    done(result)
                except Exception as problem:
                    self.say(str(problem))
            else:
                self.say(error or "Task canceled.")
        self.task = Job(label, worker, finish)
        self.task.progressChanged.connect(lambda n: self.progress.setValue(int(n)))
        QgsApplication.taskManager().addTask(self.task)

    def cancel(self, detach=False):
        if self.task:
            if detach:
                self.task.callback = None
                self.task.progressChanged.disconnect()
            self.task.cancel()
            if detach:
                self.task = None
                self.progress.hide()
                self.cancel_button.hide()

    def load_inputs(self):
        config = self.config()
        cache = self.cache() / "inputs" / uuid4().hex
        def worker(cancel, progress):
            if cancel():
                raise Cancelled()
            progress(10)
            result = inspect(config, cache)
            progress(100)
            return result
        def done(info):
            self.apply_inputs(config, info)
        self.launch("Checking input data", worker, done)

    def role(self, role, session=None):
        session = session or self.session
        return next((layer for layer in self.project.mapLayers().values()
                     if layer.customProperty(SESSION) == session and layer.customProperty(ROLES) == role), None)

    def add_layer(self, layer, role, visible=True, *, session=None, group_name=TITLE, top=False):
        if not layer.isValid():
            raise ValueError(f"Could not open the layer {layer.name()}.")
        session = session or self.session
        group = next((g for g in self.project.layerTreeRoot().findGroups()
                      if g.customProperty(SESSION) == session), None)
        if group is None:
            tree = self.project.layerTreeRoot()
            group = tree.insertGroup(0, group_name) if top else tree.addGroup(group_name)
            group.setCustomProperty(SESSION, session)
        existing = self.role(role, session)
        if existing:
            self.project.removeMapLayer(existing.id())
        layer.setCustomProperty(SESSION, session)
        layer.setCustomProperty(ROLES, role)
        self.project.addMapLayer(layer, False)
        node = group.insertLayer(0, layer)
        node.setItemVisibilityChecked(visible)
        return layer

    def apply_inputs(self, config, info):
        key = self.intended_sources(config)
        if self.loaded and self.loaded != key:
            self.detach_current_layers()
            if config["shape_crs"]:
                self.confirmed = (config["shape"], config["shape_crs"])
        self.loaded = key
        self.info = info
        dem = self.add_layer(QgsRasterLayer(config["dem"], "Source DTED"), "dem", top=config["limit_water"])
        if info["reference"]:
            layer = QgsRasterLayer(config["reference"], "Validation Reference")
            elevation_visible(layer, info["reference"].get("display_range"))
            self.add_layer(layer, "reference", False)
        if info["imagery"]:
            self.add_layer(QgsRasterLayer(info["imagery"], "RGB Satellite Imagery"), "image", False)
        shape = info["shape"]
        crs_label = "CRS unconfirmed"
        if shape and shape["confirmed"]:
            coast = QgsVectorLayer(shape["path"], "Source Shoreline", "ogr")
            crs = QgsCoordinateReferenceSystem.fromWkt(shape["crs"])
            coast.setCrs(crs)
            coast.setReadOnly(True)
            crs_label = crs.authid() or crs.description()
            shoreline_visible(coast, "#f5b94f", 0.40)
            self.add_layer(coast, "shore")
        source = info["dem"]
        image_text = f"Imagery tiles {len(info['images'])}"
        details = (f"{source['authority']} · {source['size'][0]:,} × {source['size'][1]:,}\n"
                   f"{source['type']} · PixelIs{source['pixel'] or 'Area'} · {source['unit'] or 'Unit unspecified'}\n"
                   f"NoData {source['nodata']}\nGrid transform " + ", ".join(f"{x:.9g}" for x in source["affine"]) +
                   f"\n\n{image_text}")
        if shape:
            details += f" · Shoreline features {shape['count']:,} · {crs_label}"
        if info["reference"]:
            ref = info["reference"]
            details += f"\nReference {ref['size'][0]:,} × {ref['size'][1]:,} · PixelIs{ref['pixel']}"
        self.input_details.setText(details)
        if config["limit_water"]:
            areas = QgsVectorLayer(config["verified_water"], "Verified DTED Sea Areas", "ogr")
            areas.setReadOnly(True)
            areas.setOpacity(0.45)
            self.add_layer(areas, "verified_water")
            self.say("Verified sea areas loaded. Choose the output elevation in Analysis and Export.")
        elif shape["confirmed"]:
            self.ensure_sketch_layers()
            self.say("Inputs checked. Open Map Editing to start editing over the DTED.")
        else:
            self.say("The shoreline has no .prj file. Enter and confirm the CRS supplied by the data provider.")
        self.canvas.setExtent(self.canvas.mapSettings().layerExtentToOutputExtent(dem, dem.extent()))
        self.canvas.refresh()
        self.remember()

    def detach_current_layers(self):
        self.end_tool()
        self.clear_vertex()
        self.disconnect_edits()
        # Existing project layers remain available as their previous work group.
        self.reset_session()
        self.reviews.clear()
        self.summary.clear()

    def vector_result(self, folder, table, label):
        return QgsVectorLayer(str(Path(folder) / PACKAGE) + "|layername=" + table, label, "ogr")

    def open_edit_layer(self, folder):
        edited = self.vector_result(folder, "edited", "Edited Shoreline")
        shoreline_visible(edited, "#11b9d7", 0.46)
        previous = self.role("edited")
        active = previous and self.iface.activeLayer() == previous
        edited.setCustomProperty(GENERATED_SHORE, geometry_signature(self.line_geometries(edited)))
        if previous:
            self.clear_vertex()
            self.disconnect_edit("edited")
        self.add_layer(edited, "edited")
        self.attach_edit(edited, "edited")
        if active:
            self.iface.setActiveLayer(edited)

    def start_dted_edit(self):
        config = self.config()
        if not self.info or self.intended_sources(config) != self.loaded or not self.role("dem"):
            self.say("Click Check Inputs and Load Map first.")
            return
        if config["limit_water"]:
            self.say("The selected sea polygons define the DTED range. Open Analysis and Export to apply an elevation.")
            return
        if self.role("edited"):
            self.show_dted_edit()
            return
        info = self.info
        folder = self.cache() / "work" / uuid4().hex

        def work(cancel, progress):
            return prepare_coast_edit(config, info, folder, cancel, progress)

        def done(result):
            self.open_edit_layer(result)
            self.show_dted_edit()

        self.launch("Preparing the editable shoreline within the DTED extent", work, done)

    def show_dted_edit(self):
        tree = self.project.layerTreeRoot()
        for role in ("dem", "shore", "edited", "image", "reference", "result", "mask", "changes"):
            layer = self.role(role)
            node = tree.findLayer(layer.id()) if layer else None
            if node:
                if role in ("dem", "edited"):
                    node.setItemVisibilityCheckedParentRecursive(True)
                else:
                    node.setItemVisibilityChecked(False)
        if self.edit_target("edited"):
            self.say("The editable shoreline is displayed over the DTED. You can edit vertices or add shorelines.")
        self.canvas.refresh()
        self.remember()

    def dted_elevation(self):
        dem = self.role("dem")
        if not dem:
            self.say("Load the source DTED on the map first.")
            return

        def sample(layer, points):
            point = points[0]
            samples = []
            for raster_layer, label in ((layer, "Source DTED"), (self.role("result"), "Output Elevation")):
                if raster_layer:
                    elevation, ok = raster_layer.dataProvider().sample(point, 1)
                    samples.append(f"{label} {elevation:g} m" if ok and isfinite(elevation) else f"{label} NoData")
            mask = self.role("mask")
            if mask:
                value, ok = mask.dataProvider().sample(point, 1)
                classes = {LAND: "Land", SEA: "Sea", UNKNOWN: "Unclassified", KEEP: "Keep Original", NODATA: "NoData"}
                samples.append("Class " + (classes.get(value, "NoData") if ok else "NoData"))
            self.say("\n".join(samples))

        self.sketch("dem", "point", "Click the map to inspect the elevation at that point.", sample, edit=False)

    def save_shoreline(self):
        layer = self.role("edited")
        if not layer:
            self.say("Click Start DTED Editing first.")
            return
        dialog = QFileDialog(self.dock, "Save Edited Shoreline", "", "GeoPackage (*.gpkg)")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("gpkg")
        dialog.selectFile("edited-coastline.gpkg")
        if not dialog.exec():
            return
        path = Path(dialog.selectedFiles()[0]).resolve()
        if path.suffix.lower() != ".gpkg":
            self.say("Save the shoreline with the .gpkg extension.")
            return
        source_paths = [Path(self.fields[key].text()).resolve() for key in SOURCE_FIELDS if self.fields[key].text()]
        source_paths.append(Path(layer.source().split("|", 1)[0]).resolve())
        if path in source_paths:
            self.say("Choose a filename different from the input or the file being edited.")
            return
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = "edited_coastline"
        options.skipAttributeCreation = True
        source_crs = QgsCoordinateReferenceSystem.fromWkt(self.info["shape"]["crs"])
        if layer.crs() != source_crs:
            options.ct = QgsCoordinateTransform(layer.crs(), source_crs, self.project)
        try:
            with TemporaryDirectory(dir=path.parent) as folder:
                pending = Path(folder) / path.name
                error, message, *_ = QgsVectorFileWriter.writeAsVectorFormatV3(
                    layer, str(pending), self.project.transformContext(), options)
                if error != QgsVectorFileWriter.WriterError.NoError:
                    raise OSError(message)
                pending.replace(path)
        except OSError as error:
            self.say(f"Could not save the shoreline.\n{error}")
            return
        self.say(f"Saved the edited shoreline.\n{path}")

    def ensure_sketch_layers(self):
        crs = QgsCoordinateReferenceSystem.fromWkt(self.info["work"])
        for role, name, geometry, fields in (
            ("seeds", "Classification Seeds", "Point", "&field=class:integer"),
            ("manual", "Manual Classification", "Polygon", "&field=class:integer&field=sequence:integer"),
        ):
            layer = self.role(role)
            if layer is None:
                layer = QgsVectorLayer(f"{geometry}?crs={crs.authid() or 'EPSG:32652'}{fields}", name, "memory")
                layer.setCrs(crs)
                self.add_layer(layer, role)
            self.attach_edit(layer, role)

    def attach_edit(self, layer, role):
        if role in self.edit_layers:
            return
        self.edit_layers[role] = layer
        def change(*_):
            self.edit_generation += 1
        def command():
            self.history.append((datetime.now(timezone.utc).isoformat(), role, "Edit"))
        stack = layer.undoStack()
        connections = [(stack.indexChanged, change), (layer.editCommandEnded, command),
                       (layer.afterCommitChanges, change), (layer.afterRollBack, change)]
        for sig, slot in connections:
            sig.connect(slot)
        self.edit_connections[role] = connections

    def disconnect_edit(self, role):
        for signal, slot in reversed(self.edit_connections.pop(role, [])):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.edit_layers.pop(role, None)

    def disconnect_edits(self):
        for role in list(self.edit_connections):
            self.disconnect_edit(role)

    def line_geometries(self, layer):
        return [shapely.from_wkb(bytes(feature.geometry().asWkb()))
                for feature in layer.getFeatures() if feature.hasGeometry()]

    def snapshots(self):
        previous_mask = Path(self.preview) / MASK if self.preview else None
        result = {"history": list(self.history),
                  "previous_mask": str(previous_mask) if previous_mask and previous_mask.is_file() else None}
        for role, layer in self.edit_layers.items():
            if role == "edited":
                lines = self.line_geometries(layer)
                if geometry_signature(lines) != layer.customProperty(GENERATED_SHORE):
                    result["lines"] = lines
                continue
            rows = [(shapely.from_wkb(bytes(f.geometry().asWkb())), f) for f in layer.getFeatures() if f.hasGeometry()]
            if role == "manual":
                result["overrides"] = [(geom, int(f["class"]), int(f["sequence"])) for geom, f in rows]
            elif role == "seeds":
                result["seeds"] = [(geom, int(f["class"])) for geom, f in rows]
        return result

    def run_preview(self):
        self.run_output(self.cache() / "previews", False)

    def save_as(self):
        folder = QFileDialog.getExistingDirectory(self.dock, "Select Output Folder")
        if folder:
            self.run_output(folder, True)

    def run_output(self, folder, final):
        config = self.config()
        dted_level = self.dted_level.currentData() if final else None
        if not self.info or self.intended_sources(config) != self.loaded:
            self.say("Click Check Inputs and Load Map first.")
            return
        if not config["limit_water"] and not self.info["shape"]["confirmed"]:
            self.say("Confirm the shoreline CRS.")
            return
        key = ([config[k] for k in config if k not in OUTPUT_FIELDS], self.edit_generation)
        reused = self.preview if self.preview_key == key and self.preview and Path(self.preview).is_dir() else None
        edits = self.snapshots()
        info = self.info
        def work(cancel, progress):
            return output_bundle(config, info, edits, folder, reused, cancel, progress, dted_level=dted_level)
        def done(result):
            self.preview, report = result
            self.preview_key = key
            refresh_edit = not reused and "lines" not in edits and self.edit_generation == key[1] and not self.tool
            self.apply_result(self.preview, report, final, refresh_edit, config["limit_water"])
        self.launch("Exporting and validating results" if final else "Classifying sea areas and building the preview", work, done)

    def apply_result(self, folder, report, final, refresh_edit=False, limit_water=False):
        limits = report["display_range_m"]
        style_notice = ""
        for name, label, role, show in (
            (OUTPUT, "Output Elevation", "result", False), (MASK, "Sea Mask", "mask", False),
            (CHANGES, "Changed Pixels", "changes", False),
        ):
            layer = QgsRasterLayer(str(Path(folder) / name), label)
            if name == OUTPUT:
                elevation_visible(layer, limits)
                if final and limits and limits != report["elevation_range_m"]:
                    _, saved = layer.saveNamedStyle(str(Path(folder) / "elevation.qml"), QgsMapLayer.StyleCategory.Symbology)
                    if not saved:
                        style_notice = "\nQGIS could not save the reference display range for reopening."
            self.add_layer(layer, role, show)
        auto = self.vector_result(folder, "automatic", "Automatic Shoreline")
        shoreline_visible(auto, "#5bd4c0", 0.40)
        self.add_layer(auto, "automatic", False)
        if not limit_water and (refresh_edit or not self.role("edited")):
            self.open_edit_layer(folder)
        review = self.vector_result(folder, "review", "Review Areas")
        shoreline_visible(review, "#de922d", 0.55)
        self.add_layer(review, "review", False)
        self.fill_reviews()
        baseline = report["input_sea_value"]
        display = f"\nDisplay range {limits[0]:g} to {limits[1]:g} m" if limits else ""
        self.summary.setText(f"Input Sea Elevation {str(baseline) + ' m' if baseline is not None else 'Unknown'}\n"
                             f"Changed pixels {report['changed_pixels']:,}\n"
                             f"Unclassified area {report['unknown_area_m2']:,.1f} m²\n"
                             f"{report['seconds']:.1f} s · {report['workers']} CPU workers"
                             + display
                             + ("\nReused the previous sea mask" if report["reused_mask"] else "")
                             + (f"\nDTED tiles saved {len(report['dted_tiles'])}" if report.get("dted_tiles") else ""))
        self.say(f"Saved the validated results.\n{folder}{style_notice}" if final else
                 "Preview ready. Review the map or export the results.")
        self.canvas.refresh()
        self.remember()

    def fill_reviews(self):
        self.reviews.clear()
        layer = self.role("review")
        if not layer:
            return
        for index, feature in enumerate(layer.getFeatures(), 1):
            item = QListWidgetItem(str(feature["reason"]))
            item.setData(Qt.ItemDataRole.UserRole, int(feature.id()))
            self.reviews.addItem(item)

    def zoom_review(self, item, _previous):
        layer = self.role("review")
        if not layer or item is None:
            return
        feature = layer.getFeature(item.data(Qt.ItemDataRole.UserRole))
        if not feature.isValid():
            return
        extent = feature.geometry().boundingBox()
        extent.grow(max(max(extent.width(), extent.height()) * 0.15, 15))
        self.project.layerTreeRoot().findLayer(layer.id()).setItemVisibilityChecked(True)
        self.canvas.setExtent(self.canvas.mapSettings().layerExtentToOutputExtent(layer, extent))
        self.canvas.refresh()

    def edit_target(self, role):
        layer = self.role(role)
        if layer is None:
            self.say("Load the inputs, then click Start DTED Editing.")
            return None
        if not layer.isEditable() and not layer.startEditing():
            self.say("Could not start editing the layer.")
            return None
        self.iface.setActiveLayer(layer)
        return layer

    def clear_vertex(self):
        if self.native_vertex and self.canvas.mapTool() is self.native_vertex:
            self.canvas.setMapTool(self.previous_tool) if self.previous_tool else self.canvas.unsetMapTool(self.native_vertex)
        self.native_vertex = None
        self.previous_tool = None

    def end_tool(self):
        if self.tool:
            tool = self.tool
            self.tool = None
            if self.canvas.mapTool() is tool:
                self.canvas.setMapTool(self.previous_tool) if self.previous_tool else self.canvas.unsetMapTool(tool)
            self.previous_tool = None
            tool.dispose()

    def vertex_tool(self):
        layer = self.role("edited")
        if not layer:
            self.say("Load the inputs, then click Start DTED Editing.")
            return
        self.start_vertex(layer, "The QGIS Vertex Tool is active. Use it to move, add, or delete vertices on the map.")

    def start_vertex(self, layer, message):
        self.end_tool()
        self.clear_vertex()
        if not layer.isEditable() and not layer.startEditing():
            self.say("Could not start editing the layer.")
            return
        self.iface.setActiveLayer(layer)
        self.previous_tool = self.canvas.mapTool()
        action = self.iface.mainWindow().findChild(QAction, "mActionVertexToolActiveLayer")
        if action is None:
            action = self.iface.actionVertexTool()
        action.trigger()
        self.native_vertex = self.canvas.mapTool()
        self.say(message)

    def sketch(self, role, kind, message, apply, edit=True):
        layer = self.edit_target(role) if edit else self.role(role)
        if layer is None:
            return
        self.start_sketch(layer, kind, message, apply, snap=edit)

    def start_sketch(self, layer, kind, message, apply, snap):
        self.end_tool()
        self.clear_vertex()
        self.previous_tool = self.canvas.mapTool()
        def done(points):
            self.end_tool()
            if points:
                try:
                    transform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), layer.crs(), self.project)
                    apply(layer, [transform.transform(point) for point in points])
                except Exception as error:
                    self.say(str(error))
        self.tool = SketchTool(self.canvas, kind, done, snap=snap)
        self.canvas.setMapTool(self.tool)
        self.say(message)

    def add_feature(self, layer, geometry, values, command):
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        for name, value in values.items():
            feature[name] = value
        layer.beginEditCommand(command)
        if layer.addFeature(feature):
            layer.endEditCommand()
            layer.triggerRepaint()
            self.say("Edit applied. Click Preview DTED Elevation to refresh the preview.")
        else:
            layer.destroyEditCommand()
            self.say("Could not apply the edit.")

    def seed(self, value):
        self.sketch("seeds", "point", "Click inside a land or sea area to place a classification seed.",
                    lambda layer, points: self.add_feature(layer, QgsGeometry.fromPointXY(points[0]), {"class": value}, "Add Seed"))

    def polygon(self, value):
        def add(layer, points):
            polygon = QgsGeometry.fromPolygonXY([points + [points[0]]])
            if not polygon.isGeosValid():
                raise ValueError("Draw a polygon without self-intersections.")
            order = max((int(f["sequence"]) for f in layer.getFeatures()), default=0) + 1
            self.add_feature(layer, polygon, {"class": value, "sequence": order}, "Add Manual Classification Area")
        self.sketch("manual", "polygon", "Click the map to draw a polygon. Right-click to finish.", add)

    def replace_tool(self):
        def replace(layer, points):
            xy = [(p.x(), p.y()) for p in points]
            first, last = shapely.Point(xy[0]), shapely.Point(xy[-1])
            spacing = max(2.0, self.search.value())
            nearby = QgsRectangle(first.x - spacing, first.y - spacing, first.x + spacing, first.y + spacing)
            from qgis.core import QgsFeatureRequest
            candidates = []
            for feature in layer.getFeatures(QgsFeatureRequest().setFilterRect(nearby)):
                line = shapely.from_wkb(bytes(feature.geometry().asWkb()))
                if line.geom_type == "LineString" and max(line.distance(first), line.distance(last)) <= spacing:
                    candidates.append((line.distance(first) + line.distance(last), feature.id(), line))
            if not candidates:
                raise ValueError("Snap both endpoints to the same shoreline.")
            _, fid, line = min(candidates, key=lambda item: item[0])
            start, end = line.project(first), line.project(last)
            if start > end:
                start, end, xy = end, start, list(reversed(xy))
            if end - start < 0.01:
                raise ValueError("Choose two distinct endpoints.")
            left, right = substring(line, 0, start), substring(line, end, line.length)
            edited = shapely.LineString([*left.coords, *xy[1:-1], *right.coords])
            if not edited.is_simple:
                raise ValueError("The replacement segment would create a self-intersection in the shoreline.")
            qgeometry = QgsGeometry()
            qgeometry.fromWkb(bytes(edited.wkb))
            layer.beginEditCommand("Replace Segment")
            if not layer.changeGeometry(fid, qgeometry):
                layer.destroyEditCommand()
                raise ValueError("Could not replace the shoreline segment.")
            layer.endEditCommand()
            layer.triggerRepaint()
            self.say("Segment replaced. Click Preview DTED Elevation to refresh the preview.")
        self.sketch("edited", "line", "Snap the replacement segment's endpoints to the same shoreline. Right-click to finish.", replace)

    def add_shoreline(self):
        def add(layer, points):
            line = QgsGeometry.fromPolylineXY(points)
            if line.length() <= 0 or not line.isSimple():
                raise ValueError("Draw a shoreline with distinct points and no self-intersections.")
            self.add_feature(layer, line, {}, "Add Shoreline")

        self.sketch("edited", "line", "Draw the shoreline, then right-click to finish.", add)

    def undo_redo(self, redo):
        layer = self.iface.activeLayer()
        if layer and layer in self.edit_layers.values():
            stack = layer.undoStack()
            if redo and stack.canRedo():
                stack.redo()
                word = "Redo"
            elif not redo and stack.canUndo():
                stack.undo()
                word = "Undo"
            else:
                return
            self.history.append((datetime.now(timezone.utc).isoformat(), layer.customProperty(ROLES), word))
            layer.triggerRepaint()

    def remember(self, *_):
        if not hasattr(self, "fields"):
            return
        def records(role):
            layer = self.role(role)
            return [(bytes(f.geometry().asWkb()).hex(), [f[name] for name in names])
                    for names in [("class", "sequence") if role == "manual" else ("class",) if role == "seeds" else ()]
                    for f in layer.getFeatures() if f.hasGeometry()] if layer else None
        def stored(role):
            layer = self.role(role)
            return records(role) if layer and (layer.isEditable() or layer.providerType() == "memory") else None
        state = dict(session=self.session, config=self.config(), supplied_crs=self.shape_crs.text(), loaded=self.loaded,
                     preview=self.preview, info=self.info, history=self.history,
                     sketches={role: stored(role) for role in ("seeds", "manual", "edited")})
        self.project.writeEntry(SCOPE, "/session", json.dumps(state, ensure_ascii=False))
        self.satellite.remember()

    def restore_project(self, *_):
        self.epoch += 1
        self.end_tool()
        self.clear_vertex()
        self.cancel(detach=True)
        self.disconnect_edits()
        self.reset_session()
        self.reviews.clear()
        self.summary.clear()
        state = json.loads(self.project.readEntry(SCOPE, "/session", "{}")[0])
        values = state.get("config") or {}
        for name, field in self.fields.items():
            field.setText(values.get(name, ""))
        self.shape_crs.setText(state.get("supplied_crs", ""))
        self.work_crs.setText(values.get("work_crs", DEFAULT_WORK_CRS))
        if values.get("shape_crs"):
            self.confirmed = (values["shape"], values["shape_crs"])
            self.confirm_button.setText("Shoreline CRS Confirmed")
        for combo, key, default in ((self.input_value, "input_value", "auto"), (self.output_value, "output_value", -50),
                                    (self.curve, "curve", DEFAULT_CURVE)):
            combo.setCurrentIndex(combo.findData(values.get(key, default)))
        for widget, key, default in ((self.search, "search", 10), (self.join, "join", 3),
                                     (self.transition, "transition", DEFAULT_TRANSITION)):
            widget.setValue(values.get(key, default))
        self.refine.setChecked(values.get("refine", True))
        self.narrow.setChecked(values.get("narrow", True))
        self.limit_water.setChecked(values.get("limit_water", False))
        self.output_selected()
        self.satellite.restore()
        self.input_details.setText(host_versions())
        self.say("Choose a source DTED and a shoreline file.")
        if not state:
            return
        self.session = state["session"]
        self.history = state["history"]
        self.preview = state.get("preview")
        self.loaded = tuple(state["loaded"]) if state.get("loaded") else None
        if self.loaded is not None and len(self.loaded) == len(SOURCE_FIELDS) + 2:
            self.loaded += (False,)
        self.info = state.get("info")
        if self.info and self.info["shape"] and self.info["shape"]["confirmed"] and not self.limit_water.isChecked():
            self.ensure_sketch_layers()
        for role, stored in state.get("sketches", {}).items():
            layer = self.role(role)
            if layer is None:
                continue
            self.attach_edit(layer, role)
            if stored is None:
                continue
            existing = [(bytes(f.geometry().asWkb()).hex(), f.attributes()[-len(stored[0][1]):] if stored and stored[0][1] else [])
                        for f in layer.getFeatures() if f.hasGeometry()]
            if existing != [(wkb, attrs) for wkb, attrs in stored]:
                layer.startEditing()
                layer.beginEditCommand("Restore Shoreline Edits")
                layer.deleteFeatures([f.id() for f in layer.getFeatures()])
                for wkb, attrs in stored:
                    geometry = QgsGeometry()
                    geometry.fromWkb(bytes.fromhex(wkb))
                    feature = QgsFeature(layer.fields())
                    feature.setGeometry(geometry)
                    for name, value in zip(("class", "sequence"), attrs):
                        feature[name] = value
                    layer.addFeature(feature)
                layer.endEditCommand()
        self.fill_reviews()
        self.say("Saved work reopened. If the input files have moved, select their new locations.")
