from collections import defaultdict
from hashlib import sha256
import math
import numpy as np
from osgeo import gdal
import shapely
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .inputs import footprint, reference, reproject, transform, vector


LAND, SEA, UNKNOWN, KEEP, NODATA = 1, 2, 3, 4, 255


def geometry_signature(geometries):
    digest = sha256()
    for geometry in sorted(geom.wkb for geom in geometries):
        digest.update(len(geometry).to_bytes(8, "little"))
        digest.update(geometry)
    return digest.hexdigest()


def parts(geometry, kinds):
    if geometry.is_empty:
        return
    if geometry.geom_type in kinds:
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from parts(child, kinds)


class RasterSampler:
    def __init__(self, ds, source, bands, alpha=None):
        self.ds = ds
        self.to_raster = transform(source, ds.GetProjection())
        self.inverse = gdal.InvGeoTransform(ds.GetGeoTransform())
        self.bands = bands
        self.alpha = alpha

    def read(self, coordinates):
        xy = np.asarray(coordinates).reshape(-1, 2)
        i = self.inverse
        x, y = self.to_raster(xy[:, 0], xy[:, 1])
        col = np.floor(i[0] + i[1] * x + i[2] * y).astype(int)
        row = np.floor(i[3] + i[4] * x + i[5] * y).astype(int)
        values = np.full((len(xy), len(self.bands)), np.nan)
        groups = defaultdict(list)
        for k in np.flatnonzero((col >= 0) & (col < self.ds.RasterXSize) &
                                (row >= 0) & (row < self.ds.RasterYSize)):
            groups[(col[k] // 256, row[k] // 256)].append(k)
        for indices in groups.values():
            c, r = col[indices], row[indices]
            x0, y0 = int(c.min()), int(r.min())
            shape = (x0, y0, int(c.max()) - x0 + 1, int(r.max()) - y0 + 1)
            at = (r - y0, c - x0)
            mask = (self.ds.GetRasterBand(self.alpha) if self.alpha else
                    self.ds.GetRasterBand(self.bands[0]).GetMaskBand())
            valid = mask.ReadAsArray(*shape)[at] > 0
            for n, band in enumerate(self.bands):
                values[np.asarray(indices)[valid], n] = self.ds.GetRasterBand(band).ReadAsArray(*shape)[at][valid]
        return values


def water_probability(rgb):
    red, green, blue = np.moveaxis(rgb.astype(float), -1, 0)
    total = np.maximum(red + green + blue, 1)
    return np.clip(0.5 + 2.2 * (blue - red) / total + 0.8 * (blue - green) / total, 0, 1)


def load_lines(info, domain, check):
    shape = info["shape"]
    if not shape["confirmed"]:
        raise ValueError("Enter and confirm the shoreline CRS before starting analysis.")
    source, work = shape["crs"], info["work"]
    requested = reproject(shapely.segmentize(domain, 1000), work, source)
    result = []
    review = []
    with vector(shape["path"]) as layer:
        layer.SetSpatialFilterRect(*requested.bounds)
        for n, feature in enumerate(layer):
            if n % 250 == 0:
                check()
            ogr_geom = feature.GetGeometryRef()
            if ogr_geom is None or ogr_geom.IsEmpty():
                continue
            geom = reproject(shapely.force_2d(shapely.from_wkb(bytes(ogr_geom.ExportToWkb()))), source, work)
            if not geom.is_valid or not geom.is_simple:
                review.extend((line, "Intersecting or invalid shoreline") for line in parts(geom, {"LineString"}))
            clipped = shapely.make_valid(geom).intersection(domain)
            result.extend(parts(clipped, {"LineString"}))
    return result, review


def join_endpoints(lines, distance):
    if not lines or distance <= 0:
        return lines
    endpoints = [(i, end, Point(line.coords[end])) for i, line in enumerate(lines)
                 if not line.is_ring for end in (0, -1)]
    tree = STRtree([end[2] for end in endpoints])
    network = STRtree(lines)
    choices = {}
    for i, (line_id, end, point) in enumerate(endpoints):
        if len(network.query(point, predicate="intersects")) > 1:
            continue
        targets = [int(j) for j in tree.query(point, predicate="dwithin", distance=distance)
                   if endpoints[j][0] != line_id and not point.equals(endpoints[j][2])]
        if len(targets) != 1:
            continue
        j = targets[0]
        coords = lines[line_id].coords
        tangent = np.asarray(point.coords[0]) - np.asarray(coords[1 if end == 0 else -2])
        delta = np.asarray(endpoints[j][2].coords[0]) - np.asarray(point.coords[0])
        magnitude = np.linalg.norm(tangent)
        if magnitude > 0 and np.dot(tangent, delta) >= 0.65 * magnitude * np.linalg.norm(delta):
            choices[i] = j
    bridges = []
    for i, j in choices.items():
        if i < j and choices.get(j) == i:
            first, last = np.asarray([endpoints[k][2].coords[0] for k in (i, j)])
            tangents = []
            for k in (i, j):
                line_id, end, point = endpoints[k]
                tangent = np.asarray(point.coords[0]) - np.asarray(lines[line_id].coords[1 if end == 0 else -2])
                tangents.append(tangent / np.linalg.norm(tangent))
            span = np.linalg.norm(last - first)
            controls = [first, first + tangents[0] * span / 3, last + tangents[1] * span / 3, last]
            t = np.linspace(0, 1, max(3, math.ceil(span) + 1))[:, None]
            bridge = LineString((1 - t)**3 * controls[0] + 3 * t * (1 - t)**2 * controls[1] +
                                3 * t**2 * (1 - t) * controls[2] + t**3 * controls[3])
            if len(network.query(bridge, predicate="intersects")) == 2 and all(
                    bridge.intersection(lines[endpoints[k][0]]).equals(endpoints[k][2]) for k in (i, j)):
                bridges.append(bridge)
    return lines + bridges


def image_refine(lines, sampler, width, check):
    if not sampler or width <= 0:
        return lines, [], []
    result, review, distances = [], [], []
    original = STRtree(lines)
    offsets = np.linspace(-width, width, 2 * math.ceil(width) + 1)
    center = len(offsets) // 2
    penalty = 0.035 * abs(offsets[:, None] - offsets)
    for i, line in enumerate(lines):
        check()
        # Sample in batches while solving over the whole line.
        chain = np.arange(0, line.length, 2.0)
        chain = np.append(chain, line.length)
        if len(chain) < 4:
            result.append(line)
            continue
        xy = shapely.get_coordinates(shapely.line_interpolate_point(line, chain))
        before = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(0, chain - 2)))
        after = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(line.length, chain + 2)))
        tangent = after - before
        normal = tangent[:, ::-1] * [-1, 1] / np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-6)
        costs = np.full((len(chain), len(offsets)), np.inf)
        anchored = np.zeros(len(chain), dtype=bool)
        reasons = []
        for start in range(0, len(chain), 128):
            check()
            side = normal[start:start + 128, None, :]
            points = xy[start:start + 128, None, :] + side * offsets[None, :, None]
            size = points.shape[:2]
            samples = [sampler.read(p).reshape(*size, 3) / 255 for p in
                       (points - 3 * side, points - side, points + side, points + 3 * side)]
            a, b, c, d = samples
            color = np.linalg.norm((a + b - c - d) / 2, axis=2)
            brightness = abs((a + b - c - d).mean(axis=2)) / 2
            texture = abs(np.linalg.norm(a - b, axis=2) - np.linalg.norm(c - d, axis=2))
            contrast = color + 0.3 * brightness + 0.2 * texture
            for j, row in enumerate(contrast):
                index = start + j
                valid = np.isfinite(row)
                if not valid.all():
                    reason = "Imagery gap or edge"
                else:
                    best = int(np.argmax(row))
                    alternatives = row[abs(offsets - offsets[best]) > 2.2]
                    separation = row[best] - alternatives.max() if len(alternatives) else row[best]
                    reason = "Low imagery contrast or possible tidal flat" if row[best] < 0.12 or separation < 0.035 else ""
                if reason:
                    costs[index, center] = 0
                    anchored[index] = True
                    reasons.append((index, reason))
                else:
                    costs[index] = -row + 0.015 * abs(offsets)
        for boundary in (0, -1):
            costs[boundary] = np.inf
            costs[boundary, center] = 0
            anchored[boundary] = True
        score = costs[0].copy()
        ancestors = np.empty(costs.shape, dtype=np.int32)
        for index in range(1, len(chain)):
            transitions = score[:, None] + penalty
            ancestors[index] = transitions.argmin(axis=0)
            score = transitions.min(axis=0) + costs[index]
        path = [int(score.argmin())]
        for index in range(len(chain) - 1, 0, -1):
            path.append(int(ancestors[index, path[-1]]))
        path.reverse()
        displacement = offsets[path]
        adjustable = ~(anchored[:-2] | anchored[1:-1] | anchored[2:])
        center_displacement = displacement[1:-1]
        center_displacement[adjustable] = (
            (displacement[:-2] + 2 * center_displacement + displacement[2:]) / 4
        )[adjustable]
        # Shift the original vertices as well so a zero shift never simplifies bends.
        vertex_xy = np.asarray(line.coords)
        vertex_chain = np.r_[0, np.cumsum(np.linalg.norm(np.diff(vertex_xy, axis=0), axis=1))]
        dense = np.union1d(vertex_chain, chain)
        base = shapely.get_coordinates(shapely.line_interpolate_point(line, dense))
        shift = np.column_stack([np.interp(dense, chain, normal[:, n] * displacement) for n in (0, 1)])
        moved = LineString(base + shift) if np.any(displacement) else line
        corridor = shapely.union(line, moved).convex_hull if line.is_ring else line.buffer(width)
        neighbors = [lines[k] for k in original.query(corridor, predicate="intersects") if k != i]
        nearby_moved = any(line.distance(other) < 2 * width and not line.touches(other) for other in neighbors)
        invalid = (not moved.is_simple or (line.is_ring and
                   abs(shapely.Polygon(moved).area - shapely.Polygon(line).area) >
                   0.25 * shapely.Polygon(line).area))
        if np.any(displacement) and (invalid or nearby_moved or
                any(not moved.intersection(other).difference(line.intersection(other)).is_empty for other in neighbors)):
            moved = line
            displacement[:] = 0
            review.append((line, "Topology check (island, intersection, or narrow channel)"))
        result.append(moved)
        distances.extend(abs(displacement).tolist())
        start = last = None
        previous = ""
        for index, reason in reasons + [(None, "")]:
            if previous and (reason != previous or index != last + 1):
                first = max(0, start - 1)
                end = min(len(chain) - 1, last + 1)
                review.append((LineString(xy[first:end + 1]), previous))
            if reason and (reason != previous or index != last + 1):
                start = index
            previous, last = reason, index
    return result, review, distances


def trusted_polygons(path, work):
    if not path:
        return []
    with vector(path) as layer:
        srs = layer.GetSpatialRef()
        if srs is None:
            raise ValueError("The verified sea area polygons must have a CRS.")
        for feature in layer:
            g = feature.GetGeometryRef()
            if g is not None:
                yield from parts(reproject(shapely.make_valid(shapely.from_wkb(bytes(g.ExportToWkb()))),
                                           srs.ExportToWkt(), work), {"Polygon"})


def coast_domain(info, dem, config):
    base = reproject(footprint(dem.GetGeoTransform(), 0, 0, dem.RasterXSize, dem.RasterYSize),
                     info["dem"]["wkt"], info["work"])
    search = config["search"] if config.get("refine") else 0
    return base, base.buffer(max(200, search * 3 + config["join"]))


def coast_network(info, domain, join, check):
    lines, review = load_lines(info, domain, check)
    lines = list(parts(shapely.line_merge(shapely.union_all(lines)), {"LineString"}))
    joined = join_endpoints(lines, join)
    return list(parts(shapely.line_merge(shapely.MultiLineString(joined)), {"LineString"})), review


def classify(info, dem, config, edits, image, check):
    base, domain = coast_domain(info, dem, config)
    search = config["search"] if config.get("refine") else 0
    lines, review = coast_network(info, domain, config["join"], check)
    automatic, image_review, movement = image_refine(lines, image, search, check)
    review.extend(image_review)
    edited = edits.get("lines")
    active = automatic if edited is None else edited
    water = list(trusted_polygons(config.get("verified_water"), info["work"]))
    clipped_water = [p.intersection(domain) for p in water if p.intersects(domain)]
    network = shapely.union_all([domain.boundary, *active, *(p.boundary for p in clipped_water)])
    faces, cuts, dangles, bad = shapely.polygonize_full(list(parts(network, {"LineString"})))
    unresolved_lines = []
    for obj, reason in [(cuts, "Shoreline gap or estuary"), (dangles, "Unconnected endpoint"), (bad, "Invalid polygon boundary")]:
        for line in parts(obj, {"LineString"}):
            review.append((line, reason))
            unresolved_lines.append(line)
    uncertainty = STRtree(unresolved_lines)
    heights = RasterSampler(dem, info["work"], [1])
    face_list = list(parts(faces, {"Polygon"}))
    points = edits.get("seeds", [])
    seed_tree = STRtree([p for p, _ in points])
    water_tree = STRtree(clipped_water)
    decisions, seed_samples = [], []
    for face in face_list:
        check()
        within = face.intersection(base)
        if within.is_empty or within.area <= 0:
            continue
        seeds = [points[index] for index in seed_tree.query(face, predicate="contains")]
        evidence = {value for _, value in seeds}
        anchor = within.representative_point()
        if len(water_tree.query(anchor, predicate="within")):
            evidence.add(SEA)
        conflict = len(evidence) > 1
        reason = "Conflicting land and sea seeds" if conflict else "Insufficient classification evidence or estuary"
        value = next(iter(evidence)) if len(evidence) == 1 else UNKNOWN
        sample_points = []
        if not evidence and not len(uncertainty.query(face, predicate="intersects")):
            bounds = within.bounds
            grid = np.linspace(0.12, 0.88, 7)
            x, y = np.meshgrid(bounds[0] + grid * (bounds[2] - bounds[0]),
                               bounds[1] + grid * (bounds[3] - bounds[1]))
            selected = shapely.contains_xy(within, x.ravel(), y.ravel())
            sample_points = np.column_stack((x.ravel()[selected], y.ravel()[selected]))
            if len(sample_points) >= 5:
                z = heights.read(sample_points).ravel()
                valid = np.isfinite(z)
                rgb = water_probability(image.read(sample_points)) if image else np.full(len(z), np.nan)
                covered = valid & np.isfinite(rgb)
                opens = face.boundary.intersection(domain.boundary).length > 5
                near_coast = len(active) > 0 and face.boundary.difference(domain.boundary).length > 5
                if (covered.sum() >= 5 and (rgb[covered] > 0.68).mean() >= 0.85 and opens and
                        near_coast and np.ptp(z[covered]) <= 1 and max(z[covered]) <= 2):
                    value = SEA
                elif ((valid.sum() >= 5 and np.quantile(z[valid], 0.75) > 3) or
                      (covered.sum() >= 5 and (rgb[covered] < 0.37).mean() >= 0.85)):
                    value = LAND
        if value == SEA:
            for point in [p for p, label in seeds if label == SEA] + [anchor]:
                local = heights.read([(point.x + dx, point.y + dy)
                                      for dx, dy in [(0, 0), (-6, 0), (6, 0), (0, -6), (0, 6)]
                                      if face.contains(Point(point.x + dx, point.y + dy))]).ravel()
                seed_samples.extend(local[np.isfinite(local)].tolist())
        if value == UNKNOWN:
            review.append((within.boundary, reason))
        decisions.append((face, value, reason if value == UNKNOWN else ""))
    for geom, label, order in sorted(edits.get("overrides", []), key=lambda record: record[2]):
        for poly in parts(shapely.make_valid(geom), {"Polygon"}):
            decisions.append((poly, label, ""))
    return automatic, active, decisions, review, seed_samples, movement
