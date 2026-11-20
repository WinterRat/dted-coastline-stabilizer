"""Build the ZIP used by QGIS's Install from ZIP command."""

from configparser import ConfigParser
from pathlib import Path
from tomllib import loads
from zipfile import ZIP_DEFLATED, ZipFile


def build():
    root = Path(__file__).resolve().parent
    package = root / "dted_coastline_stabilizer"
    metadata_path = package / "metadata.txt"
    metadata = ConfigParser()
    metadata.read(metadata_path, encoding="utf8")
    assert metadata["general"]["qgisMinimumVersion"] == "4.2.2"
    project = loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if metadata["general"]["version"] != project["version"]:
        raise ValueError("The plugin version in metadata.txt must match pyproject.toml")
    sources = sorted(package.glob("*.py"))
    assert package / "__init__.py" in sources
    for source in sources:
        compile(source.read_bytes(), str(source), "exec")
    destination = root / f"{package.name}.zip"
    pending = destination.with_suffix(".zip.tmp")
    try:
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as archive:
            for source in [metadata_path, package / "user-guide-ko.html", *sources]:
                archive.write(source, source.relative_to(root).as_posix())
        with ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise ValueError("The QGIS plugin ZIP failed its integrity check")
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    print(build())
