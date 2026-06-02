# src/lcsh_benchmark/scaleup/sources.py
"""Per-source bulk-MARC chunk enumeration. Returns chunk descriptors
{source, chunk_id, url, container}; no parsing. Network is injected for tests.

container values: 'gz_marcxml'/'tar_marcxml' (MARCXML, read via pymarc map_xml)
and 'gz_marc'/'tar_marc' (binary MARC21, read via pymarc MARCReader). The
'local' source ingests manually-downloaded files from
data/raw/v2/incoming/<source>/ (container inferred from extension).
"""
import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path

COLUMBIA_URL = ("https://lito.cul.columbia.edu/extracts/"
                "ColumbiaLibraryCatalog/full/extract-%03d.xml.gz")
# Princeton serves 31 tarballs via 0-indexed download endpoints
# (download/0 .. download/30); each yields open_dataset_NN.tar.gz (application/gzip).
# Verified live 2026-05-25.
PRINCETON_URL = "https://lib-jobs.princeton.edu/open-marc-records/download/%d"
PRINCETON_N = 31
HARVARD_DOI = "doi:10.7910/DVN/LZDQYN"
HARVARD_API = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
               "?persistentId=" + HARVARD_DOI)
HARVARD_FILE = "https://dataverse.harvard.edu/api/access/datafile/%s"
# LoC MDSConnect retrospective files (coverage <=2014); the canonical list
# is published at loc.gov/cds; enumerated explicitly when run for real.
LOC_FILES: list[str] = []


def _http_exists(url: str, n: int) -> bool:
    req = urllib.request.Request(url % n, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def princeton_chunks() -> Iterator[dict]:
    for i in range(PRINCETON_N):   # 0-indexed download endpoints
        yield {"source": "princeton", "chunk_id": f"{i:02d}",
               "url": PRINCETON_URL % i, "container": "tar_marcxml"}


def columbia_chunks(exists=None, max_n: int = 400) -> Iterator[dict]:
    exists = exists or (lambda url, n: _http_exists(url, n))
    for i in range(1, max_n + 1):
        if not exists(COLUMBIA_URL, i):
            break
        yield {"source": "columbia", "chunk_id": f"{i:03d}",
               "url": COLUMBIA_URL % i, "container": "gz_marcxml"}


def harvard_chunks(fetch_json=None) -> Iterator[dict]:
    fetch_json = fetch_json or (lambda url: _http_json(url))
    data = fetch_json(HARVARD_API)
    files = data["data"]["latestVersion"]["files"]
    for f in files:
        fid = str(f["dataFile"]["id"])
        yield {"source": "harvard", "chunk_id": fid,
               "url": HARVARD_FILE % fid, "container": "gz_marcxml"}


def loc_chunks() -> Iterator[dict]:
    for i, url in enumerate(LOC_FILES, 1):
        yield {"source": "loc", "chunk_id": f"{i:03d}",
               "url": url, "container": "gz_marcxml"}


INCOMING_DIR = "data/raw/v2/incoming"
# extension -> container, for manually-downloaded files in incoming/<source>/.
# Harvard bibdata ships a .tar.gz of BINARY MARC21 -> tar_marc; LoC XML -> gz_marcxml;
# LoC binary (utf8/mrc) -> gz_marc.
_EXT_CONTAINER = [
    (".xml.gz", "gz_marcxml"),
    (".marcxml.gz", "gz_marcxml"),
    (".mrc.gz", "gz_marc"),
    (".utf8.gz", "gz_marc"),
    (".marc8.gz", "gz_marc"),
    (".tar.gz", "tar_marc"),
    (".tgz", "tar_marc"),
]


def _container_for(filename: str) -> str | None:
    fn = filename.lower()
    for ext, cont in _EXT_CONTAINER:
        if fn.endswith(ext):
            return cont
    return None


def local_chunks(incoming_dir: str = INCOMING_DIR) -> Iterator[dict]:
    """Enumerate manually-downloaded files under incoming/<source>/<file>.
    The subdirectory name is the source; container is inferred from extension.
    Files with an unrecognized extension are skipped."""
    base = Path(incoming_dir)
    if not base.is_dir():
        return
    for src_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(p for p in src_dir.iterdir() if p.is_file()):
            cont = _container_for(f.name)
            if cont is None:
                continue
            yield {"source": src_dir.name, "chunk_id": f.name,
                   "url": f.resolve().as_uri(), "container": cont}


def iter_chunks(source: str, **kw) -> Iterator[dict]:
    return {"columbia": columbia_chunks, "princeton": princeton_chunks,
            "harvard": harvard_chunks, "loc": loc_chunks,
            "local": local_chunks}[source](**kw)
