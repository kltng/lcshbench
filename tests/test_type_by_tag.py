# tests/test_type_by_tag.py
from lcsh_benchmark.typing import type_by_tag


def test_type_by_tag_maps_each_6xx_field():
    assert type_by_tag("Sociology", "650")["type"] == "topical"
    assert type_by_tag("United States", "651")["type"] == "geographic"
    assert type_by_tag("Shakespeare, William", "600")["type"] == "name"
    assert type_by_tag("Harvard University", "610")["type"] == "name"
    assert type_by_tag("Feature films", "655")["type"] == "genre"
    assert type_by_tag("Mystery", "")["type"] == "other"


def test_type_by_tag_subdivision_and_base():
    out = type_by_tag("Sociology--Philosophy--History", "650")
    assert out["tag"] == "650"
    assert out["base"] == "sociology"
    assert out["has_subdivision"] is True
    assert out["subdivision_depth"] == 2
