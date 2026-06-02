"""A2: field-aware 6xx heading reconstruction.

The bug: `_heading_text` joined ALL subfields with `--`, which is wrong for
name/title headings (600/610/611/630), where $b/$c/$d/$q/$t/$p/$n/$k/$l are
*part of the heading*, not subdivisions. Only the subdivision subfields
$v (form) / $x (general) / $y (chronological) / $z (geographic) are joined
with `--`; main-heading subfields are joined with their cataloging punctuation
(a single space — the values already carry trailing commas/periods).
"""
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec(field):
    r = Record()
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(field)
    return r


def _heading(field):
    """Single reconstructed heading string for a field added to a bare record."""
    out = index_record(_rec(field), "test")["headings"]
    flat = [h for tag in out for h in out[tag]]
    return flat[0] if flat else ""


def test_650_topical_subdivisions_joined_with_double_dash():
    # regression: topical heading with subdivisions is unchanged
    f = Field(tag="650", indicators=[" ", "0"], subfields=[
        Subfield("a", "Science"), Subfield("z", "United States"), Subfield("x", "History")])
    assert _heading(f) == "Science--United States--History"


def test_650_keeps_b_in_main_term():
    f = Field(tag="650", indicators=[" ", "0"], subfields=[
        Subfield("a", "Concertos"), Subfield("b", "Piano"), Subfield("x", "Analysis")])
    assert _heading(f) == "Concertos Piano--Analysis"


def test_651_geographic():
    f = Field(tag="651", indicators=[" ", "0"], subfields=[
        Subfield("a", "United States"), Subfield("x", "History"), Subfield("y", "Civil War, 1861-1865")])
    assert _heading(f) == "United States--History--Civil War, 1861-1865"


def test_600_personal_name_subfields_are_part_of_the_heading():
    # THE BUG: $d (dates) must NOT be `--`-joined; it is part of the name.
    f = Field(tag="600", indicators=["1", "0"], subfields=[
        Subfield("a", "Shakespeare, William,"), Subfield("d", "1564-1616"),
        Subfield("x", "Criticism and interpretation")])
    assert _heading(f) == "Shakespeare, William, 1564-1616--Criticism and interpretation"


def test_610_corporate_name_with_subordinate_unit():
    f = Field(tag="610", indicators=["1", "0"], subfields=[
        Subfield("a", "United States."), Subfield("b", "Army"),
        Subfield("x", "History")])
    assert _heading(f) == "United States. Army--History"


def test_630_uniform_title_parts_not_subdivisions():
    f = Field(tag="630", indicators=["0", "0"], subfields=[
        Subfield("a", "Bible."), Subfield("p", "O.T."), Subfield("l", "English")])
    assert _heading(f) == "Bible. O.T. English"


def test_600_name_title_includes_t_subfield():
    f = Field(tag="600", indicators=["1", "0"], subfields=[
        Subfield("a", "Homer."), Subfield("t", "Iliad"), Subfield("x", "Translations")])
    assert _heading(f) == "Homer. Iliad--Translations"


def test_control_and_linking_subfields_dropped():
    # $0 (authority id), $2 (source), $6 (linkage), $8 must never appear
    f = Field(tag="650", indicators=[" ", "0"], subfields=[
        Subfield("a", "Science"), Subfield("2", "lcsh"),
        Subfield("0", "(DE-101)040303923"), Subfield("x", "History")])
    assert _heading(f) == "Science--History"
