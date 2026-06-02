"""A3: capture monograph status from the MARC leader.

Extraction never read the leader, so serials, maps, scores, sound recordings,
and videos could enter a dataset described as "books." A monograph is
leader/06 (type of record) in {a (language material), t (manuscript language
material)} AND leader/07 (bibliographic level) == m. index_record now exposes
`is_monograph`; the selection stage filters on it.
"""
from pymarc import Record, Field, Subfield
from lcsh_benchmark.scaleup.marc_index import index_record


def _rec(leader):
    r = Record()
    r.leader = leader
    r.add_field(Field(tag="008", data="000000s2010    xxu           000 0 eng d"))
    r.add_field(Field(tag="650", indicators=[" ", "0"], subfields=[Subfield("a", "X")]))
    return r


def _is_mono(leader):
    return index_record(_rec(leader), "test")["is_monograph"]


#                  positions: 0123456789...  06=type 07=level
_BOOK    = "00000nam a2200000 a 4500"   # type a, level m  -> monograph
_MANUSCRIPT_BOOK = "00000ntm a2200000 a 4500"  # type t, level m -> monograph
_SERIAL  = "00000nas a2200000 a 4500"   # type a, level s  -> serial
_MAP     = "00000nem a2200000 a 4500"   # type e, level m  -> cartographic
_SCORE   = "00000ncm a2200000 a 4500"   # type c, level m  -> notated music
_SOUND   = "00000njm a2200000 a 4500"   # type j, level m  -> sound recording


def test_book_is_monograph():
    assert _is_mono(_BOOK) is True
    assert _is_mono(_MANUSCRIPT_BOOK) is True


def test_serial_is_not_monograph():
    assert _is_mono(_SERIAL) is False


def test_nonbook_material_is_not_monograph():
    assert _is_mono(_MAP) is False
    assert _is_mono(_SCORE) is False
    assert _is_mono(_SOUND) is False
