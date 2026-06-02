# src/lcsh_benchmark/scaleup/composition.py
"""v2 target composition: ~5K, tiered. English capped; music (LC class M) capped."""

CORE_LANG_TARGETS = {       # ~3000 rich-input core
    "eng": 750, "ger": 300, "fre": 300, "spa": 250, "rus": 250, "chi": 250,
    "jpn": 200, "ita": 200, "ara": 200, "kor": 150, "por": 150,
}
BREADTH_LANG_TARGETS = {    # ~2000 multilingual breadth (adds smaller languages)
    "eng": 400, "ger": 200, "fre": 200, "spa": 150, "rus": 150, "chi": 150,
    "jpn": 120, "ita": 120, "ara": 120, "kor": 100, "por": 100, "pol": 80,
    "heb": 40, "hin": 40, "tur": 30,
}
MUSIC_CLASS = "M"           # LC class letter for music
MUSIC_CAP = 0.05            # max share of any tier that may be music (~2.3% pool share)
SEED = 13
