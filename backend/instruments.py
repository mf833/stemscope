"""Which AudioSet classes count as instruments, and how they group into families.

A family's presence is the max confidence over its classes, so "Electric guitar",
"Acoustic guitar" and "Strum" all feed a single "Guitar" lane and event stream.
Families that correspond to a Demucs stem use that stem's loudness to detect
"gets louder / quieter" changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    classes: tuple[str, ...]
    color: str
    stem: str | None = None
    plural: bool = False  # "Drums enter" vs "Guitar enters"


FAMILIES: list[Family] = [
    Family("Vocals", ("Singing", "Male singing", "Female singing", "Child singing", "Choir", "Rapping",
                      "Humming", "Yodeling", "Chant", "Synthetic singing", "Beatboxing", "Whistling"),
           "#ec5f74", stem="vocals", plural=True),
    Family("Drums", ("Drum kit", "Drum", "Snare drum", "Bass drum", "Hi-hat", "Cymbal", "Drum roll",
                     "Rimshot", "Drum machine"),
           "#f08a3e", stem="drums", plural=True),
    Family("Bass", ("Bass guitar", "Double bass"), "#e9b949", stem="bass"),
    Family("Guitar", ("Guitar", "Electric guitar", "Acoustic guitar", "Steel guitar, slide guitar", "Strum",
                      "Tapping (guitar technique)"),
           "#62c972", stem="guitar"),
    Family("Piano", ("Piano", "Electric piano", "Keyboard (musical)", "Harpsichord"), "#a98bfa", stem="piano"),
    Family("Organ", ("Organ", "Electronic organ", "Hammond organ"), "#38bdf8"),
    Family("Synth", ("Synthesizer", "Sampler"), "#22d3ee"),
    Family("Strings", ("Bowed string instrument", "String section", "Violin, fiddle", "Cello", "Pizzicato",
                       "Orchestra"),
           "#e879f9", plural=True),
    Family("Brass", ("Brass instrument", "Trumpet", "Trombone", "French horn"), "#fbbf24"),
    Family("Woodwinds", ("Wind instrument, woodwind instrument", "Flute", "Saxophone", "Clarinet"),
           "#a3e635", plural=True),
    Family("Percussion", ("Tambourine", "Maraca", "Rattle (instrument)", "Wood block", "Clapping", "Tabla",
                          "Timpani", "Gong"),
           "#f7b27a"),
    Family("Bells & mallets", ("Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone",
                               "Steelpan", "Tubular bells", "Bell", "Jingle bell", "Chime", "Wind chime",
                               "Church bell"),
           "#5eead4", plural=True),
    Family("Banjo", ("Banjo",), "#9be3a8"),
    Family("Mandolin", ("Mandolin",), "#9be3a8"),
    Family("Ukulele", ("Ukulele",), "#9be3a8"),
    Family("Sitar", ("Sitar",), "#9be3a8"),
    Family("Harp", ("Harp",), "#f0abfc"),
    Family("Zither", ("Zither",), "#9be3a8"),
    Family("Harmonica", ("Harmonica",), "#f9d65c"),
    Family("Accordion", ("Accordion",), "#f9d65c"),
    Family("Bagpipes", ("Bagpipes",), "#facc15", plural=True),
    Family("Didgeridoo", ("Didgeridoo",), "#f9d65c"),
    Family("Theremin", ("Theremin",), "#22d3ee"),
    Family("Turntable scratching", ("Scratching (performance technique)",), "#cbd5e1"),
]

# Classes too broad to describe *which* kind of instrument it is ("mostly electric guitar").
GENERIC = {
    "Singing", "Guitar", "Strum", "Tapping (guitar technique)", "Drum", "Snare drum", "Bass drum", "Hi-hat",
    "Cymbal", "Drum roll", "Rimshot", "Keyboard (musical)", "Organ", "Bowed string instrument",
    "Brass instrument", "Wind instrument, woodwind instrument", "Mallet percussion", "Bell",
}

SHORT_NAMES = {
    "Steel guitar, slide guitar": "slide guitar",
    "Tapping (guitar technique)": "guitar tapping",
    "Keyboard (musical)": "keyboard",
    "Violin, fiddle": "violin",
    "Wind instrument, woodwind instrument": "woodwind",
    "Marimba, xylophone": "marimba / xylophone",
    "Rattle (instrument)": "rattle",
    "Scratching (performance technique)": "scratching",
    "Double bass": "upright bass",
}


def short_name(label: str) -> str:
    return SHORT_NAMES.get(label, label.lower())
