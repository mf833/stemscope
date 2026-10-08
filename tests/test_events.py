"""Event extraction on a synthetic arrangement with a known answer.

Run:  .venv/bin/python -m unittest discover tests
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend import events
from backend.instruments import FAMILIES

HOP = 0.25
DUR = 180.0


def build(sensitivity: float = 0.5) -> dict:
    labels = sorted({c for fam in FAMILIES for c in fam.classes}) + ["Music", "Speech"]
    index = {label: i for i, label in enumerate(labels)}
    n = int(DUR / HOP)
    t = np.arange(n) * HOP
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 0.04, (n, len(labels)))  # background noise floor

    def play(label, start, end, conf):
        mask = (t >= start) & (t < end)
        probs[mask, index[label]] = conf + rng.uniform(-0.05, 0.05, mask.sum())

    play("Drum kit", 0, DUR, 0.45)
    play("Guitar", 12, 160, 0.50)
    play("Electric guitar", 12, 160, 0.40)
    play("Singing", 30, 75, 0.7)
    play("Male singing", 30, 75, 0.5)
    play("Singing", 76.5, 120, 0.7)  # 1.5 s dropout should be bridged
    play("Singing", 140, 170, 0.6)
    play("Piano", 50, 51, 0.6)  # 1 s blip should be ignored
    play("Bass guitar", 12, DUR, 0.22)

    level = {s: np.full(n, -80.0) for s in ["vocals", "drums", "bass", "guitar", "piano", "other"]}
    level["drums"][:] = -20
    level["bass"][t >= 12] = -24
    level["guitar"][(t >= 12) & (t < 160)] = -30
    level["guitar"][(t >= 65) & (t < 160)] = -21  # +9 dB at 1:05
    level["vocals"][(t >= 30) & (t < 120)] = -18
    level["vocals"][(t >= 95) & (t < 120)] = -26  # -8 dB at 1:35
    level["vocals"][(t >= 140) & (t < 170)] = -18
    level = {k: (v + rng.normal(0, 0.8, n)).astype(np.float32) for k, v in level.items()}

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        np.savez_compressed(d / "analysis.npz", hop=HOP, probs=probs.astype(np.float16), labels=np.array(labels))
        np.savez_compressed(d / "levels.npz", hop=HOP, **level)
        return events.build_report(d, DUR, sensitivity)


def story(report: dict, family: str) -> list[tuple[str, float | None]]:
    entry = next((s for s in report["summary"] if s["family"] == family), None)
    return [(p["text"], p["time"]) for p in entry["parts"]] if entry else []


class EventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build()

    def assertStory(self, family, expected):
        got = story(self.report, family)
        self.assertEqual([text for text, _ in got], [text for text, _ in expected], got)
        for (_, t_got), (_, t_exp) in zip(got, expected):
            if t_exp is None:
                self.assertIsNone(t_got)
            else:
                self.assertAlmostEqual(t_got, t_exp, delta=0.75)

    def test_guitar_enters_gets_louder_drops_out(self):
        self.assertStory("Guitar", [("enters at", 12), ("gets louder at", 65), ("drops out at", 160)])

    def test_vocals_gap_bridged_and_quieter_section(self):
        self.assertStory("Vocals", [
            ("enter at", 30), ("get quieter at", 95), ("drop out at", 120),
            ("come back in at", 140), ("drop out at", 170),
        ])

    def test_drums_throughout(self):
        self.assertStory("Drums", [("play throughout", None)])

    def test_blip_ignored(self):
        self.assertEqual(story(self.report, "Piano"), [])

    def test_enter_detail_names_specific_instrument(self):
        enter = next(e for e in self.report["events"] if e["family"] == "Guitar" and e["kind"] == "enter")
        self.assertIn("mostly electric guitar", enter["detail"])

    def test_low_sensitivity_drops_weak_bass(self):
        self.assertTrue(story(self.report, "Bass"))
        self.assertEqual(story(build(sensitivity=0.0), "Bass"), [])


if __name__ == "__main__":
    unittest.main()
