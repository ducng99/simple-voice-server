"""Underdog Woof: a small English text-to-speech model with two built-in voices, Juno and Mira. Runs on CPU
with numpy only.

    from underdog_woof import Woof
    tts = Woof()                      # Juno; or Woof(voice="mira")
    wav = tts("Your order ships on March 3rd.")   # float32 mono, 24 kHz
    tts.save("out.wav", wav)
"""
import os

import numpy as np

from .normalize import Normalizer
from .runtime import RNG, SR, WoofNP, write_wav
from .textsplit import split_text

__all__ = ["Woof", "SR", "VOICES"]

VOICES = ("juno", "mira")

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "woof_int8.npz")


def _stall(wav, hop=960, db=-50.0):
    """Longest run (seconds) of 40 ms frames under db dBFS between the first and last voiced frame."""
    n = len(wav) // hop
    if n < 3:
        return 0.0
    quiet = 20 * np.log10(np.sqrt((wav[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9) < db
    voiced = np.nonzero(~quiet)[0]
    if len(voiced) < 2:
        return 0.0
    best = cur = 0
    for q in quiet[voiced[0]:voiced[-1] + 1]:
        cur = cur + 1 if q else 0
        best = max(best, cur)
    return best * hop / SR


class Woof:
    """Text in, 24 kHz audio out. A rule-based normalizer first rewrites text the way it is spoken (numbers, dates,
    times, money, codes, symbols). Long text is split into sentences (and long sentences into clauses), and each piece is
    checked for runaway length and long internal silence; a failed take is redrawn up to `retries` times and the best
    take is kept."""

    def __init__(self, weights=WEIGHTS, voice="juno", temp=0.5, eos=0.9, max_spc=0.095, stall_s=1.0, retries=3, normalize=True):
        self.model = WoofNP(weights, voice=voice)
        self.norm = Normalizer(use_nemo=False) if normalize else None
        self.temp, self.eos = temp, eos
        self.max_spc, self.stall_s, self.retries = max_spc, stall_s, retries
        self.sr = SR

    def _piece(self, text):
        cap = int(self.model.frames_max)
        best = None
        for _ in range(1 + self.retries):
            lat = self.model.latents(text, temp=self.temp, eos=self.eos)
            m = self.model
            wav = m.dec(lat * m.std + m.mean)
            run = _stall(wav)
            spc = len(wav) / SR / max(1, len(text))
            bad = (run >= self.stall_s) + (spc > self.max_spc or len(lat) >= cap)
            score = (bad, run, spc)
            if best is None or score < best[0]:
                best = (score, wav)
            if not bad:
                break
        return best[1]

    def __call__(self, text, seed=None):
        if seed is not None:
            RNG.seed(seed)
        if self.norm is not None:
            text = self.norm(text)
        pieces = split_text(text) or [text]
        wavs = [self._piece(p) for p in pieces]
        return (wavs[0] if len(wavs) == 1 else np.concatenate(wavs)).astype(np.float32)

    @staticmethod
    def save(path, wav):
        write_wav(path, wav)
