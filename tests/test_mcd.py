import numpy as np
import pytest

from speechopt.mcd import K, freqt, mcd, mcd_aligned, mel_cepstrum


SR = 22050


def _voice(
    f0: float = 180.0,
    dur: float = 1.5,
    formants: tuple[int, ...] = (700, 1200, 2600),
    seed: int = 0,
) -> np.ndarray:
    """Crude vowel: an impulse train through resonators, plus a little noise."""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    x = np.zeros(n)
    x[:: int(SR / f0)] = 1.0
    from scipy.signal import lfilter

    for f in formants:
        r = np.exp(-np.pi * 120 / SR)
        x = lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x)
    x = x / np.abs(x).max() * 0.5
    return x + 1e-3 * rng.standard_normal(n)


def test_freqt_alpha_zero_is_truncation() -> None:
    c = np.random.default_rng(1).standard_normal((3, 40))
    np.testing.assert_allclose(freqt(c, 24, 0.0), c[:, :25], atol=1e-12)


def test_freqt_round_trip() -> None:
    c = 0.8 ** np.arange(30) * np.random.default_rng(2).standard_normal(30)
    back = freqt(freqt(c, 400, 0.455), 29, -0.455)
    np.testing.assert_allclose(back[0], c, atol=1e-6)


def test_freqt_matches_warped_log_spectrum() -> None:
    # Defining property: the warped cepstrum evaluated at the warped frequency
    # reproduces the original log spectrum.
    a = 0.455
    c = 0.7 ** np.arange(12) * np.random.default_rng(3).standard_normal(12)
    mc = freqt(c, 300, a)[0]
    w = np.linspace(0.01, np.pi - 0.01, 200)
    w_warp = w + 2 * np.arctan(a * np.sin(w) / (1 - a * np.cos(w)))
    orig = c[0] + (c[1:, None] * np.cos(np.arange(1, 12)[:, None] * w)).sum(0)
    warped = mc[0] + (mc[1:, None] * np.cos(np.arange(1, 301)[:, None] * w_warp)).sum(0)
    np.testing.assert_allclose(warped, orig, atol=1e-6)


def test_mcd_formula_on_known_difference() -> None:
    a = np.zeros((5, 25))
    b = np.zeros((5, 25))
    b[:, 1] = 0.3
    b[:, 0] = 7.0  # c0 must be ignored
    assert mcd_aligned(a, b) == pytest.approx(K * np.sqrt(2.0) * 0.3)


def test_identical_audio_is_zero() -> None:
    x = _voice()
    assert mcd(x, x) == pytest.approx(0.0, abs=1e-9)


def test_gain_change_is_ignored() -> None:
    x = _voice()
    assert mcd(x, 0.5 * x) == pytest.approx(0.0, abs=1e-6)


def test_cepstrum_is_insensitive_to_pitch_but_sensitive_to_formants() -> None:
    same_formants_other_pitch = mcd(_voice(f0=180), _voice(f0=210, seed=1))
    other_formants = mcd(_voice(f0=180), _voice(f0=180, formants=(300, 2300, 3000), seed=1))
    assert same_formants_other_pitch < other_formants / 3


def test_dtw_handles_time_stretch() -> None:
    x = _voice(dur=1.0)
    y = np.concatenate(
        [_voice(formants=(700, 1200, 2600), dur=0.5), _voice(formants=(300, 2300, 3000), dur=0.5)]
    )
    y_slow = np.concatenate(
        [_voice(formants=(700, 1200, 2600), dur=0.8), _voice(formants=(300, 2300, 3000), dur=0.8)]
    )
    assert mcd(y, y_slow) < mcd(y, x)


def test_shape() -> None:
    assert mel_cepstrum(_voice()).shape[1] == 25
