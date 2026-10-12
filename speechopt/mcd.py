"""Mel-cepstral distortion (MCD) in plain numpy.

pysptk/pyworld have no Apple Silicon wheels and cannot be compiled on this machine, so the
mel-cepstrum is computed here as follows. The numbers are comparable between systems in this
study but NOT with published SPTK-based MCD values.

1. Trim leading/trailing silence (librosa, top_db=30).
2. STFT with a 1024-point Hann window and 256-sample hop (the Tacotron 2 frame grid).
3. Real cepstrum of the log-amplitude spectrum, liftered to the first 60 coefficients
   (2.7 ms at 22.05 kHz, shorter than any speaking pitch period) to drop pitch harmonics.
4. Warped to a mel-cepstrum of order 24 with the all-pass recursion used by SPTK's `freqt`
   (alpha = 0.455 for 22.05 kHz).
5. c0 (energy) is excluded; frames are aligned with DTW (Euclidean on c1..c24), and
   MCD = (10 / ln 10) * sqrt(2 * sum_d (c_d - c'_d)^2) is averaged over the aligned path.
"""
import librosa
import numpy as np

ALPHA_22K = 0.455
ORDER = 24
LIFTER = 60
N_FFT = 1024
HOP = 256
K = 10.0 / np.log(10.0)


def freqt(c, order, alpha):
    """Frequency-warp minimum-phase cepstra c (frames x M) to `order`+1 coefficients."""
    c = np.atleast_2d(np.asarray(c, dtype=np.float64))
    g = np.zeros((c.shape[0], order + 1))
    b = 1.0 - alpha * alpha
    for i in range(c.shape[1] - 1, -1, -1):
        d = g.copy()
        g[:, 0] = c[:, i] + alpha * d[:, 0]
        if order >= 1:
            g[:, 1] = b * d[:, 0] + alpha * d[:, 1]
        for j in range(2, order + 1):
            g[:, j] = d[:, j - 1] + alpha * (d[:, j] - g[:, j - 1])
    return g


def mel_cepstrum(y, order=ORDER, alpha=ALPHA_22K, lifter=LIFTER):
    """Frames x (order+1) mel-cepstra of a mono float waveform."""
    spec = np.abs(librosa.stft(np.asarray(y, dtype=np.float64), n_fft=N_FFT, hop_length=HOP, window="hann", center=True))
    log_amp = np.log(np.maximum(spec, 1e-8)).T  # frames x (N_FFT/2+1)
    real_cep = np.fft.irfft(log_amp, n=N_FFT, axis=1)[:, :lifter]
    c = real_cep.copy()
    c[:, 1:] *= 2.0  # two-sided real cepstrum -> one-sided minimum-phase convention
    return freqt(c, order, alpha)


def mcd_aligned(mc_a, mc_b):
    """MCD (dB) between two equal-length mel-cepstrum sequences, excluding c0."""
    diff = mc_a[:, 1:] - mc_b[:, 1:]
    return float(np.mean(K * np.sqrt(2.0 * np.sum(diff * diff, axis=1))))


def mcd(y_ref, y_syn, top_db=30):
    """DTW-aligned MCD (dB) between a reference and a synthesised waveform at 22.05 kHz."""
    a = mel_cepstrum(librosa.effects.trim(np.asarray(y_ref, dtype=np.float64), top_db=top_db)[0])
    b = mel_cepstrum(librosa.effects.trim(np.asarray(y_syn, dtype=np.float64), top_db=top_db)[0])
    _, wp = librosa.sequence.dtw(X=a[:, 1:].T, Y=b[:, 1:].T, metric="euclidean")
    wp = wp[::-1]
    return mcd_aligned(a[wp[:, 0]], b[wp[:, 1]])
