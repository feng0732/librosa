#!/usr/bin/env python
# Tests for mel-spectrogram inversion (mel_to_stft / mel_to_audio).
#
# The mel projection W (shape n_mels x 1+n_fft/2) compresses a linear-frequency
# spectrum onto far fewer bands, so recovering a linear spectrum is an
# underdetermined non-negative least-squares problem rather than an exact
# inverse.  These tests therefore assert the documented *approximation*
# contract -- non-negativity, stable shapes/dtypes, a small re-projection
# residual in mel space, and interpretable band positions -- while explicitly
# demonstrating that an exact, unique linear spectrum cannot be recovered.
# Tight numerical bounds are applied only to quantities the contract actually
# guarantees (the re-projected mel bands); ambiguity intrinsic to the problem
# is asserted separately from the solver's reconstruction bias.

# Disable cache
import contextlib
import os

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

import numpy as np
import pytest

import librosa

SR = 22050
N_FFT = 2048
N_MELS = 128
N_FRAMES = 4
# Observed-domain (mel space) relative-error budget.  Empirically the NNLS
# solver achieves relative residuals of a few percent or better here; the
# budget stays deliberately looser than the measured values.
MEL_REL_TOL = 5e-2
# Fixed-point (invert -> re-project -> invert) drift is solver-tolerance
# limited, not exact; measured around 1-2% in relative norm.
FIXED_POINT_REL_TOL = 5e-2


@pytest.fixture(scope="module")
def mel_basis():
    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, dtype=np.float64)


@pytest.fixture(scope="module")
def fftfreqs():
    return librosa.fft_frequencies(sr=SR, n_fft=N_FFT)


def _relative_mel_error(mel_basis, stft_inverse, mels, power):
    """Re-project an inverted spectrum into mel space and return the
    Frobenius-norm relative residual against the input mel spectrogram."""
    reprojected = np.einsum("...ft,mf->...mt", stft_inverse**power, mel_basis)
    return np.linalg.norm((reprojected - mels).ravel()) / np.linalg.norm(mels.ravel())


# ---------------------------------------------------------------------------
# Requirement 1: energy conventions, consistent re-projected units, and the
# non-uniqueness of the underdetermined inverse.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("power", [1.0, 2.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mel_to_stft_energy_convention_consistency(power, dtype, mel_basis, rng):
    """Magnitude (power=1) and power (power=2) mel inputs each invert to a
    non-negative linear magnitude spectrum whose re-projection carries the
    same units (energy convention) as the supplied mel spectrogram."""
    basis = mel_basis.astype(dtype)
    n_bins = 1 + N_FFT // 2

    # Start from a non-negative power spectrum, then apply the same
    # S -> S**power mapping that feature.melspectrogram uses.
    power_spec = (rng.standard_normal(size=(n_bins, N_FRAMES)) ** 2).astype(dtype)
    linear_input = power_spec if power == 2.0 else np.sqrt(power_spec)
    mels = basis @ linear_input

    stft_inverse = librosa.feature.inverse.mel_to_stft(mels, sr=SR, n_fft=N_FFT, power=power)

    # Documented output contract: non-negative magnitude spectrum with the
    # linear-frequency bin count and the input precision.
    assert stft_inverse.shape == (n_bins, N_FRAMES)
    assert stft_inverse.dtype == dtype
    assert np.all(stft_inverse >= 0)

    # Re-projection must reproduce the observed mel bands: the only place a
    # tight error bound is justified, because the mel bands are the data that
    # were actually observed.
    assert _relative_mel_error(basis, stft_inverse, mels, power) <= MEL_REL_TOL

    # Re-projected units match the input convention: re-applying the inverse
    # exponent returns an array in the same space as the input mel spectrum.
    reprojected = basis @ (stft_inverse**power)
    assert reprojected.shape == mels.shape
    assert reprojected.dtype == dtype
    assert np.all(reprojected >= 0)


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_mel_to_stft_reprojection_fixed_point(power, mel_basis, rng):
    """Repeated invert -> project cycles converge to a fixed point instead of
    manufacturing information.  The residual drift is solver tolerance, not
    zero, so it is bounded relatively rather than asserted exact."""
    n_bins = 1 + N_FFT // 2
    power_spec = rng.standard_normal(size=(n_bins, N_FRAMES)) ** 2
    linear_input = power_spec if power == 2.0 else np.sqrt(power_spec)
    mels = mel_basis @ linear_input

    s1 = librosa.feature.inverse.mel_to_stft(mels, sr=SR, n_fft=N_FFT, power=power)
    m1 = mel_basis @ (s1**power)
    s2 = librosa.feature.inverse.mel_to_stft(m1, sr=SR, n_fft=N_FFT, power=power)
    m2 = mel_basis @ (s2**power)

    # A second cycle must barely move either representation, and must not
    # increase the mel-space fit error relative to the first cycle.
    assert np.linalg.norm(m2 - m1) / np.linalg.norm(m1) <= FIXED_POINT_REL_TOL
    assert np.linalg.norm(s2 - s1) / np.linalg.norm(s1) <= FIXED_POINT_REL_TOL
    first_cycle_error = np.linalg.norm(m1 - mels) / np.linalg.norm(mels)
    second_cycle_error = np.linalg.norm(m2 - mels) / np.linalg.norm(mels)
    assert second_cycle_error <= first_cycle_error + FIXED_POINT_REL_TOL


def test_mel_inversion_is_underdetermined_and_nonunique(mel_basis, rng):
    """n_mels (128) < n_fft/2+1 (1025) and the projection is rank deficient:
    distinct non-negative linear spectra can share one mel image, so that
    image cannot identify a unique original spectrum.  This test separates
    the problem's intrinsic ambiguity from the solver's reconstruction
    residual."""
    basis = mel_basis
    n_bins = 1 + N_FFT // 2

    # The right null space {x : Wx = 0} is what makes the inverse non-unique.
    _u, singular, vh = np.linalg.svd(basis, full_matrices=True)
    rank_tol = singular.max() * max(basis.shape) * np.finfo(np.float64).eps
    rank = int((singular > rank_tol).sum())
    assert rank == N_MELS
    assert rank < n_bins
    null_space = vh[rank:]
    assert null_space.shape[0] == n_bins - rank

    # Two strictly non-negative power spectra differing only along W's null
    # space.  A positive floor keeps every entry away from zero so the null
    # perturbation can be large while preserving positivity.
    x0 = rng.standard_normal(size=(n_bins, N_FRAMES)) ** 2 + 1.0
    direction = null_space.T @ rng.standard_normal(size=(null_space.shape[0], N_FRAMES))
    coefficient = np.zeros(N_FRAMES)
    for frame in range(N_FRAMES):
        d = direction[:, frame]
        base = x0[:, frame]
        # Largest positive/negative step keeping base + c*d non-negative.
        ratios_pos = np.where(d < 0, base / np.maximum(-d, 1e-300), np.inf)
        c_pos = ratios_pos[np.isfinite(ratios_pos)].min() if np.isfinite(ratios_pos).any() else np.inf
        ratios_neg = np.where(d > 0, base / np.maximum(d, 1e-300), np.inf)
        c_neg = ratios_neg[np.isfinite(ratios_neg)].min() if np.isfinite(ratios_neg).any() else np.inf
        c = 0.9 * max(c_pos, c_neg)
        coefficient[frame] = c if c_pos >= c_neg else -c
    x_alt = x0 + direction * coefficient

    # The candidates are genuinely different and both admissible.
    assert np.all(x_alt > 0)
    assert np.linalg.norm(x_alt - x0) / np.linalg.norm(x0) > 0.05

    # Yet they project onto exactly the same mel spectrogram.
    mels_0 = basis @ x0
    mels_alt = basis @ x_alt
    np.testing.assert_allclose(mels_alt, mels_0, atol=1e-12)

    # One inversion of the shared mel image.  It fits the observed mel bands
    # well (small reconstruction bias) ...
    stft_inverse = librosa.feature.inverse.mel_to_stft(mels_0, sr=SR, n_fft=N_FFT, power=2.0)
    assert _relative_mel_error(basis, stft_inverse, mels_0, 2.0) <= MEL_REL_TOL

    # ... but cannot be close to both distinct candidates at once.  Both
    # distances stay well above the mel-space residual, proving that a small
    # re-projection error does not imply recovery of the unique original.
    dist_0 = np.linalg.norm(np.sqrt(x0) - stft_inverse) / np.linalg.norm(np.sqrt(x0))
    dist_alt = np.linalg.norm(np.sqrt(x_alt) - stft_inverse) / np.linalg.norm(np.sqrt(x_alt))
    assert dist_0 > MEL_REL_TOL
    assert dist_alt > MEL_REL_TOL


# ---------------------------------------------------------------------------
# Requirement 2: interpretable band positions and bounded error for sparse
# and mixed timbres, plus multichannel shape / precision stability.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f0", [220.0, 1000.0, 4000.0])
def test_sparse_tone_band_localization(f0, fftfreqs):
    """A pure tone reconstructs with its energy concentrated around the
    source band.  The tight bound applies only after mel re-projection; the
    linear-spectrum shape is asserted via band localization, not equality."""
    y = librosa.tone(f0, sr=SR, duration=0.25)
    stft_power = np.abs(librosa.stft(y, n_fft=N_FFT)) ** 2
    mels = librosa.feature.melspectrogram(S=stft_power, sr=SR, n_fft=N_FFT)
    stft_inverse = librosa.feature.inverse.mel_to_stft(mels, sr=SR, n_fft=N_FFT, power=2.0)

    assert stft_inverse.shape == (1 + N_FFT // 2, stft_power.shape[1])
    assert np.all(stft_inverse >= 0)

    # Bounded reconstruction bias in the observed (mel) domain.
    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    assert _relative_mel_error(basis, stft_inverse, mels, 2.0) <= MEL_REL_TOL

    # Interpretability: the dominant reconstructed FFT bin and most of the
    # power lie within +/-20% of the generating frequency.
    power = (stft_inverse**2).mean(axis=1)
    dominant_bin = int(np.argmax(power))
    assert 0.8 * f0 <= fftfreqs[dominant_bin] <= 1.2 * f0

    band_mask = (fftfreqs >= 0.8 * f0) & (fftfreqs <= 1.2 * f0)
    assert power[band_mask].sum() / power.sum() >= 0.5


def test_mixed_timbre_band_separation(fftfreqs):
    """A mixture of well-separated tones preserves each source band in the
    reconstruction, with the mel-domain error bounded."""
    component_freqs = [440.0, 1500.0, 5000.0]
    y = sum(librosa.tone(f, sr=SR, duration=0.25) for f in component_freqs)
    stft_power = np.abs(librosa.stft(y, n_fft=N_FFT)) ** 2
    mels = librosa.feature.melspectrogram(S=stft_power, sr=SR, n_fft=N_FFT)
    stft_inverse = librosa.feature.inverse.mel_to_stft(mels, sr=SR, n_fft=N_FFT, power=2.0)

    basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    assert _relative_mel_error(basis, stft_inverse, mels, 2.0) <= MEL_REL_TOL

    power = (stft_inverse**2).mean(axis=1)
    # Every component frequency must retain a detectable local band; +/-15%
    # windows are disjoint for the chosen, well-separated components.
    for f0 in component_freqs:
        band_mask = (fftfreqs >= 0.85 * f0) & (fftfreqs <= 1.15 * f0)
        assert band_mask.any()
        assert power[band_mask].sum() / power.sum() >= 0.05
        dominant_in_band = fftfreqs[band_mask][np.argmax(power[band_mask])]
        assert 0.85 * f0 <= dominant_in_band <= 1.15 * f0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lead_shape", [(2,), (2, 3)])
def test_mel_to_stft_multichannel_stability(lead_shape, dtype, mel_basis, rng):
    """Batched/channel inputs keep their leading dimensions and precision,
    and each channel equals the result of inverting it independently."""
    n_bins = 1 + N_FFT // 2
    basis = mel_basis.astype(dtype)
    stft_orig = (rng.standard_normal(size=(*lead_shape, n_bins, N_FRAMES)) ** 2).astype(dtype)
    mels = np.einsum("...ft,mf->...mt", stft_orig, basis)

    stft_inverse = librosa.feature.inverse.mel_to_stft(mels, sr=SR, n_fft=N_FFT, power=2.0)

    assert stft_inverse.shape == (*lead_shape, n_bins, N_FRAMES)
    assert stft_inverse.dtype == dtype
    assert np.all(stft_inverse >= 0)

    # The NNLS solve is deterministic and block-exact: vectorized and
    # independent per-channel calls agree exactly.
    for index in np.ndindex(lead_shape):
        single = librosa.feature.inverse.mel_to_stft(mels[index], sr=SR, n_fft=N_FFT, power=2.0)
        np.testing.assert_array_equal(stft_inverse[index], single)

    # Distinct channels must not collapse onto one another.
    assert not np.allclose(stft_inverse[lead_shape[0] - 1], stft_inverse[0])


def test_mel_to_audio_multichannel_shape_and_validity():
    """Griffin-Lim inversion of a stereo mel spectrogram returns the same
    channel count and requested length, with finite valid audio per channel,
    and the reconstruction is reproducible under a fixed RNG seed."""
    n_fft = 512
    hop_length = 128
    y = np.vstack(
        [
            librosa.tone(440.0, sr=SR, duration=0.5),
            librosa.tone(880.0, sr=SR, duration=0.5),
        ]
    )
    # Build the mel spectrogram with the same frame parameters used for the
    # Griffin-Lim inversion so frame counts stay consistent.
    mels = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=32)

    y_inv = librosa.feature.inverse.mel_to_audio(
        mels,
        sr=SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=2,
        length=y.shape[-1],
        rng=0,
    )

    assert y_inv.shape == y.shape
    assert y_inv.dtype == np.float32
    assert np.isfinite(y_inv).all()
    for channel in y_inv:
        assert librosa.util.valid_audio(channel)
    assert not np.allclose(y_inv[0], y_inv[1])

    # A fixed seed makes the phase initialization reproducible.
    y_inv_again = librosa.feature.inverse.mel_to_audio(
        mels,
        sr=SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=2,
        length=y.shape[-1],
        rng=0,
    )
    np.testing.assert_array_equal(y_inv, y_inv_again)

    # A different seed yields a different phase realization.
    y_inv_other = librosa.feature.inverse.mel_to_audio(
        mels,
        sr=SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=2,
        length=y.shape[-1],
        rng=7,
    )
    assert not np.allclose(y_inv, y_inv_other)
