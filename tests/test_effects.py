#!/usr/bin/env python
"""Unit tests for the effects module"""

# Disable cache
import contextlib
import os

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

from contextlib import nullcontext as dnr

import numpy as np
import pytest

import librosa


@pytest.fixture(scope="module")
def y_multi():
    sr = 44100
    y = librosa.chirp(fmin=100, fmax=2000, sr=sr, duration=1.5)
    y = np.vstack([y, y[::-1]])
    return y, sr


@pytest.fixture(scope="module", params=[22050, 44100])
def ysr(request):
    sr = request.param
    y = librosa.chirp(fmin=100, fmax=2000, sr=sr, duration=1.5)
    return y, sr


@pytest.mark.parametrize(
    "rate,ctx",
    [
        (0.25, dnr()),
        (1.0, dnr()),
        (2.0, dnr()),
        (4.0, dnr()),
        (-1, pytest.raises(librosa.ParameterValueError)),
        (0, pytest.raises(librosa.ParameterValueError)),
    ],
)
@pytest.mark.parametrize("n_fft", [2048, 2049])
def test_time_stretch(ysr, rate, ctx, n_fft):

    with ctx:
        y, sr = ysr
        ys = librosa.effects.time_stretch(y, rate=rate, n_fft=n_fft)

        orig_duration = librosa.get_duration(y=y, sr=sr)
        new_duration = librosa.get_duration(y=ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert np.allclose(orig_duration, rate * new_duration, rtol=1e-2, atol=1e-3)


def test_time_stretch_multi(y_multi):
    y, _sr = y_multi

    # compare each channel
    C0 = librosa.effects.time_stretch(y[0], rate=1.1)
    C1 = librosa.effects.time_stretch(y[1], rate=1.1)
    Call = librosa.effects.time_stretch(y, rate=1.1)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("n_steps", [-1.5, 1.5, 5])
@pytest.mark.parametrize(
    "bins_per_octave,ctx",
    [
        (12, dnr()),
        (24, dnr()),
        (-1, pytest.raises(librosa.ParameterValueError)),
        (0, pytest.raises(librosa.ParameterValueError)),
    ],
)
@pytest.mark.parametrize("n_fft", [2048, 2049])
def test_pitch_shift(ysr, n_steps, bins_per_octave, ctx, n_fft):

    with ctx:
        y, sr = ysr
        ys = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave, n_fft=n_fft)

        orig_duration = librosa.get_duration(y=y, sr=sr)
        new_duration = librosa.get_duration(y=ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert orig_duration == new_duration


def test_pitch_shift_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.effects.pitch_shift(y[0], sr=sr, n_steps=1)
    C1 = librosa.effects.pitch_shift(y[1], sr=sr, n_steps=1)
    Call = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)

    # Check each channel
    # Relaxing precision here due to architecture sensitivities on linux-arm64
    assert np.allclose(C0, Call[0], atol=1e-6, rtol=1e-6)
    assert np.allclose(C1, Call[1], atol=1e-6, rtol=1e-6)

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
def test_time_stretch_pitch_shift_pad_mode(ysr, pad_mode):
    # pad_mode must go to stft only; istft does not accept it
    y, sr = ysr

    ys = librosa.effects.time_stretch(y, rate=1.5, pad_mode=pad_mode)
    yp = librosa.effects.pitch_shift(y, sr=sr, n_steps=2, pad_mode=pad_mode)

    assert np.all(np.isfinite(ys))
    assert yp.shape == y.shape

    if pad_mode != "constant":
        ys_const = librosa.effects.time_stretch(y, rate=1.5, pad_mode="constant")
        assert not np.allclose(ys, ys_const)


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_mono(align_zeros):

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1], dtype=float)
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2], dtype=float)
    intervals = np.asarray([[2, 4], [6, 8], [0, 2], [8, 10], [4, 6]])

    y_out = librosa.effects.remix(y, intervals, align_zeros=align_zeros)
    assert np.allclose(y_out, y_t)


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_stereo(align_zeros):

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1], dtype=float)
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2], dtype=float)
    y = np.vstack([y, y])
    y_t = np.vstack([y_t, y_t])

    intervals = np.asarray([[2, 4], [6, 8], [0, 2], [8, 10], [4, 6]])

    y_out = librosa.effects.remix(y, intervals, align_zeros=align_zeros)
    assert np.allclose(y_out, y_t)


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_tuples(align_zeros):
    # intervals given as a list of (start, end) tuples
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1], dtype=float)
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2], dtype=float)
    intervals = [(2, 4), (6, 8), (0, 2), (8, 10), (4, 6)]

    y_out = librosa.effects.remix(y, intervals, align_zeros=align_zeros)
    assert np.allclose(y_out, y_t)


def test_hpss(ysr):

    y, _sr = ysr

    y_harm, y_perc = librosa.effects.hpss(y)

    # Make sure that the residual energy is generally small
    y_residual = y - y_harm - y_perc

    rms_orig = librosa.feature.rms(y=y)
    rms_res = librosa.feature.rms(y=y_residual)

    assert np.percentile(rms_orig, 0.01) > np.percentile(rms_res, 0.99)


def test_hpss_multi(y_multi):
    y, _sr = y_multi

    # compare each channel
    CH0, CP0 = librosa.effects.hpss(y[0])
    CH1, CP1 = librosa.effects.hpss(y[1])
    CHall, CPall = librosa.effects.hpss(y)

    # Check each channel
    assert np.allclose(CH0, CHall[0])
    assert np.allclose(CP0, CPall[0])
    assert np.allclose(CH1, CHall[1])
    assert np.allclose(CP1, CPall[1])

    # Verify that they're not all the same
    assert not np.allclose(CHall[0], CHall[1])
    assert not np.allclose(CPall[0], CPall[1])


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_percussive(ysr, window):

    y, _sr = ysr

    _yh1, yp1 = librosa.effects.hpss(y, window=window)

    yp2 = librosa.effects.percussive(y, window=window)

    assert np.allclose(yp1, yp2)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_harmonic(ysr, window):

    y, _sr = ysr

    yh1, _yp1 = librosa.effects.hpss(y, window=window)

    yh2 = librosa.effects.harmonic(y, window=window)

    assert np.allclose(yh1, yh2)


@pytest.mark.parametrize(
    "effect_fn",
    [librosa.effects.hpss, librosa.effects.harmonic, librosa.effects.percussive],
)
def test_effects_window(ysr, effect_fn):
    # The window must reach the STFT.  Asserting only that the outputs differ
    # would also pass if it reached stft but not istft, so the reconstruction
    # check below and component equivalence tests above pin synthesis as well.
    y, _sr = ysr

    def first(result):
        return result[0] if isinstance(result, tuple) else result

    y_hann = first(effect_fn(y=y, window="hann"))
    y_hamming = first(effect_fn(y=y, window="hamming"))
    y_blackman = first(effect_fn(y=y, window="blackman"))

    assert not np.allclose(y_hann, y_hamming)
    assert not np.allclose(y_hann, y_blackman)


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
def test_hpss_window_reconstruction(ysr, window):
    # Analysis and synthesis must use the same window, otherwise the harmonic
    # and percussive parts no longer sum back to the input.
    y, _sr = ysr

    y_harm, y_perc = librosa.effects.hpss(y=y, window=window)

    assert np.allclose(y_harm + y_perc, y, atol=1e-6)


@pytest.fixture(scope="module", params=[False, True], ids=["mono", "stereo"])
def y_trim(request):
    # construct 5 seconds of stereo silence
    # Stick a sine wave in the middle three seconds

    sr = 22050
    trim_duration = 3.0
    y = np.sin(2 * np.pi * 440.0 * np.arange(0, trim_duration * sr) / sr)
    y = librosa.util.pad_center(y, size=5 * sr)

    if request.param:
        y = np.vstack([y, np.zeros_like(y)])
    return y


@pytest.mark.parametrize("top_db", [60, 40, 20])
@pytest.mark.parametrize("ref", [1, np.max])
@pytest.mark.parametrize("trim_duration", [3.0])
def test_trim(y_trim, top_db, ref, trim_duration):

    yt, idx = librosa.effects.trim(y_trim, top_db=top_db, ref=ref)

    # Test for index position
    fidx = [slice(None)] * y_trim.ndim
    fidx[-1] = slice(*idx.tolist())
    assert np.allclose(yt, y_trim[tuple(fidx)])

    # Verify logamp
    rms = librosa.feature.rms(y=librosa.to_mono(yt), center=False)
    logamp = librosa.power_to_db(rms**2, ref=ref, top_db=None)
    assert np.all(logamp > -top_db)

    # Verify logamp
    rms_all = librosa.feature.rms(y=librosa.to_mono(y_trim)).squeeze()
    logamp_all = librosa.power_to_db(rms_all**2, ref=ref, top_db=None)

    start = int(librosa.samples_to_frames(idx[0]))
    stop = int(librosa.samples_to_frames(idx[1]))
    assert np.all(logamp_all[:start] <= -top_db)
    assert np.all(logamp_all[stop:] <= -top_db)

    # Verify duration
    duration = librosa.get_duration(y=yt)
    assert np.allclose(duration, trim_duration, atol=1e-1), duration


def test_trim_empty():

    y = np.zeros(1)

    yt, idx = librosa.effects.trim(y, ref=1)

    assert yt.size == 0
    assert idx[0] == 0
    assert idx[1] == 0


def test_trim_multi(y_multi):
    # Test for https://github.com/librosa/librosa/issues/1489
    y, _sr = y_multi
    librosa.effects.trim(y=y)


def test_split_multi(y_multi):
    # Test for https://github.com/librosa/librosa/issues/1489
    y, _sr = y_multi

    librosa.effects.split(y=y)


@pytest.fixture(
    scope="module",
    params=[0, 1, 2, 3],
    ids=["constant", "end-silent", "full-signal", "gaps"],
)
def y_split_idx(request):

    sr = 8192
    y = np.ones(5 * sr)

    if request.param == 0:
        # Constant
        idx_true = np.asarray([[0, 5 * sr]])

    elif request.param == 1:
        # end-silent
        y[::2] *= -1
        y[4 * sr :] = 0
        idx_true = np.asarray([[0, 4 * sr]])

    elif request.param == 2:
        # begin-silent
        y[::2] *= -1
        idx_true = np.asarray([[0, 5 * sr]])
    else:
        # begin and end are silent
        y[::2] *= -1

        # Zero out all but two intervals
        y[:sr] = 0
        y[2 * sr : 3 * sr] = 0
        y[4 * sr :] = 0

        # The true non-silent intervals
        idx_true = np.asarray([[sr, 2 * sr], [3 * sr, 4 * sr]])

    return y, idx_true


@pytest.mark.parametrize("frame_length", [1024, 2048, 4096])
@pytest.mark.parametrize("hop_length", [256, 512, 1024])
@pytest.mark.parametrize("top_db", [20, 60, 80])
def test_split(y_split_idx, frame_length, hop_length, top_db):

    y, idx_true = y_split_idx

    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    assert np.all(intervals <= y.shape[-1])

    int_match = librosa.util.match_intervals(intervals, idx_true)

    for i in range(len(intervals)):
        i_true = idx_true[int_match[i]]

        assert np.all(np.abs(i_true - intervals[i]) <= frame_length), intervals[i]


@pytest.mark.parametrize("coef", [0.5, 0.99])
@pytest.mark.parametrize("zi", [None, 0, [0]])
@pytest.mark.parametrize("return_zf", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preemphasis(coef, zi, return_zf: bool, dtype):
    x = np.arange(10, dtype=dtype)

    if return_zf:
        y, _zf = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)
    else:
        y = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)

    assert np.allclose(y[1:], x[1:] - coef * x[:-1])
    assert x.dtype == y.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preemphasis_continue(dtype):

    # Compare pre-emphasis computed in parts to that of the whole sequence in one go
    x = np.arange(64, dtype=dtype)

    y1, zf1 = librosa.effects.preemphasis(x[:32], return_zf=True)
    y2, zf2 = librosa.effects.preemphasis(x[32:], return_zf=True, zi=zf1)

    y_all, zf_all = librosa.effects.preemphasis(x, return_zf=True)

    assert np.allclose(y_all, np.concatenate([y1, y2]))
    assert np.allclose(zf2, zf_all)
    assert x.dtype == y_all.dtype


def test_preemphasis_multi(y_multi):
    y, _sr = y_multi

    # compare each channel
    C0, zf0 = librosa.effects.preemphasis(y[0], return_zf=True)
    C1, zf1 = librosa.effects.preemphasis(y[1], return_zf=True)
    Call, zf = librosa.effects.preemphasis(y, return_zf=True)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert np.allclose(zf0, zf[0])
    assert np.allclose(zf1, zf[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])
    assert not np.allclose(zf[0], zf[1])


def test_deemphasis_multi(y_multi):
    y, _sr = y_multi

    # compare each channel
    C0, zf0 = librosa.effects.deemphasis(y[0], return_zf=True)
    C1, zf1 = librosa.effects.deemphasis(y[1], return_zf=True)
    Call, zf = librosa.effects.deemphasis(y, return_zf=True)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert np.allclose(zf0, zf[0])
    assert np.allclose(zf1, zf[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])
    assert not np.allclose(zf[0], zf[1])


@pytest.mark.parametrize("coef", [0.5, 0.99])
@pytest.mark.parametrize("zi", [None, 0, [0]])
@pytest.mark.parametrize("return_zf", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_deemphasis(coef, zi, return_zf, dtype):
    x = np.arange(10, dtype=dtype)

    y = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)

    if return_zf:
        y, _zf = y

    y_deemph = librosa.effects.deemphasis(y, coef=coef, zi=zi)

    assert np.allclose(x, y_deemph)
    assert x.dtype == y_deemph.dtype


def _sharp_stereo(dtype=np.float64):
    """Deterministic stereo signal with channel-specific seam transients."""
    rng = np.random.default_rng(116)
    y = 0.25 * rng.standard_normal((2, 72))

    y[0, 15:18] += np.asarray([4.0, -3.5, 2.5])
    y[1, 31:34] += np.asarray([-3.0, 4.5, -2.5])
    y[0, 47:50] += np.asarray([-2.0, 3.0, -1.5])
    y[1, 55:58] += np.asarray([2.5, -3.0, 2.0])

    return y.astype(dtype, copy=False)


def _block_edges(n, cuts):
    boundaries = (0,) + tuple(cuts) + (n,)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _preemphasis_blocks(y, cuts, coef=0.97, zi=None):
    blocks = []
    state = zi

    for start, stop in _block_edges(y.shape[-1], cuts):
        kwargs = {"coef": coef, "return_zf": True}
        if state is not None:
            kwargs["zi"] = state
        block, state = librosa.effects.preemphasis(y[..., start:stop], **kwargs)
        blocks.append(block)

    return np.concatenate(blocks, axis=-1), state


def _deemphasis_blocks(y, cuts, coef=0.97, zi=None):
    blocks = []
    state = zi

    for start, stop in _block_edges(y.shape[-1], cuts):
        block, state = librosa.effects.deemphasis(
            y[..., start:stop], coef=coef, zi=state, return_zf=True
        )
        blocks.append(block)

    return np.concatenate(blocks, axis=-1), state


@pytest.mark.parametrize("cuts", [(2,), (16,), (32,), (69,), (16, 40), (2, 23, 41, 69)])
@pytest.mark.parametrize("coef", [0.5, 0.97])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preemphasis_stereo_blocks_match_whole(cuts, coef, dtype):
    x = _sharp_stereo(dtype)
    rtol = atol = 1e-6 if dtype == np.float32 else 1e-12

    y_whole, zf_whole = librosa.effects.preemphasis(x, coef=coef, return_zf=True)
    y_block, zf_block = _preemphasis_blocks(x, cuts, coef=coef)

    assert y_block.shape == x.shape
    assert y_block.dtype == x.dtype
    np.testing.assert_allclose(y_block, y_whole, rtol=rtol, atol=atol)
    np.testing.assert_allclose(zf_block, zf_whole, rtol=rtol, atol=atol)

    # Every cut must preserve the causal sample equation at the new block edge.
    for cut in cuts:
        expected = x[..., cut] - coef * x[..., cut - 1]
        np.testing.assert_allclose(y_block[..., cut], expected, rtol=rtol, atol=atol)

    # Compare against independent mono streams to catch cross-channel states.
    for channel in range(x.shape[0]):
        y_mono, zf_mono = librosa.effects.preemphasis(
            x[channel], coef=coef, return_zf=True
        )
        np.testing.assert_allclose(y_block[channel], y_mono, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            zf_block[channel], zf_mono, rtol=rtol, atol=atol
        )

    restored_block = librosa.effects.deemphasis(y_block, coef=coef)
    restored_whole = librosa.effects.deemphasis(y_whole, coef=coef)
    np.testing.assert_allclose(restored_block, x, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        restored_block, restored_whole, rtol=rtol, atol=atol
    )


def test_preemphasis_deemphasis_stereo_roundtrip_invariant_to_every_cut():
    x = _sharp_stereo()
    coef = 0.97

    y_whole, zf_whole = librosa.effects.preemphasis(x, coef=coef, return_zf=True)
    restored_whole = librosa.effects.deemphasis(y_whole, coef=coef)

    for cut in range(2, x.shape[-1] - 1):
        y_block, zf_block = _preemphasis_blocks(x, (cut,), coef=coef)

        np.testing.assert_allclose(y_block, y_whole, atol=1e-12)
        np.testing.assert_allclose(zf_block, zf_whole, atol=1e-12)

        restored = librosa.effects.deemphasis(y_block, coef=coef)
        np.testing.assert_allclose(restored, x, atol=1e-12)
        np.testing.assert_allclose(restored, restored_whole, atol=1e-12)


@pytest.mark.parametrize("cuts", [(2,), (16,), (69,), (7, 23, 41)])
def test_preemphasis_stereo_explicit_state_is_channelwise(cuts):
    x = _sharp_stereo()
    coef = 0.97
    zi = np.array([[1.1], [-0.8]], dtype=x.dtype)

    y_whole, zf_whole = librosa.effects.preemphasis(
        x, coef=coef, zi=zi, return_zf=True
    )
    y_block, zf_block = _preemphasis_blocks(x, cuts, coef=coef, zi=zi)

    assert zf_block.shape == (2, 1)
    np.testing.assert_allclose(y_block, y_whole)
    np.testing.assert_allclose(zf_block, zf_whole)
    assert not np.allclose(zf_block[0], zf_block[1])

    for channel in range(2):
        y_mono, zf_mono = librosa.effects.preemphasis(
            x[channel], coef=coef, zi=float(zi[channel, 0]), return_zf=True
        )
        np.testing.assert_allclose(y_block[channel], y_mono)
        np.testing.assert_allclose(zf_block[channel], zf_mono)


@pytest.mark.parametrize("cuts", [(2,), (16,), (32,), (69,), (16, 40), (2, 23, 41, 69)])
@pytest.mark.parametrize("coef", [0.5, 0.97])
def test_deemphasis_stereo_blocks_match_whole(cuts, coef):
    rng = np.random.default_rng(272)
    y = 0.25 * rng.standard_normal((2, 72))
    y[:, 16] += np.asarray([3.0, -3.0])
    y[:, 40] += np.asarray([-2.5, 2.5])
    zi = np.array([[0.9], [-0.7]])

    y_whole, zf_whole = librosa.effects.deemphasis(
        y, coef=coef, zi=zi, return_zf=True
    )
    y_block, zf_block = _deemphasis_blocks(y, cuts, coef=coef, zi=zi)

    assert y_block.shape == y.shape
    assert zf_block.shape == (2, 1)
    np.testing.assert_allclose(y_block, y_whole)
    np.testing.assert_allclose(zf_block, zf_whole)

    for channel in range(2):
        y_mono, zf_mono = librosa.effects.deemphasis(
            y[channel], coef=coef, zi=float(zi[channel, 0]), return_zf=True
        )
        np.testing.assert_allclose(y_block[channel], y_mono)
        np.testing.assert_allclose(zf_block[channel], zf_mono)


@pytest.mark.parametrize("cuts", [(2,), (16,), (32,), (69,), (16, 40), (2, 23, 41, 69)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stereo_blockwise_roundtrip_continues_zero_state(cuts, dtype):
    x = _sharp_stereo(dtype)
    coef = 0.97
    tol = 1e-6 if dtype == np.float32 else 1e-12
    zi = np.zeros((2, 1), dtype=dtype)

    emphasized, zf_emphasized = _preemphasis_blocks(x, cuts, coef=coef, zi=zi)
    emphasized_whole, zf_whole = librosa.effects.preemphasis(
        x, coef=coef, zi=zi, return_zf=True
    )
    np.testing.assert_allclose(emphasized, emphasized_whole, rtol=tol, atol=tol)
    np.testing.assert_allclose(zf_emphasized, zf_whole, rtol=tol, atol=tol)

    restored, zf_restored = _deemphasis_blocks(
        emphasized, cuts, coef=coef, zi=np.zeros_like(zi)
    )
    restored_whole, zf_restored_whole = librosa.effects.deemphasis(
        emphasized_whole, coef=coef, zi=np.zeros_like(zi), return_zf=True
    )

    np.testing.assert_allclose(restored, x, rtol=tol, atol=tol)
    np.testing.assert_allclose(
        restored, restored_whole, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        zf_restored, zf_restored_whole, rtol=tol, atol=tol
    )


@pytest.mark.parametrize("active_channel", [0, 1])
@pytest.mark.parametrize("cuts", [(3,), (32,), (62,), (15, 31, 47)])
def test_preemphasis_block_seam_transients_do_not_crosstalk(active_channel, cuts):
    n = 64
    silent_channel = 1 - active_channel
    x = np.zeros((2, n))

    for cut in cuts:
        x[active_channel, cut] = 4.0
        x[active_channel, cut - 1] = -3.0

    y_block, zf_block = _preemphasis_blocks(x, cuts)

    np.testing.assert_array_equal(y_block[silent_channel], np.zeros(n))
    np.testing.assert_array_equal(zf_block[silent_channel], np.zeros(1))

    y_mono, zf_mono = _preemphasis_blocks(x[active_channel], cuts)
    np.testing.assert_allclose(y_block[active_channel], y_mono)
    np.testing.assert_allclose(zf_block[active_channel], zf_mono)

    restored_whole = librosa.effects.deemphasis(y_block)
    np.testing.assert_array_equal(
        restored_whole[silent_channel], np.zeros(n)
    )
    np.testing.assert_allclose(
        restored_whole[active_channel], x[active_channel], atol=1e-12
    )

    restored_blocks, _ = _deemphasis_blocks(
        y_block, cuts, zi=np.zeros((2, 1))
    )
    np.testing.assert_array_equal(
        restored_blocks[silent_channel], np.zeros(n)
    )
    np.testing.assert_allclose(
        restored_blocks[active_channel], x[active_channel], atol=1e-12
    )


def test_preemphasis_stereo_blocks_on_smooth_fixture(y_multi):
    y, _sr = y_multi
    n = y.shape[-1]
    cuts = (n // 3, 2 * n // 3)

    y_whole, zf_whole = librosa.effects.preemphasis(y, return_zf=True)
    y_block, zf_block = _preemphasis_blocks(y, cuts)

    np.testing.assert_allclose(y_block, y_whole)
    np.testing.assert_allclose(zf_block, zf_whole)

    restored = librosa.effects.deemphasis(y_block)
    np.testing.assert_allclose(restored, y, atol=1e-12)


@pytest.mark.parametrize("mode", ["sine", "triangle", "square"])
@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2])
def test_tremolo(ysr, mode, depth, phase):
    y, sr = ysr
    yt = librosa.effects.tremolo(y, sr=sr, rate=5.0, depth=depth, mode=mode, phase=phase)

    assert yt.shape == y.shape
    assert yt.dtype == y.dtype

    if depth == 0.0:
        assert np.allclose(y, yt)


def test_tremolo_multi(y_multi):
    y, sr = y_multi
    C0 = librosa.effects.tremolo(y[0], sr=sr, rate=5.0)
    C1 = librosa.effects.tremolo(y[1], sr=sr, rate=5.0)
    Call = librosa.effects.tremolo(y, sr=sr, rate=5.0)

    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])


def test_tremolo_errors():
    y = np.ones(100)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.tremolo(y, sr=0)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.tremolo(y, sr=22050, rate=-1)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.tremolo(y, sr=22050, depth=1.5)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.tremolo(y, sr=22050, mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterTypeError):
        librosa.effects.tremolo(np.ones(100, dtype=int), sr=22050)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tremolo_dtype(dtype):
    y = np.ones(22050, dtype=dtype)
    yt = librosa.effects.tremolo(y, sr=22050)
    assert yt.dtype == dtype


@pytest.mark.parametrize("mode", ["sine", "triangle"])
@pytest.mark.parametrize("depth", [0.0, 0.5])
@pytest.mark.parametrize("kind", ["linear", "cubic"])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2])
def test_vibrato(ysr, mode, depth, kind, phase):
    y, sr = ysr
    yv = librosa.effects.vibrato(y, sr=sr, rate=5.0, depth=depth, mode=mode, kind=kind, phase=phase)

    assert yv.shape == y.shape
    assert yv.dtype == y.dtype


def test_vibrato_multi(y_multi):
    y, sr = y_multi
    C0 = librosa.effects.vibrato(y[0], sr=sr, rate=5.0)
    C1 = librosa.effects.vibrato(y[1], sr=sr, rate=5.0)
    Call = librosa.effects.vibrato(y, sr=sr, rate=5.0)

    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])


def test_vibrato_errors():
    y = np.ones(100)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.vibrato(y, sr=-1)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.vibrato(y, sr=22050, rate=0)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.vibrato(y, sr=22050, depth=-0.1)
    with pytest.raises(librosa.ParameterValueError):
        librosa.effects.vibrato(y, sr=22050, mode="square")  # type: ignore[arg-type]
    with pytest.raises(librosa.ParameterTypeError):
        librosa.effects.vibrato(np.ones(100, dtype=int), sr=22050)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vibrato_dtype(dtype):
    y = np.ones(22050, dtype=dtype)
    yv = librosa.effects.vibrato(y, sr=22050)
    assert yv.dtype == dtype


def test_vibrato_params():
    y = np.ones(22050)
    # Test hop_length=None
    yv_none = librosa.effects.vibrato(y, sr=22050, hop_length=None)
    assert yv_none.shape == y.shape

    # Test custom win_length and n_fft
    yv_win = librosa.effects.vibrato(y, sr=22050, n_fft=1024, win_length=512)
    assert yv_win.shape == y.shape

    # Test pad_mode
    yv_pad = librosa.effects.vibrato(y, sr=22050, pad_mode="reflect")
    assert yv_pad.shape == y.shape

    # Test depth=0.0 returns exact copy
    yv_zero = librosa.effects.vibrato(y, sr=22050, depth=0.0)
    assert np.allclose(y, yv_zero)


def test_vibrato_short():
    # Signal producing < 2 frames in STFT should early return copy
    y = np.ones(100)
    with pytest.warns(UserWarning, match=r"n_fft=.*is too large for input signal"):
        yv = librosa.effects.vibrato(y, sr=22050)
    assert np.allclose(y, yv)
