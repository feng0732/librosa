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


@pytest.fixture
def stereo_sample_timeline():
    # frame_length=1 makes trim/split boundaries exact sample coordinates.
    y = np.zeros((2, 48), dtype=np.float64)
    y[0, 4:10] = 1.0
    y[1, 20:27] = 0.16
    return y


@pytest.mark.parametrize("ref", [1.0, np.max])
def test_trim_multichannel_shared_sample_timeline(stereo_sample_timeline, ref):
    y = stereo_sample_timeline
    params = dict(top_db=20, ref=ref, frame_length=1, hop_length=1)

    y_trim, trim_idx = librosa.effects.trim(y, **params)
    intervals = librosa.effects.split(y, **params)

    expected_intervals = np.asarray([[4, 10], [20, 27]])
    np.testing.assert_array_equal(trim_idx, [4, 27])
    np.testing.assert_array_equal(intervals, expected_intervals)

    # The index belongs to the original joint time axis. Slicing both channels
    # with it must preserve their one-to-one sample correspondence.
    np.testing.assert_array_equal(y_trim, y[:, 4:27])
    np.testing.assert_array_equal(y_trim[0], y[0, 4:27])
    np.testing.assert_array_equal(y_trim[1], y[1, 4:27])

    # A talkspurt on either channel keeps the shared interval open.
    for sample in (7, 23):
        assert np.any((intervals[:, 0] <= sample) & (sample < intervals[:, 1]))


def test_trim_split_multichannel_all_silence():
    y = np.zeros((2, 16), dtype=np.float64)
    params = dict(top_db=20, ref=1.0, frame_length=1, hop_length=1)

    y_trim, trim_idx = librosa.effects.trim(y, **params)
    intervals = librosa.effects.split(y, **params)

    np.testing.assert_array_equal(trim_idx, [0, 0])
    assert y_trim.shape == (2, 0)
    assert intervals.shape == (0, 2)


@pytest.fixture
def stereo_interview_timeline():
    frame_length = 128
    hop_length = 32
    speech_intervals = np.asarray(
        [
            [4 * hop_length, 12 * hop_length],
            [20 * hop_length, 28 * hop_length],
        ]
    )

    y = np.zeros((2, 32 * hop_length), dtype=np.float64)
    y[0, speech_intervals[0, 0] : speech_intervals[0, 1]] = 1.0
    y[1, speech_intervals[1, 0] : speech_intervals[1, 1]] = 0.20

    return y, frame_length, hop_length, speech_intervals


@pytest.mark.parametrize("ref", [1.0, np.max])
def test_trim_split_multichannel_reference_boundaries(stereo_interview_timeline, ref):
    y, frame_length, hop_length, _ = stereo_interview_timeline
    params = dict(
        top_db=20,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    y_trim, trim_idx = librosa.effects.trim(y, **params)
    intervals = librosa.effects.split(y, **params)

    expected_intervals = np.asarray([[96, 448], [640, 928]])
    np.testing.assert_array_equal(trim_idx, [96, 928])
    np.testing.assert_array_equal(intervals, expected_intervals)

    # Trim and split expose the same shared timeline endpoints.
    assert trim_idx[0] == intervals[0, 0]
    assert trim_idx[1] == intervals[-1, 1]
    np.testing.assert_array_equal(y_trim, y[:, trim_idx[0] : trim_idx[1]])


@pytest.mark.parametrize("ref", [1.0, np.max])
@pytest.mark.parametrize("top_db", [16, 18, 20, 22, 30])
def test_trim_and_split_boundaries_map_to_original_samples(stereo_interview_timeline, ref, top_db):
    y, frame_length, hop_length, speech_intervals = stereo_interview_timeline
    params = dict(
        top_db=top_db,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    y_trim, trim_idx = librosa.effects.trim(y, **params)
    intervals = librosa.effects.split(y, **params)

    assert intervals.shape == (2, 2)
    np.testing.assert_array_equal(trim_idx, [intervals[0, 0], intervals[-1, 1]])
    np.testing.assert_array_equal(y_trim, y[:, trim_idx[0] : trim_idx[1]])

    n_samples = y.shape[-1]
    assert np.all((0 <= intervals) & (intervals <= n_samples))
    assert np.all(intervals % hop_length == 0)

    for interval, speech_interval in zip(intervals, speech_intervals, strict=True):
        speech_start, speech_end = speech_interval
        speech_center = (speech_start + speech_end) // 2

        # The returned sample coordinates stay within one analysis frame of
        # the designed talkspurt boundaries and contain its center.
        assert abs(interval[0] - speech_start) <= frame_length
        assert abs(interval[1] - speech_end) <= frame_length
        assert interval[0] <= speech_center < interval[1]

    # Prefix, inter-interval gap, and suffix remain outside every segment.
    for sample in (2 * hop_length, 16 * hop_length, 30 * hop_length):
        assert not np.any((intervals[:, 0] <= sample) & (sample < intervals[:, 1]))


@pytest.mark.parametrize("ref", [1.0, np.max])
def test_multichannel_threshold_boundaries_are_nested(stereo_interview_timeline, ref):
    y, frame_length, hop_length, _ = stereo_interview_timeline
    top_db_values = [16, 18, 20, 22, 30]
    results = []

    for top_db in top_db_values:
        trim_idx = librosa.effects.trim(
            y,
            top_db=top_db,
            ref=ref,
            frame_length=frame_length,
            hop_length=hop_length,
        )[1]
        intervals = librosa.effects.split(
            y,
            top_db=top_db,
            ref=ref,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        results.append((trim_idx, intervals))

    # Increasing top_db makes the criterion more permissive.  Stricter
    # intervals must therefore be nested in the more permissive timeline.
    for result_index in range(len(results) - 1):
        strict_idx, strict_intervals = results[result_index]
        loose_idx, loose_intervals = results[result_index + 1]

        assert loose_idx[0] <= strict_idx[0]
        assert loose_idx[1] >= strict_idx[1]

        for start, end in strict_intervals:
            contained = (loose_intervals[:, 0] <= start) & (end <= loose_intervals[:, 1])
            assert np.any(contained)


def test_reference_level_change_keeps_multichannel_speech(
    stereo_interview_timeline,
):
    y_full_scale, frame_length, hop_length, speech_intervals = stereo_interview_timeline
    y = 0.5 * y_full_scale  # Loud/ref-peak and quiet channels become 0.5 and 0.10.
    params = dict(top_db=24, frame_length=frame_length, hop_length=hop_length)

    y_absolute, idx_absolute = librosa.effects.trim(y, ref=1.0, **params)
    intervals_absolute = librosa.effects.split(y, ref=1.0, **params)
    y_peak, idx_peak = librosa.effects.trim(y, ref=np.max, **params)
    intervals_peak = librosa.effects.split(y, ref=np.max, **params)

    # Neither reference convention may remove either original talkspurt core.
    for intervals in (intervals_absolute, intervals_peak):
        for speech_start, speech_end in speech_intervals:
            center = (speech_start + speech_end) // 2
            assert np.any((intervals[:, 0] <= center) & (center < intervals[:, 1]))

    # np.max follows the 0.5 RMS peak, so it is more permissive than ref=1.
    assert idx_peak[0] <= idx_absolute[0]
    assert idx_peak[1] >= idx_absolute[1]
    for start, end in intervals_absolute:
        assert np.any((intervals_peak[:, 0] <= start) & (end <= intervals_peak[:, 1]))

    # The reference must affect the threshold, while both outputs still map
    # exactly back to channel-aligned samples in the original recording.
    assert not np.array_equal(idx_peak, idx_absolute)
    assert not np.array_equal(intervals_peak, intervals_absolute)
    np.testing.assert_array_equal(y_absolute, y[:, idx_absolute[0] : idx_absolute[1]])
    np.testing.assert_array_equal(y_peak, y[:, idx_peak[0] : idx_peak[1]])


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
