#!/usr/bin/env python
"""Unit tests for beat-driven remixing of stereo audio.

These tests exercise the editing chain used to score a dance video: beat
analysis produces frame indices, frames are converted to sample positions,
cut points are optionally snapped to zero crossings, and the selected
intervals are concatenated (possibly out of order and with repeats) by
`librosa.effects.remix`.

The synthetic fixture is a stereo signal assembled from equal-length
"beats" whose boundaries are exact zero crossings of the mono downmix.
The right channel crosses zero at every segment midpoint as well, so the
fixture can distinguish downmix-based snapping from per-channel snapping.
This makes it possible to check that the whole chain resolves to the
intended cut points, that boundaries never drift into a neighboring
segment, and that every output sample is drawn from the same source time
in both the left and right channels.
"""

# Disable cache
import contextlib
import os

with contextlib.suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")

import itertools

import numpy as np
import pytest

import librosa

SEG_LEN = 8192
SR = 22050

# One amplitude per beat.  Signs alternate so that the mono downmix crosses
# zero exactly at every segment boundary.  The left channel dominates the
# downmix, while the right channel carries a different waveform (a half-wave
# square, which also crosses zero at each segment midpoint) so that a channel
# mix-up or a per-channel snapping step is visible in the output.
LEFT_BASE = np.array([0.60, -0.45, 0.85, -0.25, 0.55, -0.75, 0.35, -0.95])
RIGHT_BASE = np.array([0.10, 0.15, 0.12, 0.09, 0.14, 0.11, 0.13, 0.08])

HOP_LENGTHS = [256, 300, 512, 1024, 1536]


def _dance_track():
    """Build the stereo test signal and its ground-truth segment boundaries.

    Each channel is assembled from ``LEFT_BASE.size`` segments of ``SEG_LEN``
    samples.  Left-channel samples vary slightly within a segment (a tiny
    ramp) so that no two positions in the channel share a value.

    Returns
    -------
    y : np.ndarray [shape=(2, n)]
    sr : int
    bounds : np.ndarray [shape=(n_segments + 1,)]
        Sample positions of every segment boundary, including 0 and len(y).
    """
    pos = np.arange(SEG_LEN, dtype=float)
    half_wave = np.where(pos < SEG_LEN // 2, 1.0, -1.0)
    left = np.concatenate([amp * (1.0 + pos / 2**23) for amp in LEFT_BASE])
    right = np.concatenate([amp * half_wave for amp in RIGHT_BASE])
    y = np.vstack([left, right])
    bounds = np.arange(LEFT_BASE.size + 1) * SEG_LEN
    return y, SR, bounds


def _slices(y, intervals):
    """Concatenate the requested source slices, as remix is expected to."""
    return np.concatenate([y[..., int(start) : int(end)] for start, end in intervals], axis=-1)


def _nearest(candidates, positions):
    """Nearest-value lookup with deterministic tie-breaking on the lower index."""
    candidates = np.asarray(candidates)
    positions = np.asarray(positions)
    distances = np.abs(positions[:, np.newaxis] - candidates[np.newaxis, :])
    return candidates[distances.argmin(axis=1)]


def _snap_positions(y, positions):
    """Map positions to the nearest zero crossing (or end-of-signal).

    This is the documented behavior of ``remix(align_zeros=True)``: zero
    crossings are taken from the mono downmix, and the end of the signal is
    treated as a zero crossing.
    """
    zeros = np.append(np.flatnonzero(librosa.zero_crossings(librosa.to_mono(y))), y.shape[-1])
    return _nearest(zeros, positions)


@pytest.fixture(scope="module")
def dance_track():
    return _dance_track()


@pytest.fixture(scope="module", params=HOP_LENGTHS, ids=[str(h) for h in HOP_LENGTHS])
def beat_chain(request, dance_track):
    """Beat analysis -> frame/sample conversion for a given hop length."""
    y, sr, bounds = dance_track
    hop_length = request.param
    bpm = 60.0 * sr / SEG_LEN
    mono = librosa.to_mono(y)

    tempo, frames = librosa.beat.beat_track(y=mono, sr=sr, hop_length=hop_length, bpm=bpm, trim=False, units="frames")
    tempo_samples, samples = librosa.beat.beat_track(
        y=mono, sr=sr, hop_length=hop_length, bpm=bpm, trim=False, units="samples"
    )
    return y, sr, bounds, bpm, hop_length, frames, samples, tempo, tempo_samples


def test_fixture_boundaries_are_zero_crossings(dance_track):
    # The rest of the module relies on this: each beat boundary is a zero
    # crossing of the mono downmix, and there are no crossings elsewhere.
    y, _sr, bounds = dance_track
    np.testing.assert_array_equal(np.flatnonzero(librosa.zero_crossings(librosa.to_mono(y))), bounds[:-1])

    # The right channel has extra crossings, so per-channel snapping would
    # disagree with downmix snapping (see test_remix_aligns_to_mono_downmix).
    right_zeros = np.flatnonzero(librosa.zero_crossings(y[1]))
    assert not np.array_equal(right_zeros, bounds[:-1])

    # The channels are genuinely different waveforms.
    assert not np.array_equal(y[0], y[1])


@pytest.mark.parametrize("align_zeros", [False, True])
@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("n_intervals", [1, 3])
def test_remix_keeps_inputs_unchanged(dance_track, align_zeros, as_array, n_intervals):
    y, _sr, bounds = dance_track
    # Off-boundary requests, so that zero-crossing alignment actually moves them.
    requests = [
        (int(bounds[2]) + 13, int(bounds[3])),
        (int(bounds[0]), int(bounds[1]) + 7),
        (int(bounds[2]) + 13, int(bounds[3])),
    ][:n_intervals]

    if as_array:
        request_in = np.asarray(requests, dtype=np.int64)
        request_snapshot = request_in.copy()
    else:
        request_in = list(requests)
        request_snapshot = list(requests)

    y_before = y.copy()
    y_out = librosa.effects.remix(y, request_in, align_zeros=align_zeros)

    # Neither the waveform nor the slicing request may be modified.
    np.testing.assert_array_equal(y, y_before)
    assert y.shape == y_before.shape
    assert y.dtype == y_before.dtype
    if as_array:
        np.testing.assert_array_equal(request_in, request_snapshot)
        assert request_in.dtype == request_snapshot.dtype
    else:
        assert request_in == request_snapshot

    # The output is an independent buffer: editing it cannot alter the source,
    # even when only one interval was requested.
    assert not np.shares_memory(y_out, y)
    y_out[...] = 0
    np.testing.assert_array_equal(y, y_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_remix_preserves_dtype(dance_track, dtype):
    y, _sr, bounds = dance_track
    y = y.astype(dtype)
    y_out = librosa.effects.remix(y, [(int(bounds[2]), int(bounds[3]))], align_zeros=False)

    assert y_out.dtype == dtype
    np.testing.assert_array_equal(y_out, y[..., bounds[2] : bounds[3]])


@pytest.mark.parametrize("align_zeros", [False, True])
@pytest.mark.parametrize("split_offset", [0, 137])
def test_remix_adjacent_subsegments_equal_merged(dance_track, align_zeros, split_offset):
    # The same selection, expressed either as one interval or as two adjacent
    # sub-intervals, must concatenate to identical content and length.
    y, _sr, bounds = dance_track
    start = int(bounds[1]) + 11
    stop = int(bounds[4]) - 9
    interior = int(bounds[2]) + split_offset

    merged = librosa.effects.remix(y, [(start, stop)], align_zeros=align_zeros)
    part_a = librosa.effects.remix(y, [(start, interior)], align_zeros=align_zeros)
    part_b = librosa.effects.remix(y, [(interior, stop)], align_zeros=align_zeros)
    split = librosa.effects.remix(y, [(start, interior), (interior, stop)], align_zeros=align_zeros)

    np.testing.assert_array_equal(split, merged)
    np.testing.assert_array_equal(np.concatenate([part_a, part_b], axis=-1), merged)

    assert split.shape[-1] == merged.shape[-1]
    assert part_a.shape[-1] + part_b.shape[-1] == merged.shape[-1]

    if not align_zeros:
        # Without zero-crossing alignment, no boundary may move at all.
        np.testing.assert_array_equal(merged, _slices(y, [(start, stop)]))
        assert merged.shape[-1] == stop - start


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_repeated_selection_length_and_content(dance_track, align_zeros):
    y, _sr, bounds = dance_track
    # The request is deliberately off-boundary for the aligned case.
    beat = [(int(bounds[3]) + 123, int(bounds[4]) - 77)]

    single = librosa.effects.remix(y, beat, align_zeros=align_zeros)
    doubled = librosa.effects.remix(y, beat * 2, align_zeros=align_zeros)
    tripled = librosa.effects.remix(y, beat * 3, align_zeros=align_zeros)

    np.testing.assert_array_equal(doubled, np.concatenate([single] * 2, axis=-1))
    np.testing.assert_array_equal(tripled, np.concatenate([single] * 3, axis=-1))

    assert doubled.shape[-1] == 2 * single.shape[-1]
    assert tripled.shape[-1] == 3 * single.shape[-1]


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_repeated_adjacent_splits_equal_repeated_merged(dance_track, align_zeros):
    # Repeating a selection twice must not depend on whether that selection is
    # requested as one interval or as two adjacent pieces.
    y, _sr, bounds = dance_track
    whole = (int(bounds[2]), int(bounds[5]))
    halves = [(int(bounds[2]), int(bounds[3])), (int(bounds[3]), int(bounds[5]))]
    order = [0, 1, 0, 1]

    merged_out = librosa.effects.remix(y, [whole, whole], align_zeros=align_zeros)
    split_out = librosa.effects.remix(y, [halves[k] for k in order], align_zeros=align_zeros)

    np.testing.assert_array_equal(split_out, merged_out)
    assert split_out.shape[-1] == merged_out.shape[-1]


def test_remix_stereo_time_mapping_reorder_and_repeats(dance_track):
    # After reordering and repeated references, output sample t must come from
    # y[..., idx[t]] on *both* channels: one shared time mapping per frame.
    y, _sr, bounds = dance_track
    segments = [(int(s), int(e)) for s, e in itertools.pairwise(bounds)]
    order = [6, 1, 6, 0, 7, 3]
    requests = [segments[k] for k in order]

    y_out = librosa.effects.remix(y, requests, align_zeros=False)

    idx = np.concatenate([np.arange(start, stop) for start, stop in requests])
    assert y_out.shape == (y.shape[0], idx.size)
    np.testing.assert_array_equal(y_out[0], y[0][idx])
    np.testing.assert_array_equal(y_out[1], y[1][idx])

    # The channels stay distinct: no L/R swap or interleaving.
    assert not np.array_equal(y_out[0], y_out[1])

    # Nothing is pulled in from unreferenced times.
    referenced = np.concatenate([np.arange(*segments[k]) for k in sorted(set(order))])
    np.testing.assert_array_equal(np.unique(idx), referenced)

    # The edit is not the identity mapping.
    assert not np.array_equal(y_out, y[..., : idx.size])


def test_remix_zero_crossing_snapping_moves_boundaries(dance_track):
    y, _sr, bounds = dance_track
    requests = [(int(bounds[1]) + 300, int(bounds[3]) - 300), (int(bounds[5]), int(bounds[6]) + 250)]

    snapped_positions = _snap_positions(y, np.ravel(requests))
    # Snapping must land exactly on the segment boundaries, not just nearby.
    np.testing.assert_array_equal(snapped_positions, bounds[[1, 3, 5, 6]])

    snapped = librosa.effects.remix(y, requests, align_zeros=True)
    expected_intervals = list(zip(snapped_positions[::2], snapped_positions[1::2], strict=True))
    np.testing.assert_array_equal(snapped, _slices(y, expected_intervals))
    assert snapped.shape[-1] == sum(int(stop) - int(start) for start, stop in expected_intervals)

    # Boundary movement actually happened, and the unsnapped edit is different.
    raw = librosa.effects.remix(y, requests, align_zeros=False)
    np.testing.assert_array_equal(raw, _slices(y, requests))
    assert raw.shape[-1] == sum(int(stop) - int(start) for start, stop in requests)
    assert not np.array_equal(np.ravel(requests), snapped_positions)
    assert snapped.shape[-1] != raw.shape[-1]


def test_remix_aligns_to_mono_downmix(dance_track):
    # Zero-crossing alignment is defined on the mono downmix.  The right
    # channel has its own crossings at the segment midpoints, which would give
    # a different (wrong) pair of cut points.
    y, _sr, bounds = dance_track
    request = [(5000, 20000)]

    mono_snapped = _snap_positions(y, np.ravel(request))
    np.testing.assert_array_equal(mono_snapped, bounds[[1, 2]])

    right_zeros = np.append(np.flatnonzero(librosa.zero_crossings(y[1])), y.shape[-1])
    right_snapped = _nearest(right_zeros, np.ravel(request))
    assert not np.array_equal(right_snapped, mono_snapped)
    assert (right_snapped[1] - right_snapped[0]) != (mono_snapped[1] - mono_snapped[0])

    y_out = librosa.effects.remix(y, request, align_zeros=True)
    np.testing.assert_array_equal(y_out, _slices(y, [tuple(mono_snapped)]))
    assert y_out.shape[-1] == mono_snapped[1] - mono_snapped[0]


def test_beat_frames_to_samples_consistency(beat_chain):
    # Frame -> sample conversion must be the same mapping at every hop length,
    # and it must be exactly frames * hop_length.
    y, _sr, _bounds, bpm, hop_length, frames, samples, tempo, tempo_samples = beat_chain

    assert frames.size > 0
    assert frames.size == samples.size
    np.testing.assert_allclose(np.atleast_1d(tempo), bpm)
    np.testing.assert_allclose(np.atleast_1d(tempo_samples), np.atleast_1d(tempo))

    np.testing.assert_array_equal(librosa.frames_to_samples(frames, hop_length=hop_length), samples)
    np.testing.assert_array_equal(samples, frames.astype(int) * hop_length)
    np.testing.assert_array_equal(librosa.samples_to_frames(samples, hop_length=hop_length), frames)

    assert np.all(np.diff(samples) > 0)
    assert samples.min() >= 0
    assert samples.max() <= y.shape[-1]


def test_beat_boundaries_stay_within_segments(beat_chain):
    # Whatever the hop length, a detected beat is closer to one true segment
    # boundary than half a segment, and snapping resolves it to that boundary.
    # Consequently a cut can never land inside a different sound segment.
    y, _sr, bounds, _bpm, _hop_length, _frames, samples, _tempo, _tempo_samples = beat_chain

    snapped = _snap_positions(y, samples)
    assert np.all(np.abs(snapped - samples) < SEG_LEN // 2)
    assert np.all(np.isin(snapped, bounds))
    np.testing.assert_array_equal(snapped, bounds)

    # Snapping is strictly monotone: no cut is lost or reordered.
    assert np.all(np.diff(snapped) > 0)


def test_beat_driven_remix_preserves_channel_time(beat_chain):
    # Full editing chain: beat positions -> snapped cut points -> reordered,
    # repeated intervals.  Both channels must follow the same time mapping.
    y, _sr, _bounds, _bpm, _hop_length, _frames, samples, _tempo, _tempo_samples = beat_chain

    snapped = _snap_positions(y, samples)
    selection = list(itertools.pairwise(snapped))

    # The untouched beat grid reconstructs the source exactly.
    grid = librosa.effects.remix(y, selection, align_zeros=True)
    np.testing.assert_array_equal(grid, y)
    assert grid.shape[-1] == y.shape[-1]

    # Reordering with a repeated reference (segment 5 appears twice).
    order = [5, 2, 5, 0, 7, 3, 1]
    requests = [selection[k] for k in order]
    y_out = librosa.effects.remix(y, requests, align_zeros=True)

    idx = np.concatenate([np.arange(int(start), int(stop)) for start, stop in requests])
    assert y_out.shape[-1] == idx.size
    np.testing.assert_array_equal(y_out[0], y[0][idx])
    np.testing.assert_array_equal(y_out[1], y[1][idx])
    assert not np.array_equal(y_out[0], y_out[1])


def test_beat_driven_chain_with_estimated_tempo(dance_track):
    # Same chain without pinning the tempo: the tracker must recover the beat
    # rate and the resulting cuts must still fall on real segment boundaries.
    y, sr, bounds = dance_track
    hop_length = 512
    mono = librosa.to_mono(y)

    tempo, frames = librosa.beat.beat_track(y=mono, sr=sr, hop_length=hop_length, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    expected_bpm = 60.0 * sr / SEG_LEN
    assert np.isclose(tempo, expected_bpm, rtol=5e-2)

    samples = librosa.frames_to_samples(frames, hop_length=hop_length)
    snapped = _snap_positions(y, samples)
    np.testing.assert_array_equal(snapped, bounds)

    selection = list(itertools.pairwise(snapped))
    y_out = librosa.effects.remix(y, selection, align_zeros=True)
    np.testing.assert_array_equal(y_out, y)
