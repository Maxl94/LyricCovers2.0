import os
from typing import List, Tuple

import numpy as np
import torch
import torchaudio


def get_audio_info(audio_path: str, sample_rate: int = 44100) -> Tuple[torch.Tensor, int, float]:
    """
    Get audio information including exact duration.

    Returns:
        Tuple of (waveform, sample_rate, duration_in_seconds)
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    duration_samples = waveform.shape[1]
    duration_seconds = duration_samples / sample_rate

    print(f"Audio length: {duration_seconds:.2f} seconds")
    print(f"Total samples: {duration_samples}")
    return waveform, sample_rate, duration_seconds


def smart_random_audio_splits(
    audio_path: str,
    segment_length: int = 30,
    num_segments_multiplier: float = 1.0,
    max_overlap_ratio: float = 0.75,
    sample_rate: int = 44100,
    seed: int = None,
    max_attempts: int = 1000,
    ensure_end_coverage: bool = True,
) -> List[Tuple[int, int]]:
    """
    Generate random audio splits with controlled overlap and guaranteed end coverage.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load audio file and get exact duration
    waveform, sr, duration_seconds = get_audio_info(audio_path, sample_rate)
    total_samples = int(duration_seconds * sample_rate)

    # Convert segment length to samples
    segment_samples = segment_length * sample_rate

    # Calculate number of possible full segments and desired number of segments
    possible_segments = int(np.ceil(duration_seconds / segment_length))
    desired_segments = int(possible_segments * num_segments_multiplier)

    print(f"Possible segments: {possible_segments}")
    print(f"Desired segments: {desired_segments}")

    # Calculate minimum step size based on overlap constraint
    min_step = int(segment_samples * (1 - max_overlap_ratio))

    # Initialize list to store selected segments
    selected_segments = []

    # Always include the last segment first if ensure_end_coverage is True
    if ensure_end_coverage:
        last_segment_start = max(0, total_samples - segment_samples)
        last_segment = (last_segment_start, total_samples)
        selected_segments.append(last_segment)
        print(f"Added last segment: {last_segment_start/sample_rate:.2f}s - {total_samples/sample_rate:.2f}s")
        desired_segments -= 1

    # Start with evenly spaced segments as a base
    if desired_segments > 0:
        available_space = total_samples - segment_samples
        base_step = max(
            min_step,
            available_space // (desired_segments - 1) if desired_segments > 1 else available_space,
        )

        # Place initial segments with even spacing
        for i in range(min(desired_segments, int(available_space / min_step) + 1)):
            start_sample = i * base_step
            if start_sample + segment_samples <= total_samples:
                # Check overlap with existing segments
                proposed_segment = (start_sample, start_sample + segment_samples)
                if not any(
                    _check_overlap(proposed_segment, existing_segment, max_overlap_ratio)
                    for existing_segment in selected_segments
                ):
                    selected_segments.append(proposed_segment)

    # Try to add remaining segments with random placement
    remaining_segments = desired_segments - (len(selected_segments) - (1 if ensure_end_coverage else 0))
    if remaining_segments > 0:
        # Generate all possible start points
        step = min_step // 2  # Smaller step for fine-grained placement
        possible_starts = list(range(0, total_samples - segment_samples + 1, step))

        # Try to place remaining segments
        attempts = 0
        while remaining_segments > 0 and attempts < max_attempts and possible_starts:
            start_sample = np.random.choice(possible_starts)
            end_sample = min(start_sample + segment_samples, total_samples)

            # Check overlap with existing segments
            proposed_segment = (start_sample, end_sample)
            valid_segment = not any(
                _check_overlap(proposed_segment, existing_segment, max_overlap_ratio)
                for existing_segment in selected_segments
            )

            if valid_segment:
                selected_segments.append(proposed_segment)
                remaining_segments -= 1
                # Remove nearby start points to maintain minimum spacing
                possible_starts = [s for s in possible_starts if abs(s - start_sample) >= min_step // 2]

            attempts += 1

    # Sort segments by start time
    selected_segments.sort(key=lambda x: x[0])

    # Validate coverage
    covered_ranges = _merge_overlapping_ranges(selected_segments)
    total_coverage = sum(end - start for start, end in covered_ranges)
    coverage_ratio = total_coverage / total_samples

    print(f"Coverage ratio: {coverage_ratio:.2%}")
    print(f"Number of segments: {len(selected_segments)}")
    print(f"First segment: {selected_segments[0][0]/sample_rate:.2f}s - {selected_segments[0][1]/sample_rate:.2f}s")
    print(f"Last segment: {selected_segments[-1][0]/sample_rate:.2f}s - {selected_segments[-1][1]/sample_rate:.2f}s")

    return selected_segments, sr


def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int], max_overlap_ratio: float) -> bool:
    """Check if two segments overlap more than the maximum allowed ratio."""
    start1, end1 = segment1
    start2, end2 = segment2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        segment_length = min(end1 - start1, end2 - start2)
        overlap_ratio = overlap_length / segment_length
        return overlap_ratio > max_overlap_ratio

    return False


def _merge_overlapping_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping ranges to calculate total coverage."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)

    return merged


def random_audio_splits(
    audio_path: str,
    segment_length: int = 30,
    num_segments_multiplier: float = 2,
    max_overlap_ratio: float = 0.5,
    sample_rate: int = 44100,
    seed: int = None,
) -> List[Tuple[int, int]]:
    """
    Generate random audio splits with controlled overlap.

    Args:
        audio_path: Path to the audio file
        segment_length: Length of each segment in seconds (default: 30)
        num_segments_multiplier: Multiplier for number of segments (default: 1.0)
        max_overlap_ratio: Maximum allowed overlap between segments (default: 0.75)
        sample_rate: Audio sample rate (default: 44100)
        seed: Random seed for reproducibility

    Returns:
        List of tuples containing start and end sample indices
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load audio file
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    # Convert segment length to samples
    segment_samples = segment_length * sample_rate
    total_samples = waveform.shape[1]

    # Calculate number of possible full segments and desired number of segments
    possible_segments = total_samples // segment_samples
    desired_segments = int(possible_segments * num_segments_multiplier)

    # Initialize list to store selected segments
    selected_segments = []
    max_attempts = desired_segments * 10  # Limit attempts to avoid infinite loops
    attempts = 0

    while len(selected_segments) <= desired_segments and attempts < max_attempts:
        # Generate random start point
        start_sample = np.random.randint(0, total_samples - segment_samples)
        end_sample = start_sample + segment_samples

        # Check if this segment overlaps too much with existing segments
        valid_segment = True
        for existing_start, existing_end in selected_segments:
            overlap_start = max(start_sample, existing_start)
            overlap_end = min(end_sample, existing_end)

            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                segment_length = segment_samples
                overlap_ratio = overlap_length / segment_length

                if overlap_ratio > max_overlap_ratio:
                    valid_segment = False
                    break

        if valid_segment:
            selected_segments.append((start_sample, end_sample))

        attempts += 1

    # Sort segments by start time for better organization
    selected_segments.sort(key=lambda x: x[0])

    return selected_segments, sr


def extract_segments(
    audio_path: str,
    segments: List[Tuple[int, int]],
    output_prefix: str = "segment_",
    sample_rate: int = 44100,
    output_dir: str = "output",
) -> List[str]:
    """
    Extract and save audio segments based on the provided segment indices.

    Args:
        audio_path: Path to the audio file
        segments: List of (start, end) sample indices
        output_prefix: Prefix for output files
        sample_rate: Audio sample rate

    Returns:
        List of paths to the saved segment files
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for i, (start, end) in enumerate(segments):
        segment = waveform[:, start:end]
        output_path = os.path.join(output_dir, f"{output_prefix}{i}.wav")
        torchaudio.save(output_path, segment, sample_rate)
        output_paths.append(output_path)

    return output_paths


# Example usage
if __name__ == "__main__":
    # Example parameters
    audio_path = "input.wav"
    segment_length = 30  # seconds
    num_segments_multiplier = 1.5  # 50% more segments than possible non-overlapping segments
    max_overlap_ratio = 0.75  # maximum 75% overlap allowed

    # Generate random segments
    segments = random_audio_splits(
        audio_path=audio_path,
        segment_length=segment_length,
        num_segments_multiplier=num_segments_multiplier,
        max_overlap_ratio=max_overlap_ratio,
    )

    # Extract and save segments
    output_files = extract_segments(audio_path, segments)
