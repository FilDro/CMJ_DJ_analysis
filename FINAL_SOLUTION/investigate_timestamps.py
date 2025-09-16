#!/usr/bin/env python3
"""
Investigate timestamps and actual sampling frequency
"""

import pandas as pd
import numpy as np

# Load one file to examine timestamps
file_path = "/Users/filip/Desktop/Making stuff/F Labs/Force Data/Data 5.09/CMJ/janusz_cmj3.csv"
df = pd.read_csv(file_path)

print("=== TIMESTAMP INVESTIGATION ===")
print(f"File: {file_path}")
print(f"Total samples: {len(df)}")

# Examine timestamps
timestamps = df['Timestamp_s']
print(f"\nTimestamp range: {timestamps.iloc[0]:.6f} to {timestamps.iloc[-1]:.6f}")
print(f"Total duration: {timestamps.iloc[-1] - timestamps.iloc[0]:.6f} seconds")

# Calculate actual sampling frequency
time_diffs = timestamps.diff().dropna()
avg_time_diff = time_diffs.mean()
actual_sampling_freq = 1.0 / avg_time_diff

print(f"\nTiming Analysis:")
print(f"Average time between samples: {avg_time_diff:.6f} seconds")
print(f"Calculated sampling frequency: {actual_sampling_freq:.1f} Hz")
print(f"Assumed sampling frequency: 500 Hz")
print(f"FREQUENCY ERROR: {abs(actual_sampling_freq - 500):.1f} Hz")

# Show some sample timestamps to see the pattern
print(f"\nFirst 10 timestamps:")
for i in range(10):
    if i > 0:
        diff = timestamps.iloc[i] - timestamps.iloc[i-1]
        print(f"  {i}: {timestamps.iloc[i]:.6f} (diff: {diff:.6f}s)")
    else:
        print(f"  {i}: {timestamps.iloc[i]:.6f}")

print(f"\nLast 10 timestamps:")
for i in range(len(timestamps)-10, len(timestamps)):
    if i > len(timestamps)-11:
        diff = timestamps.iloc[i] - timestamps.iloc[i-1]
        print(f"  {i}: {timestamps.iloc[i]:.6f} (diff: {diff:.6f}s)")
    else:
        print(f"  {i}: {timestamps.iloc[i]:.6f}")

# Check for timing irregularities
print(f"\nTiming Irregularities:")
print(f"Min time diff: {time_diffs.min():.6f}s ({1/time_diffs.min():.1f} Hz)")
print(f"Max time diff: {time_diffs.max():.6f}s ({1/time_diffs.max():.1f} Hz)")
print(f"Std dev of time diffs: {time_diffs.std():.6f}s")

# Count how many samples are at different frequencies
freq_bins = [1/td if td > 0 else 0 for td in time_diffs]
unique_freqs = np.unique(np.round(freq_bins, 1))
print(f"\nFrequency distribution:")
for freq in unique_freqs[:10]:  # Show top 10
    count = sum(1 for f in freq_bins if abs(f - freq) < 0.1)
    percent = (count / len(freq_bins)) * 100
    print(f"  ~{freq:.1f} Hz: {count} samples ({percent:.1f}%)")

# Recalculate jump height with correct frequency
print(f"\n=== RECALCULATING WITH CORRECT FREQUENCY ===")

# Load force data
force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']

# Find minimum force period (same logic as before)
smoothed_force = force_series.rolling(window=10, center=True).mean().fillna(force_series)
min_force_idx = smoothed_force.idxmin()
min_force_value = smoothed_force.iloc[min_force_idx]
threshold = max(min_force_value * 3, 100)

# Expand around minimum
flight_start = min_force_idx
flight_end = min_force_idx
max_samples = int(1.5 * actual_sampling_freq)  # Use actual frequency

while (flight_start > 0 and
       smoothed_force.iloc[flight_start - 1] < threshold and
       flight_end - flight_start < max_samples):
    flight_start -= 1

while (flight_end < len(smoothed_force) - 1 and
       smoothed_force.iloc[flight_end + 1] < threshold and
       flight_end - flight_start < max_samples):
    flight_end += 1

# Calculate flight time using ACTUAL timestamps
flight_time_correct = timestamps.iloc[flight_end] - timestamps.iloc[flight_start]
jump_height_correct = 0.125 * 9.81 * (flight_time_correct ** 2)

print(f"Original calculation (500 Hz assumed):")
flight_time_wrong = (flight_end - flight_start + 1) / 500
jump_height_wrong = 0.125 * 9.81 * (flight_time_wrong ** 2)
print(f"  Flight samples: {flight_end - flight_start + 1}")
print(f"  Flight time: {flight_time_wrong:.3f}s")
print(f"  Jump height: {jump_height_wrong:.3f}m ({jump_height_wrong*100:.1f}cm)")

print(f"\nCorrected calculation (using timestamps):")
print(f"  Flight samples: {flight_end - flight_start + 1}")
print(f"  Flight time: {flight_time_correct:.3f}s")
print(f"  Jump height: {jump_height_correct:.3f}m ({jump_height_correct*100:.1f}cm)")

print(f"\nCORRECTION FACTOR: {flight_time_wrong / flight_time_correct:.2f}x")