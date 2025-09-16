#!/usr/bin/env python3
"""
Compare flight time vs impulse-momentum methods
"""

import pandas as pd
import numpy as np

def analyze_single_file_comparison(file_path):
    """Compare both methods on single file"""
    print(f"\n=== COMPARING METHODS ON {file_path.split('/')[-1]} ===")

    # Load data
    df = pd.read_csv(file_path)
    force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
    timestamps = df['Timestamp_s']
    time_series = timestamps - timestamps.iloc[0]

    # Calculate sampling stats
    time_diffs = timestamps.diff().dropna()
    avg_sampling_period = time_diffs.mean()
    actual_freq = 1.0 / avg_sampling_period

    print(f"Data: {len(df)} samples, {time_series.iloc[-1]:.1f}s duration, {actual_freq:.0f}Hz avg")

    # METHOD 1: FLIGHT TIME
    print(f"\n--- FLIGHT TIME METHOD ---")

    # Body weight
    window_duration = 1.0
    min_std = float('inf')
    best_bw = None

    current_time = 0
    while current_time + window_duration < min(5.0, time_series.iloc[-1]):
        start_idx = None
        end_idx = None

        for i, t in enumerate(time_series):
            if start_idx is None and t >= current_time:
                start_idx = i
            if end_idx is None and t >= current_time + window_duration:
                end_idx = i
                break

        if start_idx and end_idx and end_idx > start_idx:
            window = force_series[start_idx:end_idx]
            if 400 < window.mean() < 1200 and window.std() < min_std:
                min_std = window.std()
                best_bw = window.mean()

        current_time += 0.25

    if best_bw is None:
        best_bw = 700

    print(f"Body weight: {best_bw:.0f}N")

    # Flight detection
    smoothed = force_series.rolling(window=10, center=True).mean().fillna(force_series)
    min_idx = smoothed.idxmin()
    threshold = max(smoothed.iloc[min_idx] * 3, 100)

    flight_start = min_idx
    flight_end = min_idx
    max_expand = int(1.5 * actual_freq)

    while (flight_start > 0 and
           smoothed.iloc[flight_start - 1] < threshold and
           flight_end - flight_start < max_expand):
        flight_start -= 1

    while (flight_end < len(smoothed) - 1 and
           smoothed.iloc[flight_end + 1] < threshold and
           flight_end - flight_start < max_expand):
        flight_end += 1

    # Flight time from timestamps
    flight_time = timestamps.iloc[flight_end] - timestamps.iloc[flight_start]
    flight_height = 0.125 * 9.81 * (flight_time ** 2)

    print(f"Flight time: {flight_time:.3f}s")
    print(f"Jump height: {flight_height:.3f}m ({flight_height*100:.1f}cm)")

    # METHOD 2: IMPULSE-MOMENTUM (simplified)
    print(f"\n--- IMPULSE-MOMENTUM METHOD ---")

    # Find movement onset (simple version)
    threshold = best_bw * 0.95
    movement_start = None
    for i, force in enumerate(force_series):
        if force < threshold:
            movement_start = i
            break

    if movement_start is None:
        print("No movement detected")
        return

    print(f"Movement onset: {time_series.iloc[movement_start]:.2f}s")

    # Find minimum after movement
    search_end_time = time_series.iloc[movement_start] + 2.0
    search_end_idx = len(force_series) - 1
    for i, t in enumerate(time_series):
        if t >= search_end_time:
            search_end_idx = i
            break

    min_segment = force_series[movement_start:search_end_idx]
    min_local_idx = min_segment.idxmin()

    # Find peak after minimum
    peak_search_time = time_series.iloc[min_local_idx] + 1.0
    peak_search_idx = len(force_series) - 1
    for i, t in enumerate(time_series):
        if t >= peak_search_time:
            peak_search_idx = i
            break

    peak_segment = force_series[min_local_idx:peak_search_idx]
    peak_local_idx = peak_segment.idxmax()

    # Calculate impulse
    prop_force = force_series[min_local_idx:peak_local_idx + 1]
    prop_times = time_series[min_local_idx:peak_local_idx + 1]
    net_forces = prop_force - best_bw

    impulse = 0.0
    for i in range(len(net_forces) - 1):
        dt = prop_times.iloc[i + 1] - prop_times.iloc[i]
        impulse += 0.5 * (net_forces.iloc[i] + net_forces.iloc[i + 1]) * dt

    body_mass = best_bw / 9.81
    takeoff_vel = impulse / body_mass
    impulse_height = (takeoff_vel ** 2) / (2 * 9.81) if takeoff_vel > 0 else 0

    prop_duration = time_series.iloc[peak_local_idx] - time_series.iloc[min_local_idx]

    print(f"Propulsion duration: {prop_duration:.3f}s")
    print(f"Impulse: {impulse:.1f}N⋅s")
    print(f"Takeoff velocity: {takeoff_vel:.2f}m/s")
    print(f"Jump height: {impulse_height:.3f}m ({impulse_height*100:.1f}cm)")

    print(f"\n--- COMPARISON ---")
    print(f"Flight time method: {flight_height*100:.1f}cm")
    print(f"Impulse method:     {impulse_height*100:.1f}cm")
    diff = abs(flight_height - impulse_height) * 100
    print(f"Difference:         {diff:.1f}cm")

# Test on one file
test_file = "/Users/filip/Desktop/Making stuff/F Labs/Force Data/Data 5.09/CMJ/janusz_cmj3.csv"
analyze_single_file_comparison(test_file)