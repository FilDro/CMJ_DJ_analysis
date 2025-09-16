#!/usr/bin/env python3
"""
Corrected CMJ analysis using actual timestamps instead of assumed sampling frequency
"""

import pandas as pd
import numpy as np
import glob
import os

def analyze_cmj_corrected(file_path):
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_cmj')[0]
    trial_number = filename.split('_cmj')[1].split('.')[0]

    try:
        # Load data
        df = pd.read_csv(file_path)
        force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
        timestamps = df['Timestamp_s']

        # Calculate actual sampling characteristics
        time_diffs = timestamps.diff().dropna()
        avg_sampling_period = time_diffs.mean()
        actual_freq = 1.0 / avg_sampling_period
        total_duration = timestamps.iloc[-1] - timestamps.iloc[0]

        print(f"  Data: {len(df)} samples, {total_duration:.1f}s duration, ~{actual_freq:.0f}Hz avg")

        # Body weight estimation
        window_size = int(1.0 * actual_freq)  # 1 second worth of samples
        min_std = float('inf')
        best_bw = None

        for i in range(0, len(force_series) - window_size, window_size // 4):
            window = force_series[i:i + window_size]
            mean_force = window.mean()
            std_force = window.std()
            if 400 < mean_force < 1200 and std_force < min_std:
                min_std = std_force
                best_bw = mean_force

        if best_bw is None:
            best_bw = 700

        # Flight detection using the same method but corrected timing
        smoothed_force = force_series.rolling(window=max(1, int(0.02 * actual_freq)), center=True).mean().fillna(force_series)
        min_force_idx = smoothed_force.idxmin()
        min_force_value = smoothed_force.iloc[min_force_idx]
        threshold = max(min_force_value * 3, 100)

        # Expand around minimum
        flight_start = min_force_idx
        flight_end = min_force_idx
        max_expansion = int(1.5 * actual_freq)  # Max 1.5 seconds

        # Expand backwards
        while (flight_start > 0 and
               smoothed_force.iloc[flight_start - 1] < threshold and
               flight_end - flight_start < max_expansion):
            flight_start -= 1

        # Expand forwards
        while (flight_end < len(smoothed_force) - 1 and
               smoothed_force.iloc[flight_end + 1] < threshold and
               flight_end - flight_start < max_expansion):
            flight_end += 1

        # Calculate flight time using ACTUAL TIMESTAMPS
        flight_time = timestamps.iloc[flight_end] - timestamps.iloc[flight_start]

        # Check minimum flight time
        if flight_time < 0.1:  # Less than 100ms is too short
            return {
                'Athlete': athlete_name,
                'Trial': trial_number,
                'Filename': filename,
                'Status': f'Failed: Flight too short ({flight_time:.3f}s)',
                'Jump_Height_cm': 0,
                'Flight_Time_s': 0
            }

        # Calculate jump height
        jump_height_m = 0.125 * 9.81 * (flight_time ** 2)
        jump_height_cm = jump_height_m * 100
        takeoff_velocity = 0.5 * 9.81 * flight_time

        # Get flight characteristics
        flight_segment = smoothed_force[flight_start:flight_end + 1]
        min_flight_force = flight_segment.min()
        mean_flight_force = flight_segment.mean()

        return {
            'Athlete': athlete_name,
            'Trial': trial_number,
            'Filename': filename,
            'Status': 'Success',
            'Jump_Height_cm': jump_height_cm,
            'Jump_Height_m': jump_height_m,
            'Flight_Time_s': flight_time,
            'Takeoff_Velocity_m_per_s': takeoff_velocity,
            'Body_Weight_N': best_bw,
            'Flight_Start_Time_s': timestamps.iloc[flight_start] - timestamps.iloc[0],  # Relative to start
            'Flight_End_Time_s': timestamps.iloc[flight_end] - timestamps.iloc[0],
            'Min_Flight_Force_N': min_flight_force,
            'Mean_Flight_Force_N': mean_flight_force,
            'Actual_Sampling_Hz': actual_freq,
            'Total_Duration_s': total_duration
        }

    except Exception as e:
        return {
            'Athlete': athlete_name,
            'Trial': trial_number,
            'Filename': filename,
            'Status': f'Failed: {str(e)}',
            'Jump_Height_cm': 0,
            'Flight_Time_s': 0
        }

def main():
    print("CORRECTED CMJ Analysis - Using Actual Timestamps")
    print("=" * 60)

    cmj_files = glob.glob("/Users/filip/Desktop/Making stuff/F Labs/Force Data/Data 5.09/CMJ/*.csv")
    results = []

    for file_path in sorted(cmj_files):
        print(f"\nProcessing: {os.path.basename(file_path)}")
        result = analyze_cmj_corrected(file_path)
        results.append(result)

        if result['Status'] == 'Success':
            print(f"  ✅ Jump Height: {result['Jump_Height_cm']:.1f}cm ({result['Jump_Height_m']:.3f}m)")
            print(f"     Flight Time: {result['Flight_Time_s']:.3f}s")
            print(f"     Flight Forces: {result['Min_Flight_Force_N']:.1f}N min, {result['Mean_Flight_Force_N']:.1f}N avg")
        else:
            print(f"  ❌ {result['Status']}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_excel('CORRECTED_CMJ_Final_Results.xlsx', index=False)

    # Summary
    successful = df_results[df_results['Status'] == 'Success']
    print(f"\n" + "="*60)
    print(f"FINAL SUMMARY")
    print(f"="*60)
    print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

    if len(successful) > 0:
        print(f"\n📊 JUMP HEIGHTS (corrected with actual timestamps):")
        all_heights = []
        for athlete in successful['Athlete'].unique():
            athlete_data = successful[successful['Athlete'] == athlete]
            heights_cm = athlete_data['Jump_Height_cm']
            avg_height = heights_cm.mean()
            max_height = heights_cm.max()
            min_height = heights_cm.min()
            all_heights.extend(heights_cm.tolist())
            print(f"  {athlete}: avg {avg_height:.1f}cm, range {min_height:.1f}-{max_height:.1f}cm ({len(athlete_data)} trials)")

        overall_avg = np.mean(all_heights)
        print(f"\n🎯 OVERALL AVERAGE: {overall_avg:.1f}cm")
        print(f"   Expected ~35cm: {'✅ CLOSE!' if 25 <= overall_avg <= 45 else '❌ Still off'}")

    print(f"\n💾 Results saved to: CORRECTED_CMJ_Final_Results.xlsx")

if __name__ == "__main__":
    main()