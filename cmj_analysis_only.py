#!/usr/bin/env python3
"""
CMJ Analysis Only - Simplified version for analyzing CMJ data from Data 5.09
"""

import pandas as pd
import numpy as np
import glob
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_body_mass_from_excel(athlete_name):
    """Get body mass from dane.xlsx file"""
    try:
        df = pd.read_excel('/Users/filip/Desktop/Data 10.06/dane.xlsx')

        # Clean the data - remove rows with NaN in name column
        df = df.dropna(subset=['Unnamed: 1'])

        # Rename columns for clarity
        df.columns = ['Index', 'Name', 'Mass']

        # Remove the header row WAGA
        df = df[df['Name'] != 'WAGA']

        # Look for the athlete (case insensitive)
        athlete_row = df[df['Name'].str.lower() == athlete_name.lower()]

        if not athlete_row.empty:
            mass = float(athlete_row['Mass'].iloc[0])
            logging.info(f"Found body mass for {athlete_name}: {mass} kg")
            return mass
        else:
            logging.warning(f"Body mass not found for {athlete_name}, using default 70kg")
            return 70.0

    except Exception as e:
        logging.warning(f"Could not read body mass data: {e}. Using default 70kg")
        return 70.0

def estimate_body_weight(force_series, sampling_freq=500):
    """Robust body weight estimation that handles setup periods"""
    # Method 1: Try standard approach (first 2 seconds)
    bw_samples = int(2 * sampling_freq)
    bw_period = force_series[:min(bw_samples, len(force_series))]

    # Remove outliers (values more than 2 std from mean)
    mean_bw = np.mean(bw_period)
    std_bw = np.std(bw_period)
    clean_bw = bw_period[np.abs(bw_period - mean_bw) < 2 * std_bw]

    standard_bw = np.mean(clean_bw) if len(clean_bw) > 0 else mean_bw

    # Method 2: Find sustained high force period (athlete on plate)
    # Look for period where force is consistently above 400N for at least 2 seconds
    min_force_threshold = 400  # N
    min_duration_samples = int(2 * sampling_freq)  # 2 seconds

    sustained_bw = None
    for i in range(len(force_series) - min_duration_samples):
        window = force_series[i:i+min_duration_samples]
        if all(window > min_force_threshold):
            # Found sustained period - use middle portion for body weight
            start_idx = i + int(0.5 * sampling_freq)  # Skip first 0.5s
            end_idx = i + int(1.5 * sampling_freq)    # Use next 1s
            if end_idx < len(force_series):
                sustained_bw = force_series[start_idx:end_idx].mean()
                break

    # Choose the better estimate
    if sustained_bw is not None and sustained_bw > 400:
        logging.info(f"Using sustained period body weight: {sustained_bw:.0f}N")
        return sustained_bw
    elif standard_bw > 100:
        logging.info(f"Using standard period body weight: {standard_bw:.0f}N")
        return standard_bw
    else:
        # Last resort: use median of all values above 400N
        valid_forces = force_series[force_series > 400]
        if len(valid_forces) > 0:
            median_bw = np.median(valid_forces)
            logging.info(f"Using median body weight: {median_bw:.0f}N")
            return median_bw
        else:
            logging.warning("Could not estimate body weight, using 600N default")
            return 600

def validate_flight_time(flight_time):
    """Validate flight time is realistic for human jump"""
    # Realistic flight time: 0.2-0.8 seconds
    return 0.2 <= flight_time <= 0.8

def calculate_jump_height_from_force(force_series, body_weight, sampling_freq=500):
    """Calculate jump height using improved flight detection"""
    try:
        # Use absolute threshold for true flight detection (not relative to body weight)
        takeoff_threshold = 50  # Absolute threshold for actual flight
        min_flight_samples = int(0.2 * sampling_freq)  # Minimum 200ms flight

        # Find takeoff point (where force drops below threshold)
        below_threshold = force_series < takeoff_threshold

        if not any(below_threshold):
            return None, None, None

        # Find potential flight phases
        flight_start = None
        flight_end = None

        for i in range(len(below_threshold) - min_flight_samples):
            if below_threshold[i] and not below_threshold[i-1] if i > 0 else True:
                # Potential flight start
                flight_duration = 0
                for j in range(i, len(below_threshold)):
                    if below_threshold[j]:
                        flight_duration += 1
                    else:
                        break

                # Check if this flight phase is long enough
                if flight_duration >= min_flight_samples:
                    flight_start = i
                    flight_end = i + flight_duration
                    break

        if flight_start is None or flight_end is None:
            return None, None, None

        # Calculate flight time
        flight_time = (flight_end - flight_start) / sampling_freq

        # Validate flight time
        if not validate_flight_time(flight_time):
            return None, None, None

        # Calculate jump height
        jump_height = 0.125 * 9.81 * (flight_time ** 2)

        logging.info(f"Flight detected: {flight_time:.3f}s, Height: {jump_height:.3f}m")
        return jump_height, flight_time, flight_start

    except Exception as e:
        logging.error(f"Error in jump height calculation: {e}")
        return None, None, None

def calculate_jump_height_alternative(force_series, body_weight, sampling_freq=500):
    """Alternative method using impulse-momentum"""
    try:
        # Find the movement start (significant deviation from body weight)
        baseline_threshold = 0.05 * body_weight  # 5% deviation threshold
        movement_start = None

        for i in range(len(force_series)):
            if abs(force_series[i] - body_weight) > baseline_threshold:
                movement_start = i
                break

        if movement_start is None:
            return None, None, None

        # Find takeoff (where force goes to ~0)
        takeoff_threshold = 50  # N
        takeoff_idx = None

        for i in range(movement_start, len(force_series)):
            if force_series[i] < takeoff_threshold:
                takeoff_idx = i
                break

        if takeoff_idx is None:
            return None, None, None

        # Calculate impulse during propulsion phase
        dt = 1.0 / sampling_freq
        propulsion_forces = force_series[movement_start:takeoff_idx]
        net_forces = propulsion_forces - body_weight
        impulse = np.sum(net_forces) * dt

        # Calculate takeoff velocity (impulse = change in momentum)
        mass = body_weight / 9.81
        takeoff_velocity = impulse / mass

        if takeoff_velocity <= 0:
            return None, None, None

        # Calculate jump height
        jump_height = (takeoff_velocity ** 2) / (2 * 9.81)

        logging.info(f"Alternative method: v0={takeoff_velocity:.2f}m/s, h={jump_height:.3f}m")
        return jump_height, takeoff_velocity, impulse

    except Exception as e:
        logging.error(f"Error in alternative calculation: {e}")
        return None, None, None

def analyze_cmj_final(file_path, sampling_freq=500):
    """Final robust CMJ analysis"""
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_cmj')[0]
    trial_number = filename.split('_cmj')[1].split('.')[0]

    try:
        df = pd.read_csv(file_path)
        force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
        body_mass = get_body_mass_from_excel(athlete_name)
        body_weight = estimate_body_weight(force_series, sampling_freq)

        # Try primary method first
        jump_height, flight_time, takeoff_idx = calculate_jump_height_from_force(
            force_series, body_weight, sampling_freq
        )

        # If primary method fails, try alternative
        if jump_height is None:
            jump_height, takeoff_velocity, impulse = calculate_jump_height_alternative(
                force_series, body_weight, sampling_freq
            )
            flight_time = 2 * takeoff_velocity / 9.81 if takeoff_velocity > 0 else 0
        else:
            takeoff_velocity = 0.5 * 9.81 * flight_time
            impulse = 0

        # Calculate other metrics
        peak_force = np.max(force_series)
        relative_peak_force = peak_force / body_mass

        return {
            'Athlete': athlete_name,
            'Trial': trial_number,
            'Filename': filename,
            'Status': 'Success' if jump_height is not None else 'Failed',
            'Jump_Height_m': jump_height if jump_height is not None else 0,
            'Flight_Time_s': flight_time if flight_time is not None else 0,
            'Peak_Force_N': peak_force,
            'Relative_Peak_Force_N_per_kg': relative_peak_force,
            'Body_Weight_N': body_weight,
            'Body_Mass_kg': body_mass,
            'Takeoff_Velocity_m_per_s': takeoff_velocity if 'takeoff_velocity' in locals() else 0,
            'Method': 'Flight_Time' if takeoff_idx is not None else 'Impulse_Momentum'
        }

    except Exception as e:
        logging.error(f"Error analyzing {filename}: {e}")
        return {
            'Athlete': athlete_name,
            'Trial': trial_number,
            'Filename': filename,
            'Status': f'Error: {e}',
            'Jump_Height_m': 0,
            'Flight_Time_s': 0,
            'Peak_Force_N': 0,
            'Relative_Peak_Force_N_per_kg': 0,
            'Body_Weight_N': 0,
            'Body_Mass_kg': 70,
            'Takeoff_Velocity_m_per_s': 0,
            'Method': 'None'
        }

def main():
    """Main analysis function"""
    logging.info("Starting CMJ analysis...")

    # Analyze CMJ data
    print("\n=== CMJ ANALYSIS ===")
    cmj_files = glob.glob('/Users/filip/Desktop/Making stuff/F Labs/Force Data/Data 5.09/CMJ/*.csv')
    cmj_results = []

    for file_path in sorted(cmj_files):
        result = analyze_cmj_final(file_path)
        cmj_results.append(result)
        if result['Status'] == 'Success':
            print(f"{result['Athlete']} (Trial {result['Trial']}): {result['Jump_Height_m']:.3f} m")
        else:
            print(f"{result['Athlete']} (Trial {result['Trial']}): {result['Status']}")

    # Create DataFrame and save results
    cmj_df = pd.DataFrame(cmj_results)

    # Save to Excel
    output_file = 'CMJ_Analysis_Data_5_09.xlsx'
    cmj_df.to_excel(output_file, sheet_name='CMJ_Analysis', index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    successful_results = cmj_df[cmj_df['Status'] == 'Success']

    if len(successful_results) > 0:
        print(f"Successful analyses: {len(successful_results)}/{len(cmj_results)}")

        for athlete in successful_results['Athlete'].unique():
            athlete_data = successful_results[successful_results['Athlete'] == athlete]
            avg_height = athlete_data['Jump_Height_m'].mean()
            max_height = athlete_data['Jump_Height_m'].max()
            trials = len(athlete_data)
            print(f"{athlete}: {trials} trials, Avg: {avg_height:.3f}m, Max: {max_height:.3f}m")
    else:
        print("No successful analyses")

    return cmj_df

if __name__ == "__main__":
    results_df = main()