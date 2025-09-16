"""
Final robust analysis with proper jump detection and validation
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
        max_flight_samples = int(0.6 * sampling_freq)  # Maximum 600ms flight
        
        # Look for best jump in the data (multiple attempts may be present)
        best_jump = None
        best_height = 0
        
        # Start search after initial body weight period
        search_start = int(2 * sampling_freq)  # Skip first 2 seconds
        
        for i in range(search_start, len(force_series) - min_flight_samples):
            if force_series[i] < takeoff_threshold:
                # Check sustained flight duration
                sustained_count = 0
                for j in range(i, min(i + max_flight_samples, len(force_series))):
                    if force_series[j] < takeoff_threshold:
                        sustained_count += 1
                    else:
                        break
                
                # Validate flight duration
                if min_flight_samples <= sustained_count <= max_flight_samples:
                    flight_time = sustained_count / sampling_freq
                    jump_height = 0.125 * 9.81 * flight_time**2
                    
                    # Validate with force patterns before and after flight
                    pre_samples = 100  # 200ms before
                    post_samples = 100  # 200ms after
                    
                    pre_force = force_series[max(0, i-pre_samples):i].mean() if i >= pre_samples else 0
                    post_end = i + sustained_count + post_samples
                    post_force = force_series[i+sustained_count:min(post_end, len(force_series))].mean()
                    
                    # Ensure proper push-off and landing patterns
                    if pre_force > body_weight * 0.8 and post_force > body_weight * 0.8:
                        if jump_height > best_height:
                            best_height = jump_height
                            best_jump = {
                                'takeoff_idx': i,
                                'landing_idx': i + sustained_count,
                                'flight_time': flight_time,
                                'jump_height': jump_height
                            }
        
        if best_jump:
            return best_jump['jump_height'], best_jump['flight_time'], best_jump['takeoff_idx']
        else:
            return None, None, None
    
    except Exception as e:
        logging.error(f"Error in jump height calculation: {e}")
        return None, None, None

def calculate_jump_height_alternative(force_series, body_weight, sampling_freq=500):
    """Alternative method using impulse-momentum theorem"""
    try:
        # Find movement start (unweighting phase)
        unweighting_threshold = 0.95 * body_weight
        
        movement_start = None
        for i in range(len(force_series)):
            if force_series[i] < unweighting_threshold:
                movement_start = i
                break
        
        if movement_start is None:
            return 0, 0, 0
        
        # Find minimum force point
        min_force_idx = np.argmin(force_series[movement_start:]) + movement_start
        
        # Find peak force after minimum
        peak_force_idx = np.argmax(force_series[min_force_idx:]) + min_force_idx
        
        # Calculate net impulse from minimum to peak
        net_force = force_series[min_force_idx:peak_force_idx] - body_weight
        time_step = 1.0 / sampling_freq
        
        # Simple numerical integration
        impulse = np.sum(net_force) * time_step
        
        # Estimate body mass (assuming g = 9.81)
        body_mass = body_weight / 9.81
        
        # Calculate takeoff velocity
        takeoff_velocity = impulse / body_mass if body_mass > 0 else 0
        
        # Calculate jump height
        jump_height = (takeoff_velocity ** 2) / (2 * 9.81) if takeoff_velocity > 0 else 0
        
        # Validate result
        if jump_height > 2.0:  # Unrealistic if > 2m
            return 0, 0, 0
        
        return jump_height, takeoff_velocity, impulse
    
    except Exception as e:
        logging.error(f"Error in alternative calculation: {e}")
        return 0, 0, 0

def analyze_cmj_final(file_path, sampling_freq=500):
    """Final robust CMJ analysis"""
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_cmj')[0]
    
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
        
        # Calculate asymmetry
        plate1_mean = df['Plate1_N_filtered'].mean()
        plate2_mean = df['Plate2_N_filtered'].mean()
        asymmetry = abs(plate1_mean - plate2_mean) / ((plate1_mean + plate2_mean) / 2) * 100
        
        return {
            'Athlete': athlete_name,
            'Jump_Height_m': jump_height,
            'Flight_Time_s': flight_time,
            'Peak_Force_N': peak_force,
            'Body_Weight_N': body_weight,
            'Relative_Peak_Force_N_per_kg': relative_peak_force,
            'Takeoff_Velocity_m_per_s': takeoff_velocity,
            'Body_Mass_kg': body_mass,
            'Asymmetry_Percent': asymmetry,
            'Status': 'Success'
        }
        
    except Exception as e:
        logging.error(f"Error analyzing {athlete_name}: {e}")
        return {
            'Athlete': athlete_name,
            'Status': f'Error: {str(e)}'
        }

def analyze_dj_final(file_path, sampling_freq=500):
    """Final robust DJ analysis"""
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_dj')[0]
    
    try:
        df = pd.read_csv(file_path)
        force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
        body_mass = get_body_mass_from_excel(athlete_name)
        body_weight = estimate_body_weight(force_series, sampling_freq)
        
        # Calculate jump height using same method as CMJ
        jump_height, flight_time, takeoff_idx = calculate_jump_height_from_force(
            force_series, body_weight, sampling_freq
        )
        
        if jump_height is None:
            jump_height, takeoff_velocity, impulse = calculate_jump_height_alternative(
                force_series, body_weight, sampling_freq
            )
            flight_time = 2 * takeoff_velocity / 9.81 if takeoff_velocity > 0 else 0
        else:
            takeoff_velocity = 0.5 * 9.81 * flight_time
        
        # Calculate ground contact time
        # For DJ, estimate from force above threshold
        contact_threshold = 1.2 * body_weight
        above_threshold = force_series > contact_threshold
        
        if above_threshold.any():
            first_contact = np.where(above_threshold)[0][0]
            if takeoff_idx is not None:
                gct = (takeoff_idx - first_contact) / sampling_freq
            else:
                # Estimate from force profile
                gct = 0.5  # Default estimate
        else:
            gct = 0.5  # Default estimate
        
        # Validate GCT (should be reasonable for DJ)
        if gct > 2.0 or gct < 0.1:
            gct = 0.5  # Use default
        
        # Calculate RSI
        rsi = jump_height / gct if gct > 0 else 0
        
        # Calculate other metrics
        peak_force = np.max(force_series)
        relative_peak_force = peak_force / body_mass
        
        # Calculate asymmetry
        plate1_mean = df['Plate1_N_filtered'].mean()
        plate2_mean = df['Plate2_N_filtered'].mean()
        asymmetry = abs(plate1_mean - plate2_mean) / ((plate1_mean + plate2_mean) / 2) * 100
        
        return {
            'Athlete': athlete_name,
            'Jump_Height_m': jump_height,
            'Flight_Time_s': flight_time,
            'Ground_Contact_Time_s': gct,
            'Reactive_Strength_Index': rsi,
            'Peak_Force_N': peak_force,
            'Body_Weight_N': body_weight,
            'Relative_Peak_Force_N_per_kg': relative_peak_force,
            'Takeoff_Velocity_m_per_s': takeoff_velocity,
            'Body_Mass_kg': body_mass,
            'Asymmetry_Percent': asymmetry,
            'Status': 'Success'
        }
        
    except Exception as e:
        logging.error(f"Error analyzing {athlete_name}: {e}")
        return {
            'Athlete': athlete_name,
            'Status': f'Error: {str(e)}'
        }

def analyze_mtp_final(file_path, sampling_freq=500):
    """Analyze MTP data"""
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_mtp')[0]
    
    try:
        df = pd.read_csv(file_path)
        force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
        body_mass = get_body_mass_from_excel(athlete_name)
        
        # Peak force
        peak_force = np.max(force_series)
        relative_peak_force = peak_force / body_mass
        
        # RFD calculations
        force_threshold = 50  # N
        start_indices = np.where(force_series > force_threshold)[0]
        start_idx = start_indices[0] if len(start_indices) > 0 else 0
        
        # RFD 0-50ms
        end_50ms = min(len(force_series) - 1, start_idx + int(0.05 * sampling_freq))
        rfd_50ms = (force_series[end_50ms] - force_series[start_idx]) / 0.05 if end_50ms > start_idx else 0
        
        # RFD 0-100ms
        end_100ms = min(len(force_series) - 1, start_idx + int(0.1 * sampling_freq))
        rfd_100ms = (force_series[end_100ms] - force_series[start_idx]) / 0.1 if end_100ms > start_idx else 0
        
        # Asymmetry
        plate1_mean = df['Plate1_N_filtered'].mean()
        plate2_mean = df['Plate2_N_filtered'].mean()
        asymmetry = abs(plate1_mean - plate2_mean) / ((plate1_mean + plate2_mean) / 2) * 100
        
        return {
            'Athlete': athlete_name,
            'Peak_Force_N': peak_force,
            'Relative_Peak_Force_N_per_kg': relative_peak_force,
            'RFD_0_50ms_N_per_s': rfd_50ms,
            'RFD_0_100ms_N_per_s': rfd_100ms,
            'Body_Mass_kg': body_mass,
            'Asymmetry_Percent': asymmetry,
            'Status': 'Success'
        }
        
    except Exception as e:
        logging.error(f"Error analyzing MTP for {athlete_name}: {e}")
        return {'Athlete': athlete_name, 'Status': f'Error: {str(e)}'}

def analyze_shoulderi_final(file_path, sampling_freq=500):
    """Analyze ShoulderI data"""
    filename = os.path.basename(file_path)
    athlete_name = filename.split('_i')[0]
    
    try:
        df = pd.read_csv(file_path)
        force_series = df['Plate1_N_filtered'] + df['Plate2_N_filtered']
        body_mass = get_body_mass_from_excel(athlete_name)
        
        # Peak force
        peak_force = np.max(force_series)
        relative_peak_force = peak_force / body_mass
        
        # RFD calculations
        force_threshold = 50  # N
        start_indices = np.where(force_series > force_threshold)[0]
        start_idx = start_indices[0] if len(start_indices) > 0 else 0
        
        # RFD 0-50ms
        end_50ms = min(len(force_series) - 1, start_idx + int(0.05 * sampling_freq))
        rfd_50ms = (force_series[end_50ms] - force_series[start_idx]) / 0.05 if end_50ms > start_idx else 0
        
        # RFD 0-100ms
        end_100ms = min(len(force_series) - 1, start_idx + int(0.1 * sampling_freq))
        rfd_100ms = (force_series[end_100ms] - force_series[start_idx]) / 0.1 if end_100ms > start_idx else 0
        
        # Asymmetry
        plate1_mean = df['Plate1_N_filtered'].mean()
        plate2_mean = df['Plate2_N_filtered'].mean()
        asymmetry = abs(plate1_mean - plate2_mean) / ((plate1_mean + plate2_mean) / 2) * 100
        
        return {
            'Athlete': athlete_name,
            'Peak_Force_N': peak_force,
            'Relative_Peak_Force_N_per_kg': relative_peak_force,
            'RFD_0_50ms_N_per_s': rfd_50ms,
            'RFD_0_100ms_N_per_s': rfd_100ms,
            'Body_Mass_kg': body_mass,
            'Asymmetry_Percent': asymmetry,
            'Status': 'Success'
        }
        
    except Exception as e:
        logging.error(f"Error analyzing ShoulderI for {athlete_name}: {e}")
        return {'Athlete': athlete_name, 'Status': f'Error: {str(e)}'}

def calculate_dsi(cmj_df, mtp_df):
    """Calculate Dynamic Strength Index"""
    merged_df = pd.merge(cmj_df, mtp_df, on='Athlete', suffixes=('_cmj', '_mtp'))
    merged_df['DSI'] = merged_df['Relative_Peak_Force_N_per_kg_cmj'] / merged_df['Relative_Peak_Force_N_per_kg_mtp']
    return merged_df[['Athlete', 'DSI']]

def main():
    """Main analysis function"""
    logging.info("Starting complete pentathlon analysis...")
    
    # Analyze CMJ data
    print("\n=== CMJ ANALYSIS ===")
    cmj_files = glob.glob('/Users/filip/Desktop/Making stuff/F Labs/Force Data/Data 5.09/CMJ/*.csv')
    cmj_results = []
    
    for file_path in sorted(cmj_files):
        result = analyze_cmj_final(file_path)
        cmj_results.append(result)
        if result['Status'] == 'Success':
            print(f"{result['Athlete']}: {result['Jump_Height_m']:.3f} m")
    
    # Analyze DJ data
    print("\n=== DJ ANALYSIS ===")
    dj_files = glob.glob('/Users/filip/Desktop/Data 10.06/DJ/*.csv')
    dj_results = []
    
    for file_path in sorted(dj_files):
        result = analyze_dj_final(file_path)
        dj_results.append(result)
        if result['Status'] == 'Success':
            print(f"{result['Athlete']}: {result['Jump_Height_m']:.3f} m, RSI: {result['Reactive_Strength_Index']:.3f}")
    
    # Analyze MTP data
    print("\n=== MTP ANALYSIS ===")
    mtp_files = glob.glob('/Users/filip/Desktop/Data 10.06/MTP/*.csv')
    mtp_results = []
    
    for file_path in sorted(mtp_files):
        result = analyze_mtp_final(file_path)
        mtp_results.append(result)
        if result['Status'] == 'Success':
            print(f"{result['Athlete']}: {result['Relative_Peak_Force_N_per_kg']:.2f} N/kg")
    
    # Analyze ShoulderI data
    print("\n=== SHOULDER I ANALYSIS ===")
    shoulderi_files = glob.glob('/Users/filip/Desktop/Data 10.06/ShoulderI/*.csv')
    shoulderi_results = []
    
    for file_path in sorted(shoulderi_files):
        result = analyze_shoulderi_final(file_path)
        shoulderi_results.append(result)
        if result['Status'] == 'Success':
            print(f"{result['Athlete']}: {result['Relative_Peak_Force_N_per_kg']:.2f} N/kg")
    
    # Create DataFrames
    cmj_df = pd.DataFrame(cmj_results)
    dj_df = pd.DataFrame(dj_results)
    mtp_df = pd.DataFrame(mtp_results)
    shoulderi_df = pd.DataFrame(shoulderi_results)
    
    # Calculate DSI
    print("\n=== DYNAMIC STRENGTH INDEX ===")
    dsi_df = calculate_dsi(cmj_df, mtp_df)
    for _, row in dsi_df.iterrows():
        print(f"{row['Athlete']}: {row['DSI']:.3f}")
    
    # Save to Excel
    with pd.ExcelWriter('/Users/filip/Desktop/Data 10.06/pentathlon_analysis_final.xlsx') as writer:
        cmj_df.to_excel(writer, sheet_name='CMJ_Analysis', index=False)
        dj_df.to_excel(writer, sheet_name='DJ_Analysis', index=False)
        mtp_df.to_excel(writer, sheet_name='MTP_Analysis', index=False)
        shoulderi_df.to_excel(writer, sheet_name='ShoulderI_Analysis', index=False)
        dsi_df.to_excel(writer, sheet_name='DSI_Analysis', index=False)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"CMJ successful: {len(cmj_df[cmj_df['Status'] == 'Success'])}/{len(cmj_df)}")
    print(f"DJ successful: {len(dj_df[dj_df['Status'] == 'Success'])}/{len(dj_df)}")
    print(f"MTP successful: {len(mtp_df[mtp_df['Status'] == 'Success'])}/{len(mtp_df)}")
    print(f"ShoulderI successful: {len(shoulderi_df[shoulderi_df['Status'] == 'Success'])}/{len(shoulderi_df)}")
    print(f"DSI calculations: {len(dsi_df)}")
    
    # Print summary statistics
    successful_cmj = cmj_df[cmj_df['Status'] == 'Success']
    if len(successful_cmj) > 0:
        print(f"\nCMJ Jump Height Statistics:")
        print(f"Mean: {successful_cmj['Jump_Height_m'].mean():.3f} m")
        print(f"Range: {successful_cmj['Jump_Height_m'].min():.3f} - {successful_cmj['Jump_Height_m'].max():.3f} m")
    
    print(f"\nResults saved to: pentathlon_analysis_final.xlsx")

if __name__ == "__main__":
    main()