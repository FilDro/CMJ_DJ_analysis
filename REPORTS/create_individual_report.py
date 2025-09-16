#!/usr/bin/env python3
"""
Create individual athlete-focused PDF report for pentathlon force plate analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load all analysis results"""
    cmj = pd.read_excel('pentathlon_analysis_final.xlsx', sheet_name='CMJ_Analysis')
    dj = pd.read_excel('pentathlon_analysis_final.xlsx', sheet_name='DJ_Analysis_Complete')
    mtp = pd.read_excel('pentathlon_analysis_final.xlsx', sheet_name='MTP_Analysis_Improved')
    shoulder = pd.read_excel('pentathlon_analysis_final.xlsx', sheet_name='ShoulderI_Analysis_Improved')
    
    # Standardize athlete names
    dj['Athlete'] = dj['athlete'].str.capitalize()
    
    return cmj, dj, mtp, shoulder

def create_title_page(pdf):
    """Create title page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.7, 'PENTATHLON FORCE PLATE\nINDIVIDUAL ATHLETE REPORT', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
            fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.4, 'Individual Performance Analysis\nModern Pentathlon Athletes', 
            fontsize=16, ha='center', va='center', style='italic')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_individual_athlete_page(pdf, athlete_name, cmj, dj, mtp, shoulder):
    """Create individual athlete results page"""
    fig = plt.figure(figsize=(8.5, 11))
    
    # Title
    fig.suptitle(f'ATHLETE: {athlete_name.upper()}', fontsize=20, fontweight='bold', y=0.95)
    
    # Get athlete data
    cmj_data = cmj[cmj['Athlete'].str.lower() == athlete_name.lower()]
    dj_data = dj[dj['Athlete'].str.lower() == athlete_name.lower()]
    mtp_data = mtp[mtp['Athlete'].str.lower() == athlete_name.lower()]
    shoulder_data = shoulder[shoulder['Athlete'].str.lower() == athlete_name.lower()]
    
    # Create 4 subplots for each test
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    # CMJ Results
    ax1 = fig.add_subplot(gs[0, :])
    if not cmj_data.empty:
        cmj_row = cmj_data.iloc[0]
        metrics = ['Jump Height\n(m)', 'Peak Force\n(N)', 'Relative Peak Force\n(N/kg)', 'Asymmetry\n(%)']
        values = [cmj_row['Jump_Height_m'], cmj_row['Peak_Force_N'], 
                 cmj_row['Relative_Peak_Force_N_per_kg'], cmj_row['Asymmetry_Percent']]
        
        bars = ax1.bar(metrics, values, color='skyblue', alpha=0.7)
        ax1.set_title('COUNTERMOVEMENT JUMP (CMJ)', fontweight='bold', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}' if value < 1 else f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No CMJ data available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('COUNTERMOVEMENT JUMP (CMJ)', fontweight='bold', fontsize=14)
    
    # DJ Results
    ax2 = fig.add_subplot(gs[1, :])
    if not dj_data.empty:
        dj_row = dj_data.iloc[0]
        metrics = ['Jump Height\n(m)', 'RSI', 'Ground Contact\nTime (s)', 'Peak Force\n(N)']
        values = [dj_row['jump_height'], dj_row['rsi'], 
                 dj_row['ground_contact_time'], dj_row['peak_force']]
        
        bars = ax2.bar(metrics, values, color='lightgreen', alpha=0.7)
        ax2.set_title('DROP JUMP (DJ)', fontweight='bold', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}' if value < 1 else f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No DJ data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('DROP JUMP (DJ)', fontweight='bold', fontsize=14)
    
    # MTP Results
    ax3 = fig.add_subplot(gs[2, :])
    if not mtp_data.empty:
        mtp_row = mtp_data.iloc[0]
        metrics = ['Peak Force\n(N)', 'Relative Peak Force\n(N/kg)', 'RFD 0-100ms\n(N/s)', 'Asymmetry\n(%)']
        values = [mtp_row['Peak_Force_N'], mtp_row['Relative_Peak_Force_N_per_kg'], 
                 mtp_row['RFD_0_100ms_N_per_s'], mtp_row['Asymmetry_Percent']]
        
        bars = ax3.bar(metrics, values, color='coral', alpha=0.7)
        ax3.set_title('MID-THIGH PULL (MTP)', fontweight='bold', fontsize=14)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No MTP data available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('MID-THIGH PULL (MTP)', fontweight='bold', fontsize=14)
    
    # Shoulder Results
    ax4 = fig.add_subplot(gs[3, :])
    if not shoulder_data.empty:
        shoulder_row = shoulder_data.iloc[0]
        metrics = ['Peak Force\n(N)', 'Relative Peak Force\n(N/kg)', 'RFD 0-100ms\n(N/s)', 'Asymmetry\n(%)']
        values = [shoulder_row['Peak_Force_N'], shoulder_row['Relative_Peak_Force_N_per_kg'], 
                 shoulder_row['RFD_0_100ms_N_per_s'], shoulder_row['Asymmetry_Percent']]
        
        bars = ax4.bar(metrics, values, color='gold', alpha=0.7)
        ax4.set_title('SHOULDER ISOMETRIC T-TEST', fontweight='bold', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Shoulder data available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('SHOULDER ISOMETRIC T-TEST', fontweight='bold', fontsize=14)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_top_performers_page(pdf, cmj, dj, mtp, shoulder):
    """Create top 3 performers summary page"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # CMJ Top 3
    cmj_top3 = cmj.nlargest(3, 'Jump_Height_m')
    ax1.bar(range(3), cmj_top3['Jump_Height_m'], color='skyblue', alpha=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([f"{row['Athlete']}\n{row['Jump_Height_m']:.3f}m" 
                        for _, row in cmj_top3.iterrows()], fontsize=10)
    ax1.set_title('TOP 3 - CMJ Jump Height', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Jump Height (m)')
    ax1.grid(axis='y', alpha=0.3)
    
    # DJ Top 3
    dj_top3 = dj.nlargest(3, 'rsi')
    ax2.bar(range(3), dj_top3['rsi'], color='lightgreen', alpha=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"{row['Athlete']}\n{row['rsi']:.2f}" 
                        for _, row in dj_top3.iterrows()], fontsize=10)
    ax2.set_title('TOP 3 - DJ Reactive Strength Index', fontweight='bold', fontsize=12)
    ax2.set_ylabel('RSI')
    ax2.grid(axis='y', alpha=0.3)
    
    # MTP Top 3
    mtp_top3 = mtp.nlargest(3, 'Relative_Peak_Force_N_per_kg')
    ax3.bar(range(3), mtp_top3['Relative_Peak_Force_N_per_kg'], color='coral', alpha=0.8)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([f"{row['Athlete']}\n{row['Relative_Peak_Force_N_per_kg']:.1f} N/kg" 
                        for _, row in mtp_top3.iterrows()], fontsize=10)
    ax3.set_title('TOP 3 - MTP Relative Peak Force', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Relative Peak Force (N/kg)')
    ax3.grid(axis='y', alpha=0.3)
    
    # Shoulder Top 3
    shoulder_top3 = shoulder.nlargest(3, 'Relative_Peak_Force_N_per_kg')
    ax4.bar(range(3), shoulder_top3['Relative_Peak_Force_N_per_kg'], color='gold', alpha=0.8)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels([f"{row['Athlete']}\n{row['Relative_Peak_Force_N_per_kg']:.1f} N/kg" 
                        for _, row in shoulder_top3.iterrows()], fontsize=10)
    ax4.set_title('TOP 3 - Shoulder Relative Peak Force', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Relative Peak Force (N/kg)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('TOP 3 PERFORMERS BY TEST', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_overall_summary_page(pdf, cmj, dj, mtp, shoulder):
    """Create overall summary page with key insights"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'OVERALL PERFORMANCE SUMMARY', 
            fontsize=18, fontweight='bold', ha='center', va='top')
    
    # Key metrics summary
    summary_text = f"""
KEY PERFORMANCE METRICS:

CMJ (Countermovement Jump):
• Best Jump Height: {cmj['Athlete'].iloc[cmj['Jump_Height_m'].idxmax()]} - {cmj['Jump_Height_m'].max():.3f} m
• Average Jump Height: {cmj['Jump_Height_m'].mean():.3f} m
• Range: {cmj['Jump_Height_m'].min():.3f} - {cmj['Jump_Height_m'].max():.3f} m

DJ (Drop Jump):
• Best RSI: {dj['Athlete'].iloc[dj['rsi'].idxmax()]} - {dj['rsi'].max():.2f}
• Average RSI: {dj['rsi'].mean():.2f}
• Range: {dj['rsi'].min():.2f} - {dj['rsi'].max():.2f}

MTP (Mid-Thigh Pull):
• Best Relative Force: {mtp['Athlete'].iloc[mtp['Relative_Peak_Force_N_per_kg'].idxmax()]} - {mtp['Relative_Peak_Force_N_per_kg'].max():.1f} N/kg
• Average Relative Force: {mtp['Relative_Peak_Force_N_per_kg'].mean():.1f} N/kg
• Range: {mtp['Relative_Peak_Force_N_per_kg'].min():.1f} - {mtp['Relative_Peak_Force_N_per_kg'].max():.1f} N/kg

Shoulder Isometric:
• Best Relative Force: {shoulder['Athlete'].iloc[shoulder['Relative_Peak_Force_N_per_kg'].idxmax()]} - {shoulder['Relative_Peak_Force_N_per_kg'].max():.1f} N/kg
• Average Relative Force: {shoulder['Relative_Peak_Force_N_per_kg'].mean():.1f} N/kg
• Range: {shoulder['Relative_Peak_Force_N_per_kg'].min():.1f} - {shoulder['Relative_Peak_Force_N_per_kg'].max():.1f} N/kg

ASYMMETRY ANALYSIS:
• Athletes with high MTP asymmetry (>20%): {len(mtp[mtp['Asymmetry_Percent'] > 20])}
• Athletes with high Shoulder asymmetry (>20%): {len(shoulder[shoulder['Asymmetry_Percent'] > 20])}
• Average CMJ asymmetry: {cmj['Asymmetry_Percent'].mean():.1f}%

RECOMMENDATIONS:
• Continue monitoring athletes with high asymmetry values
• Focus on bilateral strength development
• Track longitudinal changes in performance metrics
• Consider injury prevention strategies for high-asymmetry athletes
"""
    
    ax.text(0.05, 0.85, summary_text, transform=ax.transAxes, fontsize=11,
            ha='left', va='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to generate individual athlete PDF report"""
    print("Loading data...")
    cmj, dj, mtp, shoulder = load_data()
    
    # Get all unique athletes
    all_athletes = set()
    all_athletes.update(cmj['Athlete'].str.lower())
    all_athletes.update(dj['Athlete'].str.lower())
    all_athletes.update(mtp['Athlete'].str.lower())
    all_athletes.update(shoulder['Athlete'].str.lower())
    
    all_athletes = sorted(list(all_athletes))
    
    print(f"Generating individual report for {len(all_athletes)} athletes...")
    
    with PdfPages('pentathlon_individual_athlete_report.pdf') as pdf:
        # Title page
        create_title_page(pdf)
        
        # Individual athlete pages
        for athlete in all_athletes:
            print(f"Creating page for {athlete}...")
            create_individual_athlete_page(pdf, athlete, cmj, dj, mtp, shoulder)
        
        # Top performers page
        create_top_performers_page(pdf, cmj, dj, mtp, shoulder)
        
        # Overall summary page
        create_overall_summary_page(pdf, cmj, dj, mtp, shoulder)
    
    print("PDF report generated: pentathlon_individual_athlete_report.pdf")

if __name__ == "__main__":
    main()