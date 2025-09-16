#!/usr/bin/env python3
"""
Create comprehensive PDF report for pentathlon force plate analysis
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
    ax.text(0.5, 0.7, 'PENTATHLON FORCE PLATE\nANALYSIS REPORT', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
            fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.4, 'Modern Pentathlon Athletes\nForce-Time Analysis', 
            fontsize=16, ha='center', va='center', style='italic')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_summary_page(pdf, cmj, dj, mtp, shoulder):
    """Create summary statistics page"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # CMJ Jump Heights
    cmj_sorted = cmj.sort_values('Jump_Height_m', ascending=True)
    bars1 = ax1.barh(cmj_sorted['Athlete'], cmj_sorted['Jump_Height_m'], 
                     color='skyblue', alpha=0.7)
    ax1.set_xlabel('Jump Height (m)')
    ax1.set_title('CMJ Jump Heights', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    # Add values on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # DJ RSI Values
    dj_sorted = dj.sort_values('rsi', ascending=True)
    bars2 = ax2.barh(dj_sorted['Athlete'], dj_sorted['rsi'], 
                     color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Reactive Strength Index')
    ax2.set_title('DJ Reactive Strength Index', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    # Add values on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center', fontsize=8)
    
    # MTP Relative Peak Force
    mtp_sorted = mtp.sort_values('Relative_Peak_Force_N_per_kg', ascending=True)
    bars3 = ax3.barh(mtp_sorted['Athlete'], mtp_sorted['Relative_Peak_Force_N_per_kg'], 
                     color='coral', alpha=0.7)
    ax3.set_xlabel('Relative Peak Force (N/kg)')
    ax3.set_title('MTP Relative Peak Force', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    # Add values on bars
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontsize=8)
    
    # Shoulder Relative Peak Force
    shoulder_sorted = shoulder.sort_values('Relative_Peak_Force_N_per_kg', ascending=True)
    bars4 = ax4.barh(shoulder_sorted['Athlete'], shoulder_sorted['Relative_Peak_Force_N_per_kg'], 
                     color='gold', alpha=0.7)
    ax4.set_xlabel('Relative Peak Force (N/kg)')
    ax4.set_title('Shoulder Relative Peak Force', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    # Add values on bars
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_asymmetry_analysis(pdf, cmj, mtp, shoulder):
    """Create asymmetry analysis page"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # CMJ Asymmetry
    cmj_asym = cmj.sort_values('Asymmetry_Percent', ascending=True)
    bars1 = ax1.barh(cmj_asym['Athlete'], cmj_asym['Asymmetry_Percent'], 
                     color=['red' if x > 15 else 'orange' if x > 10 else 'green' 
                            for x in cmj_asym['Asymmetry_Percent']], alpha=0.7)
    ax1.set_xlabel('Asymmetry (%)')
    ax1.set_title('CMJ Asymmetry', fontweight='bold')
    ax1.axvline(x=15, color='red', linestyle='--', alpha=0.7, label='High (>15%)')
    ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='Moderate (>10%)')
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend()
    
    # MTP Asymmetry
    mtp_asym = mtp.sort_values('Asymmetry_Percent', ascending=True)
    bars2 = ax2.barh(mtp_asym['Athlete'], mtp_asym['Asymmetry_Percent'], 
                     color=['red' if x > 20 else 'orange' if x > 15 else 'green' 
                            for x in mtp_asym['Asymmetry_Percent']], alpha=0.7)
    ax2.set_xlabel('Asymmetry (%)')
    ax2.set_title('MTP Asymmetry', fontweight='bold')
    ax2.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='High (>20%)')
    ax2.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='Moderate (>15%)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()
    
    # Shoulder Asymmetry
    shoulder_asym = shoulder.sort_values('Asymmetry_Percent', ascending=True)
    bars3 = ax3.barh(shoulder_asym['Athlete'], shoulder_asym['Asymmetry_Percent'], 
                     color=['red' if x > 20 else 'orange' if x > 15 else 'green' 
                            for x in shoulder_asym['Asymmetry_Percent']], alpha=0.7)
    ax3.set_xlabel('Asymmetry (%)')
    ax3.set_title('Shoulder Asymmetry', fontweight='bold')
    ax3.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='High (>20%)')
    ax3.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='Moderate (>15%)')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend()
    
    # Asymmetry Distribution
    ax4.hist([cmj['Asymmetry_Percent'], mtp['Asymmetry_Percent'], shoulder['Asymmetry_Percent']], 
             bins=8, alpha=0.7, label=['CMJ', 'MTP', 'Shoulder'])
    ax4.set_xlabel('Asymmetry (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Asymmetry Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_correlation_analysis(pdf, cmj, dj, mtp, shoulder):
    """Create correlation analysis page"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # Merge datasets for correlation
    merged = cmj[['Athlete', 'Jump_Height_m', 'Peak_Force_N']].merge(
        dj[['Athlete', 'rsi', 'jump_height']], on='Athlete', how='inner')
    merged = merged.merge(
        mtp[['Athlete', 'Relative_Peak_Force_N_per_kg']], on='Athlete', how='inner')
    
    # CMJ vs DJ Jump Heights
    ax1.scatter(merged['Jump_Height_m'], merged['jump_height'], alpha=0.7, s=60)
    ax1.set_xlabel('CMJ Jump Height (m)')
    ax1.set_ylabel('DJ Jump Height (m)')
    ax1.set_title('CMJ vs DJ Jump Heights', fontweight='bold')
    ax1.grid(alpha=0.3)
    # Add correlation coefficient
    corr = merged['Jump_Height_m'].corr(merged['jump_height'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # CMJ Peak Force vs MTP Peak Force
    ax2.scatter(merged['Peak_Force_N'], merged['Relative_Peak_Force_N_per_kg'], alpha=0.7, s=60)
    ax2.set_xlabel('CMJ Peak Force (N)')
    ax2.set_ylabel('MTP Relative Peak Force (N/kg)')
    ax2.set_title('CMJ vs MTP Peak Force', fontweight='bold')
    ax2.grid(alpha=0.3)
    # Add correlation coefficient
    corr = merged['Peak_Force_N'].corr(merged['Relative_Peak_Force_N_per_kg'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # DJ RSI vs Jump Height
    ax3.scatter(merged['rsi'], merged['jump_height'], alpha=0.7, s=60)
    ax3.set_xlabel('DJ RSI')
    ax3.set_ylabel('DJ Jump Height (m)')
    ax3.set_title('DJ RSI vs Jump Height', fontweight='bold')
    ax3.grid(alpha=0.3)
    # Add correlation coefficient
    corr = merged['rsi'].corr(merged['jump_height'])
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Performance radar chart for top 3 athletes
    top_athletes = merged.nlargest(3, 'Jump_Height_m')
    
    # Normalize metrics (0-1 scale)
    metrics = ['Jump_Height_m', 'rsi', 'Relative_Peak_Force_N_per_kg']
    normalized = merged[metrics].copy()
    for metric in metrics:
        normalized[metric] = (normalized[metric] - normalized[metric].min()) / (normalized[metric].max() - normalized[metric].min())
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    colors = ['red', 'blue', 'green']
    for i, (_, athlete) in enumerate(top_athletes.iterrows()):
        values = normalized.loc[normalized.index == athlete.name, metrics].iloc[0].tolist()
        values += values[:1]  # Complete the circle
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=athlete['Athlete'], color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['CMJ Height', 'DJ RSI', 'MTP Force'])
    ax4.set_ylim(0, 1)
    ax4.set_title('Top 3 Athletes Performance', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_rankings_table(pdf, cmj, dj, mtp, shoulder):
    """Create rankings table page"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Create separate ranking dictionaries
    cmj_ranks = {}
    dj_ranks = {}
    mtp_ranks = {}
    shoulder_ranks = {}
    
    # CMJ rankings
    cmj_sorted = cmj.sort_values('Jump_Height_m', ascending=False)
    for i, (_, row) in enumerate(cmj_sorted.iterrows()):
        cmj_ranks[row['Athlete']] = {'rank': i+1, 'value': row['Jump_Height_m']}
    
    # DJ rankings
    dj_sorted = dj.sort_values('rsi', ascending=False)
    for i, (_, row) in enumerate(dj_sorted.iterrows()):
        dj_ranks[row['Athlete']] = {'rank': i+1, 'value': row['rsi']}
    
    # MTP rankings
    mtp_sorted = mtp.sort_values('Relative_Peak_Force_N_per_kg', ascending=False)
    for i, (_, row) in enumerate(mtp_sorted.iterrows()):
        mtp_ranks[row['Athlete']] = {'rank': i+1, 'value': row['Relative_Peak_Force_N_per_kg']}
    
    # Shoulder rankings
    shoulder_sorted = shoulder.sort_values('Relative_Peak_Force_N_per_kg', ascending=False)
    for i, (_, row) in enumerate(shoulder_sorted.iterrows()):
        shoulder_ranks[row['Athlete']] = {'rank': i+1, 'value': row['Relative_Peak_Force_N_per_kg']}
    
    # Get all athletes
    all_athletes = set(cmj['Athlete'].tolist() + dj['Athlete'].tolist() + mtp['Athlete'].tolist() + shoulder['Athlete'].tolist())
    
    # Calculate overall rankings
    overall_ranks = {}
    for athlete in all_athletes:
        ranks = []
        if athlete in cmj_ranks:
            ranks.append(cmj_ranks[athlete]['rank'])
        if athlete in dj_ranks:
            ranks.append(dj_ranks[athlete]['rank'])
        if athlete in mtp_ranks:
            ranks.append(mtp_ranks[athlete]['rank'])
        if athlete in shoulder_ranks:
            ranks.append(shoulder_ranks[athlete]['rank'])
        
        overall_ranks[athlete] = np.mean(ranks) if ranks else float('inf')
    
    # Sort by overall rank
    sorted_athletes = sorted(overall_ranks.items(), key=lambda x: x[1])
    
    # Create table data
    table_data = []
    for athlete, overall_rank in sorted_athletes:
        table_data.append([
            athlete,
            f"{cmj_ranks[athlete]['value']:.3f}" if athlete in cmj_ranks else '-',
            f"{cmj_ranks[athlete]['rank']}" if athlete in cmj_ranks else '-',
            f"{dj_ranks[athlete]['value']:.2f}" if athlete in dj_ranks else '-',
            f"{dj_ranks[athlete]['rank']}" if athlete in dj_ranks else '-',
            f"{mtp_ranks[athlete]['value']:.1f}" if athlete in mtp_ranks else '-',
            f"{mtp_ranks[athlete]['rank']}" if athlete in mtp_ranks else '-',
            f"{shoulder_ranks[athlete]['rank']}" if athlete in shoulder_ranks else '-',
            f"{overall_rank:.1f}" if overall_rank != float('inf') else '-'
        ])
    
    headers = ['Athlete', 'CMJ Height', 'CMJ Rank', 'DJ RSI', 'DJ Rank', 'MTP Force', 'MTP Rank', 'Shoulder Rank', 'Overall Rank']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the rankings
    for i in range(len(table_data)):
        overall_rank_str = table_data[i][-1]
        if overall_rank_str != '-':
            overall_rank = float(overall_rank_str)
            if overall_rank <= 3:
                color = 'lightgreen'
            elif overall_rank <= 6:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
        else:
            color = 'lightgray'
        
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('COMPREHENSIVE ATHLETE RANKINGS', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_summary_stats_page(pdf, cmj, dj, mtp, shoulder):
    """Create summary statistics page"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Calculate summary statistics
    stats_data = []
    
    # CMJ stats
    stats_data.append(['CMJ Jump Height (m)', f"{cmj['Jump_Height_m'].mean():.3f}", 
                      f"{cmj['Jump_Height_m'].std():.3f}", f"{cmj['Jump_Height_m'].min():.3f}", 
                      f"{cmj['Jump_Height_m'].max():.3f}", f"{len(cmj)}"])
    
    # DJ stats
    stats_data.append(['DJ RSI', f"{dj['rsi'].mean():.2f}", 
                      f"{dj['rsi'].std():.2f}", f"{dj['rsi'].min():.2f}", 
                      f"{dj['rsi'].max():.2f}", f"{len(dj)}"])
    
    # MTP stats
    stats_data.append(['MTP Rel. Peak Force (N/kg)', f"{mtp['Relative_Peak_Force_N_per_kg'].mean():.1f}", 
                      f"{mtp['Relative_Peak_Force_N_per_kg'].std():.1f}", f"{mtp['Relative_Peak_Force_N_per_kg'].min():.1f}", 
                      f"{mtp['Relative_Peak_Force_N_per_kg'].max():.1f}", f"{len(mtp)}"])
    
    # Shoulder stats
    stats_data.append(['Shoulder Rel. Peak Force (N/kg)', f"{shoulder['Relative_Peak_Force_N_per_kg'].mean():.1f}", 
                      f"{shoulder['Relative_Peak_Force_N_per_kg'].std():.1f}", f"{shoulder['Relative_Peak_Force_N_per_kg'].min():.1f}", 
                      f"{shoulder['Relative_Peak_Force_N_per_kg'].max():.1f}", f"{len(shoulder)}"])
    
    headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'N']
    
    table = ax.table(cellText=stats_data, colLabels=headers, cellLoc='center', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color header row
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('lightblue')
    
    ax.set_title('SUMMARY STATISTICS', fontsize=16, fontweight='bold', pad=20)
    
    # Add key findings text
    findings_text = """
KEY FINDINGS:

• Best CMJ jumper: {} ({:.3f} m)
• Best DJ RSI: {} ({:.2f})
• Highest MTP force: {} ({:.1f} N/kg)
• Highest shoulder force: {} ({:.1f} N/kg)

• {} athletes with MTP asymmetry >20%
• {} athletes with shoulder asymmetry >20%

RECOMMENDATIONS:
• Monitor high asymmetry athletes for injury risk
• Focus on bilateral strength development
• Track longitudinal changes in performance
""".format(
        cmj.loc[cmj['Jump_Height_m'].idxmax(), 'Athlete'],
        cmj['Jump_Height_m'].max(),
        dj.loc[dj['rsi'].idxmax(), 'Athlete'],
        dj['rsi'].max(),
        mtp.loc[mtp['Relative_Peak_Force_N_per_kg'].idxmax(), 'Athlete'],
        mtp['Relative_Peak_Force_N_per_kg'].max(),
        shoulder.loc[shoulder['Relative_Peak_Force_N_per_kg'].idxmax(), 'Athlete'],
        shoulder['Relative_Peak_Force_N_per_kg'].max(),
        len(mtp[mtp['Asymmetry_Percent'] > 20]),
        len(shoulder[shoulder['Asymmetry_Percent'] > 20])
    )
    
    ax.text(0.5, 0.4, findings_text, transform=ax.transAxes, fontsize=11,
            ha='center', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to generate PDF report"""
    print("Loading data...")
    cmj, dj, mtp, shoulder = load_data()
    
    print("Generating PDF report...")
    with PdfPages('pentathlon_force_analysis_report.pdf') as pdf:
        create_title_page(pdf)
        create_summary_page(pdf, cmj, dj, mtp, shoulder)
        create_asymmetry_analysis(pdf, cmj, mtp, shoulder)
        create_correlation_analysis(pdf, cmj, dj, mtp, shoulder)
        create_rankings_table(pdf, cmj, dj, mtp, shoulder)
        create_summary_stats_page(pdf, cmj, dj, mtp, shoulder)
    
    print("PDF report generated: pentathlon_force_analysis_report.pdf")

if __name__ == "__main__":
    main()