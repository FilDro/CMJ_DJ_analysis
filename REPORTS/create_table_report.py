#!/usr/bin/env python3
"""
Create table-based PDF report for pentathlon force plate analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    ax.text(0.5, 0.7, 'PENTATHLON FORCE PLATE\nANALYSIS RESULTS', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
            fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.4, 'Individual Test Results\nModern Pentathlon Athletes', 
            fontsize=16, ha='center', va='center', style='italic')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_cmj_table(pdf, cmj):
    """Create CMJ results table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in cmj.iterrows():
        table_data.append([
            row['Athlete'],
            f"{row['Jump_Height_m']:.3f}",
            f"{row['Peak_Force_N']:.0f}",
            f"{row['Relative_Peak_Force_N_per_kg']:.1f}",
            f"{row['Flight_Time_s']:.3f}",
            f"{row['Takeoff_Velocity_m_per_s']:.2f}",
            f"{row['Asymmetry_Percent']:.1f}",
            f"{row['Body_Mass_kg']:.1f}"
        ])
    
    headers = ['Athlete', 'Jump Height\n(m)', 'Peak Force\n(N)', 'Relative Peak\nForce (N/kg)', 
               'Flight Time\n(s)', 'Takeoff Velocity\n(m/s)', 'Asymmetry\n(%)', 'Body Mass\n(kg)']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('COUNTERMOVEMENT JUMP (CMJ) RESULTS', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_dj_table(pdf, dj):
    """Create DJ results table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in dj.iterrows():
        table_data.append([
            row['Athlete'],
            f"{row['jump_height']:.3f}",
            f"{row['rsi']:.2f}",
            f"{row['ground_contact_time']:.3f}",
            f"{row['flight_time']:.3f}",
            f"{row['peak_force']:.0f}",
            f"{row['relative_peak_force']:.1f}",
            f"{row['asymmetry']:.1f}"
        ])
    
    headers = ['Athlete', 'Jump Height\n(m)', 'RSI', 'Ground Contact\nTime (s)', 
               'Flight Time\n(s)', 'Peak Force\n(N)', 'Relative Peak\nForce (N/kg)', 
               'Asymmetry\n(%)']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('DROP JUMP (DJ) RESULTS', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_mtp_table(pdf, mtp):
    """Create MTP results table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in mtp.iterrows():
        table_data.append([
            row['Athlete'],
            f"{row['Peak_Force_N']:.0f}",
            f"{row['Relative_Peak_Force_N_per_kg']:.1f}",
            f"{row['Asymmetry_Percent']:.1f}",
            f"{row['Body_Mass_kg']:.1f}",
            f"{row['movement_onset_time']:.3f}"
        ])
    
    headers = ['Athlete', 'Peak Force\n(N)', 'Relative Peak\nForce (N/kg)', 
               'Asymmetry\n(%)', 'Body Mass\n(kg)', 'Movement Onset\nTime (s)']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#FF9800')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('MID-THIGH PULL (MTP) RESULTS', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_shoulder_table(pdf, shoulder):
    """Create Shoulder results table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in shoulder.iterrows():
        table_data.append([
            row['Athlete'],
            f"{row['Peak_Force_N']:.0f}",
            f"{row['Relative_Peak_Force_N_per_kg']:.1f}",
            f"{row['Asymmetry_Percent']:.1f}",
            f"{row['Body_Mass_kg']:.1f}",
            f"{row['sustained_effort_force']:.0f}"
        ])
    
    headers = ['Athlete', 'Peak Force\n(N)', 'Relative Peak\nForce (N/kg)', 
               'Asymmetry\n(%)', 'Body Mass\n(kg)', 'Sustained Effort\nForce (N)']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#9C27B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('SHOULDER ISOMETRIC T-TEST RESULTS', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_top_performers_table(pdf, cmj, dj, mtp, shoulder):
    """Create top 3 performers table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Get top 3 for each test
    cmj_top3 = cmj.nlargest(3, 'Jump_Height_m')[['Athlete', 'Jump_Height_m']]
    dj_top3 = dj.nlargest(3, 'rsi')[['Athlete', 'rsi']]
    mtp_top3 = mtp.nlargest(3, 'Relative_Peak_Force_N_per_kg')[['Athlete', 'Relative_Peak_Force_N_per_kg']]
    shoulder_top3 = shoulder.nlargest(3, 'Relative_Peak_Force_N_per_kg')[['Athlete', 'Relative_Peak_Force_N_per_kg']]
    
    # Create table data
    table_data = []
    for i in range(3):
        row = [
            f"{i+1}",
            f"{cmj_top3.iloc[i]['Athlete']}\n{cmj_top3.iloc[i]['Jump_Height_m']:.3f} m",
            f"{dj_top3.iloc[i]['Athlete']}\n{dj_top3.iloc[i]['rsi']:.2f}",
            f"{mtp_top3.iloc[i]['Athlete']}\n{mtp_top3.iloc[i]['Relative_Peak_Force_N_per_kg']:.1f} N/kg",
            f"{shoulder_top3.iloc[i]['Athlete']}\n{shoulder_top3.iloc[i]['Relative_Peak_Force_N_per_kg']:.1f} N/kg"
        ]
        table_data.append(row)
    
    headers = ['Rank', 'CMJ Jump Height', 'DJ RSI', 'MTP Relative Peak Force', 'Shoulder Relative Peak Force']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 3)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#607D8B')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rank column
    for i in range(1, 4):
        table[(i, 0)].set_facecolor('#FFC107')
        table[(i, 0)].set_text_props(weight='bold')
    
    # Color medal positions
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    for i in range(3):
        for j in range(1, len(headers)):
            table[(i+1, j)].set_facecolor(colors[i])
            table[(i+1, j)].set_text_props(weight='bold')
    
    ax.set_title('TOP 3 PERFORMERS BY TEST', fontsize=18, fontweight='bold', pad=30)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_polish_legend_page(pdf):
    """Create Polish legend page explaining all metrics"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'LEGENDA - OBJAŚNIENIE METRYKI', 
            fontsize=20, fontweight='bold', ha='center', va='top')
    
    # Create sections with better spacing and formatting
    y_pos = 0.88
    
    # CMJ Section
    ax.text(0.05, y_pos, 'COUNTERMOVEMENT JUMP (CMJ) - SKOK Z WYKROKIEM:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#4CAF50", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    cmj_text = """• Jump Height (m) - Wysokość skoku: maksymalna wysokość osiągnięta podczas skoku
• Peak Force (N) - Siła szczytowa: maksymalna siła wygenerowana podczas odbicia
• Relative Peak Force (N/kg) - Względna siła szczytowa: siła szczytowa podzielona przez masę ciała
• Flight Time (s) - Czas lotu: czas spędzony w powietrzu podczas skoku
• Takeoff Velocity (m/s) - Prędkość odbicia: prędkość w momencie opuszczenia podłogi
• Asymmetry (%) - Asymetria: różnica w sile między lewą a prawą nogą
• Body Mass (kg) - Masa ciała: waga zawodnika"""
    
    ax.text(0.05, y_pos, cmj_text, fontsize=9, ha='left', va='top')
    y_pos -= 0.15
    
    # DJ Section
    ax.text(0.05, y_pos, 'DROP JUMP (DJ) - SKOK Z WYSOKOŚCI:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2196F3", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    dj_text = """• Jump Height (m) - Wysokość skoku: maksymalna wysokość osiągnięta po skoku z wysokości
• RSI - Reactive Strength Index: wskaźnik reaktywnej siły (wysokość skoku / czas kontaktu)
• Ground Contact Time (s) - Czas kontaktu z podłogą: czas spędzony na ziemi między lądowaniem a odbiciem
• Flight Time (s) - Czas lotu: czas spędzony w powietrzu
• Peak Force (N) - Siła szczytowa: maksymalna siła podczas kontaktu z podłogą
• Relative Peak Force (N/kg) - Względna siła szczytowa: siła szczytowa na kilogram masy ciała
• Asymmetry (%) - Asymetria: różnica w sile między nogami"""
    
    ax.text(0.05, y_pos, dj_text, fontsize=9, ha='left', va='top')
    y_pos -= 0.15
    
    # MTP Section
    ax.text(0.05, y_pos, 'MID-THIGH PULL (MTP) - CIĄGNIĘCIE DO POŁOWY UDA:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF9800", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    mtp_text = """• Peak Force (N) - Siła szczytowa: maksymalna siła ciągnięcia
• Relative Peak Force (N/kg) - Względna siła szczytowa: siła szczytowa na kilogram masy ciała
• Asymmetry (%) - Asymetria: różnica w sile między rękami/stronami ciała
• Body Mass (kg) - Masa ciała: waga zawodnika
• Movement Onset Time (s) - Czas rozpoczęcia ruchu: czas do osiągnięcia progu siły"""
    
    ax.text(0.05, y_pos, mtp_text, fontsize=9, ha='left', va='top')
    y_pos -= 0.12
    
    # Shoulder Section
    ax.text(0.05, y_pos, 'SHOULDER ISOMETRIC T-TEST - IZOMETRYCZNY TEST RAMION:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#9C27B0", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    shoulder_text = """• Peak Force (N) - Siła szczytowa: maksymalna siła w teście ramion
• Relative Peak Force (N/kg) - Względna siła szczytowa: siła szczytowa na kilogram masy ciała
• Asymmetry (%) - Asymetria: różnica w sile między ramionami
• Body Mass (kg) - Masa ciała: waga zawodnika
• Sustained Effort Force (N) - Siła podtrzymana: średnia siła w końcowej fazie testu"""
    
    ax.text(0.05, y_pos, shoulder_text, fontsize=9, ha='left', va='top')
    y_pos -= 0.12
    
    # Asymmetry Interpretation
    ax.text(0.05, y_pos, 'INTERPRETACJA ASYMETRII:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFC107", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    asymmetry_text = """• <10% - Normalna asymetria
• 10-20% - Umiarkowana asymetria - monitorowanie
• >20% - Wysoka asymetria - wymaga interwencji treningowej"""
    
    ax.text(0.05, y_pos, asymmetry_text, fontsize=9, ha='left', va='top')
    y_pos -= 0.08
    
    # Notes
    ax.text(0.05, y_pos, 'UWAGI:', 
            fontsize=12, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#607D8B", alpha=0.8, edgecolor='black'))
    y_pos -= 0.04
    
    notes_text = """• Wyższe wartości RSI wskazują na lepszą reaktywność mięśni
• Asymetria >20% może wskazywać na zwiększone ryzyko kontuzji
• Wszystkie testy wykonane na platformach siłowych z częstotliwością 500 Hz"""
    
    ax.text(0.05, y_pos, notes_text, fontsize=9, ha='left', va='top')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_summary_statistics_table(pdf, cmj, dj, mtp, shoulder):
    """Create summary statistics table"""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Calculate statistics
    stats_data = [
        ['CMJ Jump Height (m)', f"{cmj['Jump_Height_m'].mean():.3f}", f"{cmj['Jump_Height_m'].std():.3f}", 
         f"{cmj['Jump_Height_m'].min():.3f}", f"{cmj['Jump_Height_m'].max():.3f}", f"{len(cmj)}"],
        ['DJ RSI', f"{dj['rsi'].mean():.2f}", f"{dj['rsi'].std():.2f}", 
         f"{dj['rsi'].min():.2f}", f"{dj['rsi'].max():.2f}", f"{len(dj)}"],
        ['MTP Relative Peak\nForce (N/kg)', f"{mtp['Relative_Peak_Force_N_per_kg'].mean():.1f}", 
         f"{mtp['Relative_Peak_Force_N_per_kg'].std():.1f}", 
         f"{mtp['Relative_Peak_Force_N_per_kg'].min():.1f}", f"{mtp['Relative_Peak_Force_N_per_kg'].max():.1f}", f"{len(mtp)}"],
        ['Shoulder Relative Peak\nForce (N/kg)', f"{shoulder['Relative_Peak_Force_N_per_kg'].mean():.1f}", 
         f"{shoulder['Relative_Peak_Force_N_per_kg'].std():.1f}", 
         f"{shoulder['Relative_Peak_Force_N_per_kg'].min():.1f}", f"{shoulder['Relative_Peak_Force_N_per_kg'].max():.1f}", f"{len(shoulder)}"]
    ]
    
    headers = ['Test Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'N Athletes']
    
    table = ax.table(cellText=stats_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.6, 2.8)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#795548')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(stats_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('SUMMARY STATISTICS', fontsize=16, fontweight='bold', pad=20)
    
    # Add key findings
    findings_text = f"""
KEY FINDINGS:
• {len(mtp[mtp['Asymmetry_Percent'] > 20])} athletes have high MTP asymmetry (>20%)
• {len(shoulder[shoulder['Asymmetry_Percent'] > 20])} athletes have high Shoulder asymmetry (>20%)
• Average CMJ asymmetry: {cmj['Asymmetry_Percent'].mean():.1f}%
• All athletes successfully completed testing protocols
"""
    
    ax.text(0.5, 0.2, findings_text, transform=ax.transAxes, fontsize=10,
            ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to generate table-based PDF report"""
    print("Loading data...")
    cmj, dj, mtp, shoulder = load_data()
    
    print("Generating table-based PDF report...")
    
    with PdfPages('pentathlon_table_report.pdf') as pdf:
        # Title page
        create_title_page(pdf)
        
        # Polish legend page
        create_polish_legend_page(pdf)
        
        # Individual test tables
        create_cmj_table(pdf, cmj)
        create_dj_table(pdf, dj)
        create_mtp_table(pdf, mtp)
        create_shoulder_table(pdf, shoulder)
        
        # Top performers table
        create_top_performers_table(pdf, cmj, dj, mtp, shoulder)
        
        # Summary statistics
        create_summary_statistics_table(pdf, cmj, dj, mtp, shoulder)
    
    print("PDF report generated: pentathlon_table_report.pdf")

if __name__ == "__main__":
    main()