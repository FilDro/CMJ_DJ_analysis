# REPORT GENERATION

## 📊 REPORT TOOLS

### `create_individual_report.py`
- **Purpose**: Generate individual athlete performance reports
- **Format**: Detailed per-athlete analysis
- **Usage**: For individual athlete feedback

### `create_pdf_report.py`
- **Purpose**: Generate PDF format reports
- **Format**: Professional PDF output
- **Usage**: For sharing and presentation

### `create_table_report.py`
- **Purpose**: Generate table format reports
- **Format**: Tabular data output
- **Usage**: For data analysis and Excel import

## 🚨 STATUS: NEEDS TIMESTAMP UPDATE

⚠️ **WARNING**: These report generation scripts likely still contain the 500Hz assumption error.

### TO FIX:
Apply the same timestamp corrections from `../FINAL_SOLUTION/corrected_cmj_final.py`:

```python
# Instead of:
time_series = np.arange(len(force_series)) / 500

# Use:
timestamps = df['Timestamp_s']
time_series = timestamps - timestamps.iloc[0]
actual_freq = 1.0 / timestamps.diff().dropna().mean()
```

## 🔧 INTEGRATION

These scripts should be updated to use the corrected analysis from:
`../FINAL_SOLUTION/corrected_cmj_final.py`

## 📋 PRIORITY

1. **High**: Update with timestamp corrections
2. **Medium**: Integrate with final solution
3. **Low**: Add new visualizations based on corrected data