
def extract_measured_radii(row):
    return {
        5: row['Diameter_5(mm)'] / 20,
        10: row['Diameter_10(mm)'] / 20,
        15: row['Diameter_15(mm)'] / 20,
        20: row['Diameter_20(mm)'] / 20,
        35: row['Diameter_mid(mm)'] / 20,
    }
