import pandas as pd



# OFO Function

def clean_ofo_survey_date(series: pd.Series) -> pd.Series:
    """
    Parse numeric survey_date values of the form:
      • YYYYMMDD (e.g. 20200809) → 2020-08-09
      • YYYYMM   (e.g. 202008)   → 2020-08-01
      • YYYY      (e.g. 2016)    → 2016-01-01
    Returns a datetime64 series, coercing any malformed entries to NaT.
    """
    def _parse(v):
        if pd.isna(v):
            return pd.NaT
        s = str(int(v))
        if len(s) == 4:
            # year only
            s = s + "0101"
        elif len(s) == 6:
            # year+month
            s = s + "01"
        elif len(s) == 8:
            # year+month+day, leave as is
            pass
        else:
            # unexpected length: pad/truncate to YYYYMMDD
            s = s.zfill(8)[:8]
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    
    return series.apply(_parse)





# FERP Functions

def load_ferp_species_table(file_path):
    """
    Loads and parses the FERP species file into a DataFrame with columns:
    ['Scientific name', 'Common name', 'Code', 'Family', 'Related', 'Genus', 'Epithet', 'Author']

    Parameters:
        file_path (str): Path to the FERP species text file

    Returns:
        pd.DataFrame: Cleaned and parsed species table
    """

    # Load and clean lines
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    lines = lines[1:]  # Skip header

    rows = []
    pending_name = None

    for line in lines:
        parts = [p.strip() for p in line.split("\t") if p.strip()]

        if len(parts) == 1:
            pending_name = parts[0]
        elif len(parts) == 4:
            if pending_name:
                sci_name = pending_name
                related = parts[0]
                pending_name = None
            else:
                sci_name = parts[0]
                related = ""

            rows.append({
                "Scientific name": sci_name,
                "Common name": parts[1],
                "Code": parts[2],
                "Family": parts[3],
                "Related": related
            })

    df = pd.DataFrame(rows)

    # Extract genus, epithet, author
    def parse_name(name):
        tokens = name.split()
        if len(tokens) >= 2:
            genus = tokens[0]
            epithet = tokens[1]
            author = " ".join(tokens[2:]) if len(tokens) > 2 else ""
        else:
            genus, epithet, author = name, "", ""
        return pd.Series([genus, epithet, author])

    df[['Genus', 'Specific epithet', 'Author']] = df['Scientific name'].apply(parse_name)

    return df
