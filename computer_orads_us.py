# -*- coding: utf-8 -*-
"""
O-RADS US Scoring Algorithm
This script classifies ovarian/adnexal lesions into O-RADS risk categories (0-5)
based on predefined ultrasound features extracted from clinical reports.
"""

import pandas as pd


def compute_O_RADS_US(lesion_data):
    """
    Calculates the O-RADS US score (0-5) based on given lesion attributes.

    Input: lesion_data (dict) containing the following keys (case-sensitive):
      - Location (str)                       e.g., "Left adnexa"
      - Size (float)                         Maximum diameter in cm
      - Physiologic (int)                    Physiologic cyst 1/0
      - Hemorrhagic (int)                    Hemorrhagic cyst 1/0
      - Dermoid (int)                        Dermoid cyst 1/0
      - Endometrioma (int)                   Endometrioma 1/0
      - Paraovarian (int)                    Paraovarian cyst 1/0
      - PeritonealInclusion (int)            Peritoneal inclusion cyst 1/0
      - Hydrosalpinx (int)                   Hydrosalpinx 1/0
      - Solid_component (int)                Solid component in cyst >=3mm 1/0
      - Solid_lesion (int)                   Solid lesion >=80% 1/0
      - Unilocular (int)                     Unilocular cyst 1/0
      - Bilocular (int)                      Bilocular cyst 1/0
      - Multilocular (int)                   Multilocular cyst 1/0
      - Smooth (int)                         Smooth margins 1/0
      - Irregular (int)                      Irregular margins 1/0
      - Shadowing (int)                      Acoustic shadowing 1/0
      - Papillary_projection (int)           Number of papillary projections (0 if none)
      - ColorScore (int or None)             Color Doppler score (1-4), can be None
      - Ascites_or_PeritNod (int)            Ascites or peritoneal nodularity 1/0

    Output: dict {
      "O-RADS": int (0-5, or 6 as a catch-all),
      "reason": str (Brief explanation of the triggered rule)
    }
    """

    # Extract values with safe defaults
    get = lambda k, d=0: lesion_data.get(k, d)
    Size = lesion_data.get('Size', None)
    Physiologic = int(bool(get('Physiologic', 0)))
    Hemorrhagic = int(bool(get('Hemorrhagic', 0)))
    Dermoid = int(bool(get('Dermoid', 0)))
    Endometrioma = int(bool(get('Endometrioma', 0)))
    Paraovarian = int(bool(get('Paraovarian', 0)))
    PeritonealInclusion = int(bool(get('PeritonealInclusion', 0)))
    Hydrosalpinx = int(bool(get('Hydrosalpinx', 0)))
    Solid_component = int(bool(get('Solid_component', 0)))
    Solid_lesion = int(bool(get('Solid_lesion', 0)))
    Unilocular = int(bool(get('Unilocular', 0)))
    Bilocular = int(bool(get('Bilocular', 0)))
    Multilocular = int(bool(get('Multilocular', 0)))
    Irregular = int(bool(get('Irregular', 0)))
    Shadowing = int(bool(get('Shadowing', 0)))
    pps = int(get('Papillary_projection', 0) or 0)
    CS = get('ColorScore', None)  # 1-4 or None
    Ascites = int(bool(get('Ascites_or_PeritNod', 0)))

    # Helper function to format the return value
    def out(score, reason):
        return {"O-RADS": int(score), "reason": reason}

    # 1. Conditions for score 0 (Incomplete evaluation)
    essential_present = (Size is not None) or any([
        Solid_component, Solid_lesion, Unilocular, Bilocular, Multilocular,
        Physiologic, Hemorrhagic, Dermoid, Endometrioma, Paraovarian,
        PeritonealInclusion, Hydrosalpinx, Ascites
    ])
    if not essential_present:
        return out(0, "Incomplete: no size and no descriptive features")

    # 2. Physiologic cysts
    if Physiologic and (Size is not None):
        if Size <= 3:
            return out(1, "Physiologic cyst <=3 cm -> O-RADS 1")
        else:
            return out(2, "Physiologic cyst >3 cm -> O-RADS 2")

    # 3. Specific typical benign lesions -> O-RADS 2
    if Paraovarian or PeritonealInclusion or Hydrosalpinx:
        return out(2, "Typical benign lesion (Paraovarian/PeritonealInclusion/Hydrosalpinx) -> O-RADS 2")

    # 4. Typical benign lesions with size thresholds
    if Hemorrhagic or Dermoid or Endometrioma:
        if Size < 10:
            return out(2, "Typical benign lesion (Hemorrhagic/Dermoid/Endometrioma) size < 10 cm -> O-RADS 2")
        else:
            return out(3, "Typical benign lesion (Hemorrhagic/Dermoid/Endometrioma) size >= 10 cm -> O-RADS 3")

    # 5. High-risk immediate triggers -> O-RADS 5
    if Ascites:
        return out(5, "Ascites or peritoneal nodularity present -> O-RADS 5")

    # 6. Solid lesion handling (>=80% solid)
    if Solid_lesion:
        # Irregular solid -> high risk
        if Irregular:
            return out(5, "Solid lesion & irregular -> O-RADS 5")
        # High blood flow (CS=4) -> high risk (with or without shadowing)
        if CS == 4:
            return out(5, "Solid lesion & ColorScore=4 -> O-RADS 5")
        # Solid, avascular/low flow with shadowing -> lower risk (O-RADS 3)
        if Shadowing:
            return out(3, "Solid lesion with shadowing (and not irregular/high-CS) -> O-RADS 3")
        # CS=1 (avascular) -> lower risk
        if CS == 1:
            return out(3, "Solid lesion with CS=1 (avascular) -> O-RADS 3")
        # Non-shadowing, moderate flow -> O-RADS 4
        if CS in (2, 3):
            return out(4, "Solid lesion non-shadowing with CS 2-3 -> O-RADS 4")

    # 7. Lesions with solid components (but <80% solid overall)
    if Solid_component:
        # Unilocular with solid component -> O-RADS 4 or 5 based on papillae
        if Unilocular:
            if pps >= 4:
                return out(5, "Unilocular with >=4 papillary projections -> O-RADS 5")
            else:
                return out(4, "Unilocular with <4 papillary projections -> O-RADS 4")

        # Bi- or multilocular with solid component -> O-RADS 4 or 5 based on color score
        if Bilocular or Multilocular:
            if CS in (3, 4):
                return out(5, "Bi-/Multilocular with solid component CS 3-4 -> O-RADS 5")
            else:
                return out(4, "Bi-/Multilocular with solid component CS 1-2 -> O-RADS 4")

        # Catch-all rules for unclassified locularity with solid components
        if pps >= 4:
            return out(5, "Solid_component with >=4 papillary projections -> O-RADS 5")
        if CS in (3, 4):
            return out(5, "Solid_component with CS 3-4 -> O-RADS 5")

        return out(4, "Solid_component (default intermediate) -> O-RADS 4")

    # 8. Cystic lesions without solid components

    # Irregular inner wall
    if Irregular:
        if Unilocular:
            return out(3, "Unilocular cyst with irregular inner wall -> O-RADS 3")
        if Bilocular or Multilocular:
            return out(4, "Bi-/Multilocular cyst with irregular inner wall -> O-RADS 4")

    # Uni- or bilocular smooth cysts
    if Unilocular or Bilocular:
        if not Irregular:
            if (Size is not None) and (Size >= 10):
                return out(3, "Uni-/Bilocular smooth cyst >= 10 cm -> O-RADS 3")
            if (Size is not None) and (Size < 10):
                return out(2, "Uni-/Bilocular smooth cyst < 10 cm -> O-RADS 2")

    # Multilocular smooth cysts
    if Multilocular:
        if not Irregular:
            if (Size is not None) and (Size < 10):
                if (CS is None) or (CS < 4):
                    return out(3, "Multilocular smooth < 10 cm and CS<4 -> O-RADS 3")
            else:
                if (Size is not None) and (Size >= 10):
                    return out(4, "Multilocular smooth >= 10 cm -> O-RADS 4")
            if CS == 4:
                return out(4, "Multilocular smooth with CS=4 -> O-RADS 4")

    # 9. Fallback (If no rules matched)
    return out(6, "Default catch-all -> O-RADS 6")


def row_to_lesion_data(row):
    """
    Converts a row from the dataframe into a lesion_data dictionary.
    Note: Column headers map to the proprietary dataset format.
    """
    lesion_data = {
        "Location": row.get("病灶位置"),
        "Size": float(row.get("病灶最大直径大小（cm）")) if pd.notna(row.get("病灶最大直径大小（cm）")) else None,

        # Lesion types (set to 1 if not null)
        "Physiologic": 1 if pd.notna(row.get("生理性囊肿")) else 0,
        "Hemorrhagic": 1 if pd.notna(row.get("出血性囊肿")) else 0,
        "Dermoid": 1 if pd.notna(row.get("畸胎瘤")) else 0,
        "Endometrioma": 1 if pd.notna(row.get("巧克力囊肿")) else 0,
        "Paraovarian": 1 if pd.notna(row.get("卵巢旁囊肿")) else 0,
        "PeritonealInclusion": 1 if pd.notna(row.get("腹膜包裹囊肿")) else 0,
        "Hydrosalpinx": 1 if pd.notna(row.get("输卵管积水")) else 0,

        # Ultrasound features (set to 1 if not null)
        "Solid_component": 1 if pd.notna(row.get("有实性成分")) else 0,
        "Solid_lesion": 1 if pd.notna(row.get("实性病变")) else 0,
        "Unilocular": 1 if pd.notna(row.get("单房囊性")) else 0,
        "Bilocular": 1 if pd.notna(row.get("双房囊性")) else 0,
        "Multilocular": 1 if pd.notna(row.get("多房囊性")) else 0,
        "Irregular": 1 if pd.notna(row.get("不规则")) else 0,
        "Shadowing": 1 if pd.notna(row.get("阴影")) else 0,
        "Ascites_or_PeritNod": 1 if pd.notna(row.get("腹水、腹膜结节")) else 0,

        # Numerical features
        "Papillary_projection": int(row.get("乳头状突起的个数")) if pd.notna(row.get("乳头状突起的个数")) else 0,
        "ColorScore": int(row.get("彩色多普勒血流评分")) if pd.notna(row.get("彩色多普勒血流评分")) else None
    }
    return lesion_data


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Define relative paths for GitHub repository
    INPUT_FILE = "data/majority_vote_result.xlsx"
    OUTPUT_FILE = "data/majority_vote_result_scored.xlsx"

    try:
        # Read the dataset
        df = pd.read_excel(INPUT_FILE)

        # Calculate scores
        results = []
        for idx, row in df.iterrows():
            lesion_dict = row_to_lesion_data(row)
            score_output = compute_O_RADS_US(lesion_dict)
            results.append(score_output)

        # Append results to the DataFrame
        df["O-RADS"] = [r["O-RADS"] for r in results]
        df["reason"] = [r["reason"] for r in results]

        # Save the updated file
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Successfully processed {len(df)} records and saved to {OUTPUT_FILE}")

    except FileNotFoundError:
        print(
            f"Error: Could not find the input file at {INPUT_FILE}. Please ensure the data directory is set up correctly.")