"""T03: select six systems, join schools, recode missing values and audit.

Run from the repository root: python 02_data-analysis/scripts/t03_prepare_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ["ARE", "SAU", "QAT", "JOR", "MAR", "PSE"]
STUDENT_PREDICTORS = [
    "ESCS", "PAREDINT", "HISEI", "HOMEPOS", "ANXMAT", "MATHEFF",
    "MATHEF21", "BELONG", "BULLIED", "DISCLIM", "TEACHSUP",
]
SCHOOL_PREDICTORS = ["EDUSHORT", "STAFFSHORT", "MCLSIZE", "STRATIO", "SCHLTYPE", "SCHSIZE"]
PREDICTORS = STUDENT_PREDICTORS + SCHOOL_PREDICTORS
SCHOOL_KEYS = ["CNT", "CNTSCHID"]
STUDENT_KEYS = ["CNT", "CNTSTUID"]
PLAUSIBLE_VALUES = [f"PV{i}MATH" for i in range(1, 11)]
WEIGHTS = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]
MISSING_LABELS = {"valid skip", "not applicable", "invalid", "no response"}


def main():
    """Produce the prepared data and small audit tables without fitting models."""
    output = ROOT / "02_data-analysis/outputs/t03"
    output.mkdir(parents=True, exist_ok=True)
    dictionary = pd.read_csv(ROOT / "data/data_dictionary.csv")

    # Read only needed columns and rows to keep memory use low.
    students = pd.read_parquet(
        ROOT / "data/parquet/CY08MSP_STU_QQQ",
        columns=SCHOOL_KEYS + ["CNTSTUID"] + STUDENT_PREDICTORS + PLAUSIBLE_VALUES + WEIGHTS,
        filters=[("CNT", "in", SYSTEMS)],
    )
    schools = pd.read_parquet(
        ROOT / "data/parquet/CY08MSP_SCH_QQQ",
        columns=SCHOOL_KEYS + SCHOOL_PREDICTORS,
        filters=[("CNT", "in", SYSTEMS)],
    )
    if set(students["CNT"].unique()) != set(SYSTEMS):
        raise ValueError("One or more requested systems are absent from the student data.")
    if students[SCHOOL_KEYS + ["CNTSTUID"]].isna().any().any():
        raise ValueError("Student join keys or identifiers contain missing values.")
    if schools[SCHOOL_KEYS].isna().any().any():
        raise ValueError("School join keys contain missing values.")
    if students.duplicated(STUDENT_KEYS).any():
        raise ValueError("Duplicate student identifiers found; inspect before proceeding.")
    if schools.duplicated(SCHOOL_KEYS).any():
        raise ValueError("Duplicate school identifiers found; inspect before proceeding.")

    # A left join retains students even if school information is unavailable.
    # CNT is included to prevent matching schools across different systems.
    data = students.merge(
        schools, on=SCHOOL_KEYS, how="left", validate="many_to_one", indicator=True,
    )
    unmatched = data.loc[data["_merge"] == "left_only", SCHOOL_KEYS].drop_duplicates()
    unmatched.to_csv(output / "unmatched_schools.csv", index=False)
    unmatched_students = int(data["_merge"].eq("left_only").sum())
    data = data.drop(columns="_merge")
    if len(data) != len(students):
        raise ValueError("The school join changed the number of students.")

    # Missing codes differ by variable: never replace 95/99 across all columns.
    # The repository extraction already converted most documented codes to NaN.
    recoding = []
    for variable in PREDICTORS:
        source = "student" if variable in STUDENT_PREDICTORS else "school"
        metadata = dictionary.loc[
            (dictionary["dataset"] == source) & (dictionary["variable"] == variable)
        ]
        if len(metadata) != 1:
            raise ValueError(f"Expected one dictionary entry for {source}/{variable}.")
        labels = json.loads(metadata.iloc[0]["value_labels"])
        codes = [float(code) for code, label in labels.items()
                 if label.strip().lower() in MISSING_LABELS]
        before = int(data[variable].isna().sum())
        replaced = int(data[variable].isin(codes).sum())
        data[variable] = data[variable].replace(codes, np.nan)
        recoding.append({
            "variable": variable, "source": source,
            "documented_missing_codes": json.dumps(codes),
            "already_missing": before, "newly_recoded": replaced,
            "total_missing": int(data[variable].isna().sum()),
            "percent_missing": float(data[variable].isna().mean() * 100),
        })

    # Exactly one indicator per predictor, including those with no missing values.
    indicators = data[PREDICTORS].isna().astype("uint8").add_suffix("_missing")
    data = pd.concat([data, indicators], axis=1)
    pd.DataFrame(recoding).to_csv(output / "missingness.csv", index=False)
    country_summary = data.groupby("CNT").agg(
        students=("CNTSTUID", "size"), schools=("CNTSCHID", "nunique"),
        missing_or_nonpositive_weights=("W_FSTUWT", lambda s: int((s.isna() | s.le(0)).sum())),
    ).reindex(SYSTEMS)
    country_summary.to_csv(output / "system_summary.csv")
    data.groupby("CNT")[PREDICTORS].agg(lambda s: s.isna().mean() * 100).to_csv(
        output / "missingness_by_system.csv"
    )
    audit = {
        "systems": SYSTEMS, "students_before_join": len(students),
        "students_after_join": len(data), "school_records_loaded": len(schools),
        "schools_with_students": len(data[SCHOOL_KEYS].drop_duplicates()),
        "duplicate_students": int(data.duplicated(STUDENT_KEYS).sum()),
        "unmatched_students": unmatched_students, "unmatched_schools": len(unmatched),
        "predictors": PREDICTORS, "missing_indicators": len(indicators.columns),
        "mathematics_plausible_values": PLAUSIBLE_VALUES, "weight_columns": len(WEIGHTS),
        "newly_recoded_values": sum(row["newly_recoded"] for row in recoding),
        "missing_mathematics_values": int(data[PLAUSIBLE_VALUES].isna().sum().sum()),
        "missing_or_nonpositive_weight_values": int((data[WEIGHTS].isna() | data[WEIGHTS].le(0)).sum().sum()),
    }
    data.to_parquet(output / "pisa_six_systems_prepared.parquet", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(country_summary.to_string())
    print(f"\nSaved {len(data):,} students and {len(data.columns)} columns to {output}")
    print(f"Unmatched students: {unmatched_students}; newly recoded values: {audit['newly_recoded_values']}")
    return data, audit


if __name__ == "__main__":
    main()
