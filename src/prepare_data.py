from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    filename: str,
) -> None:
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )


# 1. Clean the crop recommendation data
recommendation_path = RAW_DIR / "Crop_recommendation.csv"
recommendation = pd.read_csv(recommendation_path)
recommendation.columns = recommendation.columns.astype(str).str.strip()

recommendation_columns = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
]
require_columns(
    recommendation,
    set(recommendation_columns),
    recommendation_path.name,
)

recommendation = recommendation[recommendation_columns].copy()

for column in recommendation_columns[:-1]:
    recommendation[column] = pd.to_numeric(
        recommendation[column],
        errors="coerce",
    )

recommendation["label"] = (
    recommendation["label"]
    .astype("string")
    .str.strip()
    .str.lower()
)
recommendation = recommendation.dropna(subset=recommendation_columns)
recommendation.to_csv(
    PROCESSED_DIR / "crop_recommendation_clean.csv",
    index=False,
)


# 2. Clean Andhra Pradesh production data through 2014
production_path = RAW_DIR / "Crop Production data.csv"
production = pd.read_csv(production_path, low_memory=False)
production.columns = production.columns.astype(str).str.strip()

production_columns = {
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production",
}
require_columns(production, production_columns, production_path.name)

for column in ["State_Name", "District_Name", "Season", "Crop"]:
    production[column] = clean_text(production[column])

old_ap = production[
    production["State_Name"].str.casefold() == "andhra pradesh"
].copy()

old_ap["Crop_Year"] = pd.to_numeric(
    old_ap["Crop_Year"],
    errors="coerce",
)
old_ap["Area"] = pd.to_numeric(old_ap["Area"], errors="coerce")
old_ap["Production"] = pd.to_numeric(
    old_ap["Production"],
    errors="coerce",
)
old_ap = old_ap.dropna(
    subset=[
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
        "Production",
    ]
)
old_ap = old_ap[
    (old_ap["Crop_Year"] <= 2014)
    & (old_ap["Area"] > 0)
    & (old_ap["Production"] >= 0)
].copy()
old_ap["Data_Source"] = "Crop Production data.csv"


# 3. Add APY production rows from 2015–2019
apy_path = RAW_DIR / "APY.csv"
apy = pd.read_csv(apy_path, low_memory=False)
apy.columns = apy.columns.astype(str).str.strip()

apy_columns = {
    "State",
    "District",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production",
}
require_columns(apy, apy_columns, apy_path.name)

for column in ["State", "District", "Season", "Crop"]:
    apy[column] = clean_text(apy[column])

apy["Crop_Year"] = pd.to_numeric(apy["Crop_Year"], errors="coerce")
apy["Area"] = pd.to_numeric(apy["Area"], errors="coerce")
apy["Production"] = pd.to_numeric(
    apy["Production"],
    errors="coerce",
)

newer_ap = apy[
    (apy["State"].str.casefold() == "andhra pradesh")
    & apy["Crop_Year"].between(2015, 2019)
].copy()

newer_ap = newer_ap.dropna(
    subset=[
        "District",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
        "Production",
    ]
)
newer_ap = newer_ap[
    (newer_ap["Area"] > 0) & (newer_ap["Production"] >= 0)
].copy()

newer_ap = newer_ap.rename(columns={"District": "District_Name"})
newer_ap["State_Name"] = "Andhra Pradesh"
newer_ap["Data_Source"] = "APY.csv, 2015–2019"

production_district_aliases = {
    "SPSR NELLORE": "NELLORE",
    "VISAKHAPATANAM": "VISAKHAPATNAM",
    "Y.S.R.": "KADAPA",
    "Y.S.R": "KADAPA",
    "KADAPA": "KADAPA",
}


def standardize_production_district(value: object) -> str:
    name = str(value).strip().upper()
    return production_district_aliases.get(name, name)


old_ap["District_Name"] = old_ap["District_Name"].map(
    standardize_production_district
)
newer_ap["District_Name"] = newer_ap["District_Name"].map(
    standardize_production_district
)

production_ap = pd.concat(
    [old_ap, newer_ap],
    ignore_index=True,
    sort=False,
)
production_ap["Yield_tonnes_per_hectare"] = (
    production_ap["Production"] / production_ap["Area"]
)

production_ap.to_csv(
    PROCESSED_DIR / "andhra_pradesh_production_1997_2019.csv",
    index=False,
)

yield_lookup = (
    production_ap
    .groupby(["District_Name", "Crop", "Season"], as_index=False)
    .agg(
        Area_hectares=("Area", "sum"),
        Production_tonnes=("Production", "sum"),
        first_year=("Crop_Year", "min"),
        last_year=("Crop_Year", "max"),
        source_rows=("Crop_Year", "size"),
    )
)
yield_lookup["Yield_tonnes_per_hectare"] = (
    yield_lookup["Production_tonnes"] / yield_lookup["Area_hectares"]
)
yield_lookup.to_csv(
    PROCESSED_DIR / "andhra_pradesh_yield_lookup.csv",
    index=False,
)


# 4. Prepare Andhra Pradesh market prices, including July 2026 files
price_paths = [
    RAW_DIR / "market_prices" / "2025.csv",
    RAW_DIR / "market_prices" / "2026.csv",
    PROCESSED_DIR / "andhra_pradesh_prices_july_2026.csv",
]
price_parts = []

required_price_columns = {
    "State",
    "District",
    "Market",
    "Commodity",
    "Arrival_Date",
    "Modal_Price",
}

price_district_aliases = {
    "Anantapur": "Ananthapuramu",
    "Chittor": "Chittoor",
    "Cuddapah": "Kadapa",
    "Vijayanagaram": "Vizianagaram",
}

for price_path in price_paths:
    if not price_path.exists():
        print(f"Skipping missing file: {price_path.name}")
        continue

    # Read large price files in batches.
    for chunk in pd.read_csv(
        price_path,
        chunksize=200_000,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()
        require_columns(
            chunk,
            required_price_columns,
            price_path.name,
        )

        chunk["State"] = clean_text(chunk["State"])
        ap_chunk = chunk[
            chunk["State"].str.casefold() == "andhra pradesh"
        ].copy()

        if ap_chunk.empty:
            continue

        ap_chunk = ap_chunk.rename(
            columns={"Modal_Price": "Modal Price"}
        )

        for column in ["District", "Market", "Commodity"]:
            ap_chunk[column] = clean_text(ap_chunk[column])

        # Make spelling variants match the app's 26 district names.
        ap_chunk["District"] = ap_chunk["District"].replace(
            price_district_aliases
        )

        ap_chunk["Arrival_Date"] = pd.to_datetime(
            ap_chunk["Arrival_Date"],
            errors="coerce",
        )
        ap_chunk["Modal Price"] = pd.to_numeric(
            ap_chunk["Modal Price"],
            errors="coerce",
        )
        ap_chunk = ap_chunk.dropna(
            subset=[
                "District",
                "Market",
                "Commodity",
                "Arrival_Date",
                "Modal Price",
            ]
        )

        price_parts.append(
            ap_chunk[
                [
                    "State",
                    "District",
                    "Market",
                    "Commodity",
                    "Arrival_Date",
                    "Modal Price",
                ]
            ]
        )

if not price_parts:
    raise FileNotFoundError(
        "No Andhra Pradesh price rows were found in the configured files."
    )

ap_prices = pd.concat(price_parts, ignore_index=True)
ap_prices.to_csv(
    PROCESSED_DIR / "andhra_pradesh_prices_available.csv",
    index=False,
)


# 5. Print a summary
print(f"Recommendation rows prepared: {len(recommendation):,}")
print(f"AP production rows from 1997–2014: {len(old_ap):,}")
print(f"APY rows added for 2015–2019: {len(newer_ap):,}")
print(f"Combined AP production rows: {len(production_ap):,}")
print(f"Historic AP districts: {production_ap['District_Name'].nunique()}")
print(f"Available AP price rows from 2025–2026: {len(ap_prices):,}")
print(f"Price district names found: {ap_prices['District'].nunique()}")
print(f"Prepared files saved in: {PROCESSED_DIR}")