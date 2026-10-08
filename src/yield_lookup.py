from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "andhra_pradesh_production_1997_2019.csv"
)

DISTRICT_ALIASES = {
    "Ananthapuramu": "ANANTAPUR",
    "Kadapa": "KADAPA",
    "Nellore": "NELLORE",
}

CROP_ALIASES = {
    "rice": ["rice", "paddy"],
    "maize": ["maize", "corn"],
    "chickpea": ["chickpea", "bengal gram"],
    "kidneybeans": ["kidney bean", "rajma"],
    "pigeonpeas": ["pigeon pea", "pigeonpeas", "arhar/tur", "red gram"],
    "mothbeans": ["moth bean"],
    "mungbean": ["mung bean", "green gram", "moong"],
    "blackgram": ["black gram", "urd"],
    "lentil": ["lentil", "masoor"],
    "pomegranate": ["pomegranate"],
    "banana": ["banana"],
    "mango": ["mango"],
    "grapes": ["grape", "grapes"],
    "watermelon": ["watermelon"],
    "muskmelon": ["muskmelon"],
    "apple": ["apple"],
    "orange": ["orange"],
    "papaya": ["papaya"],
    "coconut": ["coconut"],
    "cotton": ["cotton", "kapas"],
    "jute": ["jute"],
    "coffee": ["coffee"],
}


def normalize(value: object) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def historical_district_name(current_district: str) -> str:
    for current_name, historic_name in DISTRICT_ALIASES.items():
        if normalize(current_district) == normalize(current_name):
            return historic_name

    return current_district.strip().upper()


def get_yield_comparison(
    district: str,
    crop: str,
    season: str,
) -> dict[str, float | str | None]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Production data not found: {DATA_PATH}. "
            "Run src\\prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    for column in ["District_Name", "Crop", "Season"]:
        data[column] = data[column].astype("string").str.strip()

    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data["Production"] = pd.to_numeric(data["Production"], errors="coerce")
    data = data.dropna(
        subset=["District_Name", "Crop", "Season", "Area", "Production"]
    )
    data = data[data["Area"] > 0]

    crop_key = normalize(crop)
    crop_terms = CROP_ALIASES.get(crop_key, [crop_key])
    crop_terms = [normalize(term) for term in crop_terms]

    normalized_crops = data["Crop"].map(normalize)
    crop_rows = data[
        normalized_crops.map(
            lambda name: any(term in name for term in crop_terms)
        )
    ]

    if crop_rows.empty:
        if crop_key == "coffee":
            return {
                "district_yield_tonnes_per_hectare": None,
                "ap_historical_average_tonnes_per_hectare": 0.179,
                "message": (
                    "Coffee Board AP-wide productivity benchmark: "
                    "179 kg/ha (0.179 t/ha). This is a statewide reference, "
                    "not a district-specific Coffee yield."
                ),
            }

        return {
            "district_yield_tonnes_per_hectare": None,
            "ap_historical_average_tonnes_per_hectare": None,
            "message": (
                f"No historic production records were found for {crop}. "
                "The available data cannot calculate a crop-specific yield."
            ),
        }

    season_rows = crop_rows[
        crop_rows["Season"].map(normalize).eq(normalize(season))
    ]

    if season_rows.empty:
        average_rows = crop_rows
        average_message = (
            f"No {season} records were found for {crop}; "
            "showing the AP historical average for this crop across seasons."
        )
        district_yield = None
    else:
        average_rows = season_rows
        average_message = (
            f"No district-specific {crop} {season} yield was found; "
            "showing the AP historical average for this crop and season."
        )

        historic_name = normalize(historical_district_name(district))
        district_rows = season_rows[
            season_rows["District_Name"].map(normalize).eq(historic_name)
        ]

        if district_rows.empty:
            district_yield = None
        else:
            district_yield = (
                district_rows["Production"].sum()
                / district_rows["Area"].sum()
            )

    ap_average = (
        average_rows["Production"].sum()
        / average_rows["Area"].sum()
    )

    if district_yield is not None:
        message = (
            "District yield is calculated from historical production "
            "divided by area."
        )
    else:
        message = average_message

    return {
        "district_yield_tonnes_per_hectare": (
            round(district_yield, 3)
            if district_yield is not None
            else None
        ),
        "ap_historical_average_tonnes_per_hectare": round(ap_average, 3),
        "message": message,
    }


if __name__ == "__main__":
    result = get_yield_comparison(
        district="Guntur",
        crop="Rice",
        season="Kharif",
    )
    print(result)