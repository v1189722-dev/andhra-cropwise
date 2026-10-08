# CropWise AP

CropWise AP is a Streamlit prototype for crop recommendation and agricultural estimates across the 26 Andhra Pradesh districts.

## Features

- Recommends a crop from soil and weather inputs using a Random Forest model.
- Shows a historical yield value when crop and season records are available.
- Estimates market price with an XGBoost model.
- Displays a map highlighting the selected Andhra Pradesh district.

## Run the app

Open PowerShell in the project folder and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py