# SalesPulse AI — Sales Analytics and Forecasting Demo

A Streamlit demo with interactive KPI reporting, category/region breakdowns, CSV upload, Random Forest short-term forecasting, holdout evaluation and downloadable reports.

## Run locally

1. Install Python 3.10 or newer and open this folder in VS Code.
2. Create a virtual environment:
   - Windows PowerShell: `py -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Run `python -m pip install -r requirements.txt`.
4. Run `python -m streamlit run app.py`.
5. Open the local address printed in the terminal (usually http://localhost:8501).

## Deploy using Streamlit Community Cloud

1. Create a new GitHub repository, such as `salespulse-ai-demo`.
2. Upload `app.py`, `requirements.txt`, `sample_sales.csv`, `.gitignore` and this README to the repository root. Do **not** upload real client data or secrets.
3. Sign in to https://share.streamlit.io/ and select **Create app** (the interface may change).
4. Choose your GitHub repository, branch and `app.py` as the entrypoint.
5. Deploy and test the public app link using the sample dataset.
6. If deployment fails, open the app logs and check dependency installation and paths.

## Input CSV schema

Required columns: `order_id,date,product,category,region,quantity,unit_price`.

- `date`: ISO date `YYYY-MM-DD` preferred.
- `quantity`: positive number.
- `unit_price`: nonnegative GBP amount, without currency symbol.
- Each row represents an order line; the same `order_id` can appear on multiple lines.
- `revenue` is calculated as `quantity * unit_price`; discounts, returns, VAT and shipping are not modelled.
- Forecasting requires at least 45 calendar days in the filtered date range.

## Model and limitations

A RandomForestRegressor learns calendar features (elapsed days, weekday, month, weekend). Data is aggregated to daily revenue with zero-filled missing dates. The most recent ~20% of dates are held out for evaluation. The model is refitted on the full series to produce a 7–30 day forecast. The app displays mean absolute error (MAE) and a basic seven-day-mean baseline error; it warns when the ML model is worse.

This is a demonstration, **not** a validated production forecasting service. It cannot model promotions, returns, changing stock, holidays, or causal drivers. Random forests may be poor at extrapolating long-term trends. Never claim forecasts are certain or that the app is GDPR-compliant by default.

## Demo script (3 minutes)

1. Show that the sample data is synthetic.
2. Filter by region and category; point out live KPI updates.
3. Open the business breakdown tab and identify the strongest category.
4. Show the forecast, holdout MAE and simple baseline comparison; explain limitations.
5. Download a summary CSV and ask the prospect how they currently prepare this report.

## Before working with real clients

Agree scope, ownership, hosting, access control, data protection responsibilities and maintenance. Streamlit Community Cloud public demo apps are not a suitable default for confidential client data. Use a private, secured deployment and authorised data-processing arrangements where required.
