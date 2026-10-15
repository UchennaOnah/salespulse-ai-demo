# SalesPulse AI 2.0

A polished Streamlit sales analytics demonstration with synthetic data, interactive filters, category and regional breakdowns, deterministic business insights, and a Random Forest short-term revenue forecast.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Activation is optional. Open http://localhost:8501 after Streamlit starts.

## Deploy to your existing Streamlit app

1. Back up your current GitHub `salespulse-ai-demo` repository or create a new branch.
2. Upload **the contents** of this folder to the repository root, replacing `app.py`, `requirements.txt`, `README.md` and `sample_sales.csv`. Include the `assets/logo.svg` file in an `assets` directory. Do not upload `.venv`, `.env` or secrets.
3. Commit the changes to the branch configured in Streamlit Community Cloud. The existing app should redeploy automatically; check its build logs if it does not.
4. Keep the entrypoint set to `app.py`.

## Expected CSV format

`order_id,date,product,category,region,quantity,unit_price`

Dates should be parseable; quantity positive and unit_price nonnegative. One order can have multiple product lines.

## Forecasting limitations

The Random Forest uses calendar features, a chronological holdout and a simple 7-day-mean baseline. It is a demonstration model, not a validated commercial demand forecast. It does not model holidays, promotions, stockouts or exogenous variables. It may not outperform the baseline; the app reports this explicitly. Only use authorised, non-sensitive data in the public demo.

## Branding

The editable vector logo is `assets/logo.svg`. The `assets/design_preview.png` is an illustrative visual concept, not a screenshot of the running application.
