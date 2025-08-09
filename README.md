
# Marketing Campaign Analysis

An end-to-end, modular project for analyzing marketing campaign data with:
- Python package structure (`src/`)
- Logging & error handling
- Unit tests (pytest)
- Streamlit app (`app/streamlit_app.py`)
- Deployable on Streamlit Cloud

## Quickstart

```bash
# 1) Create & activate env (example with venv)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run tests
pytest

# 4) Run Streamlit app
streamlit run app/streamlit_app.py
```

## Project Structure

```
marketing-campaign-app/
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ logging_conf.py
│  ├─ data_loader.py
│  ├─ cleaning.py
│  ├─ features.py
│  └─ viz.py
├─ tests/
│  └─ test_cleaning.py
├─ data/                  # place local CSVs here (gitignored)
├─ notebooks/             # exploratory notebooks
├─ .streamlit/config.toml
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Configuration

- Place environment variables in `.env` (optional). See `sample.env`.
- Update defaults in `src/config.py`.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io, create a new app and point it to `app/streamlit_app.py` on `main`.
3. Add any secrets (e.g., DB connection strings) under **App settings → Secrets** as TOML.
4. Click **Deploy**.

## License

MIT
