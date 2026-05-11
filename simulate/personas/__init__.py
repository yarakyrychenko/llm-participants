"""
simulate.personas — demographic persona database for human-ai-eval.

Quick start:
    import pandas as pd
    from simulate import load_prompts

    # Load all personas as a DataFrame
    import json, os
    with open(os.path.join(os.path.dirname(__file__), "personas.json")) as f:
        db = json.load(f)

    df = pd.DataFrame(db["personas"])

    # Filter by country
    us = df[df["country_code"] == "US"]
    india = df[df["country_code"] == "IN"]

    # Load international prompt templates
    prompts = load_prompts(os.path.join(os.path.dirname(__file__), "prompts.json"))

See build_personas.py to regenerate or extend the database.
See extract_attributes.py to extract demographics from free-text profiles.
"""

import json
import os
import string

_DB_PATH = os.path.join(os.path.dirname(__file__), "personas.json")
_US_DB_PATH = os.path.join(os.path.dirname(__file__), "us_personas.json")
_US_FULL_DB_PATH = os.path.join(os.path.dirname(__file__), "us_personas_full.json")
_GSS_DB_PATH = os.path.join(os.path.dirname(__file__), "gss_personas.csv")

__all__ = [
    "load_default_personas",
    "load_metadata",
    "load_gss_personas",
    "load_personas",
    "load_us_personas_full",
    "load_us_personas",
]


def load_personas(country_code: str = None):
    """
    Load persona database as a pandas DataFrame.

    Parameters
    ----------
    country_code : str, optional
        ISO 2-letter code to filter (e.g. 'US', 'IN', 'DE', 'BR', 'NG', 'JP').
        If None, returns all 180 personas.

    Returns
    -------
    pandas.DataFrame with columns: id, country, country_code, age, gender, education,
        employment_status, income_level, urban_rural, marital_status, religion,
        political_orientation, political_party, ethnicity, race, party
    """
    import pandas as pd
    with open(_DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    df = pd.DataFrame(db["personas"])
    if country_code:
        df = df[df["country_code"] == country_code.upper()].reset_index(drop=True)
    return df


def load_us_personas():
    """Load the packaged US persona database as a pandas DataFrame."""
    import pandas as pd

    with open(_US_DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    return pd.DataFrame(db["personas"])


def load_us_personas_full():
    """Load the full packaged US persona database as a pandas DataFrame."""
    import pandas as pd

    with open(_US_FULL_DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    return pd.DataFrame(db["personas"])


def load_gss_personas():
    """
    Load the packaged GSS persona database as a pandas DataFrame.

    The returned frame keeps the original GSS columns used by the `gss_prompt`
    template and adds common aliases used by sampling and legacy prompt helpers.
    """
    import pandas as pd

    df = pd.read_csv(_GSS_DB_PATH)
    df = df.copy()

    if "profile_id" in df.columns:
        df["gss_id"] = df["id"].astype(str)
        df["id"] = df["profile_id"].astype(str)
    else:
        df["id"] = df["id"].astype(str)

    def clean_title(value):
        if pd.isna(value):
            return value
        return string.capwords(str(value).strip())

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = df["sex"].map(clean_title)
    df["race"] = df["racecen1"].map(clean_title)
    df["region"] = df["region"].map(clean_title)
    df["education"] = df["degree"].map(clean_title)
    df["income_level"] = df["income16"]
    df["political_party"] = df["partyid"]
    df["party"] = df["partyid"]
    df["political_orientation"] = df["polviews"]
    df["religion"] = df["relig"]
    df["bio"] = ""
    df["country"] = "United States"
    df["country_code"] = "US"
    return df


def load_default_personas():
    """Return the default synthetic participant dataframe for simulations."""
    return load_us_personas()


def load_metadata() -> dict:
    """Return database metadata (sources, notes, field descriptions)."""
    with open(_DB_PATH, encoding="utf-8") as f:
        return json.load(f)["metadata"]
