"""
extract_attributes.py — Extract structured demographic attributes from free-text persona profiles.

Two modes:
1. LLM-based extraction (requires Anthropic API key or any OpenAI-compatible client)
2. Rule-based extraction with regex patterns (no API required, lower coverage)

Usage:
    from simulate.personas.extract_attributes import extract_attributes, extract_attributes_batch

    # Single profile (LLM mode)
    profile_text = "I'm a 34-year-old woman from Mumbai. I work as a software engineer..."
    attrs = extract_attributes(profile_text, client=client, model="claude-sonnet-4-6")

    # Batch (LLM mode)
    profiles = [...]
    results = extract_attributes_batch(profiles, client=client)

    # Rule-based (no API)
    attrs = extract_attributes_regex(profile_text)

Compatible with format_personalization() in simulate/agent.py — returned dict keys
match [TAG] placeholders: age, gender, education, party, race, political_orientation, etc.
"""

import re
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Target schema: fields returned by extract_attributes()
# All fields are optional (None if not found).
# ---------------------------------------------------------------------------

ATTRIBUTE_SCHEMA = {
    "age": "integer (18–90)",
    "gender": "string, one of: Man | Woman | Non-binary | Other",
    "education": "string, highest level completed, localized to country",
    "employment_status": "string: Employed full-time | Part-time | Self-employed | Student | Unemployed | Retired | Homemaker | Other",
    "income_level": "string: Low | Middle | High (relative to country of residence)",
    "urban_rural": "string: Urban | Suburban | Semi-urban | Rural",
    "marital_status": "string: Single | Married | Divorced | Widowed | Partnered",
    "religion": "string, specific tradition if mentioned",
    "political_orientation": "string: Far left | Left | Centre-left | Centre | Centre-right | Right | Far right | Apolitical",
    "political_party": "string, specific party if mentioned, else null",
    "ethnicity": "string, ethnic/racial identity as described",
    "country": "string, country of residence",
    "country_code": "string, ISO 2-letter country code",
}

EXTRACTION_SYSTEM_PROMPT = """\
You are a demographic attribute extractor. Given a free-text persona description,
extract structured demographic attributes in JSON format.

Return ONLY a valid JSON object with these keys (use null for unknown fields):
{schema}

Rules:
- age: integer only, infer from context if not explicit (e.g., "college student" → ~20)
- gender: normalize to Man / Woman / Non-binary / Other
- income_level: Low / Middle / High relative to the person's country
- political_orientation: use the 7-point scale (Far left → Far right) or Apolitical
- If a field cannot be determined, use null
- Do NOT invent information not present in the text
- Return only JSON, no explanation
""".format(schema=json.dumps(ATTRIBUTE_SCHEMA, indent=2))


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

def extract_attributes(
    profile_text: str,
    client=None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Extract demographic attributes from a free-text profile using an LLM.

    Parameters
    ----------
    profile_text : str
        Free-text persona description (e.g., "I'm a 34-year-old engineer from Mumbai...")
    client : OpenAI-compatible client (e.g., anthropic client via openai shim, or openai.OpenAI())
        Must support client.chat.completions.create(). If None, falls back to regex extraction.
    model : str
        Model identifier (default: claude-sonnet-4-6)

    Returns
    -------
    dict with keys from ATTRIBUTE_SCHEMA. Missing fields are None.
    Also adds backward-compat aliases: race (=ethnicity), party (=political_party).
    """
    if client is None:
        return extract_attributes_regex(profile_text)

    messages = [
        {"role": "user", "content": f"Extract demographic attributes from this profile:\n\n{profile_text}"}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=500,
            messages=messages,
            system=EXTRACTION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
    except TypeError:
        # Some clients take system as a separate param; try with messages format
        all_messages = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}] + messages
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=500,
            messages=all_messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

    try:
        attrs = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        attrs = json.loads(match.group()) if match else {}

    # Coerce age to int
    if attrs.get("age") is not None:
        try:
            attrs["age"] = int(attrs["age"])
        except (ValueError, TypeError):
            attrs["age"] = None

    # Add backward-compat aliases
    attrs.setdefault("race", attrs.get("ethnicity"))
    attrs.setdefault("party", attrs.get("political_party"))

    return attrs


def extract_attributes_batch(
    profiles: list[str],
    client=None,
    model: str = "claude-sonnet-4-6",
    verbose: bool = False,
) -> list[dict]:
    """
    Extract attributes from a list of free-text profiles.

    Parameters
    ----------
    profiles : list of str
    client : OpenAI-compatible client
    model : str
    verbose : bool — print progress

    Returns
    -------
    list of dicts (same order as input)
    """
    results = []
    for i, text in enumerate(profiles):
        if verbose:
            print(f"  Extracting {i+1}/{len(profiles)}...", end="\r")
        attrs = extract_attributes(text, client=client, model=model)
        results.append(attrs)
    if verbose:
        print(f"  Done. Extracted {len(results)} profiles.        ")
    return results


# ---------------------------------------------------------------------------
# Rule-based extraction (no API required)
# ---------------------------------------------------------------------------

# Regex patterns for common demographic mentions
_AGE_PATTERNS = [
    r'\b(\d{1,2})[- ]?year[s]?[- ]?old\b',
    r'\bage[d]?\s+(\d{1,2})\b',
    r'\bI(?:\'m| am) (\d{1,2})\b',
    r'\b(\d{1,2})[- ]?yo\b',
]

_GENDER_PATTERNS = {
    "Man": [r'\b(?:he|him|his)\b', r'\bman\b', r'\bgentleman\b', r'\bmale\b', r'\bfather\b', r'\bhusband\b', r'\bson\b', r'\bbrother\b'],
    "Woman": [r'\b(?:she|her|hers)\b', r'\bwoman\b', r'\blady\b', r'\bfemale\b', r'\bmother\b', r'\bwife\b', r'\bdaughter\b', r'\bsister\b'],
    "Non-binary": [r'\b(?:they|them|their)\b', r'\bnon.?binary\b', r'\bnonbinary\b', r'\benby\b', r'\bgenderqueer\b'],
}

_EDUCATION_KEYWORDS = {
    "Graduate degree": [r'\bphd\b', r'\bph\.d\b', r'\bdoctorate\b', r'\bmaster\'?s?\b', r'\bmba\b', r'\bmd\b', r'\bjd\b', r'\bgraduate degree\b', r'\bpostgrad\b'],
    "Bachelor's degree": [r'\bbachelor\'?s?\b', r'\bb\.?[as]\b', r'\buniversity graduate\b', r'\bcollege graduate\b'],
    "Some college": [r'\bsome college\b', r'\bcollege dropout\b', r'\bstudent\b', r'\buniversity student\b', r'\bcommunity college\b'],
    "High school diploma / GED": [r'\bhigh school\b', r'\bged\b', r'\bsecondary school\b', r'\b12th grade\b'],
    "Less than high school": [r'\bdropped out\b', r'\belementary\b', r'\bprimary school\b', r'\bno formal education\b'],
}

_POLITICAL_KEYWORDS = {
    "Very liberal": [r'\bvery liberal\b', r'\bfar left\b', r'\bsocialist\b', r'\bmarxist\b'],
    "Liberal": [r'\bliberal\b', r'\bprogressive\b', r'\bleft.?wing\b', r'\bdemocrat\b', r'\bleft\b'],
    "Moderate": [r'\bmoderate\b', r'\bcentrist\b', r'\bindependent\b', r'\bcentre\b', r'\bcenter\b'],
    "Conservative": [r'\bconservative\b', r'\bright.?wing\b', r'\brepublican\b', r'\bright\b'],
    "Very conservative": [r'\bvery conservative\b', r'\bfar right\b', r'\bextreme right\b'],
}

_EMPLOYMENT_KEYWORDS = {
    "Student": [r'\bstudent\b', r'\bstudying\b', r'\bin college\b', r'\bin university\b'],
    "Retired": [r'\bretired\b', r'\bpensioner\b', r'\bretirement\b'],
    "Unemployed": [r'\bunemployed\b', r'\bout of work\b', r'\bjobless\b', r'\blooking for work\b'],
    "Self-employed": [r'\bself.?employed\b', r'\bbusiness owner\b', r'\bfreelancer\b', r'\bentrepreneur\b', r'\brun(?:ning)? my own\b'],
    "Homemaker": [r'\bhomemaker\b', r'\bstay.?at.?home\b', r'\bnot working\b'],
    "Employed full-time": [r'\bwork(?:s|ing)? full.?time\b', r'\bfull.?time employee\b', r'\bfull.?time job\b'],
    "Part-time": [r'\bpart.?time\b'],
}

_URBAN_RURAL_KEYWORDS = {
    "Urban": [r'\bcity\b', r'\burban\b', r'\bmetropolis\b', r'\bdowntown\b', r'\bcitydweller\b'],
    "Suburban": [r'\bsuburb\b', r'\bsuburban\b', r'\boutskirts\b'],
    "Rural": [r'\brural\b', r'\bcountryside\b', r'\bfarm\b', r'\bsmall town\b', r'\bvillage\b'],
}

_MARITAL_KEYWORDS = {
    "Single": [r'\bsingle\b', r'\bnever married\b', r'\bnot married\b'],
    "Married": [r'\bmarried\b', r'\bspouse\b', r'\bwife\b', r'\bhusband\b', r'\bpartner\b'],
    "Divorced": [r'\bdivorced\b', r'\bseparated\b', r'\bex.?wife\b', r'\bex.?husband\b'],
    "Widowed": [r'\bwidowed\b', r'\bwidow\b', r'\bwidower\b'],
}

_COUNTRY_KEYWORDS = {
    "United States": [r'\bUSA\b', r'\bUnited States\b', r'\bAmerica\b', r'\bAmerican\b', r'\bU\.S\.\b'],
    "India": [r'\bIndia\b', r'\bIndian\b', r'\bDelhiite\b', r'\bMumbaikar\b', r'\bBengali\b'],
    "Germany": [r'\bGermany\b', r'\bGerman\b', r'\bDeutschland\b', r'\bBerlin\b', r'\bMunich\b'],
    "Brazil": [r'\bBrazil\b', r'\bBrazilian\b', r'\bBrasil\b', r'\bSão Paulo\b', r'\bRio\b'],
    "Nigeria": [r'\bNigeria\b', r'\bNigerian\b', r'\bLagos\b', r'\bAbuja\b', r'\bKano\b'],
    "Japan": [r'\bJapan\b', r'\bJapanese\b', r'\bTokyo\b', r'\bOsaka\b', r'\bKyoto\b'],
}

_COUNTRY_CODES = {
    "United States": "US", "India": "IN", "Germany": "DE",
    "Brazil": "BR", "Nigeria": "NG", "Japan": "JP",
}


def extract_attributes_regex(profile_text: str) -> dict:
    """
    Rule-based demographic extraction using regex patterns.
    Lower coverage than LLM-based extraction but requires no API.

    Returns dict with same keys as extract_attributes(). None for undetected fields.
    """
    text = profile_text.lower()

    # Age
    age = None
    for pattern in _AGE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                candidate = int(match.group(1))
                if 10 <= candidate <= 100:
                    age = candidate
                    break
            except (ValueError, IndexError):
                pass

    # Gender
    gender = None
    for label, patterns in _GENDER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                gender = label
                break
        if gender:
            break

    # Education
    education = None
    for level, patterns in _EDUCATION_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                education = level
                break
        if education:
            break

    # Political orientation
    political_orientation = None
    for orientation, patterns in _POLITICAL_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                political_orientation = orientation
                break
        if political_orientation:
            break

    # Employment
    employment_status = None
    for status, patterns in _EMPLOYMENT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                employment_status = status
                break
        if employment_status:
            break

    # Urban/rural
    urban_rural = None
    for label, patterns in _URBAN_RURAL_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                urban_rural = label
                break
        if urban_rural:
            break

    # Marital status
    marital_status = None
    for status, patterns in _MARITAL_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                marital_status = status
                break
        if marital_status:
            break

    # Country
    country = None
    country_code = None
    for name, patterns in _COUNTRY_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, profile_text, re.IGNORECASE):
                country = name
                country_code = _COUNTRY_CODES.get(name)
                break
        if country:
            break

    attrs = {
        "age": age,
        "gender": gender,
        "education": education,
        "employment_status": employment_status,
        "income_level": None,
        "urban_rural": urban_rural,
        "marital_status": marital_status,
        "religion": None,
        "political_orientation": political_orientation,
        "political_party": None,
        "ethnicity": None,
        "country": country,
        "country_code": country_code,
    }

    # Backward-compat aliases
    attrs["race"] = attrs["ethnicity"]
    attrs["party"] = attrs["political_party"]

    return attrs


# ---------------------------------------------------------------------------
# Utility: merge extracted attributes into an existing persona dict
# ---------------------------------------------------------------------------

def merge_attributes(persona: dict, extracted: dict, overwrite: bool = False) -> dict:
    """
    Merge extracted attributes into an existing persona dict.

    Parameters
    ----------
    persona : dict — existing persona (e.g., from personas.json)
    extracted : dict — result of extract_attributes()
    overwrite : bool — if True, extracted values overwrite existing ones;
                       if False (default), only fill in None/missing fields

    Returns
    -------
    dict — updated persona
    """
    result = dict(persona)
    for key, value in extracted.items():
        if value is not None:
            if overwrite or result.get(key) is None:
                result[key] = value
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_profiles = [
        "I'm a 34-year-old woman living in Chicago. I work as a nurse and have a bachelor's degree. "
        "I'm married with two kids and consider myself a moderate Democrat.",

        "27 year old male, student at IIT Delhi, single. Hindu. Supports BJP.",

        "Rentner, 68 Jahre alt, aus Bayern. Katholisch, CDU-Wähler. Wohne in einem kleinen Dorf.",

        "Sou uma mulher de 42 anos de São Paulo. Trabalho como professora. Evangélica, apoiei o PT nas últimas eleições.",
    ]

    print("Rule-based extraction (no API required):")
    print("-" * 50)
    for profile in sample_profiles:
        attrs = extract_attributes_regex(profile)
        # Show only non-None values
        found = {k: v for k, v in attrs.items() if v is not None and k not in ("race", "party")}
        print(f"Profile: {profile[:60]}...")
        print(f"Extracted: {json.dumps(found, indent=2)}")
        print()
