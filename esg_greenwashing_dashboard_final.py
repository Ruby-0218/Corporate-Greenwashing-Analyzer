from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote_plus

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import yfinance as yf
except Exception:
    yf = None


# 0. Basic settings
st.set_page_config(
    layout="wide",
    page_title="Corporate Greenwashing Analyzer",
)
alt.data_transformers.disable_max_rows()



# 1. CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap');

    :root {
        --primary-color: #2f4a5f;
        --text-main: #263746;
        --text-muted: #5d6f7e;
        --line: #e0e0e0;
        --danger: #c94c4c;
        --soft-bg: #f7f8fa;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: Georgia, 'Times New Roman', serif !important;
        color: var(--text-main);
    }

    .stApp { background: #ffffff; }

    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

    .main .block-container {
        padding-top: 3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 1450px;
    }

    .stApp h1 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: clamp(2.4rem, 5vw, 4.2rem) !important;
        font-weight: 700 !important;
        line-height: 1.08 !important;
        color: #1a1a1a !important;
        margin-bottom: 0.8rem !important;
    }

    .stApp h2 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: clamp(1.75rem, 3vw, 2.65rem) !important;
        font-weight: 500 !important;
        color: #1a1a1a !important;
        margin-top: 2.2rem !important;
        margin-bottom: 0.9rem !important;
    }

    .stApp h3 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        margin-top: 1.6rem !important;
    }

    .stApp p, .stMarkdown p, .stMarkdown li {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: 1.04rem !important;
        color: #555555 !important;
        line-height: 1.7 !important;
        max-width: 920px;
    }

    .project-author {
        color: #5d6f7e;
        font-family: 'Roboto', sans-serif;
        font-size: 0.95rem;
        margin-top: -0.35rem;
        margin-bottom: 2.1rem;
    }

    .intro-box {
        background: linear-gradient(135deg, #f8fafb 0%, #ffffff 100%);
        border: 1px solid #e1e6eb;
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        margin: 1rem 0 2rem 0;
        max-width: 1100px;
        box-shadow: 0 8px 22px rgba(38, 55, 70, 0.04);
    }
    .intro-box p { margin-bottom: 0 !important; max-width: 1040px !important; }

    /* top navigation: equal-size square-like tabs */
    .stApp div[data-testid="stHorizontalBlock"]:has(button[kind="primary"], button[kind="secondary"]) {
        align-items: stretch !important;
        border-bottom: 1px solid #d9d9d9 !important;
        gap: 0 !important;
        margin-bottom: 2.2rem !important;
        overflow: visible !important;
        position: relative !important;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(button[kind="primary"], button[kind="secondary"])::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 1px;
        box-shadow: 0 -6px 9px -4px rgba(0, 0, 0, 0.35);
        pointer-events: none;
    }

    button[kind="primary"], button[data-testid="baseButton-primary"] {
        height: 64px !important;
        min-height: 64px !important;
        max-height: 64px !important;
        width: 100% !important;
        background-color: #e6e6e6 !important;
        border: 1px solid #bdbdbd !important;
        border-radius: 0 !important;
        box-shadow: 6px 0 8px -6px rgba(0, 0, 0, 0.42) !important;
        color: #263746 !important;
        justify-content: flex-start !important;
        align-items: center !important;
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        margin-bottom: -1px !important;
        padding: 0.75rem 0.9rem !important;
        text-align: left !important;
        white-space: normal !important;
    }
    button[kind="secondary"], button[data-testid="baseButton-secondary"] {
        height: 64px !important;
        min-height: 64px !important;
        max-height: 64px !important;
        width: 100% !important;
        border: 1px solid #d9d9d9 !important;
        background: #ffffff !important;
        color: #555555 !important;
        justify-content: flex-start !important;
        align-items: center !important;
        font-family: Georgia, 'Times New Roman', serif !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        margin-bottom: -1px !important;
        padding: 0.75rem 0.9rem !important;
        border-radius: 0 !important;
        box-shadow: 6px 0 8px -6px rgba(0, 0, 0, 0.35) !important;
        text-align: left !important;
        white-space: normal !important;
    }
    button[kind="secondary"]:hover {
        border-color: #b7c4cf !important;
        color: #2f4a5f !important;
        background: #f5f5f5 !important;
    }

    /* equal KPI cards */
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e1e5ea;
        border-radius: 12px;
        padding: 1.15rem 1.15rem;
        height: 150px;
        min-height: 150px;
        max-height: 150px;
        width: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .metric-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(2rem, 3.2vw, 3rem);
        font-weight: 700;
        color: #263746;
        line-height: 1;
        white-space: nowrap;
    }
    .metric-label {
        font-family: 'Roboto', sans-serif;
        color: #667789;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.65rem;
        line-height: 1.35;
        min-height: 2.1em;
    }
    .metric-note {
        font-family: Georgia, 'Times New Roman', serif;
        color: #6f7782;
        font-size: 0.86rem;
        line-height: 1.25;
        margin-top: 0.45rem;
    }

    /* designed chart explanation card */
    .insight-card {
        background: linear-gradient(135deg, #f8fafb 0%, #ffffff 100%);
        border: 1px solid #e1e6eb;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-top: 0.75rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 22px rgba(38, 55, 70, 0.05);
        max-width: 1050px;
    }
    .insight-eyebrow {
        font-family: 'Roboto', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #5d6f7e;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .insight-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.18rem;
        color: #263746;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .insight-card p {
        margin-bottom: 0 !important;
        font-size: 0.98rem !important;
        color: #555555 !important;
        line-height: 1.65 !important;
        max-width: 980px !important;
    }

    .next-page-wrap { margin-top: 2rem; margin-bottom: 1rem; }
    .next-page-link {
        display: inline-block;
        border: 1px solid #d9d9d9;
        background: #ffffff;
        color: #5d6f7e !important;
        font-family: 'Roboto', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }
    .next-page-link:hover {
        border-color: #b7c4cf;
        color: #2f4a5f !important;
        background: #f5f5f5;
    }

    hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# 2. Helper functions
def render_top_navigation(pages: list[str], active_page: str):
    nav_cols = st.columns(len(pages), gap="small")
    for idx, nav_page in enumerate(pages):
        button_type = "primary" if nav_page == active_page else "secondary"
        if nav_cols[idx].button(nav_page, key=f"top_nav_{idx}", type=button_type, use_container_width=True):
            if nav_page != active_page:
                st.query_params["page"] = nav_page
                st.rerun()


def render_next_page_button(next_page: str):
    href = f"?page={quote_plus(next_page)}#top"
    st.markdown(
        f"""
        <div class="next-page-wrap">
            <a class="next-page-link" href="{href}" target="_self">Next → {escape(next_page)}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value, label, note=None):
    note_html = f'<div class="metric-note">{escape(str(note))}</div>' if note else '<div class="metric-note">&nbsp;</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-value">{escape(str(value))}</div>
        <div class="metric-label">{escape(str(label))}</div>
        {note_html}
    </div>
    """


def insight(title: str, text: str, eyebrow: str = "Dashboard Insight"):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-eyebrow">{escape(eyebrow)}</div>
            <div class="insight-title">{escape(title)}</div>
            <p>{escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    try:
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.{digits}f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.{digits}f}K"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"


def pct(x, digits=1):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def find_data_file():
    possible_paths = [
        Path("data/esg_greenwashing_energy_utilities_industrials_2010_2024.csv"),
        Path("esg_greenwashing_energy_utilities_industrials_2010_2024.csv"),
        Path("./data/esg_greenwashing_energy_utilities_industrials_2010_2024.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    st.error("找不到資料檔。請確認 CSV 放在：data/esg_greenwashing_energy_utilities_industrials_2010_2024.csv")
    st.stop()


def get_first_existing_col(df, candidates):
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for col in cols:
            if cand in col:
                return col
    return None


def normalize_binary(series):
    if series is None:
        return 0
    s = series.copy()
    if s.dtype == "O":
        s = s.astype(str).str.strip().str.lower()
        return s.map(
            {
                "1": 1,
                "yes": 1,
                "y": 1,
                "true": 1,
                "signed": 1,
                "committed": 1,
                "verified": 1,
                "0": 0,
                "no": 0,
                "n": 0,
                "false": 0,
                "none": 0,
                "not signed": 0,
                "not committed": 0,
                "not verified": 0,
            }
        ).fillna(0).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


@st.cache_data
def load_and_prepare_data():
    data_path = find_data_file()
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]

    year_col = get_first_existing_col(df, ["year", "calendar_year"])
    company_col = get_first_existing_col(df, ["company", "company_name", "name"])
    ticker_col = get_first_existing_col(df, ["ticker", "symbol"])
    sector_col = get_first_existing_col(df, ["sector", "industry"])
    country_col = get_first_existing_col(df, ["country", "headquarters_country"])
    revenue_col = get_first_existing_col(df, ["revenue_usd_bn", "revenue"])
    esg_col = get_first_existing_col(df, ["esg_score", "esg"])
    greenwash_col = get_first_existing_col(df, ["greenwashing_flag", "greenwash", "flag"])

    s1_col = get_first_existing_col(df, ["scope1_emissions_mt_co2e", "scope1"])
    s2_col = get_first_existing_col(df, ["scope2_emissions_mt_co2e", "scope2"])
    s3_col = get_first_existing_col(df, ["scope3_emissions_mt_co2e", "scope3"])
    s1s2_col = get_first_existing_col(df, ["total_s1_s2_mt_co2e", "total_s1_s2", "scope1_scope2"])

    netzero_col = get_first_existing_col(df, ["net_zero_target", "net_zero", "netzero"])
    sbti_col = get_first_existing_col(df, ["sbti_committed", "sbti"])
    verified_col = get_first_existing_col(df, ["third_party_verified", "third_party_verification", "verified", "verification"])
    cdp_col = get_first_existing_col(df, ["cdp_climate_rating", "cdp_rating", "cdp"])

    df["year"] = safe_numeric(df[year_col]) if year_col else np.nan
    df["company"] = df[company_col].astype(str) if company_col else "Unknown"
    df["sector"] = df[sector_col].astype(str) if sector_col else "Unknown"
    df["country"] = df[country_col].astype(str) if country_col else "Unknown"

    if ticker_col:
        df["ticker"] = df[ticker_col].astype(str)
    else:
        df["ticker"] = np.nan

    ticker_map = {
        "3m": "MMM", "american electric power": "AEP", "arcelormittal": "MT", "basf": "BASFY",
        "bp": "BP", "caterpillar": "CAT", "chevron": "CVX", "conocophillips": "COP",
        "cummins": "CMI", "dominion energy": "D", "duke energy": "DUK", "e.on": "EONGY",
        "emerson electric": "EMR", "enel": "ENLAY", "eni": "E", "equinor": "EQNR",
        "exxonmobil": "XOM", "general electric": "GE", "holcim": "HCMLY", "honeywell": "HON",
        "iberdrola": "IBDRY", "nextera energy": "NEE", "occidental petroleum": "OXY",
        "pioneer natural resources": "PXD", "rwe": "RWEOY", "shell": "SHEL", "siemens": "SIEGY",
        "southern company": "SO", "totalenergies": "TTE", "xcel energy": "XEL",
    }
    mapped_ticker = df["company"].str.lower().map(ticker_map)
    df["ticker"] = df["ticker"].replace(["nan", "None", ""], np.nan).fillna(mapped_ticker)

    df["revenue_usd_bn"] = safe_numeric(df[revenue_col]) if revenue_col else np.nan
    df["esg_score"] = safe_numeric(df[esg_col]) if esg_col else np.nan
    df["greenwashing_flag"] = normalize_binary(df[greenwash_col]) if greenwash_col else 0

    df["scope1_emissions"] = safe_numeric(df[s1_col]) if s1_col else np.nan
    df["scope2_emissions"] = safe_numeric(df[s2_col]) if s2_col else np.nan
    df["scope3_emissions"] = safe_numeric(df[s3_col]) if s3_col else np.nan

    if s1s2_col:
        df["total_s1_s2_emissions"] = safe_numeric(df[s1s2_col])
    else:
        df["total_s1_s2_emissions"] = df["scope1_emissions"].fillna(0) + df["scope2_emissions"].fillna(0)

    df["total_emissions"] = (
        df["scope1_emissions"].fillna(0)
        + df["scope2_emissions"].fillna(0)
        + df["scope3_emissions"].fillna(0)
    )
    if df["total_emissions"].sum(skipna=True) == 0:
        df["total_emissions"] = df["total_s1_s2_emissions"]

    df["net_zero_target"] = normalize_binary(df[netzero_col]) if netzero_col else 0
    df["sbti_committed"] = normalize_binary(df[sbti_col]) if sbti_col else 0
    df["third_party_verified"] = normalize_binary(df[verified_col]) if verified_col else 0
    df["cdp_rating"] = df[cdp_col].astype(str) if cdp_col else "N/A"

    df["carbon_intensity"] = df["total_emissions"] / df["revenue_usd_bn"].replace(0, np.nan)
    df["operational_carbon_intensity"] = df["total_s1_s2_emissions"] / df["revenue_usd_bn"].replace(0, np.nan)
    df["scope3_share"] = df["scope3_emissions"] / df["total_emissions"].replace(0, np.nan)

    df = df.sort_values(["company", "year"]).reset_index(drop=True)
    df["emissions_yoy_change"] = df.groupby("company")["total_emissions"].pct_change()
    df["esg_yoy_change"] = df.groupby("company")["esg_score"].diff()
    df["carbon_intensity_yoy_change"] = df.groupby("company")["carbon_intensity"].pct_change()

    for col in ["emissions_yoy_change", "carbon_intensity_yoy_change"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    emission_reduction_score = (50 - df["emissions_yoy_change"].fillna(0) * 100).clip(0, 100)
    df["credibility_score"] = (
        0.35 * df["esg_score"].fillna(df["esg_score"].median())
        + 0.20 * emission_reduction_score
        + 12 * df["sbti_committed"]
        + 10 * df["net_zero_target"]
        + 10 * df["third_party_verified"]
        - 18 * df["greenwashing_flag"]
    ).clip(0, 100)

    df["greenwashing_label"] = np.where(df["greenwashing_flag"] == 1, "Flagged", "Clean")
    df["commitment_group"] = np.where(
        (df["net_zero_target"] == 1) | (df["sbti_committed"] == 1),
        "Has climate commitment",
        "No climate commitment",
    )

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


df = load_and_prepare_data()


# 3. Header and routing
pages = [
    "Overview",
    "Emissions Intelligence",
    "Commitment Tracker",
    "Greenwashing Explorer",
    "Sector Benchmarking",
    "Prediction Model",
    "Live Financials",
]

query_page = st.query_params.get("page")
page = query_page if query_page in pages else pages[0]
if query_page != page:
    st.query_params["page"] = page

st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown("<h1>Corporate Greenwashing Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="project-author">ESG Intelligence Dashboard | Energy, Utilities & Industrials, 2010–2024</div>',
    unsafe_allow_html=True,
)
render_top_navigation(pages, page)


def filter_data_interactive(base_df, key_prefix="global"):
    min_year = int(base_df["year"].min())
    max_year = int(base_df["year"].max())
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.4, 1.4])
    with c1:
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year), key=f"{key_prefix}_year_range")
    with c2:
        sectors = sorted(base_df["sector"].dropna().unique().tolist())
        selected_sectors = st.multiselect("Sector", sectors, default=sectors, key=f"{key_prefix}_sector")
    with c3:
        countries = sorted(base_df["country"].dropna().unique().tolist())
        selected_countries = st.multiselect("Country", countries, default=countries, key=f"{key_prefix}_country")
    with c4:
        flag_choice = st.selectbox("Greenwashing status", ["All", "Flagged only", "Clean only"], key=f"{key_prefix}_flag")

    filtered = base_df[
        (base_df["year"].between(year_range[0], year_range[1]))
        & (base_df["sector"].isin(selected_sectors))
        & (base_df["country"].isin(selected_countries))
    ].copy()
    if flag_choice == "Flagged only":
        filtered = filtered[filtered["greenwashing_flag"] == 1]
    elif flag_choice == "Clean only":
        filtered = filtered[filtered["greenwashing_flag"] == 0]
    return filtered


# Page 1: Overview
if page == "Overview":
    st.markdown(
        """
        <div class="intro-box">
        <p>
        Many companies report stronger ESG performance over time, but climate credibility depends on whether these improvements are matched by actual emissions reductions. 
        This dashboard analyzes the gap between ESG scores, climate commitments, Scope 1/2/3 emissions, and greenwashing flags.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered_df = filter_data_interactive(df, "overview")
    total_records = len(filtered_df)
    company_count = filtered_df["company"].nunique()
    flagged_records = int(filtered_df["greenwashing_flag"].sum())
    flag_rate = flagged_records / total_records if total_records > 0 else 0
    avg_esg = filtered_df["esg_score"].mean()
    total_emissions = filtered_df["total_emissions"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(metric_card(company_count, "Companies"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card(total_records, "Company-year observations"), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card(flagged_records, "Flagged records", pct(flag_rate)), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card(format_num(avg_esg, 1), "Average ESG score"), unsafe_allow_html=True)
    with k5:
        st.markdown(metric_card(format_num(total_emissions, 1), "Total emissions Mt CO₂e"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## ESG Score Is Rising — But Are Emissions Falling?")
    insight(
        title="Are ESG scores improving faster than real emissions performance?",
        text=(
            "This section compares average ESG scores with average total emissions over time. "
            "If ESG scores rise while emissions remain flat or increase, the result may suggest a gap between corporate ESG disclosure and actual climate performance."
        ),
        eyebrow="Analysis Purpose",
    )

    yearly = (
        filtered_df.groupby("year", as_index=False)
        .agg(
            avg_esg=("esg_score", "mean"),
            avg_total_emissions=("total_emissions", "mean"),
            avg_carbon_intensity=("carbon_intensity", "mean"),
            greenwashing_rate=("greenwashing_flag", "mean"),
        )
        .dropna(subset=["year"])
    )
    c1, c2 = st.columns(2)
    with c1:
        esg_line = (
            alt.Chart(yearly)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("avg_esg:Q", title="Average ESG Score", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("avg_esg:Q", title="Avg ESG", format=".2f"), alt.Tooltip("greenwashing_rate:Q", title="Greenwashing Rate", format=".1%")],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(esg_line, use_container_width=True)
    with c2:
        emissions_line = (
            alt.Chart(yearly)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("avg_total_emissions:Q", title="Average Total Emissions"),
                tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("avg_total_emissions:Q", title="Avg Emissions", format=".2f"), alt.Tooltip("avg_carbon_intensity:Q", title="Avg Carbon Intensity", format=".2f")],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(emissions_line, use_container_width=True)

    st.markdown("## Greenwashing Heatmap")
    insight(
        title="Which firms are repeatedly flagged for greenwashing?",
        text=(
            "Each cell represents one company-year observation. Red cells indicate flagged records, while blue cells indicate clean records. "
            "Companies with repeated red cells may require deeper due diligence because the signal appears persistent rather than temporary."
        ),
        eyebrow="How to Read This Chart",
    )
    heatmap_df = filtered_df.dropna(subset=["company", "year"]).copy()
    if len(heatmap_df) > 0:
        heatmap = (
            alt.Chart(heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("company:N", title=None, sort=alt.EncodingSortField(field="greenwashing_flag", op="sum", order="descending")),
                color=alt.Color("greenwashing_label:N", title="Status", scale=alt.Scale(domain=["Clean", "Flagged"], range=["#d9e6ef", "#c94c4c"])),
                tooltip=["company:N", "sector:N", "country:N", "year:O", "greenwashing_label:N", alt.Tooltip("esg_score:Q", title="ESG", format=".1f"), alt.Tooltip("total_emissions:Q", title="Total emissions", format=".2f")],
            )
            .properties(height=520)
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

    st.markdown("## ESG Paradox Map")
    insight(
        title="High ESG score does not always mean low emissions.",
        text=(
            "This scatter plot compares ESG score against total emissions. Companies in the upper-right area combine high ESG scores with high emissions, "
            "which may indicate a potential ESG credibility gap. Bubble size represents revenue, and red points represent flagged observations."
        ),
        eyebrow="Chart Interpretation",
    )
    year_options = sorted(filtered_df["year"].dropna().astype(int).unique().tolist())
    if year_options:
        selected_year = st.select_slider("Select year for ESG vs emissions comparison", options=year_options, value=max(year_options))
        scatter_df = filtered_df[filtered_df["year"] == selected_year].dropna(subset=["esg_score", "total_emissions", "revenue_usd_bn"])
        if len(scatter_df) > 0:
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(opacity=0.78)
                .encode(
                    x=alt.X("esg_score:Q", title="ESG Score", scale=alt.Scale(zero=False)),
                    y=alt.Y("total_emissions:Q", title="Total Emissions Mt CO₂e"),
                    size=alt.Size("revenue_usd_bn:Q", title="Revenue USD bn", scale=alt.Scale(range=[80, 1200])),
                    color=alt.Color("greenwashing_label:N", title="Greenwashing", scale=alt.Scale(domain=["Clean", "Flagged"], range=["#4E79A7", "#E15759"])),
                    tooltip=["company:N", "sector:N", "country:N", "year:O", alt.Tooltip("esg_score:Q", title="ESG", format=".1f"), alt.Tooltip("total_emissions:Q", title="Total emissions", format=".2f"), alt.Tooltip("revenue_usd_bn:Q", title="Revenue USD bn", format=".2f"), "greenwashing_label:N"],
                )
                .properties(height=470)
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("No valid observations for the selected year.")

    render_next_page_button("Emissions Intelligence")


# Page 2: Emissions Intelligence
elif page == "Emissions Intelligence":
    st.markdown("## Emissions Intelligence")
    st.markdown("Scope 1, Scope 2, and Scope 3 emissions reveal very different climate stories. In many Energy companies, Scope 3 emissions dominate because downstream use of sold products carries the largest climate impact.")
    filtered_df = filter_data_interactive(df, "emissions")

    st.markdown("## Scope 1 / 2 / 3 Breakdown")
    insight(
        title="Where do companies' emissions actually come from?",
        text=(
            "This chart breaks total emissions into Scope 1, Scope 2, and Scope 3 categories. "
            "For many energy-related companies, Scope 3 can dominate because downstream use of sold products creates the largest climate impact."
        ),
        eyebrow="Emissions Structure",
    )
    year_options = sorted(filtered_df["year"].dropna().astype(int).unique().tolist())
    if year_options:
        selected_year = st.select_slider("Choose year", options=year_options, value=max(year_options), key="emissions_year")
        year_df = filtered_df[filtered_df["year"] == selected_year].copy()
        scope_cols = ["scope1_emissions", "scope2_emissions", "scope3_emissions"]
        scope_df = year_df[["company", "sector", "country"] + scope_cols].melt(id_vars=["company", "sector", "country"], value_vars=scope_cols, var_name="scope", value_name="emissions")
        scope_df["scope"] = scope_df["scope"].map({"scope1_emissions": "Scope 1", "scope2_emissions": "Scope 2", "scope3_emissions": "Scope 3"})
        scope_df = scope_df.dropna(subset=["emissions"])
        if len(scope_df) > 0:
            top_companies = year_df.groupby("company")["total_emissions"].sum().sort_values(ascending=False).head(18).index.tolist()
            scope_plot_df = scope_df[scope_df["company"].isin(top_companies)]
            stacked = (
                alt.Chart(scope_plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("emissions:Q", title="Emissions Mt CO₂e"),
                    y=alt.Y("company:N", sort="-x", title=None),
                    color=alt.Color("scope:N", title="Scope", scale=alt.Scale(domain=["Scope 1", "Scope 2", "Scope 3"], range=["#8fb3c8", "#4e79a7", "#1f3d5a"])),
                    tooltip=["company:N", "sector:N", "scope:N", alt.Tooltip("emissions:Q", title="Emissions", format=".2f")],
                )
                .properties(height=560)
                .interactive()
            )
            st.altair_chart(stacked, use_container_width=True)
        else:
            st.info("Scope emissions are not available for this filter.")

    st.markdown("## Carbon Intensity Ranking")
    insight(
        title="Which companies emit the most per unit of revenue?",
        text=(
            "Carbon intensity adjusts emissions by company revenue. This is useful because larger companies naturally have higher absolute emissions, "
            "while intensity reveals how emission-heavy their business model is."
        ),
        eyebrow="Benchmarking Logic",
    )
    if year_options:
        ranking_df = year_df.dropna(subset=["carbon_intensity", "total_emissions", "revenue_usd_bn"]).copy()
        ranking_df = ranking_df[ranking_df["carbon_intensity"] >= 0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Highest Carbon Intensity")
            top_intensity = ranking_df.sort_values("carbon_intensity", ascending=False).head(12)
            if len(top_intensity) > 0:
                chart = alt.Chart(top_intensity).mark_bar().encode(
                    x=alt.X("carbon_intensity:Q", title="Total Emissions / Revenue"),
                    y=alt.Y("company:N", sort="-x", title=None),
                    color=alt.Color("sector:N", title="Sector"),
                    tooltip=["company:N", "sector:N", alt.Tooltip("carbon_intensity:Q", format=".2f"), alt.Tooltip("total_emissions:Q", format=".2f"), alt.Tooltip("revenue_usd_bn:Q", format=".2f")],
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)
        with c2:
            st.markdown("### Lowest Carbon Intensity")
            low_intensity = ranking_df.sort_values("carbon_intensity", ascending=True).head(12)
            if len(low_intensity) > 0:
                chart = alt.Chart(low_intensity).mark_bar().encode(
                    x=alt.X("carbon_intensity:Q", title="Total Emissions / Revenue"),
                    y=alt.Y("company:N", sort="x", title=None),
                    color=alt.Color("sector:N", title="Sector"),
                    tooltip=["company:N", "sector:N", alt.Tooltip("carbon_intensity:Q", format=".2f"), alt.Tooltip("total_emissions:Q", format=".2f"), alt.Tooltip("revenue_usd_bn:Q", format=".2f")],
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)

    st.markdown("## Scope 3 Exposure by Sector")
    insight(
        title="Which sectors depend most on value-chain emissions?",
        text=(
            "This line chart compares the average share of Scope 3 emissions by sector over time. "
            "A high Scope 3 share suggests that the firm’s climate impact is concentrated outside direct operations and should not be ignored in ESG assessment."
        ),
        eyebrow="Sector Comparison",
    )
    sector_scope = filtered_df.groupby(["year", "sector"], as_index=False).agg(scope3_share=("scope3_share", "mean")).dropna(subset=["scope3_share"])
    if len(sector_scope) > 0:
        chart = alt.Chart(sector_scope).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("scope3_share:Q", title="Average Scope 3 Share", axis=alt.Axis(format="%")),
            color=alt.Color("sector:N", title="Sector"),
            tooltip=["sector:N", "year:O", alt.Tooltip("scope3_share:Q", format=".1%")],
        ).properties(height=420).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Scope 3 share cannot be calculated for this filter.")

    render_next_page_button("Commitment Tracker")


# Page 3: Commitment Tracker
elif page == "Commitment Tracker":
    st.markdown("## Commitment Tracker")
    st.markdown("Net-zero targets, SBTi commitments, and third-party verification are important signals. But a credible commitment should be followed by measurable reductions in emissions or carbon intensity.")
    filtered_df = filter_data_interactive(df, "commitment")

    commitment_options = {"SBTi Commitment": "sbti_committed", "Net Zero Target": "net_zero_target", "Third-party Verification": "third_party_verified"}
    commitment_label = st.selectbox("Select commitment signal", list(commitment_options.keys()))
    commitment_col = commitment_options[commitment_label]
    temp = filtered_df.copy()
    temp["commitment_status"] = np.where(temp[commitment_col] == 1, f"Has {commitment_label}", f"No {commitment_label}")

    st.markdown("## Carbon Intensity After Commitment")
    insight(
        title="Do climate commitments translate into measurable reductions?",
        text=(
            "This analysis compares companies with and without selected climate commitments, such as SBTi, net-zero targets, or third-party verification. "
            "If commitments are credible, the committed group should show lower or faster-declining carbon intensity over time."
        ),
        eyebrow="Policy Question",
    )
    commit_trend = temp.groupby(["year", "commitment_status"], as_index=False).agg(avg_carbon_intensity=("carbon_intensity", "mean"), avg_esg=("esg_score", "mean"), greenwashing_rate=("greenwashing_flag", "mean")).dropna(subset=["avg_carbon_intensity"])
    if len(commit_trend) > 0:
        chart = alt.Chart(commit_trend).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("avg_carbon_intensity:Q", title="Average Carbon Intensity"),
            color=alt.Color("commitment_status:N", title="Commitment Status"),
            tooltip=["commitment_status:N", "year:O", alt.Tooltip("avg_carbon_intensity:Q", format=".2f"), alt.Tooltip("avg_esg:Q", format=".1f"), alt.Tooltip("greenwashing_rate:Q", format=".1%")],
        ).properties(height=430).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No valid carbon intensity data for this commitment signal.")

    st.markdown("## Greenwashing Rate by Commitment Status")
    insight(
        title="Are committed firms less likely to be flagged?",
        text=(
            "This bar chart compares greenwashing rates between firms with and without the selected commitment signal. "
            "A higher rate among committed firms may suggest that public climate claims need to be checked against actual emissions trajectories."
        ),
        eyebrow="Result Interpretation",
    )
    summary = temp.groupby("commitment_status", as_index=False).agg(observations=("company", "count"), greenwashing_rate=("greenwashing_flag", "mean"), avg_esg=("esg_score", "mean"), avg_total_emissions=("total_emissions", "mean"), avg_carbon_intensity=("carbon_intensity", "mean"))
    if len(summary) > 0:
        c1, c2 = st.columns([1, 1])
        with c1:
            bar = alt.Chart(summary).mark_bar().encode(
                x=alt.X("commitment_status:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("greenwashing_rate:Q", title="Greenwashing Rate", axis=alt.Axis(format="%")),
                color=alt.Color("commitment_status:N", legend=None),
                tooltip=["commitment_status:N", alt.Tooltip("greenwashing_rate:Q", format=".1%"), "observations:Q", alt.Tooltip("avg_esg:Q", format=".1f")],
            ).properties(height=360)
            st.altair_chart(bar, use_container_width=True)
        with c2:
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

    st.markdown("## Commitment Timeline by Company")
    insight(
        title="When did the company start showing commitment signals?",
        text=(
            "This timeline shows whether a selected company had net-zero, SBTi, verification, and greenwashing signals in each year. "
            "It helps compare the timing of climate commitments with the timing of greenwashing flags."
        ),
        eyebrow="Company Timeline",
    )
    selected_company = st.selectbox("Select company", sorted(filtered_df["company"].dropna().unique().tolist()), key="commit_company")
    company_commit = filtered_df[filtered_df["company"] == selected_company].copy()
    commit_melt = company_commit[["year", "net_zero_target", "sbti_committed", "third_party_verified", "greenwashing_flag"]].melt(id_vars=["year"], var_name="signal", value_name="value")
    commit_melt["signal"] = commit_melt["signal"].map({"net_zero_target": "Net Zero", "sbti_committed": "SBTi", "third_party_verified": "Third-party Verified", "greenwashing_flag": "Greenwashing Flag"})
    commit_melt["status"] = np.where(commit_melt["value"] == 1, "Yes", "No")
    timeline = alt.Chart(commit_melt).mark_rect().encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("signal:N", title=None),
        color=alt.Color("status:N", title="Status", scale=alt.Scale(domain=["No", "Yes"], range=["#e8edf2", "#2f4a5f"])),
        tooltip=["signal:N", "year:O", "status:N"],
    ).properties(height=210)
    st.altair_chart(timeline, use_container_width=True)

    render_next_page_button("Greenwashing Explorer")


# Page 4: Greenwashing Explorer
elif page == "Greenwashing Explorer":
    st.markdown("## Greenwashing Company Explorer")
    st.markdown("Select a company to inspect whether ESG scores, emissions, and climate commitments move in the same direction. This turns the dashboard into a company-level ESG due diligence tool.")
    company_list = sorted(df["company"].dropna().unique().tolist())
    default_idx = company_list.index("ExxonMobil") if "ExxonMobil" in company_list else 0
    selected_company = st.selectbox("Select company", company_list, index=default_idx)
    company_df = df[df["company"] == selected_company].sort_values("year").copy()
    latest = company_df.dropna(subset=["year"]).sort_values("year").iloc[-1]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(metric_card(str(latest["sector"]), "Sector"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card(str(latest["country"]), "Country"), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card(format_num(latest["esg_score"], 1), "Latest ESG score"), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card(format_num(latest["total_emissions"], 1), "Latest total emissions"), unsafe_allow_html=True)
    with k5:
        st.markdown(metric_card(format_num(latest["credibility_score"], 1), "Credibility score"), unsafe_allow_html=True)

    st.markdown("## ESG Score and Emissions Trajectory")
    insight(
        title="Does this company’s ESG story match its emissions trend?",
        text=(
            "This company-level view helps evaluate whether ESG scores, emissions, and climate commitments move in the same direction. "
            "A rising ESG score combined with flat or increasing emissions may indicate a potential credibility gap."
        ),
        eyebrow="Company Due Diligence",
    )
    esg_chart = alt.Chart(company_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("esg_score:Q", title="ESG Score", scale=alt.Scale(zero=False)),
        tooltip=["company:N", "year:O", alt.Tooltip("esg_score:Q", format=".1f"), "greenwashing_label:N"],
    ).properties(height=340).interactive()
    emissions_chart = alt.Chart(company_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("total_emissions:Q", title="Total Emissions Mt CO₂e"),
        tooltip=["company:N", "year:O", alt.Tooltip("total_emissions:Q", format=".2f"), alt.Tooltip("carbon_intensity:Q", format=".2f"), "greenwashing_label:N"],
    ).properties(height=340).interactive()
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(esg_chart, use_container_width=True)
    with c2:
        st.altair_chart(emissions_chart, use_container_width=True)

    st.markdown("## Scope Breakdown Over Time")
    insight(
        title="Which emission scope drives this company’s climate footprint?",
        text=(
            "This stacked area chart shows how Scope 1, Scope 2, and Scope 3 emissions evolve over time for the selected company. "
            "If Scope 3 dominates, operational emissions alone are not enough to assess climate credibility."
        ),
        eyebrow="Scope Analysis",
    )
    scope_company = company_df[["year", "scope1_emissions", "scope2_emissions", "scope3_emissions"]].melt(id_vars=["year"], var_name="scope", value_name="emissions")
    scope_company["scope"] = scope_company["scope"].map({"scope1_emissions": "Scope 1", "scope2_emissions": "Scope 2", "scope3_emissions": "Scope 3"})
    scope_company = scope_company.dropna(subset=["emissions"])
    if len(scope_company) > 0:
        area = alt.Chart(scope_company).mark_area(opacity=0.82).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("emissions:Q", title="Emissions Mt CO₂e"),
            color=alt.Color("scope:N", title="Scope"),
            tooltip=["scope:N", "year:O", alt.Tooltip("emissions:Q", format=".2f")],
        ).properties(height=390).interactive()
        st.altair_chart(area, use_container_width=True)
    else:
        st.info("Scope breakdown is unavailable for this company.")

    st.markdown("## Company Commitment and Flag Timeline")
    insight(
        title="Do flags appear before or after public climate commitments?",
        text=(
            "The timeline compares climate signals and greenwashing flags in the same company-year view. "
            "This helps identify whether commitments coincide with cleaner performance or whether concerns continue after public pledges."
        ),
        eyebrow="Timeline Interpretation",
    )
    timeline_df = company_df[["year", "net_zero_target", "sbti_committed", "third_party_verified", "greenwashing_flag"]].melt(id_vars=["year"], var_name="signal", value_name="value")
    timeline_df["signal"] = timeline_df["signal"].map({"net_zero_target": "Net Zero", "sbti_committed": "SBTi", "third_party_verified": "Third-party Verified", "greenwashing_flag": "Greenwashing Flag"})
    timeline_df["status"] = np.where(timeline_df["value"] == 1, "Yes", "No")
    chart = alt.Chart(timeline_df).mark_rect().encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("signal:N", title=None),
        color=alt.Color("status:N", title="Status", scale=alt.Scale(domain=["No", "Yes"], range=["#e8edf2", "#c94c4c"])),
        tooltip=["signal:N", "year:O", "status:N"],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

    render_next_page_button("Sector Benchmarking")


# Page 5: Sector Benchmarking
elif page == "Sector Benchmarking":
    st.markdown("## Sector Benchmarking")
    st.markdown("This page compares Energy, Utilities, and Industrials firms by ESG scores, emissions intensity, and greenwashing frequency. The peer map is designed to identify companies that combine high emissions intensity with high ESG scores.")
    filtered_df = filter_data_interactive(df, "sector")

    st.markdown("## Sector-level Greenwashing Rate")
    insight(
        title="Which sectors show higher greenwashing frequency?",
        text=(
            "This chart compares the share of flagged company-year observations across sectors. "
            "A higher rate suggests that firms in that sector more often show a mismatch between stated ESG progress and emissions-related indicators."
        ),
        eyebrow="Sector Risk",
    )
    sector_summary = filtered_df.groupby("sector", as_index=False).agg(observations=("company", "count"), companies=("company", "nunique"), avg_esg=("esg_score", "mean"), avg_total_emissions=("total_emissions", "mean"), avg_carbon_intensity=("carbon_intensity", "mean"), greenwashing_rate=("greenwashing_flag", "mean")).dropna(subset=["sector"])
    c1, c2 = st.columns(2)
    with c1:
        if len(sector_summary) > 0:
            chart = alt.Chart(sector_summary).mark_bar().encode(
                x=alt.X("sector:N", title="Sector"),
                y=alt.Y("greenwashing_rate:Q", title="Greenwashing Rate", axis=alt.Axis(format="%")),
                color=alt.Color("sector:N", legend=None),
                tooltip=["sector:N", "companies:Q", "observations:Q", alt.Tooltip("greenwashing_rate:Q", format=".1%"), alt.Tooltip("avg_esg:Q", format=".1f")],
            ).properties(height=360)
            st.altair_chart(chart, use_container_width=True)
    with c2:
        if len(sector_summary) > 0:
            st.dataframe(sector_summary.round(3), use_container_width=True, hide_index=True)

    st.markdown("## ESG Score Distribution by Sector")
    insight(
        title="Do some sectors consistently receive higher ESG scores?",
        text=(
            "This box plot compares the distribution of ESG scores across sectors. "
            "It helps distinguish sector-level ESG scoring patterns from company-specific climate performance."
        ),
        eyebrow="ESG Distribution",
    )
    box_df = filtered_df.dropna(subset=["sector", "esg_score"]).copy()
    if len(box_df) > 0:
        box = alt.Chart(box_df).mark_boxplot(size=48).encode(
            x=alt.X("sector:N", title="Sector"),
            y=alt.Y("esg_score:Q", title="ESG Score", scale=alt.Scale(zero=False)),
            color=alt.Color("sector:N", legend=None),
            tooltip=["sector:N"],
        ).properties(height=410)
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("No ESG score data available.")

    st.markdown("## Sector Peer Map")
    insight(
        title="Which companies stand out from their sector peers?",
        text=(
            "The peer map compares carbon intensity and ESG score across companies in the same year. "
            "Firms with both high carbon intensity and high ESG scores may be worth deeper investigation, especially if they are also flagged for greenwashing."
        ),
        eyebrow="Sector Benchmarking",
    )
    year_options = sorted(filtered_df["year"].dropna().astype(int).unique().tolist())
    if year_options:
        selected_year = st.select_slider("Select year for peer map", options=year_options, value=max(year_options), key="sector_peer_year")
        peer_df = filtered_df[filtered_df["year"] == selected_year].dropna(subset=["carbon_intensity", "esg_score", "total_emissions", "sector", "company"]).copy()
        peer_df = peer_df[(peer_df["carbon_intensity"] >= 0) & (peer_df["total_emissions"] >= 0)]
        if len(peer_df) > 0:
            upper = peer_df["carbon_intensity"].quantile(0.98)
            peer_df["carbon_intensity_plot"] = peer_df["carbon_intensity"].clip(upper=upper) if pd.notna(upper) and upper > 0 else peer_df["carbon_intensity"]
            peer_map = alt.Chart(peer_df).mark_circle(opacity=0.78).encode(
                x=alt.X("carbon_intensity_plot:Q", title="Carbon Intensity"),
                y=alt.Y("esg_score:Q", title="ESG Score", scale=alt.Scale(zero=False)),
                size=alt.Size("total_emissions:Q", title="Total Emissions", scale=alt.Scale(range=[100, 1300])),
                color=alt.Color("sector:N", title="Sector"),
                shape=alt.Shape("greenwashing_label:N", title="Greenwashing"),
                tooltip=["company:N", "sector:N", "country:N", "greenwashing_label:N", alt.Tooltip("esg_score:Q", format=".1f"), alt.Tooltip("carbon_intensity:Q", title="Carbon intensity", format=".2f"), alt.Tooltip("total_emissions:Q", title="Total emissions", format=".2f"), alt.Tooltip("revenue_usd_bn:Q", title="Revenue", format=".2f")],
            ).properties(height=520).interactive()
            st.altair_chart(peer_map, use_container_width=True)
        else:
            st.info("Peer map 沒有可畫的資料。通常是 carbon_intensity、ESG score 或 emissions 有缺值。")

    st.markdown("## Country-level Summary")
    insight(
        title="How do countries differ in ESG and greenwashing signals?",
        text=(
            "This table aggregates company-year observations by headquarters country. "
            "It is useful for identifying geographic patterns in ESG scores, emissions intensity, and flagged observations."
        ),
        eyebrow="Country Summary",
    )
    country_summary = filtered_df.groupby("country", as_index=False).agg(observations=("company", "count"), companies=("company", "nunique"), avg_esg=("esg_score", "mean"), avg_carbon_intensity=("carbon_intensity", "mean"), greenwashing_rate=("greenwashing_flag", "mean")).sort_values("greenwashing_rate", ascending=False)
    st.dataframe(country_summary.round(3), use_container_width=True, hide_index=True)

    render_next_page_button("Prediction Model")


# Page 6: Prediction Model
elif page == "Prediction Model":
    st.markdown("## Machine Learning Greenwashing Prediction")
    st.markdown("This page trains a Random Forest classifier to predict the binary greenwashing flag. The goal is not to claim causal inference, but to identify which variables are most useful in detecting a credibility gap.")
    model_df = df.copy()
    selected_features = [
        "revenue_usd_bn", "scope1_emissions", "scope2_emissions", "scope3_emissions",
        "total_s1_s2_emissions", "total_emissions", "carbon_intensity", "operational_carbon_intensity",
        "scope3_share", "esg_score", "net_zero_target", "sbti_committed", "third_party_verified",
        "emissions_yoy_change", "esg_yoy_change", "carbon_intensity_yoy_change",
    ]
    selected_features = [c for c in selected_features if c in model_df.columns]
    ml_df = model_df[selected_features + ["greenwashing_flag", "company", "year", "sector"]].copy()
    y = ml_df["greenwashing_flag"].astype(int)
    X_raw = ml_df[selected_features].replace([np.inf, -np.inf], np.nan)

    if y.nunique() < 2:
        st.warning("The current dataset contains only one class in the greenwashing flag, so the model cannot be trained.")
    else:
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X_raw), columns=selected_features, index=X_raw.index)
        stratify_arg = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify_arg)
        model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(metric_card(format_num(acc, 2), "Accuracy"), unsafe_allow_html=True)
        with k2:
            st.markdown(metric_card(format_num(prec, 2), "Precision"), unsafe_allow_html=True)
        with k3:
            st.markdown(metric_card(format_num(rec, 2), "Recall"), unsafe_allow_html=True)
        with k4:
            st.markdown(metric_card(format_num(f1, 2), "F1 Score"), unsafe_allow_html=True)
        with k5:
            st.markdown(metric_card(format_num(auc, 2), "ROC-AUC"), unsafe_allow_html=True)

        st.markdown("## Feature Importance")
        insight(
            title="Which variables are most useful for detecting greenwashing?",
            text=(
                "Feature importance shows which variables the Random Forest model uses most when predicting greenwashing flags. "
                "This does not prove causality, but it helps identify the indicators that are most informative for ESG credibility screening."
            ),
            eyebrow="Model Interpretation",
        )
        importance_df = pd.DataFrame({"feature": selected_features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        importance_df["feature_label"] = importance_df["feature"].str.replace("_", " ", regex=False).str.title()
        feat_chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X("importance:Q", title="Relative Importance"),
            y=alt.Y("feature_label:N", sort="-x", title=None),
            tooltip=["feature_label:N", alt.Tooltip("importance:Q", format=".3f")],
        ).properties(height=480)
        st.altair_chart(feat_chart, use_container_width=True)

        st.markdown("## Confusion Matrix")
        insight(
            title="How does the model separate clean and flagged observations?",
            text=(
                "The confusion matrix compares actual labels with model predictions. "
                "False negatives mean flagged observations were missed, while false positives mean clean observations were classified as risky."
            ),
            eyebrow="Model Evaluation",
        )
        cm = confusion_matrix(y_test, pred)
        cm_df = pd.DataFrame(cm, index=["Actual Clean", "Actual Flagged"], columns=["Predicted Clean", "Predicted Flagged"])
        cm_long = cm_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count").rename(columns={"index": "Actual"})
        cm_chart = alt.Chart(cm_long).mark_rect().encode(
            x=alt.X("Predicted:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", title=None),
            color=alt.Color("Count:Q", title="Count"),
            tooltip=["Actual:N", "Predicted:N", "Count:Q"],
        ).properties(height=260)
        text = alt.Chart(cm_long).mark_text(size=18, fontWeight="bold").encode(x="Predicted:N", y="Actual:N", text="Count:Q", color=alt.value("black"))
        st.altair_chart(cm_chart + text, use_container_width=True)

        st.markdown("## Predicted Greenwashing Risk Table")
        insight(
            title="Which company-year observations receive the highest predicted risk?",
            text=(
                "This table ranks company-year observations by predicted greenwashing probability. "
                "It can be used as a watchlist for ESG analysts or investors who want to prioritize deeper review."
            ),
            eyebrow="Risk Watchlist",
        )
        all_risk = model.predict_proba(X)[:, 1]
        risk_df = ml_df[["company", "year", "sector", "greenwashing_flag"]].copy()
        risk_df["predicted_greenwashing_risk"] = all_risk
        risk_df = risk_df.sort_values("predicted_greenwashing_risk", ascending=False)
        selected_year = st.select_slider("Select year for risk table", options=sorted(risk_df["year"].dropna().astype(int).unique().tolist()), value=int(risk_df["year"].max()), key="risk_table_year")
        display_risk = risk_df[risk_df["year"] == selected_year].head(20).copy()
        display_risk["greenwashing_flag"] = display_risk["greenwashing_flag"].map({0: "Clean", 1: "Flagged"})
        display_risk["predicted_greenwashing_risk"] = display_risk["predicted_greenwashing_risk"].round(3)
        st.dataframe(display_risk, use_container_width=True, hide_index=True)

    render_next_page_button("Live Financials")


# Page 7: Live Financials
elif page == "Live Financials":
    st.markdown("## Live Financials")
    st.markdown("This optional page connects company-level ESG analysis with live market information. If Yahoo Finance does not return data for a ticker, the dashboard will still work without breaking the page.")
    valid_df = df.dropna(subset=["ticker"]).copy()
    valid_df = valid_df[valid_df["ticker"].astype(str).str.lower() != "nan"]

    if yf is None:
        st.warning("yfinance is not installed. Install it with: pip install yfinance")
    elif len(valid_df) == 0:
        st.warning("No valid ticker symbols are available.")
    else:
        company_list = sorted(valid_df["company"].unique().tolist())
        selected_company = st.selectbox("Select company", company_list)
        ticker = valid_df[valid_df["company"] == selected_company]["ticker"].iloc[0]
        latest_company_row = df[df["company"] == selected_company].sort_values("year").iloc[-1]
        st.markdown(f"### {selected_company} ({ticker})")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info if stock.info else {}
            previous_close = info.get("previousClose", np.nan)
            market_cap = info.get("marketCap", np.nan)
            yahoo_sector = info.get("sector", "N/A")
            summary = info.get("longBusinessSummary", "No company summary is available from Yahoo Finance.")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Previous Close", f"${previous_close:.2f}" if pd.notna(previous_close) else "N/A")
            with c2:
                st.metric("Market Cap", f"${market_cap / 1e9:.2f}B" if pd.notna(market_cap) else "N/A")
            with c3:
                st.metric("Yahoo Sector", yahoo_sector)
            with c4:
                st.metric("Dashboard ESG Score", format_num(latest_company_row["esg_score"], 1))
            insight(
                title="How does market information connect to the dashboard view?",
                text=(
                    f"In the dashboard dataset, {selected_company} has a latest total emissions value of "
                    f"{format_num(latest_company_row['total_emissions'], 2)} Mt CO₂e and a latest credibility score of "
                    f"{format_num(latest_company_row['credibility_score'], 1)}."
                ),
                eyebrow="Financial Context",
            )
            st.markdown("### Company Summary")
            st.write(summary[:1200] + ("..." if len(summary) > 1200 else ""))
        except Exception:
            st.error("Failed to fetch Yahoo Finance data. This may be caused by an invalid ticker, network issue, or rate limit.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Dataset-side Financial ESG View")
    insight(
        title="Do revenue, emissions intensity, and greenwashing risk move together?",
        text=(
            "This scatter plot uses dataset-side financial and emissions variables, so it works even if Yahoo Finance is unavailable. "
            "It helps compare company scale, carbon intensity, and greenwashing flags in the same view."
        ),
        eyebrow="Dataset View",
    )
    finance_df = df.dropna(subset=["revenue_usd_bn", "carbon_intensity", "esg_score"]).copy()
    year_options = sorted(finance_df["year"].dropna().astype(int).unique().tolist())
    if len(year_options) > 0:
        selected_year = st.select_slider("Select year", options=year_options, value=max(year_options), key="finance_dataset_year")
        finance_year = finance_df[finance_df["year"] == selected_year].copy()
        chart = alt.Chart(finance_year).mark_circle(opacity=0.78).encode(
            x=alt.X("revenue_usd_bn:Q", title="Revenue USD bn"),
            y=alt.Y("carbon_intensity:Q", title="Carbon Intensity"),
            size=alt.Size("total_emissions:Q", title="Total Emissions", scale=alt.Scale(range=[80, 1200])),
            color=alt.Color("greenwashing_label:N", title="Greenwashing", scale=alt.Scale(domain=["Clean", "Flagged"], range=["#4E79A7", "#E15759"])),
            tooltip=["company:N", "sector:N", "country:N", "greenwashing_label:N", alt.Tooltip("revenue_usd_bn:Q", format=".2f"), alt.Tooltip("carbon_intensity:Q", format=".2f"), alt.Tooltip("esg_score:Q", format=".1f")],
        ).properties(height=480).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No financial dataset-side data available.")
