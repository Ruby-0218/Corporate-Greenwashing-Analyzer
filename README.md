# Corporate Greenwashing Analyzer
**An end-to-end ESG analytics platform detecting greenwashing patterns via Machine Learning & Interactive Visualizations.**


## Live Application
Try the interactive dashboard here: **[Corporate Greenwashing Analyzer](https://corporate-greenwashing-analyzer-any9e9omuwmcpbudtvm6h6.streamlit.app/?page=Overview)**

## Project Overview
Over the past decade, ESG (Environmental, Social, and Governance) reporting has become standard practice for large-cap companies. However, a rising ESG score does not always indicate a shrinking carbon footprint. 

This project investigates the paradox of **Corporate Greenwashing**—instances where public climate narratives and sustainability scores outpace genuine environmental progress. By merging longitudinal ESG data (2010–2024) with Machine Learning, this platform decodes greenwashing signatures across the Energy, Utilities, and Industrials sectors. 

We shift the focus from absolute emissions to **Carbon Intensity (Emissions/Revenue)** and introduce a custom **ESG Credibility Index** to provide a more rigorous benchmark for corporate sustainability.

## Data Source
The core dataset utilized in this project is sourced from Kaggle:
* **Dataset Name:** [ESG Greenwashing Detection — Energy, Utilities & Industrials (2010–2024)](https://www.kaggle.com/datasets/alitaqishah/esg-greenwashing-energy-and-industrials)
* **Author:** Syed Ali Taqi
* **Description:** A panel dataset containing 450 company-year observations across 30 major publicly listed companies. It includes Scope 1, 2, and 3 emissions, ESG scores, Net-Zero targets, SBTi sign-ons, and a binary `greenwashing_flag` derived from the gap between stated commitments and actual emission trajectories.
* *Note: Live financial metrics and current ESG risk scores are fetched dynamically via the `yfinance` API.*

## Key Features
1. **Interactive Emissions Tracking:** Multi-dimensional trajectory analysis allowing users to track ESG score inflation versus actual carbon emissions via Plotly animated scatter plots.
2. **Emissions Structure Analysis:** Deep dive into Scope 1, 2, and 3 emissions structures using stacked area charts to reveal the hidden weight of value-chain emissions.
3. **Machine Learning Greenwashing Detection:** A Random Forest Classifier trained to extract feature importances, revealing that absolute *ESG Scores* and *Carbon Intensity* are mathematically stronger predictors of greenwashing than mere "Net-Zero" pledges.
4. **Company Explorer & Benchmarking:** A SaaS-style dashboard generating a custom ESG Credibility Index and fetching real-time financial market data to evaluate how the market prices these environmental behaviors.
5. **Policy Impact Analysis:** Statistical comparison of carbon reduction rates between companies adhering to the Science Based Targets initiative (SBTi) versus non-adherers.

## Tech Stack
* **Frontend / Framework:** Streamlit
* **Data Visualization:** Plotly (Express & Graph Objects), Altair
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, MinMaxScaler)
* **Data Manipulation:** Pandas, NumPy
* **API Integration:** YFinance

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Corporate-Greenwashing-Analyzer.git](https://github.com/YourUsername/Corporate-Greenwashing-Analyzer.git)
