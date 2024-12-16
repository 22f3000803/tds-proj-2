# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "tenacity"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

# Set up constants for AI Proxy and LLM interaction
API_URL = "https://api.aiproxy.io/v1/generate"
TOKEN = os.environ.get("AIPROXY_TOKEN")
if not TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_llm(prompt, model="gpt-4o-mini", functions=None):
    """Make a call to the LLM with the given prompt."""
    payload = {"model": model, "prompt": prompt, "functions": functions}
    response = httpx.post(API_URL, json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()["output"]

# Visualization function
def save_chart(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Load CSV and perform basic analysis
def analyze_csv(filename):
    try:
        data = pd.read_csv(filename)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

    # Summary stats and missing values
    summary = data.describe(include="all")
    missing = data.isnull().sum().to_dict()

    # Correlation matrix (numerical columns only)
    correlation = data.corr(numeric_only=True)

    return data, summary, missing, correlation

# Generate visualizations
def create_visualizations(data, correlation):
    charts = []

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    heatmap_path = "correlation_heatmap.png"
    save_chart(fig, heatmap_path)
    charts.append(heatmap_path)

    # Distribution of a numerical column
    numeric_cols = data.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data[numeric_cols[0]], kde=True, ax=ax)
        ax.set_title(f"Distribution of {numeric_cols[0]}")
        hist_path = "distribution.png"
        save_chart(fig, hist_path)
        charts.append(hist_path)

    # Missing values bar chart
    missing_values = data.isnull().sum()
    if missing_values.any():
        fig, ax = plt.subplots(figsize=(8, 6))
        missing_values.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Count")
        bar_chart_path = "missing_values.png"
        save_chart(fig, bar_chart_path)
        charts.append(bar_chart_path)

    return charts

# Generate a narrative with the LLM
def generate_narrative(filename, summary, missing, correlation, charts):
    prompt = f"""
    I analyzed the dataset {filename}. Here's the context:
    - Summary statistics:
    {summary.to_string()}

    - Missing values:
    {missing}

    - Correlation matrix:
    {correlation.to_string()}

    Please write a detailed narrative describing the data, analysis, key insights, and implications. Include references to the following charts:
    {charts}
    """
    return call_llm(prompt)

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        data, summary, missing, correlation = analyze_csv(filename)
        charts = create_visualizations(data, correlation)
        narrative = generate_narrative(filename, summary, missing, correlation, charts)

        # Save narrative to README.md
        with open("README.md", "w") as f:
            f.write(narrative)

        print("Analysis complete. Results saved to README.md and PNG files.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
