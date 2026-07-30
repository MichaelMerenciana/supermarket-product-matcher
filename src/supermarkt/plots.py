"""
Functions for visualizations in notebooks/analysis.ipynb

This file is not part of the final product matching pipeline.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_affordability_vs_frac_protein_interactive(df: pd.DataFrame, threshold_protein: float = 0.2):
    fig = go.Figure()

    df_thresh = df[df["frac_kcal_protein"] > threshold_protein]

    masks = [
        ("Vegan", df_thresh["vegan"], "green"),
        ("Veggie", (~df_thresh["vegan"]) & df_thresh["veggie"], "blue"),
        ("Meats", ~df_thresh["veggie"], "red"),
    ]

    for label, mask, color in masks:
        fig.add_trace(
            go.Scatter(
                x=df_thresh.loc[mask, "frac_kcal_protein"],
                y=df_thresh.loc[mask, "protein_per_euro"],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.4,
                ),
                text=df_thresh.loc[mask, "name"],  # shown on hover
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Protein fraction: %{x:.2%}<br>"
                    "Protein/€: %{y:.1f} g<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Protein Affordability vs Percentage of kcal",
        xaxis_title="Protein fraction",
        yaxis_title="Protein/€ (g)",
        template="plotly_white",
    )

    fig.show()

def plot_frac_fat_vs_frac_protein_interactive(df: pd.DataFrame):
    fig = go.Figure()

    masks = [
        ("Vegan", df["vegan"], "green"),
        ("Veggie", (~df["vegan"]) & df["veggie"], "blue"),
        ("Meats", ~df["veggie"], "red"),
    ]

    for label, mask, color in masks:
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "frac_kcal_protein"],
                y=df.loc[mask, "frac_kcal_fat"],
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.4,
                ),
                text=df.loc[mask, "name"],  # shown on hover
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Protein fraction: %{x:.2%}<br>"
                    "Fat fraction: %{y:.2%}<br>"
                ),
            )
        )

    fig.update_layout(
        title="Relative Fat vs Protein",
        xaxis_title="Protein fraction",
        yaxis_title="Fat fraction)",
        template="plotly_white",
    )

    fig.show()

def plot_macro_frac(df : pd.DataFrame):
    df = df.copy()
    df["diet"] = "Meats"
    df.loc[df["veggie"], "diet"] = "Veggie"
    df.loc[df["vegan"], "diet"] = "Vegan"

    fig = px.scatter_3d(
        df,
        x="frac_kcal_carbs",
        y="frac_kcal_fat",
        z="frac_kcal_protein",
        color="diet",
        color_discrete_map={
            "Vegan": "green",
            "Veggie": "blue",
            "Meats": "red",
        },
        hover_data=[
            "name",
            "price",
            "protein_per_euro",
        ],
        opacity=0.3
    )

    fig.update_traces(marker=dict(size=6))

    # Display the plot
    fig.show()