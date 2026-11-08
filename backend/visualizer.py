from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.technical_analysis import compute_indicator_series

def create_stock_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create a professional multi-pane interactive chart for a stock:
    Pane 1: Candlesticks + SMA50 + SMA200 (Cyber Mint/Rose & Cyan/Purple)
    Pane 2: Volume
    Pane 3: 14-Day RSI with Overbought (70) and Oversold (30) levels
    """
    if df.empty or len(df) < 5:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient historical data to render chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#94a3b8")
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor="#080d1a",
            plot_bgcolor="#080d1a"
        )
        return fig

    # Calculate indicator series
    df_calc = compute_indicator_series(df)

    # Subplots layout
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.18, 0.24],
        subplot_titles=(
            f"<b>{symbol}</b> — Dual Momentum & Trend Alignment (SMA 50 / 200)",
            "Trading Volume & Liquidity",
            "Wilder's RSI (14-Period Momentum Oscillator)"
        )
    )

    # 1. Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=df_calc["date"],
            open=df_calc["open"],
            high=df_calc["high"],
            low=df_calc["low"],
            close=df_calc["close"],
            name="OHLC",
            increasing_line_color="#00f5a0", # Cyber Mint
            decreasing_line_color="#ff3b69", # Neon Rose
            increasing_fillcolor="#00f5a0",
            decreasing_fillcolor="#ff3b69"
        ),
        row=1, col=1
    )

    # 2. SMA50 Overlay
    fig.add_trace(
        go.Scatter(
            x=df_calc["date"],
            y=df_calc["SMA50"],
            name="SMA 50 (Short Trend)",
            line=dict(color="#38bdf8", width=2.0), # Electric Sky Cyan
            mode="lines"
        ),
        row=1, col=1
    )

    # 3. SMA200 Overlay
    fig.add_trace(
        go.Scatter(
            x=df_calc["date"],
            y=df_calc["SMA200"],
            name="SMA 200 (Long Trend)",
            line=dict(color="#c084fc", width=2.2), # Radiant Purple
            mode="lines"
        ),
        row=1, col=1
    )

    # 4. Volume Bars
    vol_colors = [
        "rgba(0, 245, 160, 0.65)" if c >= o else "rgba(255, 59, 105, 0.65)"
        for c, o in zip(df_calc["close"], df_calc["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df_calc["date"],
            y=df_calc["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.8
        ),
        row=2, col=1
    )

    # 5. RSI(14) Line
    fig.add_trace(
        go.Scatter(
            x=df_calc["date"],
            y=df_calc["RSI14"],
            name="RSI 14",
            line=dict(color="#f59e0b", width=2.0), # Amber Gold
            mode="lines"
        ),
        row=3, col=1
    )

    # Overbought (70), Neutral (50) and Oversold (30) RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3b69", line_width=1.2, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(148, 163, 184, 0.4)", line_width=1.0, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00f5a0", line_width=1.2, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(56, 189, 248, 0.05)", line_width=0, row=3, col=1)

    # Aesthetic Terminal Layout
    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=55, r=40, t=55, b=40),
        paper_bgcolor="#080d1a",
        plot_bgcolor="#080d1a",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(56, 189, 248, 0.2)",
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    fig.update_xaxes(
        gridcolor="rgba(51, 65, 85, 0.35)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#38bdf8"
    )
    fig.update_yaxes(
        gridcolor="rgba(51, 65, 85, 0.35)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#38bdf8"
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI (14)", range=[0, 100], row=3, col=1)

    return fig
