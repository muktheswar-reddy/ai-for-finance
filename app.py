import os
import io
import time
from pathlib import Path

# Disable Gradio telemetry to prevent socket timeouts in offline/sandboxed environments
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import gradio as gr

from backend.config import (
    BASE_DIR,
    DB_PATH,
    DEFAULT_START_DATE,
    DEFAULT_MAX_STOCKS,
    DEFAULT_PAUSE_SECONDS,
    DEFAULT_SKIP_EXISTING,
    FINNHUB_API_KEY,
    TWELVE_DATA_API_KEY,
    TAVILY_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL
)
import backend.database as db
import backend.data_loader as dl
import backend.technical_analysis as ta
import backend.external_apis as apis
import backend.ai_advisor as ai
import backend.visualizer as viz

# Ensure database is initialized on startup
db.init_db()

# Read custom CSS
css_path = BASE_DIR / "assets" / "custom.css"
custom_css = ""
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()

# Helper to format metric HTML cards
def render_metric_cards(total: int, buys: int, holds: int, sells: int) -> str:
    return f"""
    <div class="metrics-grid">
      <div class="kpi-card kpi-total">
        <div class="kpi-label"><span>Total Scanned</span> <span>🔍</span></div>
        <div class="kpi-val">{total}</div>
      </div>
      <div class="kpi-card kpi-buy">
        <div class="kpi-label"><span>Buy Signals</span> <span>▲</span></div>
        <div class="kpi-val">{buys}</div>
      </div>
      <div class="kpi-card kpi-hold">
        <div class="kpi-label"><span>Hold / Wait</span> <span>◆</span></div>
        <div class="kpi-val">{holds}</div>
      </div>
      <div class="kpi-card kpi-sell">
        <div class="kpi-label"><span>Sell / Risk</span> <span>▼</span></div>
        <div class="kpi-val">{sells}</div>
      </div>
    </div>
    """

# Helper to format stock summary badges
def render_decision_badge(decision: str, score: int) -> str:
    if decision == "BUY WATCH":
        badge_cls = "badge-buy"
        icon = "▲"
    elif decision == "SELL WATCH":
        badge_cls = "badge-sell"
        icon = "▼"
    else:
        badge_cls = "badge-hold"
        icon = "◆"
    return f'<div style="margin: 12px 0;"><span class="{badge_cls}">{icon} VERDICT: {decision} &nbsp;|&nbsp; QUANT SCORE: {score}</span></div>'

# Application callbacks
def handle_source_change(source_choice: str):
    is_custom_file = source_choice == "Custom File Upload"
    is_custom_symbols = source_choice == "Manual Symbol Input"
    return (
        gr.update(visible=is_custom_file),
        gr.update(visible=is_custom_symbols)
    )

def run_download_and_scan(
    source_choice: str,
    custom_file,
    custom_symbols_text: str,
    max_stocks: int,
    start_date: str,
    skip_existing: bool,
    pause_seconds: float,
    progress=gr.Progress()
):
    try:
        progress(0, desc="Resolving security universe...")
        # 1. Determine Universe
        if source_choice == "Main Board Equities (EQUITY_L.csv)":
            _, symbols = dl.load_security_document("equity_l")
        elif source_choice == "SME Equities (SME_EQUITY_L.csv)":
            _, symbols = dl.load_security_document("sme_equity_l")
        elif source_choice == "T+0 Settlement Eligible (Eligible_T0)":
            _, symbols = dl.load_security_document("eligible_t0")
        elif source_choice == "Custom File Upload":
            if not custom_file:
                return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), "⚠️ Please upload a CSV or Excel file.", gr.update(choices=[]), None
            _, symbols = dl.parse_uploaded_file(custom_file.name)
        elif source_choice == "Manual Symbol Input":
            symbols = dl.parse_custom_symbols(custom_symbols_text)
            if not symbols:
                return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), "⚠️ Please enter at least one valid ticker symbol.", gr.update(choices=[]), None
        else:
            _, symbols = dl.load_security_document("equity_l")

        if max_stocks and max_stocks > 0:
            symbols = symbols[:max_stocks]

        total_syms = len(symbols)
        if total_syms == 0:
            return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), "⚠️ No valid symbols found in selected universe.", gr.update(choices=[]), None

        # 2. Download loop
        ok_count = 0
        fail_count = 0
        skipped_count = 0

        for i, sym in enumerate(symbols):
            progress((i / total_syms) * 0.8, desc=f"Processing {sym} ({i+1}/{total_syms})")
            if skip_existing and db.downloaded_ok(sym):
                skipped_count += 1
                continue

            ok, _, _ = dl.fetch_one_stock(sym, start_date=start_date)
            if ok:
                ok_count += 1
            else:
                fail_count += 1

            if pause_seconds > 0 and (i + 1) < total_syms:
                time.sleep(pause_seconds)

        # 3. Scan all symbols in database
        progress(0.85, desc="Computing algorithmic technical scores...")
        all_data = db.get_all_daily_prices()
        scan_df = ta.scan_all_symbols(all_data)

        progress(1.0, desc="Scan complete!")
        if scan_df.empty:
            msg = f"Finished downloading. Processed: {ok_count} new, {skipped_count} skipped, {fail_count} failed. No stocks had sufficient history for scanning."
            return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), msg, gr.update(choices=[]), None

        # Calculate metrics
        total_scanned = len(scan_df)
        buys = len(scan_df[scan_df["decision"] == "BUY WATCH"])
        holds = len(scan_df[scan_df["decision"] == "HOLD / WAIT"])
        sells = len(scan_df[scan_df["decision"] == "SELL WATCH"])
        cards_html = render_metric_cards(total_scanned, buys, holds, sells)

        # Save scan results to CSV in root as notebook did
        csv_path = BASE_DIR / "indian_stock_scan_results.csv"
        scan_df.to_csv(csv_path, index=False)

        status_msg = (
            f"✅ **Scan Complete**: {total_scanned} stocks evaluated. "
            f"New downloads: {ok_count} | Skipped existing: {skipped_count} | Failures: {fail_count}."
        )

        all_symbols = scan_df["symbol"].tolist()
        return scan_df, cards_html, status_msg, gr.update(choices=all_symbols, value=all_symbols[0] if all_symbols else None), str(csv_path)

    except Exception as e:
        return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), f"❌ Error during download and scan: {str(e)}", gr.update(choices=[]), None

def run_rescan_only():
    """Run technical scan directly on already-downloaded database data."""
    try:
        all_data = db.get_all_daily_prices()
        if all_data.empty:
            return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), "ℹ️ Database is empty. Please run 'Download & Scan' first.", gr.update(choices=[]), None

        scan_df = ta.scan_all_symbols(all_data)
        if scan_df.empty:
            return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), "⚠️ Database contains rows, but none meet minimum history requirements (e.g. 50/200 days).", gr.update(choices=[]), None

        total_scanned = len(scan_df)
        buys = len(scan_df[scan_df["decision"] == "BUY WATCH"])
        holds = len(scan_df[scan_df["decision"] == "HOLD / WAIT"])
        sells = len(scan_df[scan_df["decision"] == "SELL WATCH"])
        cards_html = render_metric_cards(total_scanned, buys, holds, sells)

        csv_path = BASE_DIR / "indian_stock_scan_results.csv"
        scan_df.to_csv(csv_path, index=False)

        status_msg = f"✅ **Database Rescanned**: {total_scanned} stocks evaluated."
        all_symbols = scan_df["symbol"].tolist()
        return scan_df, cards_html, status_msg, gr.update(choices=all_symbols, value=all_symbols[0] if all_symbols else None), str(csv_path)
    except Exception as e:
        return pd.DataFrame(), render_metric_cards(0, 0, 0, 0), f"❌ Rescan failed: {str(e)}", gr.update(choices=[]), None

def filter_scan_table(full_df: pd.DataFrame, decision_filter: str, search_query: str):
    """Filter the scan results table dynamically."""
    if full_df is None or full_df.empty:
        return pd.DataFrame()
    df = full_df.copy()
    if decision_filter and decision_filter != "ALL":
        df = df[df["decision"] == decision_filter]
    if search_query:
        sq = search_query.strip().upper()
        df = df[df["symbol"].astype(str).str.contains(sq, na=False)]
    return df

def load_stock_chart(symbol: str):
    """Load interactive chart and indicators for a given symbol."""
    if not symbol:
        return None, "Please select or enter a symbol.", ""

    clean_sym = symbol.strip().upper()
    df_prices = db.get_symbol_prices(clean_sym)
    if df_prices.empty:
        # Attempt an on-demand download if missing
        ok, n, _ = dl.fetch_one_stock(clean_sym)
        if ok:
            df_prices = db.get_symbol_prices(clean_sym)

    if df_prices.empty:
        return None, f"❌ No data found for symbol '{clean_sym}'.", ""

    fig = viz.create_stock_chart(df_prices, clean_sym)
    tech = ta.analyze_group(df_prices)

    if tech:
        summary_md = f"""
### ⚡ Quantitative Indicator Profile: **{clean_sym}**
- **Effective Date**: `{tech['date']}`
- **Last Traded Close**: **₹{tech['close']}**
- **50-Day Moving Average (SMA50)**: ₹{tech['SMA50']} | **200-Day Moving Average (SMA200)**: ₹{tech['SMA200']}
- **14-Period Wilder RSI**: **{tech['RSI14']}** *(Zone: {'🔥 Overbought (>70)' if tech['RSI14'] > 70 else '❄️ Oversold (<30)' if tech['RSI14'] < 30 else '✨ Optimal Accumulation Zone (30-70)'})*
- **Trailing 20-Day Return**: **{tech['Return20D_pct']}%**
- **Composite Score**: **{tech['score']}**
- **Action Recommendation**: **{tech['decision']}**

#### 🟢 Bullish Rationale:
{tech['reasons'] if tech['reasons'] else '*No specific bullish triggers registered.*'}

#### 🔴 Bearish / Risk Factors:
{tech['risks'] if tech['risks'] else '*No downside risks flagged.*'}
"""
        badge_html = render_decision_badge(tech['decision'], tech['score'])
    else:
        summary_md = f"⚠️ Insufficient historical data to compute all indicators for {clean_sym}."
        badge_html = ""

    return fig, summary_md, badge_html

def load_stock_deepdive_and_ai(symbol: str, custom_query: str):
    """Generate both live market intelligence and Gemini AI research report."""
    if not symbol:
        return "Please specify a stock ticker symbol.", ""

    clean_sym = symbol.strip().upper()
    df_prices = db.get_symbol_prices(clean_sym)
    if df_prices.empty:
        ok, _, _ = dl.fetch_one_stock(clean_sym)
        df_prices = db.get_symbol_prices(clean_sym)

    tech = ta.analyze_group(df_prices) if not df_prices.empty else {
        "symbol": clean_sym,
        "date": "N/A",
        "close": "N/A",
        "RSI14": "N/A",
        "SMA50": "N/A",
        "SMA200": "N/A",
        "Return20D_pct": "N/A",
        "score": 0,
        "decision": "HOLD / WAIT",
        "reasons": "No local price history",
        "risks": "Missing price data"
    }

    # Fetch live web intelligence via Tavily
    news_data = apis.search_stock_news(clean_sym)

    # Generate Gemini Report
    report = ai.generate_ai_stock_report(
        tech_data=tech,
        live_quote=None,
        news_data=news_data,
        user_query=custom_query
    )

    # Format Tavily News Snippets
    news_html = ""
    if news_data.get("results"):
        news_items = []
        for item in news_data["results"][:4]:
            t = item.get("title", "")
            u = item.get("url", "#")
            c = item.get("content", "")
            news_items.append(f"- **[{t}]({u})**\n  {c}")
        news_html = "\n\n".join(news_items)
    else:
        news_html = "No real-time web news articles retrieved."

    return report, news_html

def browse_universe_doc(doc_type: str, search_query: str):
    """Load and filter eligible security documents."""
    doc_key = "equity_l" if "Main" in doc_type else ("sme_equity_l" if "SME" in doc_type else "eligible_t0")
    df, _ = dl.load_security_document(doc_key)
    if search_query:
        sq = search_query.strip().upper()
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            mask = mask | df[col].astype(str).str.upper().str.contains(sq, na=False)
        df = df[mask]
    return df

def export_all_database_files():
    """Export daily prices, scan results, and failure logs as CSV files."""
    db.init_db()
    all_data = db.get_all_daily_prices()
    failures = db.get_failures()
    scan_df = ta.scan_all_symbols(all_data) if not all_data.empty else pd.DataFrame()

    p1 = BASE_DIR / "indian_stock_daily_prices.csv"
    p2 = BASE_DIR / "indian_stock_scan_results.csv"
    p3 = BASE_DIR / "indian_stock_download_failures.csv"

    all_data.to_csv(p1, index=False)
    scan_df.to_csv(p2, index=False)
    failures.to_csv(p3, index=False)

    return str(p1), str(p2), str(p3), str(DB_PATH)

def test_api_credentials(f_key: str, t_key: str, tav_key: str, g_key: str):
    """Test all 4 API credentials."""
    r_fin = apis.check_finnhub_health(f_key)
    r_twelve = apis.check_twelve_data_health(t_key)
    r_tavily = apis.check_tavily_health(tav_key)
    r_gemini = ai.check_gemini_health(g_key)

    res = f"""
### ⚡ External API Engine Diagnostics:
- **Finnhub Market Data**: {'✅ ' + r_fin['message'] if r_fin['status'] == 'ok' else '⚠️ ' + r_fin['message']}
- **Twelve Data Engine**: {'✅ ' + r_twelve['message'] if r_twelve['status'] == 'ok' else '⚠️ ' + r_twelve['message']}
- **Tavily Web Search**: {'✅ ' + r_tavily['message'] if r_tavily['status'] == 'ok' else '⚠️ ' + r_tavily['message']}
- **Google Gemini ({GEMINI_MODEL})**: {'✅ ' + r_gemini['message'] if r_gemini['status'] == 'ok' else '⚠️ ' + r_gemini['message']}
"""
    return res

def create_app():
    # Build Modern Dark Cyber Theme
    cyber_theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"]
    ).set(
        body_background_fill="#070a12",
        body_background_fill_dark="#070a12",
        block_background_fill="#0c1220",
        block_background_fill_dark="#0c1220",
        block_border_width="1px",
        block_border_color="rgba(56, 189, 248, 0.15)",
        block_border_color_dark="rgba(56, 189, 248, 0.15)"
    )

    with gr.Blocks(title="AI for Finance | Indian Stock Terminal") as demo:

        # Header Banner
        gr.HTML(
            """
            <div class="terminal-header">
              <div class="terminal-pill-row">
                <span class="terminal-pill live"><span class="dot"></span> NSE RADAR LIVE</span>
                <span class="terminal-pill">QUANT ALGO V2.4</span>
                <span class="terminal-pill">GEMINI 2.5 FLASH</span>
                <span class="terminal-pill">TAVILY MARKET INTELLIGENCE</span>
              </div>
              <h1 class="terminal-title">⚡ AI for Finance: Indian Stock Screener & Research Terminal</h1>
              <p class="terminal-subtitle">
                High-Frequency Quantitative Momentum Engine (Dual SMA 50/200, Wilder's RSI-14, 20-Day Performance)
                fused with real-time news intelligence and Google Gemini 2.5 Flash institutional memorandums.
              </p>
            </div>
            """
        )

        with gr.Tabs() as tabs:

            # ---------------- TAB 1: SCREENER & SCANNER ----------------
            with gr.Tab("🎯 Quantitative Screener & Radar", id="tab_screener"):
                with gr.Row():
                    with gr.Column(scale=3):
                        source_choice = gr.Radio(
                            choices=[
                                "Main Board Equities (EQUITY_L.csv)",
                                "SME Equities (SME_EQUITY_L.csv)",
                                "T+0 Settlement Eligible (Eligible_T0)",
                                "Custom File Upload",
                                "Manual Symbol Input"
                            ],
                            value="Main Board Equities (EQUITY_L.csv)",
                            label="Security Universe Source",
                            info="Select official eligible documents or provide your own stocks"
                        )
                        custom_file = gr.File(
                            label="Upload Custom Security CSV or Excel File",
                            file_types=[".csv", ".xlsx", ".xls"],
                            visible=False
                        )
                        custom_symbols_text = gr.Textbox(
                            label="Enter Symbols (Comma or Space Separated)",
                            placeholder="e.g. RELIANCE, TCS, INFY, HDFCBANK, TATAMOTORS",
                            lines=2,
                            visible=False
                        )

                    with gr.Column(scale=2):
                        with gr.Accordion("⚙️ Ingestion & Scan Parameters", open=True):
                            max_stocks = gr.Slider(
                                minimum=5,
                                maximum=500,
                                value=DEFAULT_MAX_STOCKS,
                                step=5,
                                label="Max Stocks to Fetch",
                                info="Limits batch size for fast execution"
                            )
                            start_date = gr.Textbox(
                                value=DEFAULT_START_DATE,
                                label="Historical Start Date",
                                placeholder="YYYY-MM-DD"
                            )
                            skip_existing = gr.Checkbox(
                                value=DEFAULT_SKIP_EXISTING,
                                label="Skip Already Downloaded Stocks",
                                info="Speeds up execution using local SQLite database"
                            )
                            pause_seconds = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=DEFAULT_PAUSE_SECONDS,
                                step=0.1,
                                label="Rate Limit Pause (Seconds)",
                                info="Pause between requests to respect rate limits"
                            )

                with gr.Row():
                    btn_scan = gr.Button("⚡ Run Download & Quantitative Scan", elem_classes=["btn-scan-primary"], scale=2)
                    btn_rescan = gr.Button("🔄 Re-Scan Local Database", elem_classes=["btn-rescan-secondary"], scale=1)

                status_banner = gr.Markdown("Ready to scan securities.")
                metrics_html = gr.HTML(render_metric_cards(0, 0, 0, 0))

                # Table Controls
                with gr.Row():
                    filter_decision = gr.Radio(
                        choices=["ALL", "BUY WATCH", "HOLD / WAIT", "SELL WATCH"],
                        value="ALL",
                        label="Filter by Signal Verdict"
                    )
                    search_box = gr.Textbox(
                        placeholder="Search Symbol (e.g. TATA, HDFC)...",
                        label="Search Symbol",
                        scale=2
                    )

                # Dataframe Display
                scan_results_df = gr.DataFrame(
                    headers=["symbol", "date", "close", "RSI14", "SMA50", "SMA200", "Return20D_pct", "score", "decision", "reasons", "risks"],
                    datatype=["str", "str", "number", "number", "number", "number", "number", "number", "str", "str", "str"],
                    label="Algorithmic Scan Results (Sorted by Score & 20D Return)",
                    interactive=False
                )

                with gr.Row():
                    download_scan_csv = gr.File(label="Export Scan Results CSV")

            # ---------------- TAB 2: INTERACTIVE CHARTS & TECHNICAL DEEP DIVE ----------------
            with gr.Tab("📈 High-Precision Technical Charts", id="tab_chart"):
                gr.Markdown("⚡ **Quick-Launch Active Bluechips:**")
                with gr.Row():
                    btn_q_rel = gr.Button("RELIANCE", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_tcs = gr.Button("TCS", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_infy = gr.Button("INFY", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_hdfc = gr.Button("HDFCBANK", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_tata = gr.Button("TATAMOTORS", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_bpcl = gr.Button("BPCL", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_sbin = gr.Button("SBIN", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_q_itc = gr.Button("ITC", elem_classes=["quick-ticker-btn"], size="sm")

                with gr.Row():
                    selected_symbol_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Select Scanned Stock",
                        allow_custom_value=True,
                        info="Choose from scan results or type any NSE symbol (e.g. RELIANCE, TCS)",
                        scale=3
                    )
                    btn_load_chart = gr.Button("📊 Render Interactive Chart", elem_classes=["btn-chart-amber"], scale=1)

                badge_box = gr.HTML("")
                chart_plot = gr.Plot(label="Price Action, Volume & 14-Day Wilder RSI")
                stock_summary_md = gr.Markdown("Select a stock to view technical indicators.")

            # ---------------- TAB 3: GEMINI AI ADVISOR ----------------
            with gr.Tab("🧠 Gemini 2.5 Institutional Research", id="tab_ai"):
                gr.Markdown(
                    """
                    ### Institutional AI Equity Research Synthesis
                    Fuses quantitative screener signals (SMA50, SMA200, RSI, 20D Return) with real-time news catalysts 
                    (via **Tavily**) and **Google Gemini 2.5 Flash** to generate a Wall-Street-grade investment memorandum.
                    """
                )
                gr.Markdown("⚡ **Quick-Select Stock for AI Memo:**")
                with gr.Row():
                    btn_ai_rel = gr.Button("RELIANCE", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_ai_tcs = gr.Button("TCS", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_ai_infy = gr.Button("INFY", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_ai_hdfc = gr.Button("HDFCBANK", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_ai_tata = gr.Button("TATAMOTORS", elem_classes=["quick-ticker-btn"], size="sm")
                    btn_ai_bpcl = gr.Button("BPCL", elem_classes=["quick-ticker-btn"], size="sm")

                with gr.Row():
                    ai_symbol_input = gr.Textbox(
                        label="Stock Symbol",
                        placeholder="e.g. BPCL, ASHOKLEY, RELIANCE, TCS",
                        scale=1
                    )
                    ai_query_input = gr.Textbox(
                        label="Analytical Focus / Question (Optional)",
                        placeholder="e.g. Invalidation stop loss level? Earnings impact on technical setup?",
                        scale=3
                    )
                    btn_generate_ai = gr.Button("🤖 Generate Institutional Memo", elem_classes=["btn-ai-emerald"], scale=1)

                with gr.Row():
                    with gr.Column(scale=3):
                        ai_report_output = gr.Markdown("Click 'Generate Institutional Memo' to produce analysis.")
                    with gr.Column(scale=2):
                        with gr.Accordion("📰 Real-Time Web Intelligence Catalysts (Tavily)", open=True):
                            ai_news_output = gr.Markdown("Live news catalysts will appear here.")

            # ---------------- TAB 4: ELIGIBLE SECURITIES BROWSER ----------------
            with gr.Tab("📁 Eligible Securities Universe", id="tab_eligible"):
                gr.Markdown(
                    """
                    ### Official NSE Eligible Security Documents
                    Browse and search the official documents that define the trading universe:
                    - **EQUITY_L.csv**: Main Board Securities (NSE)
                    - **SME_EQUITY_L.csv**: SME Platform Securities
                    - **Eligible_T0_Securities**: Securities Eligible for T+0 Settlement Cycle
                    """
                )
                with gr.Row():
                    doc_selector = gr.Radio(
                        choices=[
                            "Main Board Equities (EQUITY_L.csv)",
                            "SME Equities (SME_EQUITY_L.csv)",
                            "T+0 Settlement Eligible (Eligible_T0)"
                        ],
                        value="Main Board Equities (EQUITY_L.csv)",
                        label="Select Document"
                    )
                    doc_search = gr.Textbox(
                        placeholder="Search by Symbol, Company Name, ISIN...",
                        label="Search Filter"
                    )

                eligible_table = gr.DataFrame(
                    label="Official Security List",
                    interactive=False
                )

            # ---------------- TAB 5: DATABASE & DOWNLOADS ----------------
            with gr.Tab("💾 Database Coverage & Data Hub", id="tab_db"):
                gr.Markdown("### SQLite Database Coverage & Download Logs")
                with gr.Row():
                    btn_refresh_db = gr.Button("🔄 Refresh Database Status", elem_classes=["btn-rescan-secondary"])
                    btn_export_all = gr.Button("📥 Generate All Export Files", elem_classes=["btn-scan-primary"])

                with gr.Row():
                    file_daily = gr.File(label="Daily Prices CSV")
                    file_scan = gr.File(label="Scan Results CSV")
                    file_fails = gr.File(label="Download Failures CSV")
                    file_sqlite = gr.File(label="SQLite Database File")

                with gr.Tabs():
                    with gr.Tab("📊 Stored Symbols Coverage"):
                        coverage_df = gr.DataFrame(label="Database Table: daily_prices")
                    with gr.Tab("⚠️ Download Failures Log"):
                        failures_df = gr.DataFrame(label="Database Table: download_log (Failures)")

                with gr.Accordion("⚠️ Database Maintenance", open=False):
                    btn_reset_db = gr.Button("🗑️ Clear / Reset SQLite Database", elem_classes=["btn-reset-danger"])
                    reset_status = gr.Markdown("")

            # ---------------- TAB 6: SETTINGS & API CONFIGURATION ----------------
            with gr.Tab("⚡ Diagnostics & API Engine", id="tab_settings"):
                gr.Markdown(
                    """
                    ### API Keys & Engine Configuration
                    The application comes pre-configured with your API credentials. You can update or test them anytime below.
                    """
                )
                with gr.Row():
                    cfg_finnhub = gr.Textbox(
                        value=FINNHUB_API_KEY,
                        label="Finnhub API Key",
                        type="password"
                    )
                    cfg_twelve = gr.Textbox(
                        value=TWELVE_DATA_API_KEY,
                        label="Twelve Data API Key",
                        type="password"
                    )
                with gr.Row():
                    cfg_tavily = gr.Textbox(
                        value=TAVILY_API_KEY,
                        label="Tavily Web Search API Key",
                        type="password"
                    )
                    cfg_gemini = gr.Textbox(
                        value=GEMINI_API_KEY,
                        label="Google Gemini API Key",
                        type="password"
                    )

                btn_test_apis = gr.Button("🔍 Test All API Connections", elem_classes=["btn-rescan-secondary"])
                api_status_md = gr.Markdown("Click above to verify API connectivity.")

        # Reactive Bindings

        # 1. Source selector toggle
        source_choice.change(
            fn=handle_source_change,
            inputs=[source_choice],
            outputs=[custom_file, custom_symbols_text]
        )

        # State variable to hold full scan dataframe for filtering
        state_scan_df = gr.State(pd.DataFrame())

        # 2. Run scan
        def scan_and_cache(*args):
            scan_df, cards_html, status_msg, dropdown_update, csv_file = run_download_and_scan(*args)
            return scan_df, scan_df, cards_html, status_msg, dropdown_update, csv_file

        btn_scan.click(
            fn=scan_and_cache,
            inputs=[
                source_choice,
                custom_file,
                custom_symbols_text,
                max_stocks,
                start_date,
                skip_existing,
                pause_seconds
            ],
            outputs=[
                scan_results_df,
                state_scan_df,
                metrics_html,
                status_banner,
                selected_symbol_dropdown,
                download_scan_csv
            ]
        )

        # 3. Rescan local database
        def rescan_and_cache():
            scan_df, cards_html, status_msg, dropdown_update, csv_file = run_rescan_only()
            return scan_df, scan_df, cards_html, status_msg, dropdown_update, csv_file

        btn_rescan.click(
            fn=rescan_and_cache,
            inputs=[],
            outputs=[
                scan_results_df,
                state_scan_df,
                metrics_html,
                status_banner,
                selected_symbol_dropdown,
                download_scan_csv
            ]
        )

        # 4. Filter scan table
        filter_decision.change(
            fn=filter_scan_table,
            inputs=[state_scan_df, filter_decision, search_box],
            outputs=[scan_results_df]
        )
        search_box.change(
            fn=filter_scan_table,
            inputs=[state_scan_df, filter_decision, search_box],
            outputs=[scan_results_df]
        )

        # 5. Clicking a row in the scan table populates the dropdown and AI input
        def on_table_select(evt: gr.SelectData, current_df: pd.DataFrame):
            try:
                row_idx = evt.index[0]
                sym = str(current_df.iloc[row_idx]["symbol"])
                return sym, sym
            except Exception:
                return gr.update(), gr.update()

        scan_results_df.select(
            fn=on_table_select,
            inputs=[scan_results_df],
            outputs=[selected_symbol_dropdown, ai_symbol_input]
        )

        # 6. Render Chart
        btn_load_chart.click(
            fn=load_stock_chart,
            inputs=[selected_symbol_dropdown],
            outputs=[chart_plot, stock_summary_md, badge_box]
        )

        # Quick ticker bindings for charts
        for btn, sym in [
            (btn_q_rel, "RELIANCE"),
            (btn_q_tcs, "TCS"),
            (btn_q_infy, "INFY"),
            (btn_q_hdfc, "HDFCBANK"),
            (btn_q_tata, "TATAMOTORS"),
            (btn_q_bpcl, "BPCL"),
            (btn_q_sbin, "SBIN"),
            (btn_q_itc, "ITC")
        ]:
            def make_handler(s):
                return lambda: (s, *load_stock_chart(s))
            btn.click(
                fn=make_handler(sym),
                inputs=[],
                outputs=[selected_symbol_dropdown, chart_plot, stock_summary_md, badge_box]
            )

        # 7. Generate Gemini AI Report
        btn_generate_ai.click(
            fn=load_stock_deepdive_and_ai,
            inputs=[ai_symbol_input, ai_query_input],
            outputs=[ai_report_output, ai_news_output]
        )

        # Quick ticker bindings for AI Advisor
        for btn, sym in [
            (btn_ai_rel, "RELIANCE"),
            (btn_ai_tcs, "TCS"),
            (btn_ai_infy, "INFY"),
            (btn_ai_hdfc, "HDFCBANK"),
            (btn_ai_tata, "TATAMOTORS"),
            (btn_ai_bpcl, "BPCL")
        ]:
            def make_ai_handler(s):
                return lambda: (s, *load_stock_deepdive_and_ai(s, ""))
            btn.click(
                fn=make_ai_handler(sym),
                inputs=[],
                outputs=[ai_symbol_input, ai_report_output, ai_news_output]
            )

        # 8. Browse Eligible Documents
        doc_selector.change(
            fn=browse_universe_doc,
            inputs=[doc_selector, doc_search],
            outputs=[eligible_table]
        )
        doc_search.change(
            fn=browse_universe_doc,
            inputs=[doc_selector, doc_search],
            outputs=[eligible_table]
        )

        # 9. Database views
        def refresh_db_views():
            cov = db.get_coverage()
            fails = db.get_failures()
            return cov, fails

        btn_refresh_db.click(
            fn=refresh_db_views,
            inputs=[],
            outputs=[coverage_df, failures_df]
        )

        # 10. Export database files
        btn_export_all.click(
            fn=export_all_database_files,
            inputs=[],
            outputs=[file_daily, file_scan, file_fails, file_sqlite]
        )

        # 11. Reset Database
        def handle_reset_db():
            db.reset_database()
            cov, fails = refresh_db_views()
            return "Database successfully reset and initialized.", cov, fails

        btn_reset_db.click(
            fn=handle_reset_db,
            inputs=[],
            outputs=[reset_status, coverage_df, failures_df]
        )

        # 12. Test APIs
        btn_test_apis.click(
            fn=test_api_credentials,
            inputs=[cfg_finnhub, cfg_twelve, cfg_tavily, cfg_gemini],
            outputs=[api_status_md]
        )

    return demo

# Global cyber theme definition
cyber_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"]
).set(
    body_background_fill="#070a12",
    body_background_fill_dark="#070a12",
    block_background_fill="#0c1220",
    block_background_fill_dark="#0c1220",
    block_border_width="1px",
    block_border_color="rgba(56, 189, 248, 0.15)",
    block_border_color_dark="rgba(56, 189, 248, 0.15)"
)

demo = create_app()

if __name__ == "__main__":
    port_env = os.getenv("PORT")
    port = int(port_env) if port_env else None
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    share_flag = os.getenv("GRADIO_SHARE", "True").lower() in ("true", "1", "yes")
    print(f"Starting Indian Stock Advisor on {server_name}:{port or 'dynamic'} (share={share_flag}) ...", flush=True)
    demo.queue()
    try:
        demo.launch(
            server_name=server_name,
            server_port=port,
            theme=cyber_theme,
            css=custom_css,
            share=share_flag,
            show_error=True
        )
    except OSError:
        demo.launch(
            server_name=server_name,
            theme=cyber_theme,
            css=custom_css,
            share=share_flag,
            show_error=True
        )
