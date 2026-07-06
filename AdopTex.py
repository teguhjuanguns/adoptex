import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce
from statsforecast import StatsForecast
from statsforecast.models import ARIMA as SF_ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import streamlit as st
import sys
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
from urllib.parse import quote_plus


# (opsional) matikan warning di awal
warnings.filterwarnings("ignore")

import BPTK_Py
from BPTK_Py import Model
from BPTK_Py import sd_functions as sd


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co.com/27szfzyr/Adoptex.png");
             background-attachment: scroll;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co.com/27szfzyr/Adoptex.png");
             background-attachment: scroll;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# -----------------------------------------------------------------------------
# Section: Read Parameter Tables from Google Sheets
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

SHEET_ID = "1bPCjudpmpZGvzXpjNUQYXUFYm_ouWr6cmJR2D4AoeYo"
GID1 = 0  # tab Eksogen Variable

csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID1}"

df1 = pd.read_csv(csv_url)

df1["Variable"] = df1["Variable"].str.strip()
values = df1.set_index("Variable")["Total Value"].astype(float)

# -----------------------------------------------------------------------------

GID2 = 837812078
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID2}"

dfforecast1 = pd.read_csv(csv_url)
dfforecast1 = dfforecast1[["Year", "ICE"]].dropna(how="any")

dfforecast1["Year"] = dfforecast1["Year"].astype(float)
dfforecast1["ICE"]  = dfforecast1["ICE"].astype(float)
ICE_Motorcycle_Ownership = dfforecast1.values.tolist()
print(ICE_Motorcycle_Ownership)

# -----------------------------------------------------------------------------

starttime=values["Start Time"]
stoptime=values["Stop Time"]

model = Model(starttime=starttime,stoptime=stoptime,dt=values["Time Step"],name='Lithium Accu Adoption')

def power(base_val, exponent_val):
    """Custom power function for BPTK"""
    t = sd.time()
    return lambda t: pow(base_val(t), exponent_val(t))

warnings.filterwarnings("ignore")

Real_Discount_Rate = model.constant("Real_Discount_Rate")
Real_Discount_Rate.equation = values["Real Discount Rate"]

Initial_LFP_Price = model.constant("Initial_LFP_Price")
Initial_LFP_Price.equation = values["Initial LFP Price"]

LFPPIncRate = model.constant("LFPPIncRate")
LFPPIncRate.equation = values["Lithium Accu LFP Price Increase Rate"]
LFPPInc = model.flow("LFPPInc")
Price_of_Lithium_Accu_LFP = model.stock("Price_of_Lithium_Accu_LFP")
Price_of_Lithium_Accu_LFP.initial_value = Initial_LFP_Price
Price_of_Lithium_Accu_LFP.equation = LFPPInc
LFPPInc.equation = Price_of_Lithium_Accu_LFP * LFPPIncRate

Workers = model.constant("Workers")
Workers.equation = values["Workers"]

Battery_LFP = model.constant("Battery_LFP")
Battery_LFP.equation = values["Battery LFP"]

Packaging = model.constant("Packaging")
Packaging.equation = values["Packaging"]

Manual_Book = model.constant("Manual_Book")
Manual_Book.equation = values["Manual Book"]

model.points["ICE_Motorcycle_Ownership"] = ICE_Motorcycle_Ownership

ICE_Motorcycle_Ownership = model.converter("ICE_Motorcycle_Ownership")
ICE_Motorcycle_Ownership.equation = sd.lookup(sd.time(), "ICE_Motorcycle_Ownership")

Demand_Market_Potential = model.converter("Demand_Market_Potential")
Demand_Market_Potential.equation = ICE_Motorcycle_Ownership

Variable_Cost_LFP = model.converter("Variable_Cost_LFP")
Variable_Cost_LFP.equation = (Workers + Battery_LFP + Packaging + Manual_Book) * Demand_Market_Potential

# -----------------------------------------------------------------------------

Machinery_Depreciation = model.constant("Machinery_Depreciation")
Machinery_Depreciation.equation = values["Machinery Depreciation"]

Research_Amortization = model.constant("Research_Amortization")
Research_Amortization.equation = values["Research Amortization"]

Annual_Research_and_Development = model.constant("Annual_Research_and_Development")
Annual_Research_and_Development.equation = values["Annual Research and Development"]

Production_Manager = model.constant("Production_Manager")
Production_Manager.equation = values["Production Manager"]

Building_Depreciation = model.constant("Building_Depreciation")
Building_Depreciation.equation = values["Building Depreciation"]

Electricity = model.constant("Electricity")
Electricity.equation = values["Electricity"]

Sales = model.constant("Sales")
Sales.equation = values["Sales"]

Administration_and_General_Affair = model.constant("Administration_and_General_Affair")
Administration_and_General_Affair.equation = values["Administration and General Affair"]

Fix_Cost = model.converter("Fix_Cost")
# 110000 is production capacity can reached
Fix_Cost.equation = (Machinery_Depreciation + Research_Amortization + Annual_Research_and_Development + Production_Manager + Building_Depreciation + Electricity + Sales + Administration_and_General_Affair) * (Demand_Market_Potential/110000)

# -----------------------------------------------------------------------------

Total_Production_Cost_LFP = model.converter("Total_Production_Cost_LFP")
Total_Production_Cost_LFP.equation = Variable_Cost_LFP + Fix_Cost

Cost_Per_Unit_LFP = model.converter("Cost_Per_Unit_LFP")
Cost_Per_Unit_LFP._function_string = (
    "lambda model, t: ("
    "(model.memoize('Total_Production_Cost_LFP', t) / model.memoize('Demand_Market_Potential', t)) "
    "if model.memoize('Demand_Market_Potential', t) > 0.0 else 0.0"
    ")"
)
Cost_Per_Unit_LFP.generate_function()

# -----------------------------------------------------------------------------

Corporate_Income_Tax = model.constant("Corporate_Income_Tax")
Corporate_Income_Tax.equation = values["Corporate Income Tax"]

Inflow_LFP = model.converter("Inflow_LFP")
Inflow_LFP.equation = (Price_of_Lithium_Accu_LFP * Demand_Market_Potential) - Corporate_Income_Tax*(Price_of_Lithium_Accu_LFP * Demand_Market_Potential)

Outflow_LFP = model.converter("Outflow_LFP")
Outflow_LFP.equation = Total_Production_Cost_LFP

Annual_Cashflow_LFP = model.converter("Annual_Cashflow_LFP")
Annual_Cashflow_LFP.equation = Inflow_LFP - Outflow_LFP

# -----------------------------------------------------------------------------

Cell_Tester = model.constant("Cell_Tester")
Cell_Tester.equation = values["Cell Tester"]

Module_Tester = model.constant("Module_Tester")
Module_Tester.equation = values["Module Tester"]

Research_and_Development = model.constant("Research_and_Development")
Research_and_Development.equation = values["Research and Development"]

Lithium_Chamber = model.constant("Lithium_Chamber")
Lithium_Chamber.equation = values["Lithium Chamber"]

Chamber_Construction_Project = model.constant("Chamber_Construction_Project")
Chamber_Construction_Project.equation = values["Chamber Construction Project"]

Machine_Installation = model.constant("Machine_Installation")
Machine_Installation.equation = values["Machine Installation"]

Total_Investment_Cost = model.converter("Total_Investment_Cost")
# 110000 is production capacity can reached
Total_Investment_Cost.equation = (Cell_Tester + Module_Tester + Research_and_Development + Lithium_Chamber + Chamber_Construction_Project + Machine_Installation) * (Demand_Market_Potential/110000)

# -----------------------------------------------------------------------------

Payback_Period_LFP = model.converter("Payback_Period_LFP")
Payback_Period_LFP._function_string = (
    "lambda model, t: ("
    "(model.memoize('Total_Investment_Cost', t) / model.memoize('Annual_Cashflow_LFP', t)) "
    "if model.memoize('Annual_Cashflow_LFP', t) != 0.0 else 0.0"
    ")"
)
Payback_Period_LFP.generate_function()

# -----------------------------------------------------------------------------

Machine_Lifespan = model.constant("Machine_Lifespan")
Machine_Lifespan.equation = values["Machine Lifespan"]

Real_NPV_LFP = model.converter("Real_NPV_LFP")
Real_NPV_LFP._function_string = (
    "lambda model, t: ("
    "sum("
    "model.memoize('Annual_Cashflow_LFP', t) / "
    "pow(1.0 + model.memoize('Real_Discount_Rate', t), i) "
    "for i in range(1, int(model.memoize('Machine_Lifespan', t)) + 1)"
    ") - model.memoize('Total_Investment_Cost', t)"
    ")"
)
Real_NPV_LFP.generate_function()

# -----------------------------------------------------------------------------

Present_Value_Annuity_Factor_1 = model.converter("Present_Value_Annuity_Factor_1")
Present_Value_Annuity_Factor_1._function_string = (
    "lambda model, t: ("
    "1.0 - pow(1.0 + model.memoize('i1', t), "
    "-1.0 * model.memoize('Machine_Lifespan', t))"
    ") / "
    "model.memoize('i1', t)"
)
Present_Value_Annuity_Factor_1.generate_function()

Present_Value_Annuity_Factor_2 = model.converter("Present_Value_Annuity_Factor_2")
Present_Value_Annuity_Factor_2._function_string = (
    "lambda model, t: ("
    "1.0 - pow(1.0 + model.memoize('i2', t), "
    "-1.0 * model.memoize('Machine_Lifespan', t))"
    ") / "
    "model.memoize('i2', t)"
)
Present_Value_Annuity_Factor_2.generate_function()

NPV_1_LFP = model.converter("NPV_1_LFP")
NPV_1_LFP.equation = (Annual_Cashflow_LFP * Present_Value_Annuity_Factor_1) - Total_Investment_Cost

NPV_2_LFP = model.converter("NPV_2_LFP")
NPV_2_LFP.equation = (Annual_Cashflow_LFP * Present_Value_Annuity_Factor_2) - Total_Investment_Cost

i1 = model.constant("i1")
i1.equation = values["i1"]

i2 = model.constant("i2")
i2.equation = values["i2"]

# IRR_LFP = model.converter("IRR_LFP")
# IRR_LFP._function_string = (
#     "lambda model, t: ("
#     "model.memoize('i1', t) + "
#     "(model.memoize('NPV_1_LFP', t) / "
#     "(model.memoize('NPV_1_LFP', t) - model.memoize('NPV_2_LFP', t))) * "
#     "(model.memoize('i2', t) - model.memoize('i1', t))"
#     ")"
# )
# IRR_LFP.generate_function()

IRR_LFP = model.converter("IRR_LFP")

# Wrap the entire formula in parentheses and add * 100 at the end
IRR_LFP._function_string = (
    "lambda model, t: (("
    "model.memoize('i1', t) + "
    "(model.memoize('NPV_1_LFP', t) / "
    "(model.memoize('NPV_1_LFP', t) - model.memoize('NPV_2_LFP', t))) * "
    "(model.memoize('i2', t) - model.memoize('i1', t)))*100 "
    "if model.memoize('NPV_1_LFP', t) > 0 else 0.0"
    ")"
)

IRR_LFP.generate_function()

# REMOVE THESE LINES ENTIRELY:
# IRR_LFP = model.converter("IRR_LFP")
# IRR_LFP.equation = IRR_LFP * 100

# -----------------------------------------------------------------------------

Risk_Rate = model.constant("Risk_Rate")
Risk_Rate.equation = values["Risk Rate"]

Inflation_Rate = model.constant("Inflation_Rate")
Inflation_Rate.equation = values["Inflation Rate"]

MARR = model.converter("MARR")
MARR._function_string = (
    "lambda model, t: ("
    "(1.0 + (model.memoize('Real_Discount_Rate', t) + model.memoize('Risk_Rate', t))) * "
    "(1.0 + model.memoize('Inflation_Rate', t))"
    ") - 1.0"
)
MARR.generate_function()

# -----------------------------------------------------------------------------
# Section: Register the model with BPTK
# -----------------------------------------------------------------------------

import BPTK_Py
bptk = BPTK_Py.bptk()
bptk.register_model(model)

# Setup Scenarios: Tax Sensitivity & LFP Price Sensitivity
base_tax = values["Corporate Income Tax"]
base_lfp_price = values["Initial LFP Price"]

scenario_managers_dict = {
    "smCorporate_Income_Tax": {
        "model": model,
        "scenarios": {
            "1_tax_minus_10": {
                "constants": {
                    "Corporate_Income_Tax": base_tax * 0.90
                }
            },
            "2_base": {},
            "3_tax_plus_10": {
                "constants": {
                    "Corporate_Income_Tax": base_tax * 1.10
                }
            }
        }
    },
    "smInitial_LFP_Price": {
        "model": model,
        "scenarios": {
            "1_price_minus_10": {
                "constants": {
                    "Initial_LFP_Price": base_lfp_price * 0.90
                }
            },
            "2_base": {},
            "3_price_plus_10": {
                "constants": {
                    "Initial_LFP_Price": base_lfp_price * 1.10
                }
            }
        }
    }
}
bptk.register_scenario_manager(scenario_managers_dict)

# -----------------------------------------------------------------------------
# Section: Pre-compute data series for each variable
# -----------------------------------------------------------------------------

try:
    o1 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Demand_Market_Potential"],
        series_names={
            "Demand_Market_Potential" : "Demand Market Potential (Units)",
        },
        return_df=True
    )
    o2 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Total_Investment_Cost"],
        series_names={
            "Total_Investment_Cost" : "Total Investment Cost (IDR)",
        },     
        return_df=True
    )
    o3 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Total_Production_Cost_LFP"],
        series_names={
            "Total_Production_Cost_LFP" : "Total Production Cost LFP (IDR)",
        },    
        return_df=True
    )
    o4 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Annual_Cashflow_LFP"],
        series_names={
            "Annual_Cashflow_LFP" : "Annual Cashflow LFP",
        },
        return_df=True
    )

    o5 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Payback_Period_LFP"],
        series_names={
            "Payback_Period_LFP" : "Payback Period LFP (Years)",
        },    
        return_df=True
    )
    o6 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["Real_NPV_LFP"],
        series_names={
            "Real_NPV_LFP" : "Real NPV LFP (IDR)",
        },      
        return_df=True
    )
    o7 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["IRR_LFP"],
        series_names={
            "IRR_LFP" : "IRR LFP (IDR)",
        },
        return_df=True
    )
    o8 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers="smLithium accu adoption",
        equations=["MARR"],
        series_names={},
        return_df=True,
    )

    precomputed_series = {
        "Demand_Market_Potential_LFP": o1,
        "Total_Investment_Cost": o2,
        "Total_Production_Cost_LFP": o3,
        "Annual_Cashflow_LFP": o4,
        "Payback_Period_LFP": o5,
        "Real_NPV_LFP": o6,
        "IRR_LFP": o7,
        "MARR": o8,
    }
except Exception as _exc_precompute:
    precomputed_series = {}


# -----------------------------------------------------------------------------
# Section: Streamlit UI
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #023286;    /* Sidebar background color */
    }
    [data-testid="stSidebar"] * {
        color: #000000;               /* Sidebar text color changed to white for contrast */
    }
    .justify-text {
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Navigation Menu")
st.sidebar.write("Please select a menu below:")

# 1. SIDEBAR MENU
menu_option = st.sidebar.radio(
    "Select Page",
    ("Simulation Result", "Sensitivity Analysis", "Concept Foundation"),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Button linking to the parameter database
if st.sidebar.button("Update Database"):
    st.sidebar.markdown(
        "[Open Parameter Database](https://docs.google.com/spreadsheets/d/"
        "1bPCjudpmpZGvzXpjNUQYXUFYm_ouWr6cmJR2D4AoeYo/edit?usp=sharing)"
    )

# --- Y-AXIS FORMATTER FOR GRAPHS (Shared globally) ---
def format_dynamic(x, pos):
    abs_x = abs(x)
    if abs_x >= 1e12: return f'{x * 1e-12:g}T'
    elif abs_x >= 1e9: return f'{x * 1e-9:g}B'
    elif abs_x >= 1e6: return f'{x * 1e-6:g}M'
    elif abs_x >= 1e3: return f'{x * 1e-3:g}K'
    else: return f'{x:g}'


# =============================================================================
# PAGE 1: SIMULATION RESULT PAGE
# =============================================================================

if menu_option == "Simulation Result":
    
    st.markdown(
        """
        <p class="justify-text">
            <b>AdopTex</b> is a business analytics tool integrated with data-driven Python system dynamics, machine learning, and AI to assess the potential adoption of Lithium Motorcycle Starter Pack Accumulators in Indonesia. 
            Below are the full simulation results for the financial and market parameters.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()

    # Helper function to clean and format DataFrame
    def format_dataframe(df_raw, var_name_str):
        if df_raw is None or df_raw.empty:
            return pd.DataFrame() 
        
        df_temp = df_raw.copy()
        if isinstance(df_temp.columns, pd.MultiIndex):
            df_temp.columns = ["_".join(col).strip() for col in df_temp.columns.values]
        
        match_cols = [col for col in df_temp.columns if var_name_str in col]
        var_col = match_cols[0] if match_cols else df_temp.columns[0]
        
        table_df = df_temp[[var_col]].copy().reset_index()
        table_df.columns = ["Time", var_name_str]
        table_df[var_name_str] = table_df[var_name_str].round(6)
        
        return table_df

    df_market_share_lfp = format_dataframe(o1, "Demand Market Potential LFP")
    df_inv_cost         = format_dataframe(o2, "Total Investment Cost")
    df_prod_cost_lfp    = format_dataframe(o3, "Total Production Cost LFP")
    df_cashflow_lfp     = format_dataframe(o4, "Annual Cashflow LFP")
    df_payback_lfp      = format_dataframe(o5, "Payback Period LFP")
    df_npv_lfp          = format_dataframe(o6, "Real NPV LFP")
    df_irr_lfp          = format_dataframe(o7, "IRR LFP")
    df_marr             = format_dataframe(o8, "MARR")

    list_df_lfp = [
        df_market_share_lfp, df_prod_cost_lfp, 
        df_cashflow_lfp, df_payback_lfp, df_npv_lfp, df_irr_lfp, 
        df_inv_cost, df_marr
    ]

    def merge_dataframes(df_list):
        valid_dfs = [df for df in df_list if not df.empty]
        if not valid_dfs:
            return pd.DataFrame()
        return reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), valid_dfs)

    all_df1 = merge_dataframes(list_df_lfp)
    all_dict1 = all_df1.to_dict(orient="records") if not all_df1.empty else []

    # --- HELPER FUNCTION FOR TABLE + GRAPH ---
    def render_metric_with_chart(title, df):
        st.markdown(f"#### {title}")
        if not df.empty:
            chart_df = df.set_index("Time") if "Time" in df.columns else df
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown("**Data Table**")
                st.dataframe(df, height=250, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Trend Chart**")
                st.line_chart(chart_df, height=250)
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning(f"Data for {title} is not available.")

    # --- PART 1: MARKET SHARE ---
    st.subheader("⚡ Demand Market Potential")
    render_metric_with_chart("Demand Market Potential LFP (Units)", df_market_share_lfp)

    st.divider()

    # --- PART 2: OTHER FINANCIAL METRICS ---
    st.subheader("📊 Financial & Production Metrics")
    
    render_metric_with_chart("Total Investment Cost (IDR)", df_inv_cost)
    render_metric_with_chart("Total Production Cost (IDR)", df_prod_cost_lfp)
    render_metric_with_chart("Annual Cashflow (IDR)", df_cashflow_lfp)
    render_metric_with_chart("Payback Period (Years)", df_payback_lfp)
    render_metric_with_chart("Real NPV (IDR)", df_npv_lfp)
    render_metric_with_chart("IRR (%)", df_irr_lfp)
    render_metric_with_chart("MARR (%)", df_marr)

    st.divider()

    # --- ASK AI ---
    def ask_ai_button(label: str, prompt_text: str):
        url = "https://chatgpt.com/?q=" + quote_plus(prompt_text)
        components.html(
            f"""
            <button
              id="askai_btn"
              style="
                padding: 0.5rem 0.9rem;
                border-radius: 0.5rem;
                border: 1px solid rgba(255,255,255,0.25);
                background: rgba(255,255,255,0.08);
                color: white;
                cursor: pointer;
              "
            >
              {label}
            </button>
            <script>
              const btn = document.getElementById("askai_btn");
              btn.addEventListener("click", async () => {{
                const text = {json.dumps(prompt_text)};
                const url = {json.dumps(url)};
                try {{
                  await navigator.clipboard.writeText(text);
                }} catch (e) {{ }}
                window.open(url, "_blank");
              }});
            </script>
            """,
            height=55,
        )

    question = "Please analyze the following tables systematically and in-depth based on economic analysis. Explain the main patterns, visible trends, and important insights that can be drawn from the LFP data. If relevant, include logical interpretations and implications of the data."
    prompt_text = f"--- LFP Data ---\n{all_dict1}\n\n{question}"
    ask_ai_button("Ask AI", prompt_text)


# =============================================================================
# PAGE 2: SENSITIVITY ANALYSIS PAGE
# =============================================================================
elif menu_option == "Sensitivity Analysis":
    st.title("📉 Sensitivity Analysis")
    st.write("Explore how changes in specific parameters impact the overall financial metrics of the project.")
    
    st.divider()

    # --- Helper Function for Sensitivity Plotting ---
    def render_sensitivity_metric(manager, scenarios, equation, metric_label, sensitivity_type):
        df_table = bptk.plot_scenarios(
            scenarios=scenarios,
            scenario_managers=manager,
            equations=[equation],
            return_df=True
        )
        
        # Format columns appropriately
        df_table.columns = [
            f"-10% {sensitivity_type} {metric_label}", 
            f"Base {sensitivity_type} {metric_label}", 
            f"+10% {sensitivity_type} {metric_label}"
        ]
        
        st.markdown(f"**{metric_label} Data Table**")
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        st.markdown(f"**{metric_label} Trend Chart**")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_table.plot(ax=ax, color=['green', 'blue', 'red'], linewidth=2, marker='o', markersize=4)

        # X-Axis format
        start_year = int(df_table.index.min())
        end_year = int(df_table.index.max())
        years = np.arange(start_year, end_year + 1, 1)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        
        # Y-Axis format
        ax.yaxis.set_major_formatter(FuncFormatter(format_dynamic))
        unit = "(%)" if metric_label == "IRR" else ("(Years)" if metric_label == "Payback Period" else "(IDR)")
        
        # Labels and Title
        title_context = "Corporate Income Tax" if sensitivity_type == "Tax" else "Initial LFP Price"
        ax.set_title(f"Sensitivity Analysis: Impact of {title_context} on {metric_label}", fontsize=14, pad=15)
        ax.set_ylabel(f"{metric_label} {unit}", fontsize=12)
        ax.set_xlabel("Year", fontsize=12)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), borderaxespad=0, frameon=False, columnspacing=1.0, ncol=3)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=11)
        fig.tight_layout()
        
        st.pyplot(fig)
        st.markdown("<br><br>", unsafe_allow_html=True)


    # --- PART 1: SENSITIVITY ANALYSIS (CORPORATE INCOME TAX) ---
    st.subheader("📊 Impact of Corporate Income Tax")
    tax_scenarios = ["1_tax_minus_10", "2_base", "3_tax_plus_10"]
    
    render_sensitivity_metric("smCorporate_Income_Tax", tax_scenarios, "IRR_LFP", "IRR", "Tax")
    render_sensitivity_metric("smCorporate_Income_Tax", tax_scenarios, "Real_NPV_LFP", "NPV", "Tax")
    render_sensitivity_metric("smCorporate_Income_Tax", tax_scenarios, "Payback_Period_LFP", "Payback Period", "Tax")

    st.divider()

    # --- PART 2: SENSITIVITY ANALYSIS (INITIAL LFP PRICE) ---
    st.subheader("📊 Impact of Initial LFP Price")
    price_scenarios = ["1_price_minus_10", "2_base", "3_price_plus_10"]
    
    render_sensitivity_metric("smInitial_LFP_Price", price_scenarios, "IRR_LFP", "IRR", "Price")
    render_sensitivity_metric("smInitial_LFP_Price", price_scenarios, "Real_NPV_LFP", "NPV", "Price")
    render_sensitivity_metric("smInitial_LFP_Price", price_scenarios, "Payback_Period_LFP", "Payback Period", "Price")


# =============================================================================
# PAGE 3: CONCEPT FOUNDATION
# =============================================================================
elif menu_option == "Concept Foundation":
    st.title("💡 Concept Foundation")
    st.write("Below are the foundational concepts, methodologies, and references used in the AdopTex simulation model.")
    
    st.divider()

    concept_data = {
        "Concept / Methodology": [
            "Techno-Economic Analysis",
            "WTP Probability (Probit Analysis)",
            "ML of WTP Probability Prediction",
            "Total Cost Ownership"
        ],
        "Reference Link": [
            "https://www.mdpi.com/2227-7080/6/3/73",
            "https://books.google.co.id/books/about/Probit_Analysis.html?id=Eu2pPwAACAAJ&redir_esc=y",
            "https://jbhender.github.io/Stats506/F18/GP/Group14.html",
            "https://josi.ft.unand.ac.id/index.php/josi/article/view/78"
        ]
    }
    
    df_concept = pd.DataFrame(concept_data)
    
    st.dataframe(
        df_concept,
        column_config={
            "Concept / Methodology": st.column_config.TextColumn(
                "Concept & Methodology",
                width="medium",
            ),
            "Reference Link": st.column_config.LinkColumn(
                "Reference Source",
                help="Click to open the reference document",
                display_text="Open Reference 🔗",  
                width="large"
            )
        },
        hide_index=True,          
        use_container_width=True  
    )
