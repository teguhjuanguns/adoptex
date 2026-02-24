import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
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

# -----------------------------------------------------------------------------
# Section: Read Parameter Tables from Google Sheets
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

SHEET_ID = "1Bszu5DtBg_2Oyf9PCQc2Morx8dJAAAzZd4MLKHi09-I"
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

Recharging_Time_of_Dry_Accu = model.constant("Recharging_Time_of_Dry_Accu")
Recharging_Time_of_Dry_Accu.equation = values["Recharging Time of Dry Accu"]

Recharging_Frequency_of_Dry_Accu = model.constant("Recharging_Frequency_of_Dry_Accu")
Recharging_Frequency_of_Dry_Accu.equation = values["Recharging Frequency of Dry Accu"]

SEInc_Rate = model.constant("SEInc_Rate")
SEInc_Rate.equation = values["Social External Cost Increase Rate"]
SEIncrease = model.flow("SEIncrease")
Social_External_Cost = model.stock("Social_External_Cost")
Social_External_Cost.initial_value = values["Initial Social External Cost"]
Social_External_Cost.equation = SEIncrease
SEIncrease.equation = Social_External_Cost * SEInc_Rate

Recharging_Loss_Cost_of_Dry_Accu = model.converter("Recharging_Loss_Cost_of_Dry_Accu")
Recharging_Loss_Cost_of_Dry_Accu.equation = Recharging_Time_of_Dry_Accu * Recharging_Frequency_of_Dry_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Maintenance_Frequency_of_Dry_Accu = model.constant("Maintenance_Frequency_of_Dry_Accu")
Maintenance_Frequency_of_Dry_Accu.equation = values["Maintenance Frequency of Dry Accu"]

Maintenance_Time_of_Dry_Accu = model.constant("Maintenance_Time_of_Dry_Accu")
Maintenance_Time_of_Dry_Accu.equation = values["Maintenance Time of Dry Accu"]

Maintenance_Loss_Cost_of_Dry_Accu = model.converter("Maintenance_Loss_Cost_of_Dry_Accu")
Maintenance_Loss_Cost_of_Dry_Accu.equation = Maintenance_Frequency_of_Dry_Accu * Maintenance_Time_of_Dry_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Pb = model.constant("Pb")
Pb.equation = values["Pb"]

Average_Expected_Years_Kept_of_Dry_Accu = model.constant("Average_Expected_Years_Kept_of_Dry_Accu")
Average_Expected_Years_Kept_of_Dry_Accu.equation = values["Average Expected Years Kept of Dry Accu"]

Pb_Annual_Emissions_Load_Dry_Accu = model.converter("Pb_Annual_Emissions_Load_Dry_Accu")
Pb_Annual_Emissions_Load_Dry_Accu.equation = Pb / Average_Expected_Years_Kept_of_Dry_Accu

CPb = model.constant("CPb")
CPb.equation = values["CPb"]

Health_Cost_of_Dry_Accu = model.converter("Health_Cost_of_Dry_Accu")
Health_Cost_of_Dry_Accu.equation = Pb_Annual_Emissions_Load_Dry_Accu * CPb

Social_Impact_Cost_of_Dry_Accu = model.converter("Social_Impact_Cost_of_Dry_Accu")
Social_Impact_Cost_of_Dry_Accu.equation = (Recharging_Loss_Cost_of_Dry_Accu + Maintenance_Loss_Cost_of_Dry_Accu + Health_Cost_of_Dry_Accu)

# -----------------------------------------------------------------------------

Maintenance_Price_of_Dry_Accu = model.constant("Maintenance_Price_of_Dry_Accu")
Maintenance_Price_of_Dry_Accu.equation = values["Maintenance Price of Dry Accu"]

Maintenance_Cost_of_Dry_Accu = model.converter("Maintenance_Cost_of_Dry_Accu")
Maintenance_Cost_of_Dry_Accu.equation = Maintenance_Frequency_of_Dry_Accu * Maintenance_Price_of_Dry_Accu

# -----------------------------------------------------------------------------

Capacity_of_Dry_Accu = model.constant("Capacity_of_Dry_Accu")
Capacity_of_Dry_Accu.equation = values["Capacity of Dry Accu"]

Voltage_of_Dry_Accu = model.constant("Voltage_of_Dry_Accu")
Voltage_of_Dry_Accu.equation = values["Voltage of Dry Accu"]

Charging_Efficiency_of_Dry_Accu = model.constant("Charging_Efficiency_of_Dry_Accu")
Charging_Efficiency_of_Dry_Accu.equation = values["Charging Efficiency of Dry Accu"]

EPIRate = model.constant("EPIRate")
EPIRate.equation = values["Electricity Price Annual Increase Rate"]
Electricity_Price_Increase = model.flow("Electricity_Price_Increase")
Electricity_Price = model.stock("Electricity_Price")
Electricity_Price.initial_value = values["Initial Electricity Tariff for Charging"]
Electricity_Price.equation = Electricity_Price_Increase
Electricity_Price_Increase.equation = EPIRate * Electricity_Price

Energy_Consumption_of_Dry_Accu = model.converter("Energy_Consumption_of_Dry_Accu")
Energy_Consumption_of_Dry_Accu.equation = (((Voltage_of_Dry_Accu * Capacity_of_Dry_Accu)/1000)/Charging_Efficiency_of_Dry_Accu) * Electricity_Price * Recharging_Frequency_of_Dry_Accu

Operational_Cost_of_Dry_Accu = model.converter("Operational_Cost_of_Dry_Accu")
Operational_Cost_of_Dry_Accu.equation = Energy_Consumption_of_Dry_Accu + Maintenance_Cost_of_Dry_Accu

# -----------------------------------------------------------------------------

DAPIncRate = model.constant("DAPIncRate")
DAPIncRate.equation = values["Dry Accu Price Increase Rate"]
DAPInc = model.flow("DAPInc")
Price_of_Dry_Accu = model.stock("Price_of_Dry_Accu")
Price_of_Dry_Accu.initial_value = values["Initial DA Price"]
Price_of_Dry_Accu.equation = DAPInc
DAPInc.equation = Price_of_Dry_Accu * DAPIncRate

Depreciation_Rate_of_Dry_Accu = model.constant("Depreciation_Rate_of_Dry_Accu")
Depreciation_Rate_of_Dry_Accu.equation = values["Depreciation Rate of Dry Accu"]

Real_Discount_Rate = model.constant("Real_Discount_Rate")
Real_Discount_Rate.equation = values["Real Discount Rate"]

Resale_Value_of_Dry_Accu = model.converter("Resale_Value_of_Dry_Accu")
Resale_Value_of_Dry_Accu._function_string = (
    "lambda model, t: ("
    "(pow(1.0 - model.memoize('Depreciation_Rate_of_Dry_Accu', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Dry_Accu', t)) "
    "* model.memoize('Price_of_Dry_Accu', t) "
    "* model.memoize('Real_Discount_Rate', t)) "
    "/ "
    "(pow(1.0 + model.memoize('Real_Discount_Rate', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Dry_Accu', t)) - 1.0)"
    ")"
)
Resale_Value_of_Dry_Accu.generate_function()

Life_Cycle_Cost_of_Dry_Accu = model.converter("Life_Cycle_Cost_of_Dry_Accu")
Life_Cycle_Cost_of_Dry_Accu.equation = (Price_of_Dry_Accu/Average_Expected_Years_Kept_of_Dry_Accu) + Operational_Cost_of_Dry_Accu - Resale_Value_of_Dry_Accu

# -----------------------------------------------------------------------------

Cost_Ownership_of_Dry_Accu = model.converter("Cost_Ownership_of_Dry_Accu")
Cost_Ownership_of_Dry_Accu.equation = (Social_Impact_Cost_of_Dry_Accu + Life_Cycle_Cost_of_Dry_Accu)/1e+06

Recharging_Time_of_Wet_Accu = model.constant("Recharging_Time_of_Wet_Accu")
Recharging_Time_of_Wet_Accu.equation = values["Recharging Time of Wet Accu"]

Recharging_Frequency_of_Wet_Accu = model.constant("Recharging_Frequency_of_Wet_Accu")
Recharging_Frequency_of_Wet_Accu.equation = values["Recharging Frequency of Wet Accu"]

Recharging_Loss_Cost_of_Wet_Accu = model.converter("Recharging_Loss_Cost_of_Wet_Accu")
Recharging_Loss_Cost_of_Wet_Accu.equation = Recharging_Time_of_Wet_Accu * Recharging_Frequency_of_Wet_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Maintenance_Frequency_of_Wet_Accu = model.constant("Maintenance_Frequency_of_Wet_Accu")
Maintenance_Frequency_of_Wet_Accu.equation = values["Maintenance Frequency of Wet Accu"]

Maintenance_Time_of_Wet_Accu = model.constant("Maintenance_Time_of_Wet_Accu")
Maintenance_Time_of_Wet_Accu.equation = values["Maintenance Time of Wet Accu"]

Maintenance_Loss_Cost_of_Wet_Accu = model.converter("Maintenance_Loss_Cost_of_Wet_Accu")
Maintenance_Loss_Cost_of_Wet_Accu.equation = Maintenance_Frequency_of_Wet_Accu * Maintenance_Time_of_Wet_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Average_Expected_Years_Kept_of_Wet_Accu = model.constant("Average_Expected_Years_Kept_of_Wet_Accu")
Average_Expected_Years_Kept_of_Wet_Accu.equation = values["Average Expected Years Kept of Wet Accu"]

Pb_Annual_Emissions_Load_Wet_Accu = model.converter("Pb_Annual_Emissions_Load_Wet_Accu")
Pb_Annual_Emissions_Load_Wet_Accu.equation = Pb / Average_Expected_Years_Kept_of_Wet_Accu

Health_Cost_of_Wet_Accu = model.converter("Health_Cost_of_Wet_Accu")
Health_Cost_of_Wet_Accu.equation = Pb_Annual_Emissions_Load_Wet_Accu * CPb

Social_Impact_Cost_of_Wet_Accu = model.converter("Social_Impact_Cost_of_Wet_Accu")
Social_Impact_Cost_of_Wet_Accu.equation = (Recharging_Loss_Cost_of_Wet_Accu + Maintenance_Loss_Cost_of_Wet_Accu + Health_Cost_of_Wet_Accu)

# -----------------------------------------------------------------------------

Maintenance_Price_of_Wet_Accu = model.constant("Maintenance_Price_of_Wet_Accu")
Maintenance_Price_of_Wet_Accu.equation = values["Maintenance Price of Wet Accu"]

Maintenance_Cost_of_Wet_Accu = model.converter("Maintenance_Cost_of_Wet_Accu")
Maintenance_Cost_of_Wet_Accu.equation = Maintenance_Frequency_of_Wet_Accu * Maintenance_Price_of_Wet_Accu

# -----------------------------------------------------------------------------

Capacity_of_Wet_Accu = model.constant("Capacity_of_Wet_Accu")
Capacity_of_Wet_Accu.equation = values["Capacity of Wet Accu"]

Voltage_of_Wet_Accu = model.constant("Voltage_of_Wet_Accu")
Voltage_of_Wet_Accu.equation = values["Voltage of Wet Accu"]

Charging_Efficiency_of_Wet_Accu = model.constant("Charging_Efficiency_of_Wet_Accu")
Charging_Efficiency_of_Wet_Accu.equation = values["Charging Efficiency of Wet Accu"]

Energy_Consumption_of_Wet_Accu = model.converter("Energy_Consumption_of_Wet_Accu")
Energy_Consumption_of_Wet_Accu.equation = (((Voltage_of_Wet_Accu * Capacity_of_Wet_Accu)/1000)/Charging_Efficiency_of_Wet_Accu) * Electricity_Price * Recharging_Frequency_of_Wet_Accu

Operational_Cost_of_Wet_Accu = model.converter("Operational_Cost_of_Wet_Accu")
Operational_Cost_of_Wet_Accu.equation = Energy_Consumption_of_Wet_Accu + Maintenance_Cost_of_Wet_Accu

# -----------------------------------------------------------------------------

WAPIncRate = model.constant("WAPIncRate")
WAPIncRate.equation = values["Wet Accu Price Increase Rate"]
WAPInc = model.flow("WAPInc")
Price_of_Wet_Accu = model.stock("Price_of_Wet_Accu")
Price_of_Wet_Accu.initial_value = values["Initial WA Price"]
Price_of_Wet_Accu.equation = WAPInc
WAPInc.equation = Price_of_Wet_Accu * WAPIncRate

Depreciation_Rate_of_Wet_Accu = model.constant("Depreciation_Rate_of_Wet_Accu")
Depreciation_Rate_of_Wet_Accu.equation = values["Depreciation Rate of Wet Accu"]

Resale_Value_of_Wet_Accu = model.converter("Resale_Value_of_Wet_Accu")
Resale_Value_of_Wet_Accu._function_string = (
    "lambda model, t: ("
    "(pow(1.0 - model.memoize('Depreciation_Rate_of_Wet_Accu', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Wet_Accu', t)) "
    "* model.memoize('Price_of_Wet_Accu', t) "
    "* model.memoize('Real_Discount_Rate', t)) "
    "/ "
    "(pow(1.0 + model.memoize('Real_Discount_Rate', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Wet_Accu', t)) - 1.0)"
    ")"
)
Resale_Value_of_Wet_Accu.generate_function()

Life_Cycle_Cost_of_Wet_Accu = model.converter("Life_Cycle_Cost_of_Wet_Accu")
Life_Cycle_Cost_of_Wet_Accu.equation = (Price_of_Wet_Accu/Average_Expected_Years_Kept_of_Wet_Accu) + Operational_Cost_of_Wet_Accu - Resale_Value_of_Wet_Accu

# -----------------------------------------------------------------------------

Cost_Ownership_of_Wet_Accu = model.converter("Cost_Ownership_of_Wet_Accu")
Cost_Ownership_of_Wet_Accu.equation = (Social_Impact_Cost_of_Wet_Accu + Life_Cycle_Cost_of_Wet_Accu)/1e+06

Recharging_Time_of_Lithium_Accu = model.constant("Recharging_Time_of_Lithium_Accu")
Recharging_Time_of_Lithium_Accu.equation = values["Recharging Time of Lithium Accu"]

Recharging_Frequency_of_Lithium_Accu = model.constant("Recharging_Frequency_of_Lithium_Accu")
Recharging_Frequency_of_Lithium_Accu.equation = values["Recharging Frequency of Lithium Accu"]

Recharging_Loss_Cost_of_Lithium_Accu = model.converter("Recharging_Loss_Cost_of_Lithium_Accu")
Recharging_Loss_Cost_of_Lithium_Accu.equation = Recharging_Time_of_Lithium_Accu * Recharging_Frequency_of_Lithium_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Maintenance_Frequency_of_Lithium_Accu = model.constant("Maintenance_Frequency_of_Lithium_Accu")
Maintenance_Frequency_of_Lithium_Accu.equation = values["Maintenance Frequency of Lithium Accu"]

Maintenance_Time_of_Lithium_Accu = model.constant("Maintenance_Time_of_Lithium_Accu")
Maintenance_Time_of_Lithium_Accu.equation = values["Maintenance Time of Lithium Accu"]

Maintenance_Loss_Cost_of_Lithium_Accu = model.converter("Maintenance_Loss_Cost_of_Lithium_Accu")
Maintenance_Loss_Cost_of_Lithium_Accu.equation = Maintenance_Frequency_of_Lithium_Accu * Maintenance_Time_of_Lithium_Accu * Social_External_Cost

# -----------------------------------------------------------------------------

Li = model.constant("Li")
Li.equation = values["Li"]

Average_Expected_Years_Kept_of_Lithium_Accu = model.constant("Average_Expected_Years_Kept_of_Lithium_Accu")
Average_Expected_Years_Kept_of_Lithium_Accu.equation = values["Average Expected Years Kept of Lithium Accu"]

Li_Annual_Emissions_Load_Lithium_Accu = model.converter("Li_Annual_Emissions_Load_Lithium_Accu")
Li_Annual_Emissions_Load_Lithium_Accu.equation = Li / Average_Expected_Years_Kept_of_Lithium_Accu

CLi = model.constant("CLi")
CLi.equation = values["CLi"]

Health_Cost_of_Lithium_Accu = model.converter("Health_Cost_of_Lithium_Accu")
Health_Cost_of_Lithium_Accu.equation = Li_Annual_Emissions_Load_Lithium_Accu * CLi

# -----------------------------------------------------------------------------

Social_Impact_Cost_of_Lithium_Accu = model.converter("Social_Impact_Cost_of_Lithium_Accu")
Social_Impact_Cost_of_Lithium_Accu.equation = (Recharging_Loss_Cost_of_Lithium_Accu + Maintenance_Loss_Cost_of_Lithium_Accu + Health_Cost_of_Lithium_Accu)

Maintenance_Price_of_Lithium_Accu = model.constant("Maintenance_Price_of_Lithium_Accu")
Maintenance_Price_of_Lithium_Accu.equation = values["Maintenance Price of Lithium Accu"]

Maintenance_Cost_of_Lithium_Accu = model.converter("Maintenance_Cost_of_Lithium_Accu")
Maintenance_Cost_of_Lithium_Accu.equation = Maintenance_Frequency_of_Lithium_Accu * Maintenance_Price_of_Lithium_Accu

# -----------------------------------------------------------------------------

Capacity_of_Lithium_Accu = model.constant("Capacity_of_Lithium_Accu")
Capacity_of_Lithium_Accu.equation = values["Capacity of Lithium Accu"]

Voltage_of_Lithium_Accu = model.constant("Voltage_of_Lithium_Accu")
Voltage_of_Lithium_Accu.equation = values["Voltage of Lithium Accu"]

Charging_Efficiency_of_Lithium_Accu = model.constant("Charging_Efficiency_of_Lithium_Accu")
Charging_Efficiency_of_Lithium_Accu.equation = values["Charging Efficiency of Lithium Accu"]

Energy_Consumption_of_Lithium_Accu = model.converter("Energy_Consumption_of_Lithium_Accu")
Energy_Consumption_of_Lithium_Accu.equation = (((Voltage_of_Lithium_Accu * Capacity_of_Lithium_Accu)/1000)/Charging_Efficiency_of_Lithium_Accu) * Electricity_Price * Recharging_Frequency_of_Lithium_Accu

Operational_Cost_of_Lithium_Accu = model.converter("Operational_Cost_of_Lithium_Accu")
Operational_Cost_of_Lithium_Accu.equation = Energy_Consumption_of_Lithium_Accu + Maintenance_Cost_of_Lithium_Accu

# -----------------------------------------------------------------------------

LiAPIncRate = model.constant("LiAPIncRate")
LiAPIncRate.equation = values["Lithium Accu Price Increase Rate"]
LiAPInc = model.flow("LiAPInc")
Price_of_Lithium_Accu = model.stock("Price_of_Lithium_Accu")
Price_of_Lithium_Accu.initial_value = values["Initial LiA Price"]
Price_of_Lithium_Accu.equation = LiAPInc
LiAPInc.equation = Price_of_Lithium_Accu * LiAPIncRate

Depreciation_Rate_of_Lithium_Accu = model.constant("Depreciation_Rate_of_Lithium_Accu")
Depreciation_Rate_of_Lithium_Accu.equation = values["Depreciation Rate of Lithium Accu"]

Resale_Value_of_Lithium_Accu = model.converter("Resale_Value_of_Lithium_Accu")
Resale_Value_of_Lithium_Accu._function_string = (
    "lambda model, t: ("
    "(pow(1.0 - model.memoize('Depreciation_Rate_of_Lithium_Accu', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Lithium_Accu', t)) "
    "* model.memoize('Price_of_Lithium_Accu', t) "
    "* model.memoize('Real_Discount_Rate', t)) "
    "/ "
    "(pow(1.0 + model.memoize('Real_Discount_Rate', t), "
    "model.memoize('Average_Expected_Years_Kept_of_Lithium_Accu', t)) - 1.0)"
    ")"
)
Resale_Value_of_Lithium_Accu.generate_function()

Life_Cycle_Cost_of_Lithium_Accu = model.converter("Life_Cycle_Cost_of_Lithium_Accu")
Life_Cycle_Cost_of_Lithium_Accu.equation = (Price_of_Lithium_Accu/Average_Expected_Years_Kept_of_Lithium_Accu) + Operational_Cost_of_Lithium_Accu - Resale_Value_of_Lithium_Accu

# -----------------------------------------------------------------------------

Cost_Ownership_of_Lithium_Accu = model.converter("Cost_Ownership_of_Lithium_Accu")
Cost_Ownership_of_Lithium_Accu.equation = (Social_Impact_Cost_of_Lithium_Accu + Life_Cycle_Cost_of_Lithium_Accu)/1e+06

Average_Annualized_TCO = model.converter("Average_Annualized_TCO")
Average_Annualized_TCO.equation = (Cost_Ownership_of_Dry_Accu + Cost_Ownership_of_Wet_Accu + Cost_Ownership_of_Lithium_Accu)/3

Proportion_of_Lithium_Accu_Annualized_TCO_to_Average_Annualized_TCO = model.converter("Proportion_of_Lithium_Accu_Annualized_TCO_to_Average_Annualized_TCO")
Proportion_of_Lithium_Accu_Annualized_TCO_to_Average_Annualized_TCO.equation = Cost_Ownership_of_Lithium_Accu / Average_Annualized_TCO

import BPTK_Py
bptk = BPTK_Py.bptk()
bptk.register_model(model)

import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# PART 1: MODEL BUILDING & TRAINING
# -----------------------------------------------------------------------------

# 1. Prepare the Data
# Note: Replace the dummy 'willing_to_pay' and 'total_sample' numbers with your actual survey data.
data = pd.DataFrame({
    'scenario_text': [
        '2x more expensive', '1.75x more expensive', '1.5x more expensive', '1.25x more expensive',
        'Same / Equal',
        '1.25x more efficient', '1.5x more efficient', '1.75x more efficient', '2x more efficient'
    ],
    # Converting the text scenarios into decimal ratios
    'annualized_tco_ratio': [2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.67, 0.57, 0.5],

    # DUMMY DATA: Enter the number of people Willing to Pay (WTP) here
    'willing_to_pay': [5, 10, 20, 35, 50, 65, 80, 90, 95],

    # DUMMY DATA: Enter the total number of respondents per scenario here
    'total_sample': [100, 100, 100, 100, 100, 100, 100, 100, 100]
})

# 2. Set up the Dependent Variable (Y)
# Calculate the number of people NOT willing to pay (Total Sample - WTP)
data['not_willing_to_pay'] = data['total_sample'] - data['willing_to_pay']

# Combine into a 2-column array for the Binomial GLM: [Success, Failure]
y = data[['willing_to_pay', 'not_willing_to_pay']]

# 3. Set up the Independent Variable (X)
X = data['annualized_tco_ratio']
X = sm.add_constant(X) # It is mandatory to add a constant (Intercept)

# 4. Build and Fit the Probit Model (Using GLM)
# Specify the Binomial family and the Probit link function
probit_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
result = probit_model.fit()

# Display the statistical results
print("=== Probit Model Summary ===")
print(result.summary())
print("\n")


# -----------------------------------------------------------------------------
# # PART 2: MANUAL INTEGRATION TO BPTK-Py
# -----------------------------------------------------------------------------


# 1. Extract data from BPTK (Removed return_df=True)
df_bptk = bptk.run_scenarios(
    scenario_managers=["smLithium accu adoption"],
    scenarios=["base"],
    equations=["Proportion_of_Lithium_Accu_Annualized_TCO_to_Average_Annualized_TCO"]
)

# 2. Extract Year and TCO Ratio
year_list = df_bptk.index.tolist()
list_tco_ratio = df_bptk["Proportion_of_Lithium_Accu_Annualized_TCO_to_Average_Annualized_TCO"].tolist()

# 3. Prepare prediction data (Constant + TCO Ratio)
prediction_data = pd.DataFrame({
    'const': [1.0] * len(list_tco_ratio),
    'tco_ratio': list_tco_ratio
})

# 4. Perform Prediction for WTP and Not WTP
prob_wtp_series = result.predict(prediction_data)
prob_not_wtp_series = 1.0 - prob_wtp_series # Calculate Not WTP

# 5. Convert prediction results into native Python Lists
prob_wtp_list = prob_wtp_series.values.tolist()
prob_not_wtp_list = prob_not_wtp_series.values.tolist() # Native list for Not WTP

# 6. Combine into [X, Y] coordinate format specifically for BPTK
points_wtp = []
points_not_wtp = [] # New list container for Not WTP

print("=== Probability Estimation Results per Year ===")
for i in range(len(year_list)):
    year = year_list[i]
    ratio = list_tco_ratio[i]
    wtp = prob_wtp_list[i]
    not_wtp = prob_not_wtp_list[i] # Get Not WTP value

    # Display 4 variables to the screen for easy checking
    print(f"Year {year} | TCO Ratio: {ratio:.4f} -> WTP: {wtp:.4f} | Not WTP: {not_wtp:.4f}")

    # Append to the points list in [Time, Value] format for BPTK to read
    points_wtp.append([year, wtp])
    points_not_wtp.append([year, not_wtp]) # Store Not WTP coordinates

# -----------------------------------------------------------------------------
# PART 3: INJECT BACK INTO BPTK (LOOKUP)
# -----------------------------------------------------------------------------

# 1. Register the coordinate arrays into BPTK model points
model.points["points_Probability_of_WTP"] = points_wtp
model.points["points_Probability_of_Not_WTP"] = points_not_wtp

# 2. Create WTP converter and connect it to Lookup
Probability_of_WTP = model.converter("Probability_of_WTP")
Probability_of_WTP.equation = sd.lookup(sd.time(), "points_Probability_of_WTP")

# 3. Create Not WTP converter and connect it to Lookup
Probability_of_Not_WTP = model.converter("Probability_of_Not_WTP")
Probability_of_Not_WTP.equation = sd.lookup(sd.time(), "points_Probability_of_Not_WTP")

print("\nIntegration Complete! Both probabilities (WTP & Not WTP) are now running dynamically in BPTK.")

##Techno-Economic Analysis (TEA)

Workers = model.constant("Workers")
Workers.equation = values["Workers"]

Battery = model.constant("Battery")
Battery.equation = values["Battery"]

Packaging = model.constant("Packaging")
Packaging.equation = values["Packaging"]

Manual_Book = model.constant("Manual_Book")
Manual_Book.equation = values["Manual Book"]

model.points["ICE_Motorcycle_Ownership"] = ICE_Motorcycle_Ownership

ICE_Motorcycle_Ownership = model.converter("ICE_Motorcycle_Ownership")
ICE_Motorcycle_Ownership.equation = sd.lookup(sd.time(), "ICE_Motorcycle_Ownership")

Market_Share = model.converter("Market_Share")
Market_Share.equation = Probability_of_WTP * ICE_Motorcycle_Ownership

Variable_Cost = model.converter("Variable_Cost")
Variable_Cost.equation = (Workers + Battery + Packaging + Manual_Book) * Market_Share

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
Fix_Cost.equation = Machinery_Depreciation + Research_Amortization + Annual_Research_and_Development + Production_Manager + Building_Depreciation + Electricity + Sales + Administration_and_General_Affair

# -----------------------------------------------------------------------------

Total_Production_Cost = model.converter("Total_Production_Cost")
Total_Production_Cost.equation = Variable_Cost + Fix_Cost

Cost_Per_Unit = model.converter("Cost_Per_Unit")
Cost_Per_Unit._function_string = (
    "lambda model, t: ("
    "(model.memoize('Total_Production_Cost', t) / model.memoize('Market_Share', t)) "
    "if model.memoize('Market_Share', t) > 0.0 else 0.0"
    ")"
)
Cost_Per_Unit.generate_function()

# -----------------------------------------------------------------------------

Margin_Profit = model.constant("Margin_Profit")
Margin_Profit.equation = values["Margin Profit"]

Selling_Price = model.converter("Selling_Price")
Selling_Price.equation = Cost_Per_Unit + Cost_Per_Unit * Margin_Profit

Inflow = model.converter("Inflow")
Inflow.equation = Selling_Price * Market_Share

Outflow = model.converter("Outflow")
Outflow.equation = Total_Production_Cost

Annual_Cashflow = model.converter("Annual_Cashflow")
Annual_Cashflow.equation = Inflow - Outflow

# -----------------------------------------------------------------------------

Cell_Tester = model.constant("Cell_Tester")
Cell_Tester.equation = values["Cell Tester"]

Module_Tester = model.constant("Module_Tester")
Module_Tester.equation = values["Module Tester"]

Research_and_Development = model.constant("Research_and_Development")
Research_and_Development.equation = values["Research and Development"]

Lithium_Iron_Phosphate_Battery_Chamber = model.constant("Lithium_Iron_Phosphate_Battery_Chamber")
Lithium_Iron_Phosphate_Battery_Chamber.equation = values["Lithium Iron Phosphate Battery Chamber"]

Chamber_Construction_Project = model.constant("Chamber_Construction_Project")
Chamber_Construction_Project.equation = values["Chamber Construction Project"]

Machine_Installation = model.constant("Machine_Installation")
Machine_Installation.equation = values["Machine Installation"]

Total_Investment_Cost = model.converter("Total_Investment_Cost")
Total_Investment_Cost.equation = Cell_Tester + Module_Tester + Research_and_Development + Lithium_Iron_Phosphate_Battery_Chamber + Chamber_Construction_Project + Machine_Installation

# -----------------------------------------------------------------------------

Payback_Period = model.converter("Payback_Period")
Payback_Period._function_string = (
    "lambda model, t: ("
    "(model.memoize('Total_Investment_Cost', t) / model.memoize('Annual_Cashflow', t)) "
    "if model.memoize('Annual_Cashflow', t) != 0.0 else 0.0"
    ")"
)
Payback_Period.generate_function()

# -----------------------------------------------------------------------------

Machine_Lifespan = model.constant("Machine_Lifespan")
Machine_Lifespan.equation = values["Machine Lifespan"]

Real_NPV = model.converter("Real_NPV")
Real_NPV._function_string = (
    "lambda model, t: ("
    "sum("
    "model.memoize('Annual_Cashflow', t) / "
    "pow(1.0 + model.memoize('Real_Discount_Rate', t), i) "
    "for i in range(1, int(model.memoize('Machine_Lifespan', t)) + 1)"
    ") - model.memoize('Total_Investment_Cost', t)"
    ")"
)
Real_NPV.generate_function()

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

NPV_1 = model.converter("NPV_1")
NPV_1.equation = (Annual_Cashflow * Present_Value_Annuity_Factor_1) - Total_Investment_Cost

NPV_2 = model.converter("NPV_2")
NPV_2.equation = (Annual_Cashflow * Present_Value_Annuity_Factor_2) - Total_Investment_Cost

i1 = model.constant("i1")
i1.equation = values["i1"]

i2 = model.constant("i2")
i2.equation = values["i2"]

Machine_Lifespan = model.constant("Machine_Lifespan")
Machine_Lifespan.equation = values["Machine Lifespan"]

IRR = model.converter("IRR")
IRR._function_string = (
    "lambda model, t: ("
    "model.memoize('i1', t) + "
    "(model.memoize('NPV_1', t) / "
    "(model.memoize('NPV_1', t) - model.memoize('NPV_2', t))) * "
    "(model.memoize('i2', t) - model.memoize('i1', t))"
    ")"
)
IRR.generate_function()

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

bptk.register_model(model)

scenario_manager_name = "smLithium accu adoption"

# -----------------------------------------------------------------------------
# Section: Pre‑compute data series for each variable
# -----------------------------------------------------------------------------

try:
    # Compute and store the DataFrame for each variable once.
    # Each call returns a DataFrame indexed by time.  We keep the raw
    # DataFrame; it will be tidied when displayed in the UI.
    o1 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Market_Share"],
        series_names={},
        return_df=True,
    )
    o2 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Total_Investment_Cost"],
        series_names={},
        return_df=True,
    )
    o3 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Total_Production_Cost"],
        series_names={},
        return_df=True,
    )
    o4 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Cost_Per_Unit"],
        series_names={},
        return_df=True,
    )
    o5 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Selling_Price"],
        series_names={},
        return_df=True,
    )
    o6 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Annual_Cashflow"],
        series_names={},
        return_df=True,
    )
    o7 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Payback_Period"],
        series_names={},
        return_df=True,
    )
    o8 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Real_NPV"],
        series_names={},
        return_df=True,
    )
    o9 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["IRR"],
        series_names={},
        return_df=True,
    )
    o10 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["MARR"],
        series_names={},
        return_df=True,
    )
    o11 = bptk.plot_scenarios(
        scenarios="base",
        scenario_managers=scenario_manager_name,
        equations=["Annual_Research_and_Development"],
        series_names={},
        return_df=True,
    )

    # Map variable names to their precomputed DataFrames
    precomputed_series = {
        "Market_Share": o1,
        "Total_Investment_Cost": o2,
        "Total_Production_Cost": o3,
        "Cost_Per_Unit": o4,
        "Selling_Price": o5,
        "Annual_Cashflow": o6,
        "Payback_Period": o7,
        "Real_NPV": o8,
        "IRR": o9,
        "MARR": o10,
        "Annual_Research_and_Development": o11,

    }
except Exception as _exc_precompute:
    # If any error occurs during precomputation, fall back to an empty mapping.
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

# 1. SIDEBAR MENU (Only 2 Options)
menu_option = st.sidebar.radio(
    "Select Page",
    ("Simulation Result", "Concept Foundation"),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Button linking to the parameter database
if st.sidebar.button("Update Database"):
    st.sidebar.markdown(
        "[Open Parameter Database](https://docs.google.com/spreadsheets/d/"
        "1Bszu5DtBg_2Oyf9PCQc2Morx8dJAAAzZd4MLKHi09-I/edit?usp=sharing)"
    )


# =============================================================================
# CONDITION 1: SIMULATION RESULT PAGE
# =============================================================================

import pandas as pd
import streamlit as st

# =============================================================================
# CONDITION 1: SIMULATION RESULT PAGE
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
            return pd.DataFrame() # Return empty DataFrame if there is no data
        
        df_temp = df_raw.copy()
        if isinstance(df_temp.columns, pd.MultiIndex):
            df_temp.columns = ["_".join(col).strip() for col in df_temp.columns.values]
        
        # Find the matching column
        match_cols = [col for col in df_temp.columns if var_name_str in col]
        var_col = match_cols[0] if match_cols else df_temp.columns[0]
        
        table_df = df_temp[[var_col]].copy().reset_index()
        table_df.columns = ["Time", var_name_str]
        table_df[var_name_str] = table_df[var_name_str].round(6)
        
        return table_df

    # Format all data (Ensure o1 to o11 are defined in your pre-compute code)
    df_market_share = format_dataframe(o1, "Market Share")
    df_inv_cost     = format_dataframe(o2, "Total Investment Cost")
    df_prod_cost    = format_dataframe(o3, "Total Production Cost")
    df_cpu          = format_dataframe(o4, "Cost Per Unit")
    df_price        = format_dataframe(o5, "Selling Price")
    df_cashflow     = format_dataframe(o6, "Annual Cashflow")
    df_payback      = format_dataframe(o7, "Payback Period")
    df_npv          = format_dataframe(o8, "Real NPV")
    df_irr          = format_dataframe(o9, "IRR")
    df_marr         = format_dataframe(o10, "MARR")
    df_rnd          = format_dataframe(o11, "Annual Research and Development") # New Variable Added

    # =========================================================================
    # MERGE ALL DATAFRAMES INTO A SINGLE DICTIONARY
    # =========================================================================
    
    # 1. Collect all dataframes except the base one (df_market_share) into a list
    list_df = [
        df_inv_cost, df_prod_cost, df_cpu, df_price, 
        df_cashflow, df_payback, df_npv, df_irr, df_marr, df_rnd
    ]

    # 2. Use df_market_share as the base dataframe for merging
    all_df = df_market_share.copy()

    # 3. Loop through the list and merge each dataframe based on the "Time" column
    for df in list_df:
        if not df.empty: # Ensure the dataframe is not empty before merging
            all_df = pd.merge(all_df, df, on="Time", how="outer")

    # 4. Convert the merged dataframe into a dictionary (list of records/rows)
    all_dict = all_df.to_dict(orient="records")
    
    # =========================================================================

    # --- PART 1: MARKET SHARE (Main Table & Chart) ---
    st.subheader("📊 Market Share")
    if not df_market_share.empty:
        col_table, col_chart = st.columns([1, 1.5]) # Proportion to make the chart wider
        
        with col_table:
            st.markdown("**Data Table**")
            st.dataframe(df_market_share, height=300, use_container_width=True)
            
        with col_chart:
            st.markdown("**Trend Chart**")
            chart_df = df_market_share.set_index("Time")
            st.line_chart(chart_df, height=300)
    else:
        st.warning("Market Share data is not available.")

    st.divider()

    # --- PART 2: OTHER FINANCIAL METRICS (Tables only, neatly arranged in 2 columns) ---
    st.subheader("📄 Financial & Production Metrics")
    
    col1, col2 = st.columns(2)
    
    # Left Column
    with col1:
        st.markdown("**Total Investment Cost**")
        st.dataframe(df_inv_cost, height=200, use_container_width=True)
        
        st.markdown("**Cost Per Unit**")
        st.dataframe(df_cpu, height=200, use_container_width=True)
        
        st.markdown("**Annual Cashflow**")
        st.dataframe(df_cashflow, height=200, use_container_width=True)
        
        st.markdown("**Real NPV**")
        st.dataframe(df_npv, height=200, use_container_width=True)

        st.markdown("**MARR**")
        st.dataframe(df_marr, height=200, use_container_width=True)

    # Right Column
    with col2:
        st.markdown("**Total Production Cost**")
        st.dataframe(df_prod_cost, height=200, use_container_width=True)
        
        st.markdown("**Selling Price**")
        st.dataframe(df_price, height=200, use_container_width=True)
        
        st.markdown("**Payback Period**")
        st.dataframe(df_payback, height=200, use_container_width=True)
        
        st.markdown("**IRR**")
        st.dataframe(df_irr, height=200, use_container_width=True)
        
        # Adding the new variable here to balance the columns
        st.markdown("**Annual Research and Development**")
        st.dataframe(df_rnd, height=200, use_container_width=True)

    def ask_ai_button(label: str, prompt_text: str):
        """
        One click:
        1) copy prompt_text to clipboard
        2) open ChatGPT with q= prefill
        """
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
                }} catch (e) {{
                  // Clipboard may be blocked depending on browser/hosting.
                }}

                window.open(url, "_blank");
              }});
            </script>
            """,
            height=55,
        )

    question = "Please analyze the following table systematically and in-depth based economic analysis. Explain the main patterns, visible trends, comparisons between data, and important insights that can be drawn. If relevant, include logical interpretations and implications of the data."
    prompt_text = f"{all_dict}\n\n{question}"
    ask_ai_button("Ask AI", prompt_text)

# =============================================================================
# PAGE 2: CONCEPT FOUNDATION
# =============================================================================
elif menu_option == "Concept Foundation":
    st.title("📚 Concept Foundation")
    st.write("Below are the foundational concepts, methodologies, and references used in the AdopTex simulation model.")
    
    st.divider()

    # Define the concept data based on your references
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
    
    # Display the aesthetic dataframe using Streamlit's column configuration
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
                display_text="Open Reference 🔗",  # Replaces the long URL with a clean clickable text
                width="large"
            )
        },
        hide_index=True,          # Hides the row numbers (0, 1, 2...) for a cleaner look
        use_container_width=True  # Stretches the table to fill the screen width
    )