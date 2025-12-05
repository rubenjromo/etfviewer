# app.py — Professional Streamlit ETF Portfolio Analyzer
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pycountry
import math
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="ETF Portfolio Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("📈 ETF Portfolio Analyzer — Professional Version")

# --------------------------
# Sidebar - Inputs
# --------------------------
st.sidebar.header("1) Configura tu portafolio")

st.sidebar.markdown("""
Ingrese los ETFs y pesos en formato:
