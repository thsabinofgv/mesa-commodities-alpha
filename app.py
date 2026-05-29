import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from scipy.optimize import brentq
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Alpha Trading - Risco", layout="wide", page_icon="📈")

# --- FUNÇÕES CORE E CACHE ---
@st.cache_data
def carregar_dados():
    tickers = ['CL=F', 'GC=F', 'ZS=F', 'NG=F', 'GLD', 'USO', 'SLV']
    dados = yf.download(tickers, start="2025-01-01", end="2026-05-28")['Close']
    return dados.ffill().dropna()

def black_scholes(S, K, T, r, sigma, tipo='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == 'call': return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_76(F, K, T, r, sigma, tipo='call'):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == 'call': return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else: return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def vega_bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def calcular_gregas(S, K, T, r, sigma, tipo='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100 
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    if tipo == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

# Métodos Numéricos
def vol_bissecao(S, K, T, r, preco, tipo='call', max_iter=500):
    baixo, alto = 0.0001, 5.0
    f_baixo = black_scholes(S, K, T, r, baixo, tipo) - preco
    f_alto = black_scholes(S, K, T, r, alto, tipo) - preco
    if f_baixo * f_alto > 0: return np.nan, 0
    for i in range(max_iter):
        meio = (baixo + alto) / 2.0
        f_meio = black_scholes(S, K, T, r, meio, tipo) - preco
        if abs(f_meio) < 1e-6: return meio, i+1
        if f_meio * f_baixo < 0: alto = meio
        else: baixo, f_baixo = meio, f_meio
    return meio, max_iter

def vol_newton(S, K, T, r, preco, tipo='call', max_iter=500
