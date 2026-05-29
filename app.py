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

def vol_newton(S, K, T, r, preco, tipo='call', max_iter=500):
    sigma = 0.5 
    for i in range(max_iter):
        f = black_scholes(S, K, T, r, sigma, tipo) - preco
        if abs(f) < 1e-6: return sigma, i+1
        v = vega_bs(S, K, T, r, sigma)
        if v == 0: return np.nan, i+1
        sigma = sigma - (f / v)
        if sigma <= 0: sigma = 0.0001
    return sigma, max_iter

def vol_secante(S, K, T, r, preco, tipo='call', max_iter=500):
    s0, s1 = 0.1, 0.5
    f0 = black_scholes(S, K, T, r, s0, tipo) - preco
    for i in range(max_iter):
        f1 = black_scholes(S, K, T, r, s1, tipo) - preco
        if abs(f1) < 1e-6: return s1, i+1
        if f1 - f0 == 0: return np.nan, i+1
        s_new = s1 - f1 * ((s1 - s0) / (f1 - f0))
        if s_new <= 0: s_new = 0.0001
        s0, s1, f0 = s1, s_new, f1
    return s1, max_iter

def f_obj(sigma, S, K, T, r, preco, tipo):
    return black_scholes(S, K, T, r, sigma, tipo) - preco

def vol_brent(S, K, T, r, preco, tipo='call'):
    try:
        res = brentq(f_obj, 0.0001, 5.0, args=(S, K, T, r, preco, tipo), xtol=1e-6, full_output=True)
        return res[0], res[1].iterations
    except ValueError:
        return np.nan, 0

# --- MENU LATERAL ---
st.sidebar.title("Banco Alpha Trading")
st.sidebar.markdown("---")
pagina = st.sidebar.radio("Navegação", [
    "1. Dashboard Principal",
    "2. Dados Históricos",
    "3. Precificação e Gregas",
    "4. Volatilidade Implícita",
    "5. VaR e Expected Shortfall",
    "6. Backtest e Stress Testing"
])

# Carrega os dados em background para todo o app
try:
    dados = carregar_dados()
    retornos = np.log(dados / dados.shift(1)).dropna()
    dados_ok = True
except Exception as e:
    dados_ok = False

# --- ROTEAMENTO DAS PÁGINAS ---

if pagina == "1. Dashboard Principal":
    st.title("Mesa de Commodities")
    st.subheader("Sistema de Gestão de Risco e Precificação")
    st.write("Bem-vindo ao sistema de monitoramento construído para avaliar a exposição direcional, risco não-linear e perdas potenciais da carteira.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Status dos Dados", "Online" if dados_ok else "Offline")
    col2.metric("Ativos Monitorados", "7 Commodities")
    col3.metric("Modelo Base", "Black-Scholes / 76")
    
    st.info("Utilize o menu lateral para navegar pelas análises solicitadas pela Diretoria de Risco.")

elif pagina == "2. Dados Históricos":
    st.title("Captura e Tratamento de Dados (Parte I)")
    if dados_ok:
        st.write("Dados extraídos com sucesso do Yahoo Finance.")
        
        st.subheader("Evolução Histórica (Base 100)")
        precos_norm = (dados / dados.iloc[0]) * 100
        st.line_chart(precos_norm)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volatilidade Anualizada")
            vol_anual = retornos.std() * np.sqrt(252)
            st.dataframe(vol_anual.rename("Volatilidade").to_frame().style.format("{:.2%}"))
        with col2:
            st.subheader("Matriz de Correlação")
            st.dataframe(retornos.corr().style.background_gradient(cmap='coolwarm'))
    else:
        st.error("Erro ao baixar dados. Verifique a conexão do servidor.")

elif pagina == "3. Precificação e Gregas":
    st.title("Precificação (Parte II) e Gregas (Parte VI)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parâmetros da Opção")
        S = st.number_input("Preço do Ativo/Futuro (S ou F)", value=100.0)
        K = st.number_input("Strike (K)", value=100.0)
        T_dias = st.number_input("Dias Úteis para Vencimento", value=90)
        T = T_dias / 252
        r = st.number_input("Taxa Livre de Risco (r)", value=0.05, format="%.2f")
        sigma = st.number_input("Volatilidade (sigma)", value=0.20, format="%.2f")
        tipo = st.selectbox("Tipo", ['call', 'put'])
    
    with col2:
        st.subheader("Resultados Teóricos")
        bs_price = black_scholes(S, K, T, r, sigma, tipo)
        b76_price = black_76(S, K, T, r, sigma, tipo)
        
        st.metric("Preço ETF (Black-Scholes)", f"$ {bs_price:.4f}")
        st.metric("Preço Futuro (Black-76)", f"$ {b76_price:.4f}")
        
        st.markdown("---")
        st.subheader("Letras Gregas")
        gregas = calcular_gregas(S, K, T, r, sigma, tipo)
        for k, v in gregas.items():
            st.write(f"**{k}:** {v:.4f}")

elif pagina == "4. Volatilidade Implícita":
    st.title("Métodos Numéricos e Smile (Parte III a V)")
    
    S_teste, K_teste, T_teste, r_teste = 100, 100, 1.0, 0.05
    preco_mercado = st.slider("Simular Preço de Mercado da Opção", 5.0, 20.0, 10.45)
    
    st.subheader("Comparação dos Algoritmos")
    resultados = []
    metodos = [("Bisseção", vol_bissecao), ("Newton-Raphson", vol_newton), ("Secante", vol_secante), ("Brent", vol_brent)]
    
    for nome, func in metodos:
        inicio = time.time()
        if nome == "Brent": vol, iters = func(S_teste, K_teste, T_teste, r_teste, preco_mercado)
        else: vol, iters = func(S_teste, K_teste, T_teste, r_teste, preco_mercado)
        tempo = time.time() - inicio
        
        erro = abs(black_scholes(S_teste, K_teste, T_teste, r_teste, vol) - preco_mercado) if not np.isnan(vol) else np.nan
        resultados.append({"Método": nome, "Vol Implícita": vol, "Iterações": iters, "Erro": f"{erro:.2e}", "Tempo (s)": tempo})
        
    st.dataframe(pd.DataFrame(resultados).style.format({"Vol Implícita": "{:.4f}", "Tempo (s)": "{:.6f}"}))
    
    st.markdown("---")
    st.subheader("Smile de Volatilidade Dinâmico")
    st.write("Edite os prêmios (Preço Call) na tabela abaixo simulando diferentes condições de mercado. O gráfico e a volatilidade implícita reagirão em tempo real!")
    
    # DataFrame Base
    df_base_smile = pd.DataFrame({
        "Strike": [80, 90, 100, 110, 120],
        "Preço Call": [23.5, 13.8, 6.5, 2.8, 1.5]
    })
    
    col_tabela, col_grafico = st.columns([1, 2])
    
    with col_tabela:
        # Tabela editável
        df_editado = st.data_editor(df_base_smile, hide_index=True, use_container_width=True)
    
    vols_smile = []
    for index, row in df_editado.iterrows():
        # Usando o Brent para calcular a Volatilidade para cada Strike/Preço editado
        vol_calc = vol_brent(100, row["Strike"], 90/252, 0.05, row["Preço Call"])[0]
        vols_smile.append(vol_calc)
        
    with col_grafico:
        # Gráfico interativo
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_editado["Strike"], vols_smile, marker='o', color='indigo', linewidth=2)
        ax.set_title("Curva do Smile (Vencimento de 90 dias)")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Vol. Implícita")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

elif pagina == "5. VaR e Expected Shortfall":
    st.title("Análise de Risco (Parte VII a IX)")
    
    if dados_ok:
        ativo = st.selectbox("Selecione a Commodity para Análise", ['CL=F', 'GC=F', 'ZS=F'])
        posicao = 1000000
        
        rets = retornos[ativo]
        vol_d = rets.std()
        
        st.subheader(f"VaR 1 Dia - Posição: $ 1.000.000 em {ativo}")
        
        var_hist_99 = abs(np.percentile(rets, 1) * posicao)
        var_param_99 = norm.ppf(0.99) * vol_d * posicao
        
        col1, col2 = st.columns(2)
        col1.metric("VaR Histórico (99%)", f"$ {var_hist_99:,.2f}")
        col2.metric("VaR Paramétrico (99%)", f"$ {var_param_99:,.2f}")
        
        st.markdown("---")
        st.subheader("Full Valuation VaR e Expected Shortfall (Posição Opções de Ouro GLD)")
        st.write("Recalculando prêmios para 10.000 cenários de Monte Carlo...")
        
        Z = np.random.standard_normal(10000)
        ST = 200 * np.exp((0.05 - 0.5 * 0.15**2)*(1/252) + 0.15 * np.sqrt(1/252) * Z)
        V_cenarios = black_scholes(ST, 205, (90/252) - (1/252), 0.05, 0.15) * 25000
        V0 = black_scholes(200, 205, 90/252, 0.05, 0.15) * 25000
        
        L = V0 - V_cenarios
        var_full = np.percentile(L, 99)
        es_full = np.mean(L[L > var_full])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Posição (V0)", f"$ {V0:,.2f}")
        c2.metric("VaR Full Valuation (99%)", f"$ {var_full:,.2f}")
        c3.metric("Expected Shortfall (99%)", f"$ {es_full:,.2f}")
        
    else:
        st.error("Dados não carregados.")

elif pagina == "6. Backtest e Stress Testing":
    st.title("Validação do Modelo (Parte X e XI)")
    
    if dados_ok:
        st.subheader("Backtesting do VaR (Petróleo)")
        rets = retornos['CL=F']
        posicao = 1000000
        pnl = rets * posicao
        var_movel = rets.rolling(250).quantile(0.01).shift(1) * posicao
        
        df_back = pd.DataFrame({'PnL': pnl, 'VaR': var_movel}).dropna()
        violacoes = df_back[df_back['PnL'] < df_back['VaR']]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_back.index, df_back['PnL'], label='PnL', alpha=0.6)
        ax.plot(df_back.index, df_back['VaR'], label='VaR 99%', color='red')
        ax.scatter(violacoes.index, violacoes['PnL'], color='black', marker='x', label='Violação', zorder=5)
        ax.legend()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        col1.metric("Total de Violações", len(violacoes))
        col2.metric("Dias Testados", len(df_back))
        
        st.markdown("---")
        st.subheader("Stress Testing - Posições de $ 1.000.000")
        cenarios = [
            {"Cenário": "Recessão global", "Ativo": "Petróleo", "Choque": "-25%", "Perda Projetada": posicao * -0.25},
            {"Cenário": "Fuga para segurança", "Ativo": "Ouro", "Choque": "+15%", "Perda Projetada": posicao * 0.15},
            {"Cenário": "Choque de oferta", "Ativo": "Gás Natural", "Choque": "+40%", "Perda Projetada": posicao * 0.40}
        ]
        st.table(pd.DataFrame(cenarios).style.format({"Perda Projetada": "$ {:,.2f}"}))
    else:
        st.error("Dados não carregados.")
