import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import scipy.optimize as sco
from numpy.linalg import multi_dot
from scipy.optimize import minimize
import datetime as dt

# Funciones auxiliares
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def obtener_precio_y_volumen(simbolos, start_date, end_date):
       data = yf.download(simbolos, start=start_date, end=end_date)[['Close','Volume']]
       return data.ffill().dropna()

def ratio_de_cap(data):
       ratio = data.iloc[-1,0]*data.iloc[-1,1]
       return ratio

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos(df):
    returns = df.pct_change().dropna()
    return returns

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_rendimiento_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

def calcular_beta(asset_returns, market_returns):
    covariance = np.cov(asset_returns.iloc[:,0], market_returns.iloc[:,0])[0][1]
    market_variance = np.var(market_returns.iloc[:,0])
    return covariance / market_variance if market_variance != 0 else np.nan

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan

# Nuevas funciones para VaR y CVaR
def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns < VaR].mean()
    return VaR, CVaR

def calcular_var_cvar_ventana(returns, window):
    if len(returns) < window:
        return np.nan, np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var_cvar(window_returns)

import pandas as pd
import matplotlib.pyplot as plt

def calcular_drawdown(asset):

    data = obtener_datos_acciones(asset, "2010-1-1", "2023-12-31")

    # Calcular el máximo acumulado
    data["Max_Acumulado"] = data[f"{asset}"].cummax()
    
    # Calcular el drawdown absoluto
    data["Drawdown"] = (data[f"{asset}"] - data["Max_Acumulado"]) / data["Max_Acumulado"]
    
    # Calcular el drawdown porcentual
    data["Drawdown (%)"] =  data["Drawdown"] * 100
    
    # Máximo drawdown porcentual
    max_drawdown = data["Drawdown (%)"].max()
    
    return max_drawdown

def calcular_drawdown_df(df):

    data = df

    # Calcular el máximo acumulado
    data["Max_Acumulado"] = data["Rendimiento Portafolio"].cummax()
    
    # Calcular el drawdown absoluto
    data["Drawdown"] = (data["Rendimiento Portafolio"] - data["Max_Acumulado"]) / data["Max_Acumulado"]
    
    # Calcular el drawdown porcentual
    data["Drawdown (%)"] = data["Drawdown"] * 100
    
    # Máximo drawdown porcentual
    max_drawdown = data["Drawdown (%)"].max()
    
    return max_drawdown

def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    # Crear el histograma base
    fig = go.Figure()
    
    # Calcular los bins para el histograma
    counts, bins = np.histogram(returns, bins=50)
    
    # Separar los bins en dos grupos: antes y después del VaR
    mask_before_var = bins[:-1] <= var_95
    
    # Añadir histograma para valores antes del VaR (rojo)
    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Retornos < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))
    
    # Añadir histograma para valores después del VaR (azul)
    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos > VaR',
        marker=dict(color='#FF5733')
    ))
    
    # Añadir líneas verticales para VaR y CVaR
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=2, dash='dot')
    ))
    
    # Actualizar el diseño
    fig.update_layout(
        title=title,
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        showlegend=True,
        barmode='overlay',
        bargap=0
    )
    
    return fig

def graficar_precio_normalizado(asset, benchmark):
    
    df_asset = obtener_datos_acciones(asset, "2010-1-1", "2023-12-31")
    df_benchmark = obtener_datos_acciones(benchmark, "2010-1-1", "2023-12-31")
    normalized_prices_asset = calcular_metricas(df_asset)[2]
    normalized_prices_benchmark = calcular_metricas(df_benchmark)[2]
    fig_asset = go.Figure()
    fig_asset.add_trace(go.Scatter(x=normalized_prices_asset.index, y=normalized_prices_asset[asset], name=asset,line=dict(color='red')))
    fig_asset.add_trace(go.Scatter(x=normalized_prices_benchmark.index, y=normalized_prices_benchmark[benchmark], name=benchmark,line=dict(color='orange')))
    fig_asset.update_layout(title=f'Precio Normalizado: {asset} vs {benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado',height=420)
    return st.plotly_chart(fig_asset, use_container_width=True, key="price_normalized")

def caracteristicas_y_grafica(asset,ex,i,mo,pr,pa,me,es,co):
    cont1 = st.container(border=True, height=650)
    with cont1:
     c1, c2 = st.columns([0.35,0.65])
     with c1:
        st.subheader(f"El ETF seleccionado actualmente es {asset}")
        expand1 = st.expander("**Exposición**")
        expand2 = st.expander("**Índice que sigue**")
        expand3 = st.expander("**Moneda de denominación:**")
        expand4 = st.expander("**Principales contribuyentes:**")
        expand5 = st.expander("**País o países donde invierte:**")
        expand6 = st.expander("**Métricas de riesgo:**")
        expand7 = st.expander("**Estilo:**")
        expand8 = st.expander("**Costos:**")
        with expand1:
                        st.write(f"{ex}")
        with expand2:
                        st.write(f"{i}")
        with expand3:
                        st.write(f"{mo}")
        with expand4:
                        st.write(f"{pr}")
        with expand5:
                        st.write(f"{pa}")
        with expand6:
                        st.write(f"{me}")
        with expand7:
                        st.write(f"{es}")
        with expand8:
                        st.write(f"{co}")
    with c2:     
            benchmark_options = {
                            "S&P 500": "^GSPC",
                            "Nasdaq": "^IXIC",
                            "Dow Jones": "^DJI",
                            "Russell 2000": "^RUT",
                            "ACWI": "ACWI"}
            benchmark = st.selectbox("Seleccione un benchmark", options=list(benchmark_options.keys()))
            graficar_precio_normalizado(asset, benchmark_options[benchmark])
            conte1 = st.container(border=True)
            with conte1:  
             colu1, colu2, colu3 = st.columns(3)
            with colu1:
                df = obtener_precio_y_volumen(asset,"2010-1-1", "2023-12-31")
                ratio = ratio_de_cap(df)
                st.metric("Market Cap al día de hoy",f"{ratio:,.2f}" + " USD")
            with colu2:
                df_asset = obtener_datos_acciones(asset, "2010-1-1", "2023-12-31")
                asset_returns = calcular_rendimientos(df_asset)
                df_market = obtener_datos_acciones(benchmark_options[benchmark], "2010-1-1", "2023-12-31")
                market_returns = calcular_rendimientos(df_market)
                beta = calcular_beta(asset_returns, market_returns)
                st.metric("Beta",f"{beta:,.2f}")
            with colu3:
                te = np.std(asset_returns.iloc[:,0] - market_returns.iloc[:,0]) * np.sqrt(252)
                st.metric("Tracking Error anual",f"{te:,.2f}") 

def contenido_1(boton1,boton2,boton3,boton4,boton5,asset1,asset2,asset3,asset4,asset5):
          
          if boton2:
              boton1 = False 
              @st.fragment
              def fragmento():
                      caracteristicas_y_grafica(asset2, "Bonos en mercados emergentes, denominados en dólares estadounidenses.",
                                                  "J.P. Morgan EMBI Global Core Index.",
                                                  "USD (dólares estadounidenses).",
                                                  "Bonos soberanos de países emergentes como Brasil, México, Sudáfrica, Turquía, entre otros.",
                                                  "Principalmente en mercados emergentes globales.",
                                                  "Aproximadamente 7 años, lo que lo hace más sensible a cambios en la tasa de interés y rendimiento histórico entre 4% y 6% anual.",
                                                  "Renta fija, con un enfoque en mercados emergentes.",
                                                  "Ratio de gastos de aproximadamente 0.39%.")
              fragmento()
          elif boton3:
              @st.fragment
              def fragmento():
                      caracteristicas_y_grafica(asset3,"Acciones de mercados internacionales, excluyendo Estados Unidos y Canadá.",
                                                  "MSCI ACWI ex USA Index.",
                                                  "USD (dólares estadounidenses).",
                                                  "Empresas de gran capitalización en Europa, Asia, América Latina, etc.",
                                                  "Excluye EE. UU. y Canadá, e invierte en países desarrollados y emergentes en Europa, Asia y otras regiones.",
                                                  "Rendimiento histórico entre 4% y 7% anual.",
                                                  "Acciones internacionales de mercados desarrollados y emergentes.",
                                                  "Ratio de gastos de aproximadamente 0.32%.")
              fragmento()
          elif boton4:
              @st.fragment
              def fragmento():
                      caracteristicas_y_grafica(asset4, "Acciones de mercados emergentes.",
                                                  "MSCI Emerging Markets Index.",
                                                  "USD (dólares estadounidenses).",
                                                  "Empresas de gran capitalización en mercados emergentes como China, India, Brasil, Sudáfrica, entre otros.",
                                                  "Mercados emergentes globales.",
                                                  "Rendimiento entre 5% y 10% anual.",
                                                  "Acciones de mercados emergentes, predominantemente de gran capitalización.",
                                                  "Ratio de gastos de aproximadamente 0.68%.")
              fragmento()
          elif boton5:
              @st.fragment
              def fragmento():
                      caracteristicas_y_grafica(asset5, "Commodities, como petróleo, gas natural, metales y productos agrícolas.",
                                                  "DBIQ Optimum Yield Diversified Commodity Index.",
                                                  "USD (dólares estadounidenses).",
                                                  "Futuros sobre materias primas, como petróleo (WTI), metales preciosos, gas natural, etc.",
                                                  "Global (en mercados de futuros de commodities).",
                                                  "Rendimiento historico entre 1% y 5% anual.",
                                                  "Commodities.",
                                                  "Ratio de gastos de aproximadamente 0.89%.")
              fragmento()  
          else:
              @st.fragment
              def fragmento():
                      caracteristicas_y_grafica(asset1, "Inversión en bonos de renta fija de EE. UU. (incluye bonos del Tesoro, bonos corporativos y bonos de agencias).",
                                                  "Bloomberg Barclays U.S. Aggregate Float Adjusted Index.",
                                                  "USD (dólares estadounidenses).",
                                                  "Bonos del gobierno de EE. UU., bonos corporativos de alta calidad y bonos de agencias gubernamentales como Fannie Mae y Freddie Mac.",
                                                  "Principalmente en Estados Unidos.",
                                                  "Duración de aproximadamente 6-7 años",
                                                  "No se categoriza como \"growth\" o \"value\" ya que es un ETF de bonos; es de renta fija",
                                                  "Ratio de gastos (expense ratio) de aproximadamente 0.035%.")
              fragmento()  

def graficas_metricas(grafica,asset):
    df = obtener_datos_acciones(asset, "2010-1-1", "2023-12-31")
    rend = calcular_rendimientos(df)
    var = calcular_var_cvar(rend[f"{asset}"])
    cvar = calcular_var_cvar(rend[f"{asset}"])

    if grafica == "Rendimiento Diario":
        # Crear el trace (la serie de datos) para el ETF
        trace = go.Scatter(
            x=rend.index,  # Fecha
            y=rend[f"{asset}"],  # Rendimiento diario
            mode='lines',  # Líneas para los rendimientos diarios
            name=asset,  # Nombre de la serie (ETF)
            line=dict(color='#FF5733')  # Cambiar el color de la línea
        )

        # Crear el layout del gráfico
        layout = go.Layout(
            title=f'Rendimientos Diarios de {asset} (2010-2023)',
            xaxis=dict(title='Fecha'),
            yaxis=dict(title='Rendimiento Diario'),
            template='plotly_dark'  # Puedes cambiar el tema si lo prefieres
        )

        # Crear la figura
        fig = go.Figure(data=[trace], layout=layout)

        return fig
    
    elif grafica == "Histogramas":
        return crear_histograma_distribucion(rend[f"{asset}"], var[0], cvar[0], f"{asset} Histograma")
    elif grafica == "Boxplot":
        
        # Extraer los rendimientos
        rendimientos = rend[f"{asset}"]
        
        # Crear el boxplot con Plotly
        fig = go.Figure()
        fig.add_trace(go.Box(x=rendimientos, name="Rendimientos", boxpoints="all", boxmean=True, jitter=0.3,line=dict(color='#FF5733')))
        
        # Configurar diseño
        fig.update_layout(
            title="Distribución de Rendimientos",
            yaxis_title="Rendimientos",
            xaxis_title="",
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    elif grafica == "Rendimiento Acumulado":

        # Calcular el rendimiento acumulado
        rendimiento_acumulado = (1 + rend[f"{asset}"]).cumprod() - 1
        
        # Crear la gráfica con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rend.index,
            y=rendimiento_acumulado,
            mode='lines',
            name='Rendimiento Acumulado',
            line=dict(color='#FF5733')
        ))
        
        # Configurar diseño
        fig.update_layout(
            title="Rendimiento Acumulado",
            xaxis_title="Tiempo",
            yaxis_title="Rendimiento Acumulado",
            template="plotly_white",
            showlegend=True,
        )
        
        return fig    
        
def metricas(asset):
        df = obtener_datos_acciones(asset, "2010-1-1", "2023-12-31")
        rend = calcular_rendimientos(df)
        media = rend[f"{asset}"].mean()
        sesgo = rend[f"{asset}"].skew()
        kurtosis = rend[f"{asset}"].kurt()
        var ,cvar = calcular_var_cvar(rend[f"{asset}"])
        sharpe = calcular_sharpe_ratio(rend[f"{asset}"]) 
        sortino = calcular_sortino_ratio(rend[f"{asset}"])
        drawdown = calcular_drawdown(asset)
        return media, sesgo, kurtosis, var, cvar, sharpe, sortino, drawdown


def contenido_2(boton1,boton2,boton3,boton4,boton5,asset1,asset2,asset3,asset4,asset5,grafica):
       if boton2:
        @st.fragment
        def fragmento():
              fig = graficas_metricas(grafica, asset2)
              st.plotly_chart(fig)
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                st.metric("Media",f"{metricas(asset2)[0]:,.2f}")
              with col2:
                st.metric("Sesgo",f"{metricas(asset2)[1]:,.2f}")
              with col3:
                st.metric("Kurtosis",f"{metricas(asset2)[2]:,.2f}")
              with col4:
                st.metric("VaR",f"{metricas(asset2)[3]:,.2f}")
              col5, col6, col7, col8 = st.columns(4)
              with col5:
                st.metric("CVaR",f"{metricas(asset2)[4]:,.2f}")
              with col6:
                st.metric("Sharpe Ratio",f"{metricas(asset2)[5]:,.2f}")
              with col7:
                st.metric("Sortino Ratio",f"{metricas(asset2)[6]:,.2f}")
              with col8:
                st.metric("Max Drawdown",f"{metricas(asset2)[7]:,.2f}")
        fragmento()
       elif boton3:
        @st.fragment
        def fragmento():
              fig = graficas_metricas(grafica, asset3)
              st.plotly_chart(fig)
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                st.metric("Media",f"{metricas(asset3)[0]:,.2f}")
              with col2:
                st.metric("Sesgo",f"{metricas(asset3)[1]:,.2f}")
              with col3:
                st.metric("Kurtosis",f"{metricas(asset3)[2]:,.2f}")
              with col4:
                st.metric("VaR",f"{metricas(asset3)[3]:,.2f}")
              col5, col6, col7, col8 = st.columns(4)
              with col5:
                st.metric("CVaR",f"{metricas(asset3)[4]:,.2f}")
              with col6:
                st.metric("Sharpe Ratio",f"{metricas(asset3)[5]:,.2f}")
              with col7:
                st.metric("Sortino Ratio",f"{metricas(asset3)[6]:,.2f}")
              with col8:
                st.metric("Max Drawdown",f"{metricas(asset3)[7]:,.2f}")
        fragmento()
       elif boton4:
        @st.fragment
        def fragmento():
              fig = graficas_metricas(grafica, asset4)
              st.plotly_chart(fig)
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                st.metric("Media",f"{metricas(asset4)[0]:,.2f}")
              with col2:
                st.metric("Sesgo",f"{metricas(asset4)[1]:,.2f}")
              with col3:
                st.metric("Kurtosis",f"{metricas(asset4)[2]:,.2f}")
              with col4:
                st.metric("VaR",f"{metricas(asset4)[3]:,.2f}")
              col5, col6, col7, col8 = st.columns(4)
              with col5:
                st.metric("CVaR",f"{metricas(asset4)[4]:,.2f}")
              with col6:
                st.metric("Sharpe Ratio",f"{metricas(asset4)[5]:,.2f}")
              with col7:
                st.metric("Sortino Ratio",f"{metricas(asset4)[6]:,.2f}")
              with col8:
                st.metric("Max Drawdown",f"{metricas(asset4)[7]:,.2f}")
        fragmento()
       elif boton5:
        @st.fragment
        def fragmento():
              fig = graficas_metricas(grafica, asset5)
              st.plotly_chart(fig)
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                st.metric("Media",f"{metricas(asset5)[0]:,.2f}")
              with col2:
                st.metric("Sesgo",f"{metricas(asset5)[1]:,.2f}")
              with col3:
                st.metric("Kurtosis",f"{metricas(asset5)[2]:,.2f}")
              with col4:
                st.metric("VaR",f"{metricas(asset5)[3]:,.2f}")
              col5, col6, col7, col8 = st.columns(4)
              with col5:
                st.metric("CVaR",f"{metricas(asset5)[4]:,.2f}")
              with col6:
                st.metric("Sharpe Ratio",f"{metricas(asset5)[5]:,.2f}")
              with col7:
                st.metric("Sortino Ratio",f"{metricas(asset5)[6]:,.2f}")
              with col8:
                st.metric("Max Drawdown",f"{metricas(asset5)[7]:,.2f}")
        fragmento()
       else:
        @st.fragment
        def fragmento():
              fig = graficas_metricas(grafica, asset1)
              st.plotly_chart(fig)
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                st.metric("Media",f"{metricas(asset1)[0]:,.2f}")
              with col2:
                st.metric("Sesgo",f"{metricas(asset1)[1]:,.2f}")
              with col3:
                st.metric("Kurtosis",f"{metricas(asset1)[2]:,.2f}")
              with col4:
                st.metric("VaR",f"{metricas(asset1)[3]:,.2f}")
              col5, col6, col7, col8 = st.columns(4)
              with col5:
                st.metric("CVaR",f"{metricas(asset1)[4]:,.2f}")
              with col6:
                st.metric("Sharpe Ratio",f"{metricas(asset1)[5]:,.2f}")
              with col7:
                st.metric("Sortino Ratio",f"{metricas(asset1)[6]:,.2f}")
              with col8:
                st.metric("MaxDrawdown",f"{metricas(asset1)[7]:,.2f}")
        fragmento()

def markowitz_max_sharpe(etfs):

    numofasset = len(etfs)
    df = yf.download(etfs, start="2010-1-1", end="2020-12-31")['Close']
    returns = df.pct_change().fillna(0)

    def portfolio_stats(weights,returns):

        weights = np.array(weights)[:,np.newaxis]
        port_rets = weights.T @ np.array(returns.mean() * 252)[:,np.newaxis]
        port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))

        return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights, returns)[2]
    
    # Specify constraints and bounds
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(numofasset))
    initial_wts = numofasset*[1./numofasset]

    opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
    
    # Portfolio weights
    pesos_optimos = list(zip(etfs, np.around(opt_sharpe['x']*100,2)))

    # Portfolio stats
    stats = ['Returns', 'Volatility', 'Sharpe Ratio']
    sta = list(zip(stats, np.around(portfolio_stats(opt_sharpe['x'],returns),4)))

    return sta, pesos_optimos

def markowitz(etfs,target_return):
    numofasset = len(etfs)
    df = yf.download(etfs, start="2010-1-1", end="2020-12-31")['Close']
    returns = df.pct_change().fillna(0)

    def portfolio_stats(weights):

        weights = np.array(weights)[:,np.newaxis]
        port_rets = weights.T @ np.array(returns.mean() * 252)[:,np.newaxis]
        port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))

        return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights)[2]

    # Specify constraints and bounds
    cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    bnds = tuple((0, 1) for x in range(numofasset))
    initial_wts = numofasset*[1./numofasset]

    opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    # Portfolio weights
    pesos_optimos_sharpe = np.around(opt_sharpe['x']*100,2)

    # Portfolio stats
    stats = ['Returns', 'Volatility', 'Sharpe Ratio']
    sta_sharpe = np.around(portfolio_stats(opt_sharpe['x']),4)

    # Minima varianza

    def min_variance(weights):
      return portfolio_stats(weights)[1]**2
    
    def target_min_variance(weights):
      return portfolio_stats(weights)[1]**2
    
    opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    # Optimización para mínima varianza con rendimiento objetivo
    
    cons2 = cons + [{'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - target_return}]
    opt_target_var = sco.minimize(target_min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons2)

    return opt_sharpe, opt_var, opt_target_var

def markowitz_pesos(etfs,target_return):
    numofasset = len(etfs)
    df = yf.download(etfs, start="2010-1-1", end="2020-12-31")['Close']
    returns = df.pct_change().fillna(0)

    usd_pesos = yf.download("MXN=X", start="2010-1-1", end="2020-12-31",interval="1d")['Close']
    df_ret = yf.download(["BND","EMB","ACWX","EEM","DBC"], start="2010-1-1", end="2020-12-31",interval="1d")['Close']

    # Combinar el DataFrame de ETFs con el histórico de tasa de cambio por fecha
    combined_df = df_ret.join(usd_pesos, how='inner')

    combined_df["ACWX"] = combined_df["ACWX"].fillna(0)
    combined_df["BND"] = combined_df["BND"].fillna(0)
    combined_df["DBC"] = combined_df["DBC"].fillna(0)
    combined_df["EEM"] = combined_df["EEM"].fillna(0)
    combined_df["EMB"] = combined_df["EMB"].fillna(0)

    rend_pesos = combined_df[["BND","EMB","ACWX","EEM","DBC"]]

    def portfolio_stats(weights):

        weights = np.array(weights)[:,np.newaxis]
        port_rets = weights.T @ np.array(returns.mean() * 252)[:,np.newaxis]
        port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))

        return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()
  
    def portfolio_stats_target(weights):

        weights = np.array(weights)[:,np.newaxis]
        port_rets = weights.T @ np.array(rend_pesos.mean() * 252)[:,np.newaxis]
        port_vols = np.sqrt(multi_dot([weights.T, rend_pesos.cov() * 252, weights]))

        return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights)[2]

    # Specify constraints and bounds
    cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    bnds = tuple((0, 1) for x in range(numofasset))
    initial_wts = numofasset*[1./numofasset]

    opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    # Portfolio weights
    pesos_optimos_sharpe = np.around(opt_sharpe['x']*100,2)

    # Portfolio stats
    stats = ['Returns', 'Volatility', 'Sharpe Ratio']
    sta_sharpe = np.around(portfolio_stats(opt_sharpe['x']),4)

    # Minima varianza

    def min_variance(weights):
      return portfolio_stats(weights)[1]**2
    
    def target_min_variance(weights):
      return portfolio_stats_target(weights)[1]**2
    
    opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

    # Optimización para mínima varianza con rendimiento objetivo
    
    cons2 = cons + [{'type': 'eq', 'fun': lambda x: portfolio_stats_target(x)[0] - target_return}]
    opt_target_var = sco.minimize(target_min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons2)

    return opt_sharpe, opt_var, opt_target_var
    
def grafica_dona(df):

    custom_colors = ["#FF0000", "#FF4500","#FF6347","#FF7F50","#FFA500"]

    fig = go.Figure(go.Pie(
    labels=df["ETF's"],
    values=df["Pesos Óptimos"],
    hole=0.5,
    marker=dict(colors=custom_colors)))

    return fig

def df_pesos(niv):
  nivel_rend = niv 
  etfs = ["BND","EMB","ACWX","EEM","DBC"]
  df_etfs = pd.DataFrame(etfs, columns=["ETF's"])

  opt_sharpe = markowitz(etfs,nivel_rend)[0]
  pesos_optimos_sharpe = pd.DataFrame(np.around(opt_sharpe['x']*100,2).T)
  df_sharpe = pd.concat([df_etfs, pesos_optimos_sharpe], axis=1)
  df_sharpe.reset_index()
  df_sharpe.columns = ["ETF's","Pesos Óptimos"]
  
  opt_min_var = markowitz(etfs,nivel_rend)[1]
  pesos_optimos_min_var = pd.DataFrame(np.around(opt_min_var['x']*100,2).T)
  df_min_var = pd.concat([df_etfs, pesos_optimos_min_var], axis=1)
  df_min_var.reset_index()
  df_min_var.columns = ["ETF's","Pesos Óptimos"]

  opt_min_var_target = markowitz(etfs,nivel_rend)[2]
  pesos_optimos_min_var_target = pd.DataFrame(np.around(opt_min_var_target['x']*100,2).T)
  df_min_var_target = pd.concat([df_etfs, pesos_optimos_min_var_target], axis=1)
  df_min_var_target.reset_index()
  df_min_var_target.columns = ["ETF's","Pesos Óptimos"]

  return df_sharpe, df_min_var, df_min_var_target

def backtesting(portafolio):

  etfs = ["BND","EMB","ACWX","EEM","DBC"]
  df = yf.download(etfs,"2021-1-1", "2023-12-31")["Close"]

  v_i =  df.iloc[:,0:5] @ np.array(portafolio["Pesos Óptimos"] / 100)
  rendimientos = v_i.pct_change().dropna()
  rendimientos = rendimientos.reset_index()
  rendimientos.rename(columns={0:"Rendimiento Portafolio"}, inplace=True)

  v_2021 = v_i.iloc[0]
  v_2022 = v_i.iloc[252]
  v_2023 = v_i.iloc[503]
  v_2024 = v_i.iloc[-1]

  rend_anual_2022 = (v_2022/v_2021 - 1)
  rend_anual_2023 = (v_2023/v_2022 - 1)
  rend_anual_2024 = (v_2024/v_2023 - 1)

  rend_acumulado = (1+rend_anual_2022) * (1+rend_anual_2023) * (1+rend_anual_2024) - 1

  sesgo = rendimientos["Rendimiento Portafolio"].skew()
  kurtosis = rendimientos["Rendimiento Portafolio"].kurt()
  vaR = calcular_var_cvar(rendimientos["Rendimiento Portafolio"])[0]
  cvaR = calcular_var_cvar(rendimientos["Rendimiento Portafolio"])[1]
  sharpe = calcular_sharpe_ratio(rendimientos["Rendimiento Portafolio"])
  sortino = calcular_sortino_ratio(rendimientos["Rendimiento Portafolio"])
  drawdown = calcular_drawdown_df(rendimientos)

  return rend_anual_2022, rend_anual_2023, rend_anual_2024, rend_acumulado, sesgo, kurtosis, vaR, cvaR, sharpe, sortino, drawdown

def backtesting_ew():

  etfs = ["BND","EMB","ACWX","EEM","DBC"]
  df = yf.download(etfs,"2021-1-1", "2023-12-31")["Close"]

  v_i =  df.iloc[:,0:5] @ np.array([1/5,1/5,1/5,1/5,1/5])
  rendimientos = v_i.pct_change().dropna()
  rendimientos = rendimientos.reset_index()
  rendimientos.rename(columns={0:"Rendimiento Portafolio"}, inplace=True)

  v_2021 = v_i.iloc[0]
  v_2022 = v_i.iloc[252]
  v_2023 = v_i.iloc[503]
  v_2024 = v_i.iloc[-1]

  rend_anual_2022 = (v_2022/v_2021 - 1)
  rend_anual_2023 = (v_2023/v_2022 - 1)
  rend_anual_2024 = (v_2024/v_2023 - 1)

  rend_acumulado = (1+rend_anual_2022) * (1+rend_anual_2023) * (1+rend_anual_2024) - 1

  sesgo = rendimientos["Rendimiento Portafolio"].skew()
  kurtosis = rendimientos["Rendimiento Portafolio"].kurt()
  vaR = calcular_var_cvar(rendimientos["Rendimiento Portafolio"])[0]
  cvaR = calcular_var_cvar(rendimientos["Rendimiento Portafolio"])[1]
  sharpe = calcular_sharpe_ratio(rendimientos["Rendimiento Portafolio"])
  sortino = calcular_sortino_ratio(rendimientos["Rendimiento Portafolio"])
  drawdown = calcular_drawdown_df(rendimientos)

  return rend_anual_2022, rend_anual_2023, rend_anual_2024, rend_acumulado, sesgo, kurtosis, vaR, cvaR, sharpe, sortino, drawdown

def metricas_benchmarks(benchmark):
    
    df = obtener_datos_acciones(benchmark, "2021-1-1", "2023-12-31")

    v_2021 = df.iloc[0,0]
    v_2022 = df.iloc[252,0]
    v_2023 = df.iloc[503,0]
    v_2024 = df.iloc[-1,0]

    rend_anual_2022 = (v_2022/v_2021 - 1)
    rend_anual_2023 = (v_2023/v_2022 - 1)
    rend_anual_2024 = (v_2024/v_2023 - 1)

    rend_acumulado = (1+rend_anual_2022) * (1+rend_anual_2023) * (1+rend_anual_2024) - 1

    media = metricas(benchmark)[0]
    sesgo = metricas(benchmark)[1]
    kurtosis = metricas(benchmark)[2]
    var = metricas(benchmark)[3]
    cvar = metricas(benchmark)[4]
    sharpe = metricas(benchmark)[5]
    sortino = metricas(benchmark)[6]
    drawdown = metricas(benchmark)[7]

    return rend_anual_2022, rend_anual_2023, rend_anual_2024, rend_acumulado, media, sesgo, kurtosis, var, cvar, sharpe, sortino, drawdown

def contenido_3():
    @st.fragment
    def fragmento():
        etfs = ["BND","EMB","ACWX","EEM","DBC"]
        df_etfs = pd.DataFrame(etfs, columns=["ETF's"])

        col1, col2, col3 = st.columns(3)
        with col1:
              conte1 = st.container(border=True,height=830)
              with conte1:
                st.markdown("""<h2 style='color: #FF5733;text-align:center'>Max Sharpe Ratio</h2>""", 
                unsafe_allow_html=True)
                st.markdown("<div style='margin: 91px 0;'></div>",unsafe_allow_html=True)
                opt_sharpe = markowitz(etfs,1)[0]
                pesos_optimos_sharpe = pd.DataFrame(np.around(opt_sharpe['x']*100,2).T)
                df_sharpe = pd.concat([df_etfs, pesos_optimos_sharpe], axis=1)
                df_sharpe.reset_index()
                df_sharpe.columns = ["ETF's","Pesos Óptimos"]
                st.dataframe(df_sharpe,use_container_width=True, hide_index=True)
                fig = grafica_dona(df_sharpe)
                st.plotly_chart(fig, key="1")
        with col2:
              conte2 = st.container(border=True,height=830)
              with conte2:
                st.markdown("""<h2 style='color: #FF5733;text-align:center'>Min Varianza</h2>""", 
                unsafe_allow_html=True)
                st.markdown("<div style='margin: 91px 0;'></div>",unsafe_allow_html=True)
                opt_min_var = markowitz(etfs,1)[1]
                pesos_optimos_min_var = pd.DataFrame(np.around(opt_min_var['x']*100,2).T)
                df_min_var = pd.concat([df_etfs, pesos_optimos_min_var], axis=1)
                df_min_var.reset_index()
                df_min_var.columns = ["ETF's","Pesos Óptimos"]
                st.dataframe(df_min_var,use_container_width=True, hide_index=True)
                fig = grafica_dona(df_min_var)
                st.plotly_chart(fig, key="2")
        with col3:
              @st.fragment
              def fragmento():
                conte3 = st.container(border=True,height=830)
                with conte3:
                  col1,col2 = st.columns(2)
                  with col1:
                    st.markdown("<h2 style='color: #FF5733;'>Min Varianza Target</h2>", 
                    unsafe_allow_html=True)
                  with col2:
                    niv_rend = st.number_input("Elige el nivel que quieres de rendimientos para la optimización por minima varianza",value=0.1)
                  opt_min_var_target = markowitz_pesos(etfs,niv_rend)[2]
                  pesos_optimos_min_var_target = pd.DataFrame(np.around(opt_min_var_target['x']*100,2).T)
                  df_min_var_target = pd.concat([df_etfs, pesos_optimos_min_var_target], axis=1)
                  df_min_var_target.reset_index()
                  df_min_var_target.columns = ["ETF's","Pesos Óptimos"]
                  st.markdown("<div style='margin: 48px 0;'></div>",unsafe_allow_html=True)
                  st.dataframe(df_min_var_target,use_container_width=True, hide_index=True)
                  fig = grafica_dona(df_min_var_target)
                  st.plotly_chart(fig,key="3")  
              fragmento()
    fragmento()

def contenido_4():
    @st.fragment
    def fragmento():
        col1, col2, col3 = st.columns(3)
        with col1:
          cont1 = st.container(border=True,height=600)
          with cont1:
              st.markdown("""<h2 style='color: #FF5733;text-align:center'>S&P 500</h2>""", 
              unsafe_allow_html=True)
              st.markdown("<div style='margin: 52px 0;'></div>",unsafe_allow_html=True)
              c1, c2, c3 = st.columns(3)
              with c1:
                  st.metric("Rendimientos del 2021 al 2022",f"{metricas_benchmarks("SPY")[0]:.4%}")
              with c2:
                  st.metric("Rendimientos del 2022 al 2023",f"{metricas_benchmarks("SPY")[1]:.4%}")
              with c3:
                  st.metric("Rendimientos del 2023 al 2024",f"{metricas_benchmarks("SPY")[2]:.4%}")
              c4, c5 = st.columns(2)
              with c4:
                  st.metric("Rendimiento acumulado del 2021 al 2023",f"{metricas_benchmarks("SPY")[3]:.4%}")
              with c5:
                  st.metric("Sesgo del 2021 al 2023",f"{metricas_benchmarks("SPY")[4]:.4}")
              c6, c7 = st.columns(2)
              with c6:
                  st.metric("Kurtosis del 2021 al 2023",f"{metricas_benchmarks("SPY")[5]:.4}")
              with c7:
                  st.metric("VaR del 2021 al 2023",f"{metricas_benchmarks("SPY")[6]:.4}")
              c8, c9 = st.columns(2)
              with c8:
                  st.metric("CVaR del 2021 al 2023",f"{metricas_benchmarks("SPY")[7]:.4}")
              with c9:
                  st.metric("Sharpe Ratio del 2021 al 2023",f"{metricas_benchmarks("SPY")[8]:.4}")
              c10, c11 = st.columns(2)
              with c10:
                  st.metric("Sortino Ratio del 2021 al 2023",f"{metricas_benchmarks("SPY")[9]:.4}")
              with c11:
                  st.metric("Max Drawdown del 2021 al 2023",f"{metricas_benchmarks("SPY")[10]:.4}")
          with col2:
            cont2 = st.container(border=True,height=600)
            with cont2:
              st.markdown("""<h2 style='color: #FF5733;text-align:center'>Portafolio Pesos Iguales</h2>""", 
              unsafe_allow_html=True)
              st.markdown("<div style='margin: 52px 0;'></div>",unsafe_allow_html=True)
              c1, c2, c3 = st.columns(3)
              with c1:
                  st.metric("Rendimientos del 2021 al 2022",f"{backtesting_ew()[0]:.4%}")
              with c2:
                  st.metric("Rendimientos del 2022 al 2023",f"{backtesting_ew()[1]:.4%}")
              with c3:
                  st.metric("Rendimientos del 2023 al 2024",f"{backtesting_ew()[2]:.4%}")
              c4, c5 = st.columns(2)
              with c4:
                  st.metric("Rendimiento acumulado del 2021 al 2023",f"{backtesting_ew()[3]:.4%}")
              with c5:
                  st.metric("Sesgo del 2021 al 2023",f"{backtesting_ew()[4]:.4}")
              c6, c7 = st.columns(2)
              with c6:
                  st.metric("Kurtosis del 2021 al 2023",f"{backtesting_ew()[5]:.4}")
              with c7:
                  st.metric("VaR del 2021 al 2023",f"{backtesting_ew()[6]:.4}")
              c8, c9 = st.columns(2)
              with c8:
                  st.metric("CVaR del 2021 al 2023",f"{backtesting_ew()[7]:.4}")
              with c9:
                  st.metric("Sharpe Ratio del 2021 al 2023",f"{backtesting_ew()[8]:.4}")
              c10, c11 = st.columns(2)
              with c10:
                  st.metric("Sortino Ratio del 2021 al 2023",f"{backtesting_ew()[9]:.4}")
              with c11:
                  st.metric("Max Drawdown del 2021 al 2023",f"{backtesting_ew()[10]:.4}")
          with col3:
            @st.fragment
            def fragmento():
              cont3 = st.container(border=True,height=600)
              with cont3:
                columnaa1, columnaa2, columnaa3 = st.columns(3)
                with columnaa1:
                  st.markdown("<h2 style='color: #FF5733;'>Portafolios Óptimos</h2>", 
                  unsafe_allow_html=True)
                with columnaa2:
                  niv = st.number_input("Elige el nivel que quieres de rendimientos para minima varianza",key="ks",value=0.1)
                with columnaa3:
                  seleccion = st.selectbox("Selecciona el método de optimización para calcular las metricas",options=["Sharpe Ratio","Min Varianza","Min Varianza con Rendimiento Objetivo"])
                pesos_sharpe = df_pesos(niv)[0]
                pesos_min_var = df_pesos(niv)[1]
                pesos_min_var_target = df_pesos(niv)[2]
                if seleccion == "Sharpe Ratio":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Rendimientos del 2021 al 2022",f"{backtesting(pesos_sharpe)[0]:.4%}")
                    with c2:
                        st.metric("Rendimientos del 2022 al 2023",f"{backtesting(pesos_sharpe)[1]:.4%}")
                    with c3:
                        st.metric("Rendimientos del 2023 al 2024",f"{backtesting(pesos_sharpe)[2]:.4%}")
                    c4, c5 = st.columns(2)
                    with c4:
                        st.metric("Rendimiento acumulado del 2021 al 2023",f"{backtesting(pesos_sharpe)[3]:.4%}")
                    with c5:
                        st.metric("Sesgo del 2021 al 2023",f"{backtesting(pesos_sharpe)[4]:.4}")
                    c6, c7 = st.columns(2)
                    with c6:
                        st.metric("Kurtosis del 2021 al 2023",f"{backtesting(pesos_sharpe)[5]:.4}")
                    with c7:
                        st.metric("VaR del 2021 al 2023",f"{backtesting(pesos_sharpe)[6]:.4}")
                    c8, c9 = st.columns(2)
                    with c8:
                        st.metric("CVaR del 2021 al 2023",f"{backtesting(pesos_sharpe)[7]:.4}")
                    with c9:
                        st.metric("Sharpe Ratio del 2021 al 2023",f"{backtesting(pesos_sharpe)[8]:.4}")
                    c10, c11 = st.columns(2)
                    with c10:
                        st.metric("Sortino Ratio del 2021 al 2023",f"{backtesting(pesos_sharpe)[9]:.4}")
                    with c11:
                        st.metric("Max Drawdown del 2021 al 2023",f"{backtesting(pesos_sharpe)[10]:.4}")
                elif seleccion == "Min Varianza":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Rendimientos del 2021 al 2022",f"{backtesting(pesos_min_var)[0]:.4%}")
                    with c2:
                        st.metric("Rendimientos del 2022 al 2023",f"{backtesting(pesos_min_var)[1]:.4%}")
                    with c3:
                        st.metric("Rendimientos del 2023 al 2024",f"{backtesting(pesos_min_var)[2]:.4%}")
                    c4, c5 = st.columns(2)
                    with c4:
                        st.metric("Rendimiento acumulado del 2021 al 2023",f"{backtesting(pesos_min_var)[3]:.4%}")
                    with c5:
                        st.metric("Sesgo del 2021 al 2023",f"{backtesting(pesos_min_var)[4]:.4}")
                    c6, c7 = st.columns(2)
                    with c6:
                        st.metric("Kurtosis del 2021 al 2023",f"{backtesting(pesos_min_var)[5]:.4}")
                    with c7:
                        st.metric("VaR del 2021 al 2023",f"{backtesting(pesos_min_var)[6]:.4}")
                    c8, c9 = st.columns(2)
                    with c8:
                        st.metric("CVaR del 2021 al 2023",f"{backtesting(pesos_min_var)[7]:.4}")
                    with c9:
                        st.metric("Sharpe Ratio del 2021 al 2023",f"{backtesting(pesos_min_var)[8]:.4}") 
                    c10, c11 = st.columns(2)
                    with c10:
                        st.metric("Sortino Ratio del 2021 al 2023",f"{backtesting(pesos_min_var)[9]:.4}")
                    with c11:
                        st.metric("Max Drawdown del 2021 al 2023",f"{backtesting(pesos_min_var)[10]:.4}")               
                elif seleccion == "Min Varianza con Rendimiento Objetivo":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Rendimientos del 2021 al 2022",f"{backtesting(pesos_min_var_target)[0]:.4%}")
                    with c2:
                        st.metric("Rendimientos del 2022 al 2023",f"{backtesting(pesos_min_var_target)[1]:.4%}")
                    with c3:
                        st.metric("Rendimientos del 2023 al 2024",f"{backtesting(pesos_min_var_target)[2]:.4%}")
                    c4, c5 = st.columns(2)
                    with c4:
                        st.metric("Rendimiento acumulado del 2021 al 2023",f"{backtesting(pesos_min_var_target)[3]:.4%}")
                    with c5:
                        st.metric("Sesgo del 2021 al 2023",f"{backtesting(pesos_min_var_target)[4]:.4}")
                    c6, c7 = st.columns(2)
                    with c6:
                        st.metric("Kurtosis del 2021 al 2023",f"{backtesting(pesos_min_var_target)[5]:.4}")
                    with c7:
                        st.metric("VaR del 2021 al 2023",f"{backtesting(pesos_min_var_target)[6]:.4}")
                    c8, c9 = st.columns(2)
                    with c8:
                        st.metric("CVaR del 2021 al 2023",f"{backtesting(pesos_min_var_target)[7]:.4}")
                    with c9:
                        st.metric("Sharpe Ratio del 2021 al 2023",f"{backtesting(pesos_min_var_target)[8]:.4}")
                    c10, c11 = st.columns(2)
                    with c10:
                        st.metric("Sortino Ratio del 2021 al 2023",f"{backtesting(pesos_min_var_target)[9]:.4}")
                    with c11:
                        st.metric("Max Drawdown del 2021 al 2023",f"{backtesting(pesos_min_var_target)[10]:.4}")
            fragmento()
        columnaaaa1, columnaaaa2, columnaaaa3 = st.columns(3)
        with columnaaaa1:
          with st.container():
              st.subheader("Comparación S&P500 vs Pesos Iguales vs Max Sharpe:")

              st.markdown(
                  """
                  <div class="popover-container">
                      <button class="popover-button">Mostrar información</button>
                      <div class="popover-content">
                          <strong>Conclusión general:</strong> <br>
                          Dado todo el trabajo de comparación que hemos hecho entre S&amp;P 500 (benchmark),
                          Pesos Iguales y Óptimo Sharpe Ratio, podemos decir que el S&amp;P 500 como benchmark
                          sigue siendo la referencia más sólida. A pesar de su Sharpe Ratio negativo de -0.026,
                          logró un rendimiento acumulado de +28.88%, destacándose frente a las otras dos
                          estrategias. El portafolio de Pesos Iguales tuvo un rendimiento negativo de -14.66%, lo
                          que indica que esta estrategia conservadora no aprovechó las condiciones favorables del
                          mercado, resultando en una rentabilidad nula. Por su parte, el Óptimo Sharpe Ratio,
                          diseñado para maximizar la rentabilidad ajustada al riesgo, no cumplió su objetivo, con un
                          rendimiento negativo de -15.26% y un drawdown extremadamente alto, lo que revela que
                          la optimización no fue efectiva. En resumen, el S&amp;P 500 se muestra como el benchmark
                          más eficaz, mientras que las otras dos estrategias no lograron superar su rendimiento ni
                          en rentabilidad ni en gestión del riesgo.<a href="https://drive.google.com/file/d/108stD22nyt5u7Qo6bA6m281rWzaVmBWp/view?usp=sharing" target="_blank">Ver más información</a>
                      </div>
                  </div>

                  <style>
                      .popover-container {
                          position: relative;
                          display: inline-block;
                          width: 100%; /* Ajusta al ancho del contenedor */
                      }
                      .popover-button {
                          padding: 10px 15px;
                          background-color: #FF5733;
                          color: white;
                          border: none;
                          cursor: pointer;
                          border-radius: 5px;
                          width: 100%; /* Ajusta al ancho del contenedor */
                          margin-top: 20px; /* Espaciado superior dentro del contenedor */
                          font-size: 30px;
                      }
                      .popover-content {
                          display: none;
                          position: absolute;
                          bottom: 80px; /* Controla la distancia hacia arriba del popover */
                          left: 50%;
                          transform: translateX(-50%);
                          background-color: #fff;
                          color: #333;
                          padding: 10px;
                          border-radius: 8px;
                          border: 1px solid #ccc;
                          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                          z-index: 10;
                          max-width: 1000px;
                          font-size: 25px;
                      }
                      .popover-container:hover .popover-content {
                          display: block; /* Mostrar contenido al pasar el mouse */
                          max-width: 1000px;
                          width: 100%;
                      }
                  </style>
                  """,
                  unsafe_allow_html=True,
              )
        with columnaaaa2:
          with st.container():
              st.subheader("Comparación S&P500 vs Pesos Iguales vs Min Varianza:")

              st.markdown(
                  """
                  <div class="popover-container">
                      <button class="popover-button">Mostrar información</button>
                      <div class="popover-content">
                          <strong>Conclusión general:</strong> <br>
                          En la comparación entre S&P 500 (benchmark), Pesos Iguales y Óptimo Mínima Varianza,
                          el S&P 500 también sobresale como la mejor opción. A pesar de su Sharpe Ratio
                          negativo, su retorno acumulado de +28.88% y su gestión del riesgo más efectiva lo
                          colocan por encima de las otras estrategias. El portafolio de Pesos Iguales tuvo un
                          rendimiento negativo de -14.66%, lo que muestra una falta de exposición al riesgo y una
                          incapacidad para capitalizar las oportunidades del mercado. El Óptimo Mínima Varianza, a
                          pesar de buscar minimizar la volatilidad, tuvo un rendimiento aún peor, con -15.58% de
                          retorno acumulado y un drawdown muy alto de 297.4, lo que sugiere que la optimización
                          de la volatilidad no fue efectiva en este contexto. En conclusión, el S&P 500 sigue siendo
                          la mejor opción, con un rendimiento superior y una gestión más eficiente del riesgo,
                          mientras que Pesos Iguales y Óptimo Mínima Varianza no lograron superar el benchmark
                          en este período.<a href="https://drive.google.com/file/d/1YbWzrpt6q6f8ym-4LK-2APiKhJ52OL_r/view?usp=sharing" target="_blank">Ver más información</a>
                      </div>
                  </div>

                  <style>
                      .popover-container {
                          position: relative;
                          display: inline-block;
                          width: 100%; /* Ajusta al ancho del contenedor */
                      }
                      .popover-button {
                          padding: 10px 15px;
                          background-color: #FF5733;
                          color: white;
                          border: none;
                          cursor: pointer;
                          border-radius: 5px;
                          width: 100%; /* Ajusta al ancho del contenedor */
                          margin-top: 20px; /* Espaciado superior dentro del contenedor */
                          font-size: 30px;
                      }
                      .popover-content {
                          display: none;
                          position: absolute;
                          bottom: 80px; /* Controla la distancia hacia arriba del popover */
                          left: 50%;
                          transform: translateX(-50%);
                          background-color: #fff;
                          color: #333;
                          padding: 10px;
                          border-radius: 8px;
                          border: 1px solid #ccc;
                          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                          z-index: 10;
                          max-width: 1000px;
                          font-size: 30px;
                      }
                      .popover-container:hover .popover-content {
                          display: block; /* Mostrar contenido al pasar el mouse */
                          max-width: 1000px;
                          width: 100%;
                      }
                  </style>
                  """,
                  unsafe_allow_html=True,
              )
        with columnaaaa3:
          with st.container():
              st.subheader("Comparación S&P500 vs Pesos Iguales vs Min Varianza con Rendimiento Objetivo:")

              st.markdown(
                  """
                  <div class="popover-container">
                      <button class="popover-button">Mostrar información</button>
                      <div class="popover-content">
                          <strong>Conclusión general:</strong> <br>
                          En resumen, el S&P 500 como benchmark sigue siendo la estrategia más eficaz en términos de
                          rendimiento absoluto, con un rendimiento positivo de +28.88% y un control adecuado del
                          riesgo en comparación con las otras dos estrategias. El Óptimo Mínima Varianza Target,
                          aunque tiene una curtosis más alta y un mayor VaR y CVaR, no logró proteger eficazmente el
                          capital y resultó en un rendimiento negativo de -22.67%. El portafolio de Pesos Iguales,
                          aunque sin drawdown, sufrió una pérdida negativa de -14.66% debido a su falta de exposición
                          al riesgo. En conclusión, el S&P 500 ha demostrado ser superior a ambas estrategias en
                          términos de rentabilidad y gestión del riesgo en este periodo analizado.<a href="https://drive.google.com/file/d/1rQaNjBQn1a-pyY-II7dGGxmP1P8cvSeY/view?usp=sharing" target="_blank">Ver más información</a>
                      </div>
                  </div>

                  <style>
                      .popover-container {
                          position: relative;
                          display: inline-block;
                          width: 100%; /* Ajusta al ancho del contenedor */
                      }
                      .popover-button {
                          padding: 10px 15px;
                          background-color: #FF5733;
                          color: white;
                          border: none;
                          cursor: pointer;
                          border-radius: 5px;
                          width: 100%; /* Ajusta al ancho del contenedor */
                          margin-top: 20px; /* Espaciado superior dentro del contenedor */
                          font-size: 30px;
                      }
                      .popover-content {
                          display: none;
                          position: absolute;
                          bottom: 80px; /* Controla la distancia hacia arriba del popover */
                          left: 50%;
                          transform: translateX(-50%);
                          background-color: #fff;
                          color: #333;
                          padding: 15px;
                          border-radius: 8px;
                          border: 1px solid #ccc;
                          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                          z-index: 10;
                          width: 1000px
                          max-width: 1000px;
                          font-size: 20px;
                      }
                      .popover-container:hover .popover-content {
                          display: block; /* Mostrar contenido al pasar el mouse */
                          width: 100%;
                          max-width: 1000px;
                      }
                  </style>
                  """,
                  unsafe_allow_html=True,
              )
    fragmento()
    
def black_litterman(etfs, lambd, r=0.02):
    data = yf.download(etfs, "2010-1-1", dt.datetime.now().strftime("%Y-%m-%d"))['Close']
    returns = data.pct_change().dropna()

    # Calcular matrix de covarianza
    covarianza = returns.cov() * 252

    # Definir los views

    # Parámetro tau (ajuste de incertidumbre, usaremos el recomendado)

    tau = 0.05

    # Matriz P (Opiniones)
    P = np.array([
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,-1,1,0]
    ])

    # Matriz Q (Expectativas de rendimiento)
    Q = np.array([0.04,0.07,0.05])

    # Matriz Omega (Incertidumbre asociada a las opiniones) aprox

    om = np.dot(np.dot(P,tau*covarianza),P.T)
    Omega = np.diag(np.diag(om))

    # Calculamos distribucion a priori

    pesos = np.array([1./len(etfs)]*len(etfs))

    pi = lambd * covarianza @ pesos

    # Calculamos la esperanza del retorno
    
    mu = np.linalg.inv(tau * (np.linalg.inv(covarianza)) + P.T @ np.linalg.inv(Omega) @ P) @ \
         (tau * (np.linalg.inv(covarianza) @ pi) + P.T @ np.linalg.inv(Omega) @ Q)
    

    # Funcion objetivo para maximizar
    def objetivo(w):
        return -(r + w.T @ mu - (lambd / 2) * w.T @ covarianza @ w)
    
    # Restricciones
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Suma de pesos igual a 1
    bounds = tuple((0, 1) for _ in range(len(etfs)))

    # Optimización
    resultado = minimize(objetivo, mu, method='SLSQP', bounds=bounds, constraints=constraints)

    # Vector de pesos optimos dados los views

    optimal_weights = resultado['x']

    return optimal_weights

def contenido_5():
    etfs = ["BND","EMB","ACWX","EEM","DBC"]
    df_etfs = pd.DataFrame(etfs, columns=["ETF's"])
    col1, col2 = st.columns(2)
    with col1:
        conte1 = st.container(border=True,height=898)
        with conte1:
          st.markdown("""<h2 style='color: #FF5733;text-align:center'>Pesos Óptimos Black-Literman</h2>""", 
          unsafe_allow_html=True)
          pesos_optimos = black_litterman(etfs,6)
          pesos_optimos_df = pd.DataFrame(np.around(pesos_optimos*100,2).T)
          df_black = pd.concat([df_etfs, pesos_optimos_df], axis=1)
          df_black.reset_index()
          df_black.columns = ["ETF's","Pesos Óptimos"]
          st.dataframe(df_black,use_container_width=True, hide_index=True)
          fig = grafica_dona(df_black)
          st.plotly_chart(fig, key="9383")  
    with col2:
          conte2 = st.container(border=True)
          with conte2:
            st.markdown("""<h2 style='color: #FF5733;text-align:center'>View 1 (BND)</h2>""",
                        unsafe_allow_html=True)
            st.markdown('<p style="font-size:20px;">Panorama: La Reserva Federal podría comenzar a reducir las tasas de interés en el tercer trimestre de 2024, lo que impulsaría los precios de los bonos. <br>\
                       - Esto beneficia a BND, especialmente si las curvas de rendimiento se empinan.  <br>\
                       - Rendimiento esperado: Un retorno de 4%, basado en la estabilización de los rendimientos y el ingreso por cupones</p>', unsafe_allow_html=True)
          conte3 = st.container(border=True)
          with conte3:
            st.markdown("""<h2 style='color: #FF5733;text-align:center'>View 2 (EMB)</h2>""",
                        unsafe_allow_html=True)
            st.markdown('<p style="font-size:20px;">Panorama: Aunque los mercados emergentes pueden beneficiarse de una mayor estabilidad en las tasas de interés globales y de una recuperación económica moderada en China, la falta de impulso global significativo y riesgos geopolíticos pueden limitar los retornos.  <br>\
                        - Rendimiento esperado: 7%, dependiendo del crecimiento en Asia y los flujos hacia mercados emergentes</p>', unsafe_allow_html=True)
          conte4 = st.container(border=True)
          with conte4:
            st.markdown("""<h2 style='color: #FF5733;text-align:center'>View 3 (EEM vs ACWX)</h2>""",
                        unsafe_allow_html=True)
            st.markdown('<p style="font-size:20px;">Panorama: La estabilización en Asia y la mejora de términos de intercambio podrían favorecer a EEM frente a mercados desarrollados, donde el crecimiento es más lento y la presión salarial es elevada.  <br>\
                      Expectativa de rendimiento:  <br>\
                       - EEM: apoyado por un entorno más favorable en Asia y una potencial estabilización en China.  <br>\
                       - ACWX:limitado por Europa y Japón, donde las perspectivas de crecimiento son más débiles.  <br>\
                       Esperamos una diferencia de al menos 4%-5% en EEM frente a ACWX</p>',unsafe_allow_html=True)
