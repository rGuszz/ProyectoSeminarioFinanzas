import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import fc
from datetime import datetime, timedelta
from numpy.linalg import multi_dot
import scipy.optimize as sco

# Ponemos como predeterminado la configuración de vista amplia

st.set_page_config(page_title="Proyecto", page_icon=":guardsman:", layout="wide")

# Hacemos distintas pestañas para visualizar

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Selección de activos", "Analisis de activos", "Portafolios Óptimos", "Backtesting", "Black-Litterman"])

with tab1:
    @st.fragment
    def fragmento():
        col1, col2, col3, col4, col5, col6 = st.columns([0.355,0.129,0.129,0.129,0.129,0.129])
        with col1:
            st.markdown("<h2 style='color: #FF5733;'>Nuestra selección de activos es:</h2>", 
            unsafe_allow_html=True)
        with col2:
            asset1 = "BND"
            boton1 = st.button(asset1,key="b1", use_container_width=True,)
        with col3:
            asset2 = "EMB"
            boton2 = st.button(asset2,key="b2", use_container_width=True)
        with col4:
            asset3 = "ACWX"
            boton3 = st.button(asset3,key="b3", use_container_width=True)
        with col5:
            asset4 = "EEM"
            boton4 = st.button(asset4,key="b4", use_container_width=True)
        with col6:
            asset5 = "DBC"
            boton5 = st.button(asset5,key="b5", use_container_width=True)
        fc.contenido_1(boton1,boton2,boton3,boton4,boton5,asset1,asset2,asset3,asset4,asset5)
    fragmento()
with tab2:
    @st.fragment
    def fragmento():
        @st.fragment
        def fragmento():
            columna1, columna2 = st.columns(2)
            with columna1:
                st.markdown("""<h2 style='color: #FF5733;text-align:center'>Análisis de activos</h2>""", 
              unsafe_allow_html=True)
            with columna2:
                boton_grafica = st.selectbox("Selecciona la gráfica que deseas visualizar", options= ["Rendimiento Diario",
                                                                                                        "Histogramas",
                                                                                                        "Boxplot",
                                                                                                        "Rendimiento Acumulado"])
            col2, col3, col4, col5, col6 = st.columns(5)
            with col2:
                asset1 = "BND"
                boton1 = st.button(asset1,key="b6", use_container_width=True)
            with col3:
                asset2 = "EMB"
                boton2 = st.button(asset2,key="b7", use_container_width=True)
            with col4:
                asset3 = "ACWX"
                boton3 = st.button(asset3,key="b8", use_container_width=True)
            with col5:
                asset4 = "EEM"
                boton4 = st.button(asset4,key="b9", use_container_width=True)
            with col6:
                asset5 = "DBC"
                boton5 = st.button(asset5,key="b10", use_container_width=True)
            conte1 = st.container(border=True)
            with conte1:
                fc.contenido_2(boton1,boton2,boton3,boton4,boton5,asset1,asset2,asset3,asset4,asset5,boton_grafica)
        fragmento()
    fragmento()
with tab3:
    fc.contenido_3()
with tab4:
    fc.contenido_4()
with tab5:
    fc.contenido_5()







        


    
