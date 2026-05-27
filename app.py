import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Mi cartera automática")

df = pd.read_csv("datos.csv", index_col=0)

fig = px.line(df, title="Precios históricos")

st.plotly_chart(fig, use_container_width=True)