import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Health Insight AI: Model Monitoring Dashboard.")

try:
    df = pd.read_csv("logs\metrics.csv")
    st.subheader("Latest Model Metrics")
    st.dataframe(df)

    st.line_chart(df["accuracy"])

    st.success("Metrics loaded successfully!")

except Exception as e:
    st.error(f"Error reading metrics file: {e}")

st.caption("This dashboard shows model accuracy trend over time from logs/metrics.csv file.")
