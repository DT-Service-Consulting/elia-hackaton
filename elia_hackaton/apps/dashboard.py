import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st
import webbrowser
import os

# Set Streamlit page title
st.title("Minimal Streamlit Dashboard")

# Check if the HTML file exists, if not, create an empty one
html_file = "index.html"
if not os.path.exists(html_file):
    with open(html_file, "w") as f:
        f.write("<h1>Welcome to the HTML Page</h1>")

# Button to open the HTML file
if st.button("Open HTML Page"):
    webbrowser.open(f"file://{os.path.abspath(html_file)}")

st.write("Click the button to open the HTML page.")
