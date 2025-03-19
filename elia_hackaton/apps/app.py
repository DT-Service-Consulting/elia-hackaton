import streamlit as st
import pydeck as pdk
import pandas as pd
from elia_hackaton.config import DATA_DIR

# Set up Streamlit app
st.title("Interactive Map of Belgium 🇧🇪")

equipment_df = pd.read_csv(DATA_DIR / 'Equipment.csv', index_col=0)

# Define a scatterplot layer to display on a map
scatterplot_layer = pdk.Layer(
    'ScatterplotLayer',
    data=equipment_df,
    get_position='[Longitude, Latitude]',
    get_color='[200, 30, 0, 160]',
    get_radius=2000,
)

# Define a text layer to display labels
text_layer = pdk.Layer(
    'TextLayer',
    data=equipment_df,
    get_position='[Longitude, Latitude]',
    get_text='Equipment ID',
    get_size=24,
    get_color='[0, 0, 0, 255]',
    get_angle=0,
    get_alignment_baseline='"bottom"',
)

# Set the viewport location
view_state = pdk.ViewState(
    latitude=equipment_df["Latitude"].mean(),
    longitude=equipment_df["Longitude"].mean(),
    zoom=8,
)

# Render the deck.gl map
r = pdk.Deck(layers=[scatterplot_layer, text_layer],
             initial_view_state=view_state)
st.pydeck_chart(r)
