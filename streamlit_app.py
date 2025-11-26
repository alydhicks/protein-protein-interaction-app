import streamlit as st
import pandas as pd
import numpy as np

st.title('🎈 Protein-Protein Interaction App')

st.write('This is a PPI machine learning app!')

neg_df = pd.read_csv('https://raw.githubusercontent.com/alydhicks/Protein-Files/refs/heads/main/negative_protein_sequences.csv')
neg_df.head()

pos_df = pd.read_csv('')
pos_df.head()
