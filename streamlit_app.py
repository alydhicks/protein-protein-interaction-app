import streamlit as st
import pandas as pd
import numpy as np

st.title('🎈 Protein-Protein Interaction App')

st.write('This is a PPI machine learning app!')

neg_df = pd.read_csv('https://github.com/alydhicks/protein-protein-interaction-app/blob/master/negative_protein_sequences_250.csv')
neg_df.head()

pos_df = pd.read_csv('')
