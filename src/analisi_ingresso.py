import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import warnings
warnings.filterwarnings('ignore') # Nasconde i warning fastidiosi di Python

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
import os

print("Caricamento dati e traduzione in corso...")

# Calcoliamo il percorso esatto in modo dinamico
# 1. Trova la cartella dove si trova questo script (src)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Crea il percorso per la cartella raw_data
raw_data_dir = os.path.join(script_dir, '..', 'raw_data')

# 3. Creiamo i percorsi per i due file
file_imm = os.path.join(raw_data_dir, 'bdg_serie_immatricolati.csv')
file_cod = os.path.join(raw_data_dir, 'cod_foet2013.csv')

# Carichiamo i dati usando i percorsi assoluti
df_imm = pd.read_csv(file_imm, sep=None, engine='python', encoding='latin1')
df_cod = pd.read_csv(file_cod, sep=None, engine='python', encoding='latin1')

# PULIZIA COLONNE: Rimuoviamo eventuali spazi vuoti invisibili dai titoli
df_imm.columns = df_imm.columns.str.strip()
df_cod.columns = df_cod.columns.str.strip()

# Identificazione codici STEM
df_cod['Area STEM'] = df_cod['Area STEM'].astype(str).str.strip().str.lower()
stem_prefixes = df_cod[df_cod['Area STEM'].isin(['sì', 'si', 'yes', 'y'])]['ISCED_F_1dgt'].astype(str).str.lstrip('0').unique()

# Pulizia dataset immatricolati
df_imm['COD_FoET2013_clean'] = df_imm['COD_FoET2013'].astype(str).str.lstrip('0')
df_imm['is_STEM'] = df_imm['COD_FoET2013_clean'].isin(stem_prefixes)

# Filtro Base: Anno corrente e Solo Lauree di primo livello/ciclo unico
df_base = df_imm[
    (df_imm['ANNO'] == '2024/2025') & 
    (df_imm['CorsoTIPO'].isin(['Laurea', 'Laurea Magistrale Ciclo Unico']))
]

# ==========================================
# 2. ANALISI NAZIONALE (IL TUO Z-TEST E CHI-QUADRO)
# ==========================================
# Prendiamo solo la riga del Totale Nazionale
df_nazionale = df_base[df_base['AteneoCOD'] == 'TTTTT']
tabella_nat = df_nazionale.groupby(['Genere', 'is_STEM'])['IMM'].sum().unstack(fill_value=0)
tabella_nat.columns = ['Non_STEM', 'STEM']

f_stem = tabella_nat.loc['F', 'STEM']
f_non_stem = tabella_nat.loc['F', 'Non_STEM']
m_stem = tabella_nat.loc['M', 'STEM']
m_non_stem = tabella_nat.loc['M', 'Non_STEM']

print("\n--- TEST STATISTICI COMPLETATI ---")
# Chi-Quadro
obs = np.array([[f_stem, f_non_stem], [m_stem, m_non_stem]])
chi2_stat, p_val_chi2, _, _ = chi2_contingency(obs)
print(f"P-value Chi-Quadrato: {p_val_chi2:.2e}")

# Z-Test
count = np.array([f_stem, f_non_stem]) 
nobs = np.array([f_stem + m_stem, f_non_stem + m_non_stem]) 
z_stat, p_val_z = proportions_ztest(count, nobs)
print(f"P-value Z-Test: {p_val_z:.2e}")

# ==========================================
# 3. GRAFICO 1: PIRAMIDE DELLE ISCRIZIONI (Tornado Chart)
# ==========================================
print("\nGenerazione Piramide delle Iscrizioni...")
categorie = ['Area NON-STEM\n(Umanistica, Sociale, ecc.)', 'Area STEM\n(Scienze, Ingegneria, ecc.)']
uomini = [m_non_stem, m_stem]
donne = [f_non_stem, f_stem]

fig, ax = plt.subplots(figsize=(10, 6))

# Disegniamo le barre (Uomini a sinistra come negativi, Donne a destra come positivi)
ax.barh(categorie, [-val for val in uomini], color='#2980b9', label='Uomini', height=0.6)
ax.barh(categorie, donne, color='#e74c3c', label='Donne', height=0.6)

# Formattazione per rendere i numeri negativi leggibili come positivi sull'asse
ticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(tick/1000))}k" for tick in ticks])

ax.set_title('Piramide della Segregazione Formativa (Immatricolati 2024/2025)', fontsize=14, pad=20)
ax.axvline(0, color='black', linewidth=1) # Linea centrale
ax.legend(loc='lower right')
plt.tight_layout()

# Salviamo l'immagine nella cartella src
plt.savefig('piramide_stem.png', dpi=300)
print("-> Immagine salvata come 'piramide_stem.png'")

# ==========================================
# 4. GRAFICO 2 E TABELLA: MAPPA GEOGRAFICA DEL GAP
# ==========================================
print("\nElaborazione Dati Regionali...")

# Per la mappa escludiamo il "TOTALE", le voci vuote e gli Atenei Telematici/Estero
df_regioni = df_base[
    (df_base['AteneoCOD'] != 'TTTTT') & 
    (df_base['AteneoREGIONE'].notna()) &
    (~df_base['AteneoREGIONE'].isin(['TELEMATICI', 'ESTERO']))
]

# Raggruppiamo i dati per Regione e calcoliamo quante donne sono nelle STEM
tab_reg = df_regioni.groupby(['AteneoREGIONE', 'Genere', 'is_STEM'])['IMM'].sum().reset_index()

# Isoliamo i dati STEM
stem_reg = tab_reg[tab_reg['is_STEM'] == True]
stem_pivot = stem_reg.pivot(index='AteneoREGIONE', columns='Genere', values='IMM').fillna(0)
stem_pivot['Totale_STEM'] = stem_pivot['F'] + stem_pivot['M']
stem_pivot['Percentuale_Donne_STEM'] = (stem_pivot['F'] / stem_pivot['Totale_STEM']) * 100
stem_pivot['Percentuale_Uomini_STEM'] = (stem_pivot['M'] / stem_pivot['Totale_STEM']) * 100

# --- Generazione File CSV Regionale (ORA MESSO NEL POSTO GIUSTO) ---
output_df = stem_pivot[['Totale_STEM', 'F', 'M', 'Percentuale_Donne_STEM', 'Percentuale_Uomini_STEM']].copy()
output_df = output_df.sort_values('Percentuale_Donne_STEM', ascending=False).round(2)
output_df.to_csv('percentuali_regionali_stem.csv', sep=';')
print("-> File 'percentuali_regionali_stem.csv' generato con successo!")

# --- Generazione Mappa ---
print("Generazione Mappa dell'Italia in corso...")
stem_pivot = stem_pivot.reset_index()

# Correggiamo i nomi delle regioni del MUR per farli combaciare col GeoJSON internazionale
correzioni_regioni = {
    'Trentino Alto Adige': 'Trentino-Alto Adige/Südtirol',
    'Friuli Venezia Giulia': 'Friuli-Venezia Giulia',
    'Emilia Romagna': 'Emilia-Romagna',
    "Valle d'Aosta": "Valle d'Aosta/Vallée d'Aoste"
}
stem_pivot['AteneoREGIONE'] = stem_pivot['AteneoREGIONE'].replace(correzioni_regioni)

# Scarichiamo i confini dell'Italia (GeoJSON)
url_geojson = 'https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson'
italy_geojson = requests.get(url_geojson).json()

# Creiamo la mappa interattiva
fig_map = px.choropleth_mapbox(
    stem_pivot,
    geojson=italy_geojson,
    locations='AteneoREGIONE',
    featureidkey='properties.reg_name',
    color='Percentuale_Donne_STEM',
    color_continuous_scale="Reds_r", # Più è rosso intenso, più mancano donne
    range_color=[25, 45], # Il range percentuale medio
    mapbox_style="carto-positron",
    zoom=4.5, 
    center={"lat": 41.8719, "lon": 12.5674},
    opacity=0.7,
    labels={'Percentuale_Donne_STEM': '% Donne in STEM'},
    title="Gap di Genere nelle STEM per Regione (Meno è rosso scuro, peggio è)"
)

# Apriamo la mappa nel browser
print("-> Apertura mappa nel browser. Puoi interagire con il mouse!")
fig_map.show()