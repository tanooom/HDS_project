import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import warnings
warnings.filterwarnings('ignore') 
import os

#calcolo del path della cartella attuale src e creazione di path per raw_data
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, '..', 'raw_data')

#prendiamo dai dataset quelli di interesse per effettuare le analisi relative agli ingressi
file_imm = os.path.join(raw_data_dir, 'bdg_serie_immatricolati.csv')
#il seguente file viene usato come 'traduzione' in modo da selezionare soltanto le materie STEM.
file_cod = os.path.join(raw_data_dir, 'cod_foet2013.csv')

#caricamento dei dati
df_imm = pd.read_csv(file_imm, sep=None, engine='python', encoding='latin1')
df_cod = pd.read_csv(file_cod, sep=None, engine='python', encoding='latin1')

#effettuiamo pulizia delle colonne per eliminare eventuali spazi vuoti
df_imm.columns = df_imm.columns.str.strip()
df_cod.columns = df_cod.columns.str.strip()

#in questa parte dello script si vanno a selezionare le lauree appartenenti all'area stem
#viene isolata la colonna area stem e vengono selezionate soltanto quelle con 'si'
df_cod['Area STEM'] = df_cod['Area STEM'].astype(str).str.strip().str.lower()

#prendiamo solo la colonna con il codice identificativo internazionale (senza doppioni)
stem_prefixes = df_cod[df_cod['Area STEM'].isin(['sì', 'si'])]['ISCED_F_1dgt'].astype(str).str.lstrip('0').unique()

#il codice dei corsi di laurea viene trasformato da un codice a due cifre in un codice a due cifre
df_imm['COD_FoET2013_clean'] = df_imm['COD_FoET2013'].astype(str).str.lstrip('0')
#qui creiamo una nuova colonna grazie al quale verifichiamo se uno studente appartiene ad una classe di laurea stem
df_imm['is_STEM'] = df_imm['COD_FoET2013_clean'].isin(stem_prefixes)


#applichiamo il filtro andando selezionare soltanto laurea e magistrale a ciclo unico
#non vengono selezionate lauree magistrali biennali, perché come specificato nella relazione, l'intento è stato quello di
#identificare e analizzare gli ingressi subito dopo la fine delle superiori
df_base = df_imm[
    (df_imm['ANNO'] == '2024/2025') & 
    (df_imm['CorsoTIPO'].isin(['Laurea', 'Laurea Magistrale Ciclo Unico']))
]

#TEST CHIQUADRO E Z-TEST

#creiamo un file chiamato df_nazionale in cui andiamo a prendere solo la colonna TTTTT
#il ministero inserisce questa colonna, per comodità, in cui si trova il totale nazionale
df_nazionale = df_base[df_base['AteneoCOD'] == 'TTTTT']

#prendiamo i dati e li divide in 4 catgegorie
# Uomini che fanno STEM, Uomini che non fanno STEM, Donne che fanno STEM e donne che non fanno STEM
#infine conta quanti immatricolati vi sono in ogni sezione
tabella_nat = df_nazionale.groupby(['Genere', 'is_STEM'])['IMM'].sum().unstack(fill_value=0)
tabella_nat.columns = ['Non_STEM', 'STEM']

#salva i vari dati in delle variabili 
f_stem = tabella_nat.loc['F', 'STEM']
f_non_stem = tabella_nat.loc['F', 'Non_STEM']
m_stem = tabella_nat.loc['M', 'STEM']
m_non_stem = tabella_nat.loc['M', 'Non_STEM']


# Chi-Quadro
#lo utilizziamo per comprendere se il genere e la scelta dell'università (in questo caso stem)
#sono due fenomeni collegati tra loro o magari sono completamente indipendenti

#Il test sostanzialmente è come se creasse una sorta di mondo ideale in cui prende il totale degli studenti e immagina
# come sarebbero distribuiti uomini e donne se scegliessero i corsi esattamente nello stesso modo, senza stereotipo di genere
# i dati risultanti prendono il nome di valori attesi, che infine vengono paragonati ai valori reali e ne viene misurata la distanza

#se i numeri reali sono molto diversi dai digitali il chi quadro tende ad essere alto.
#il p-value ci dice sostanzialmente se questa enorme differenza tra uomini e donne è soltanto casuale o meno

#prendo in dati isolati in precedenza e li metto in una matrice 2x2
obs = np.array([[f_stem, f_non_stem], [m_stem, m_non_stem]])
#ottengo il valore del chi quadro e il p-value
chi2_stat, p_val_chi2, _, _ = chi2_contingency(obs)

#2e per stampa in notazione scientifica
print(f"P-value Chi-Quadrato: {p_val_chi2:.2e}")



# Z-Test
# per individuare se effettivamente la % di donne stem è statisticamente differente da quella di donne che non sono nelle stem
#anche qui, se il p-value è basso vuol dire che la differenza tra le due percentuali è abbastanza grande e non è soltanto un caso.


#creo un array al cui interno inserisco soltanto i numeri relativi alle iscrizione di donne
count = np.array([f_stem, f_non_stem]) 

#vado a sommare le donne e i maschi iscritti a stem e non iscritti a stem
nobs = np.array([f_stem + m_stem, f_non_stem + m_non_stem])

#prendo il numero di donne e lo divido per il totale
#percentuale iscritti stem è donne, percentuali di iscritti non-stem è donna
#il test infine confronta le due percentuali e calcola il valore finale
z_stat, p_val_z = proportions_ztest(count, nobs)
print(f"P-value Z-Test: {p_val_z:.2e}")



#Piramide delle iscrizioni
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
# 4. GRAFICO 2: CLASSIFICA REGIONALE (Il Paradosso Meridionale)
# ==========================================
print("\nElaborazione Dati Regionali e Grafico a Barre...")

# Escludiamo il "TOTALE", le voci vuote e gli Atenei Telematici/Estero
df_regioni = df_base[
    (df_base['AteneoCOD'] != 'TTTTT') & 
    (df_base['AteneoREGIONE'].notna()) &
    (~df_base['AteneoREGIONE'].isin(['TELEMATICI', 'ESTERO']))
]

# Calcoliamo le percentuali
tab_reg = df_regioni.groupby(['AteneoREGIONE', 'Genere', 'is_STEM'])['IMM'].sum().reset_index()
stem_reg = tab_reg[tab_reg['is_STEM'] == True]
stem_pivot = stem_reg.pivot(index='AteneoREGIONE', columns='Genere', values='IMM').fillna(0)
stem_pivot['Totale_STEM'] = stem_pivot['F'] + stem_pivot['M']
stem_pivot['Percentuale_Donne_STEM'] = (stem_pivot['F'] / stem_pivot['Totale_STEM']) * 100
stem_pivot['Percentuale_Uomini_STEM'] = (stem_pivot['M'] / stem_pivot['Totale_STEM']) * 100

# Salviamo il file CSV
output_df = stem_pivot[['Totale_STEM', 'F', 'M', 'Percentuale_Donne_STEM', 'Percentuale_Uomini_STEM']].copy()
output_df = output_df.sort_values('Percentuale_Donne_STEM', ascending=False).round(2)
output_df.to_csv('percentuali_regionali_stem.csv', sep=';')
print("-> File 'percentuali_regionali_stem.csv' generato con successo!")

# --- CREAZIONE DEL GRAFICO A BARRE ---
# Ordiniamo in modo crescente per avere il valore più alto in cima al grafico
df_plot = output_df.sort_values('Percentuale_Donne_STEM', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

# Disegniamo le barre
barre = ax.barh(df_plot.index, df_plot['Percentuale_Donne_STEM'], color='#8e44ad', height=0.7)

# Calcoliamo e disegniamo la linea della Media Nazionale (38%)
media_nazionale = (f_stem / (f_stem + m_stem)) * 100
ax.axvline(media_nazionale, color='#e74c3c', linestyle='--', linewidth=2, label=f'Media Nazionale ({media_nazionale:.1f}%)')

# Aggiungiamo i numeretti esatti alla fine di ogni barra
for barra in barre:
    larghezza = barra.get_width()
    ax.text(larghezza + 0.5, barra.get_y() + barra.get_height()/2, f'{larghezza:.1f}%', 
            ha='left', va='center', fontsize=10)

ax.set_title('Percentuale di Donne Iscritte in STEM per Regione (2024/2025)', fontsize=14, pad=20)
ax.set_xlabel('% Donne su Totale Iscritti STEM')
ax.set_xlim(0, 55) # Diamo spazio per non far tagliare i numeri a destra
ax.legend(loc='lower right')
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('barre_regionali_stem.png', dpi=300)
print("-> Immagine salvata come 'barre_regionali_stem.png'")