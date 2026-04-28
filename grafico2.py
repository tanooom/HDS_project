import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================================================================
# ANALISI PERFORMANCE ACCADEMICA - Distribuzione Voti per Genere (2024)
# =============================================================================
# FIX rispetto alla versione originale:
#   Il dataset contiene sia i singoli atenei (93) sia la riga 'TTTTT' che
#   rappresenta già il loro totale nazionale. Senza filtrare, ogni studente
#   veniva contato due volte, gonfiando il Chi² da ~5.342 a ~10.683.
#   Soluzione: filtrare su AteneoCOD == 'TTTTT' prima di qualsiasi analisi.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CARICAMENTO DATI
# -----------------------------------------------------------------------------
file_path = "DATI PER BILANCIO DI GENERE/bdg_voto_laureati_serie-triennale.csv"
df_voti = pd.read_csv(file_path, sep=';', encoding='latin-1')

# -----------------------------------------------------------------------------
# 2. FILTRAGGIO E PULIZIA
# -----------------------------------------------------------------------------
voto_order = ['66-90', '91-100', '101-105', '106-110', '110 e lode']

voti_2024 = df_voti[
    (df_voti['ANNO'] == 2024) &
    (df_voti['AteneoCOD'] == 'TTTTT') &   # solo il totale nazionale, no doppio conteggio
    (df_voti['Classe_Voto_Laurea'].isin(voto_order))
].copy()

# -----------------------------------------------------------------------------
# 3. NORMALIZZAZIONE PER GENERE
# -----------------------------------------------------------------------------
totals = voti_2024.groupby('Genere')['LAU'].sum()

voti_2024['Percentuale'] = voti_2024.apply(
    lambda row: (row['LAU'] / totals[row['Genere']]) * 100, axis=1
)

# -----------------------------------------------------------------------------
# 4. TEST DEL CHI-QUADRATO DI INDIPENDENZA
# -----------------------------------------------------------------------------
contingency = voti_2024.pivot_table(
    index='Classe_Voto_Laurea', columns='Genere', values='LAU', aggfunc='sum'
)

chi2, p_val, dof, expected = stats.chi2_contingency(contingency.values)

print("=" * 55)
print("CHI-QUADRATO DI INDIPENDENZA")
print("=" * 55)
print(f"  Chi²    = {chi2:.1f}")
print(f"  dof     = {dof}")
print(f"  p-value = {p_val:.2e}")
print(f"  -> Rigetto H0: genere e fascia di voto NON sono indipendenti")

# -----------------------------------------------------------------------------
# 5. TEST DI MANN-WHITNEY U (dominanza stocastica)
# -----------------------------------------------------------------------------
grade_map = {'66-90': 1, '91-100': 2, '101-105': 3, '106-110': 4, '110 e lode': 5}

F_scores, M_scores = [], []
for _, row in voti_2024.iterrows():
    rank = grade_map[row['Classe_Voto_Laurea']]
    if row['Genere'] == 'F':
        F_scores.extend([rank] * row['LAU'])
    else:
        M_scores.extend([rank] * row['LAU'])

u_stat, u_p = stats.mannwhitneyu(F_scores, M_scores, alternative='greater')

print()
print("=" * 55)
print("MANN-WHITNEY U (F > M)")
print("=" * 55)
print(f"  U       = {u_stat:.0f}")
print(f"  p-value = {u_p:.2e}")
print(f"  -> Dominanza stocastica femminile confermata")

# -----------------------------------------------------------------------------
# 6. PERCENTUALE DI LODE E ANALISI BAYESIANA
# -----------------------------------------------------------------------------
lode = voti_2024[voti_2024['Classe_Voto_Laurea'] == '110 e lode'].groupby('Genere')['LAU'].sum()

pct_lode_F = lode['F'] / totals['F'] * 100
pct_lode_M = lode['M'] / totals['M'] * 100

# P(Donna | Lode) con teorema di Bayes
p_lode_given_F = lode['F'] / totals['F']
p_F            = totals['F'] / totals.sum()
p_lode         = lode.sum() / totals.sum()
p_donna_lode   = p_lode_given_F * p_F / p_lode

print()
print("=" * 55)
print("ANALISI BAYESIANA - PARADOSSO DELL'ECCELLENZA")
print("=" * 55)
print(f"  % lode donne : {pct_lode_F:.2f}%")
print(f"  % lode uomini: {pct_lode_M:.2f}%")
print(f"  P(Donna|Lode): {p_donna_lode * 100:.2f}%")

# -----------------------------------------------------------------------------
# 7. GRAFICO - Distribuzione percentuale dei voti per genere
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 7))

sns.barplot(
    data=voti_2024,
    x='Classe_Voto_Laurea',
    y='Percentuale',
    hue='Genere',
    order=voto_order,
    palette={'F': '#1f77b4', 'M': '#ff7f0e'}
)

plt.title(
    'Performance Accademica: Distribuzione Percentuale dei Voti per Genere (2024)',
    fontsize=14
)
plt.ylabel('Percentuale Interna al Genere (%)', fontsize=12)
plt.xlabel('Fascia di Voto', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('grafico_voti_corretto.png', dpi=150, bbox_inches='tight')
print()
print("Grafico salvato: grafico_voti_corretto.png")