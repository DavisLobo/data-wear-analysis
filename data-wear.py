import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

### Wearables data analysis project - Classification
''' 
Main goal of the project is to gain proficiency with common Data Science libraries.
Project problem is of Classification type and is required data processing prior to analysis.
We'll analyse the data to classify exercises and determine their instensity as well as estimating the calories burnt
'''

## Processamento dos dados e organizacao
id_individuo = [1,2,3,4,5,6,7,8,9]

id_atividades = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

lista_dados = [
'/Wearables/PAMAP2_Dataset/Protocol/subject101.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject102.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject103.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject104.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject105.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject106.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject107.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject108.dat',
'/Wearables/PAMAP2_Dataset/Protocol/subject109.dat'
]

col_geral = ['timestamp', 'id_atividade', 'Frequência cardiaca (bpm)']

col_dados_mao = ['mTemperatura', 'mAcel 16g (ms) 1', 'mAcel 16g (ms) 2',
                'mAcel 16g (ms) 3', 'mAcel 6g (ms) 1', 'mAcel 6g (ms) 2', 'mAcel 6g (ms) 3',
                'mGyro (rad/s) 1', 'mGyro (rad/s) 2', 'mGyro (rad/s) 3', 
                'mMag (μT) 1', 'mMag (μT) 2', 'mMag (μT) 3', 'mOrientação 1', 
                'mOrientação 2', 'mOrientação 3','mOrientação 4']

col_dados_peito = ['pTemperatura', 'pAcel 16g (ms) 1', 'pAcel 16g (ms) 2',
                'pAcel 16g (ms) 3', 'pAcel 6g (ms) 1', 'pAcel 6g (ms) 2', 'pAcel 6g (ms) 3',
                'pGyro (rad/s) 1', 'pGyro (rad/s) 2', 'pGyro (rad/s) 3', 
                'pMag (μT) 1', 'pMag (μT) 2', 'pMag (μT) 3', 'pOrientação 1', 
                'pOrientação 2', 'pOrientação 3','pOrientação 4']

col_dados_calcanhar = ['cTemperatura', 'cAcel 16g (ms) 1', 'cAcel 16g (ms) 2',
                'cAcel 16g (ms) 3', 'cAcel 6g (ms) 1', 'cAcel 6g (ms) 2', 'cAcel 6g (ms) 3',
                'cGyro (rad/s) 1', 'cGyro (rad/s) 2', 'cGyro (rad/s) 3', 
                'cMag (μT) 1', 'cMag (μT) 2', 'cMag (μT) 3', 'cOrientação 1', 
                'cOrientação 2', 'cOrientação 3', 'cOrientação 4']

colunas = col_geral + col_dados_mao + col_dados_peito + col_dados_calcanhar

dados_completos= pd.DataFrame()

for arq in lista_dados:
    procDados= pd.read_table(filepath_or_buffer=arq, header=None, sep='\s+')
    procDados.columns= colunas
    dados_completos= pd.concat([dados_completos, procDados], ignore_index=True)

def LimpaDados(dados_completos):
        dados_completos = dados_completos.drop(['cOrientação 1','cOrientação 2', 'cOrientação 3', 'cOrientação 4',
                                                'mOrientação 1','mOrientação 2', 'mOrientação 3', 'mOrientação 4',
                                                'pOrientação 1','pOrientação 2', 'pOrientação 3', 'pOrientação 4'],
                                                    axis = 1)  
        dados_completos = dados_completos.drop(dados_completos[dados_completos.id_atividade == 0].index) 
        dados_completos = dados_completos.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        dados_completos = dados_completos.interpolate() 
        
        return dados_completos

dn = LimpaDados(dados_completos)
dn.reset_index(drop = True, inplace = True)
print(dn.head(10))

# Análise dos dados

treino_Dn= dn.sample(frac=0.8, random_state=1)
teste_Dn = dn.drop(treino_Dn.index)

# Gráfico Frequência Cardiaca

fig, ax = plt.subplots(figsize=(4,4))
plt.title("FC (BPM)")
ax = sns.boxplot(y=treino_Dn["Frequência cardiaca (bpm)"])
plt.show()

## Classificar caminhada/corrida/descanso

mapa_atividades = {1: 'descanso', 2: 'descanso', 3: 'descanso',
                   4: 'caminhada', 5: 'corrida'}

df_filtrado = dn[dn['id_atividade'].isin(mapa_atividades.keys())].copy()
df_filtrado['rotulo'] = df_filtrado['id_atividade'].map(mapa_atividades)

# Magnitude do acelerômetro — invariante à rotação do sensor

for prefixo in ['m', 'p', 'c']:
    df_filtrado[f'{prefixo}Magnitude'] = np.sqrt(
        df_filtrado[f'{prefixo}Acel 16g (ms) 1']**2 +
        df_filtrado[f'{prefixo}Acel 16g (ms) 2']**2 +
        df_filtrado[f'{prefixo}Acel 16g (ms) 3']**2
    )

caracteristicas = ['Frequência cardiaca (bpm)',
                   'mMagnitude', 'pMagnitude', 'cMagnitude']

X = df_filtrado[caracteristicas]
y = df_filtrado['rotulo']

X_treino = X.loc[treino_Dn.index.intersection(X.index)]
y_treino = y.loc[X_treino.index]
X_teste  = X.loc[teste_Dn.index.intersection(X.index)]
y_teste  = y.loc[X_teste.index]

classificador = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
classificador.fit(X_treino, y_treino)

print(classification_report(y_teste, classificador.predict(X_teste)))




## Detectar intensidade do exercício

def classificar_intensidade(bpm):
    if bpm < 100:
        return 'baixa'
    elif bpm < 140:
        return 'moderada'
    else:
        return 'alta'

dn['intensidade'] = dn['Frequência cardiaca (bpm)'].apply(classificar_intensidade)

fig, eixo = plt.subplots(figsize=(12, 5))
contagem_intensidade = (dn.groupby(['id_atividade', 'intensidade'])
                          .size()
                          .unstack(fill_value=0))
contagem_intensidade.plot(kind='bar', stacked=True, ax=eixo, colormap='RdYlGn_r')
eixo.set_xlabel('ID da Atividade')
eixo.set_ylabel('Número de amostras')
eixo.set_title('Distribuição de Intensidade por Atividade')
plt.tight_layout()
plt.show()


## Estimar gasto calórico

# Valores MET baseados na literatura

met_values = {
    1: 0.9,   # lying
    2: 1.0,   # sitting
    3: 1.2,   # standing
    4: 3.5,   # walking
    5: 8.0,   # running
    6: 6.0,   # cycling
    7: 4.5,   # Nordic walking
    12: 4.0,  # ascending stairs
    13: 3.0,  # descending stairs
    16: 3.5,  # vacuum cleaning
    17: 2.3,  # ironing
    20: 7.0,  # playing soccer
    24: 10.0, # rope jumping
}

PESO_CORPORAL = 70
SAMPLE_RATE_HZ = 100       
SEC_POR_SAMPLE = 1 / SAMPLE_RATE_HZ

def estimate_calorias(row, peso_kg= PESO_CORPORAL):
    met = met_values.get(int(row['id_atividade']), 1.0)
    horas = SEC_POR_SAMPLE / 3600
    return met * peso_kg * horas     # kcal = MET × peso(kg) × tempo(horas)

dn['a'] = dn.apply(estimate_calorias, axis=1)

# Calorias por atividade

dn['id_atividade'] = dn['id_atividade'].fillna(0).astype(int)
dn['met'] = dn['id_atividade'].map(valores_met).fillna(1.0)
dn['calorias_kcal'] = dn['met'] * PESO_CORPORAL * (SEC_POR_SAMPLE / 3600)
dn.drop(columns=['met'], inplace=True)

resumo_calorias = dn.groupby('id_atividade')['calorias_kcal'].sum().reset_index()
resumo_calorias.columns = ['ID Atividade', 'Total kcal']
resumo_calorias['Atividade'] = resumo_calorias['ID Atividade'].map(id_atividades)
print(resumo_calorias.sort_values('Total kcal', ascending=False))






