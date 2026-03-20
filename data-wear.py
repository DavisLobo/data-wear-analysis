import pandas as pd
import numpy as np
import sklearn as sklearn
import seaborn as sns
from matplotlib import pyplot as plt

## Wearables data analysis project - Classification
''' 
Main goal of the project is to gain proficiency with common Data Science libraries.
Project problem is of Classification type and is required data processing prior to analysis.
We'll analyse the data to classify exercises and determine their instensity as well as estimating the calories burnt
'''

# Processamento dos dados e organizacao
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
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject101.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject102.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject103.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject104.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject105.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject106.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject107.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject108.dat',
'/home/davi/Documents/Projetos/Wearables/PAMAP2_Dataset/Protocol/subject109.dat'
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
    procDados['id_atividade']= int(arq[-5])
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

#Gráfico Frequência Cardiaca

fig, ax = plt.subplots(figsize=(4,4))
plt.title("Heart Rate")
ax = sns.boxplot(y=treino_Dn["Frequência cardiaca (bpm)"])
plt.show()

# Classificar caminhada/corrida/descanso






# Detectar intensidade do exercício






# Estimar gasto calórico








