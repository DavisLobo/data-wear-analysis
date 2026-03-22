# Análise de Dados de Wearables — PAMAP2

Projeto pessoal de Data Science com foco na exploração de um dataset massivo (3,8 milhões de linhas) e extrair informações relevantes com o auxílio de modelos de Machine Learning e EDA (Análise Exploratória de Dados). 

---

## Objetivo

Aplicar técnicas de pré-processamento, análise exploratória e aprendizado de máquina sobre dados reais de wearables, cobrindo um pipeline completo desde a leitura dos arquivos brutos até a geração de insights sobre atividade física e gasto energético para contribuir para consistência dos dados de aparelhos wearables muito utilizados por entusiastas da saúde e fitness.

---

## Dataset

**PAMAP2 Physical Activity Monitoring**
É um Dataset que contém dados de 9 indivíduos em 18 atividades diferentes, o que possibilita extrair os dados para estimar intensidade de um exercício, gasto calórico e outras métricas de performance com maior grau de precisão. Hoje nos relógios e dispositivos Wearables existe um cálculo feito para estimar as calorias de uma atividade, no entanto, é de conhecimento que grande parte desses aplicativos e dispositivos variam muito em seus dados e no cálculo feito. Um Dataset como esse possibilita maior equivalência desses dados e maior consistência principalmente o que é interessante para o público geral interessado em saúde e fitness bem como para atletas e controle de carga pela identificação da intensidade esperada de um exercício.

Disponível publicamente no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).

| Característica | Detalhe |
|---|---|
| Participantes | 9 indivíduos |
| Sensores | 3 IMUs (mão, peito, calcanhar) |
| Taxa de amostragem | 100 Hz |
| Atividades monitoradas | 18 (caminhada, corrida, ciclismo, etc.) |
| Sinais capturados | Acelerômetro, giroscópio, magnetômetro, temperatura, FC |

---

## Pipeline

```
Dados brutos (.dat)
      │
      ▼
Limpeza e pré-processamento
  • Remoção de colunas de orientação
  • Descarte de amostras de transição (id = 0)
  • Interpolação linear de valores ausentes
      │
      ▼
Análise Exploratória (EDA)
  • Distribuição da frequência cardíaca
      │
      ├──► Classificação de atividades
      │      Random Forest (caminhada / corrida / descanso)
      │
      ├──► Detecção de intensidade
      │      Zonas de FC (baixa / moderada / alta)
      │
      └──► Estimativa de gasto calórico
             Método MET
```

---

## Tarefas implementadas

### 1. Classificação de atividades
Modelo Random Forest treinado sobre frequência cardíaca e magnitude euclidiana do acelerômetro dos três sensores. A magnitude é preferida aos eixos individuais por ser invariante à orientação do dispositivo no corpo.
```
                        Classification Report

              precision    recall  f1-score   support

   caminhada       0.97      0.98      0.98     47936
     corrida       0.98      0.94      0.96     19744
    descanso       0.99      0.99      0.99    113233

    accuracy                           0.98    180913
   macro avg       0.98      0.97      0.98    180913
weighted avg       0.98      0.98      0.98    180913

```

### 2. Detecção de intensidade
Classificação baseada em limiares de frequência cardíaca, seguindo zonas padrão da fisiologia do exercício:

| Zona | FC |
|---|---|
| Baixa | < 100 bpm |
| Moderada | 100 – 139 bpm |
| Alta | ≥ 140 bpm |


<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/545b1145-7b48-48b3-98d1-560d4454137c" />


### 3. Estimativa de gasto calórico
Aplicação da fórmula MET (*Metabolic Equivalent of Task*), padrão em estudos de wearables na ausência de medição direta de VO₂:

$$\text{kcal} = \text{MET} \times \text{peso}_{kg} \times \text{tempo}_{horas}$$


```

            Calorias gastas por hora

    ID Atividade  Total kcal          Atividade
5              6  192.033333            cycling
6              7  164.593625     Nordic_walking
3              4  162.490125            walking
4              5  152.754000            running
9             16  119.337458    vacuum_cleaning
10            17  106.747472            ironing
11            24   95.977778       rope_jumping
7             12   91.168000   ascending_stairs
8             13   61.217333  descending_stairs
2              3   44.317233           standing
1              2   36.008778            sitting
0              1   33.691525              lying

```

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e limpeza de dados |
| `numpy` | Operações vetorizadas (magnitude do acelerômetro) |
| `scikit-learn` | Modelo Random Forest, métricas de avaliação |
| `seaborn` / `matplotlib` | Visualizações |

---

## Como executar

# Abra o notebook para execução interativa

jupyter notebook wearables_dados.ipynb

---

## Limitações e melhorias futuras

- A divisão treino/teste é aleatória, o que pode vazar dados do mesmo indivíduo entre os conjuntos.
- O modelo atual classifica apenas três classes. Futuramente poderia expandir para todas as 18 atividades.

---

## Referências

- Reiss, A. & Stricker, D. (2012). *Introducing a New Benchmarked Dataset for Activity Monitoring*. ISWC.
- Ainsworth, B. et al. (2011). *2011 Compendium of Physical Activities*. Medicine & Science in Sports & Exercise.
- A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. The 5th Workshop on Affect and Behaviour Related Assistance (ABRA), 2012.
