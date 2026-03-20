# Análise de Dados de Wearables — PAMAP2

Projeto acadêmico de introdução a Data Science com foco em classificação de atividades físicas a partir de dados de sensores (IMU). Desenvolvido como exercício prático com as principais bibliotecas de Python para ciência de dados.

---

## Objetivo

Aplicar técnicas de pré-processamento, análise exploratória e aprendizado de máquina sobre dados reais de wearables, cobrindo um pipeline completo desde a leitura dos arquivos brutos até a geração de insights sobre atividade física e gasto energético.

---

## Dataset

**PAMAP2 Physical Activity Monitoring**
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

### 2. Detecção de intensidade
Classificação baseada em limiares de frequência cardíaca, seguindo zonas padrão da fisiologia do exercício:

| Zona | FC |
|---|---|
| Baixa | < 100 bpm |
| Moderada | 100 – 139 bpm |
| Alta | ≥ 140 bpm |

### 3. Estimativa de gasto calórico
Aplicação da fórmula MET (*Metabolic Equivalent of Task*), padrão em estudos de wearables na ausência de medição direta de VO₂:

$$\text{kcal} = \text{MET} \times \text{peso}_{kg} \times \text{tempo}_{horas}$$

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
