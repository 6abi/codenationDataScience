import pandas as pd
import numpy as np

data = pd.read_csv('../codenation/coestatistica-1/desafio1.csv')

data.head()

def answer(df):
    states = df['estado_residencia'].unique()
    submition = pd.read_json('../codenation/coestatistica-1/submission.json', dtype='float')
    for i in states:
        state = df[(df['estado_residencia'] == i) ]['pontuacao_credito']
        submition[i]['moda'] = state.mode()
        submition[i]['mediana'] = state.median()
        submition[i]['media'] = state.mean()
        submition[i]['desvio_padrao'] = state.std()
    convertionJson(submition)
    return submition

def convertionJson(submition):
    return submition.to_json('submission.json', double_precision=15)


answer(data)
print('submission.json')
