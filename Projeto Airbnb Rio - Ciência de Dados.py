#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Bibliotecas
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Consolidar Base de Dados

meses = {'jan': 1, 'fev':2, 'mar':3, 'abr': 4, 'mai':5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
caminho_bases = pathlib.Path('dataset')
base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
    
display(base_airbnb)

# Agora vamos começar os tratamentos

base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')
print(base_airbnb['experiences_offered'].value_counts())
print((base_airbnb['host_listings_count'] == base_airbnb['host_total_listings_count']).value_counts())
print(base_airbnb['square_feet'].isnull().sum())

# Após o tratamento nas colunas, restaram 34 colunas que julgamos relevantes e iremos agora adicionar essas
# 34 colunas ao nosso Df.

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']
base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
display(base_airbnb)

# Tratar Valores Faltando
# Aqui nós exluímos as colunas que possuem valores NaN superior a 300.000 e as linhas com valores NaN.
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum()) # ->verificou quantidade de linhas com NaN.
base_airbnb = base_airbnb.dropna() # -> excluindo linhas com NaN
print(base_airbnb.shape)
print(base_airbnb.isnull().sum())

# Verificar Tipos de Dados em cada coluna.
# O modelo de previsão trata informações com números de uma forma e informações com texto de outra.

print(base_airbnb.dtypes) 
print('-'*60) 
print(base_airbnb.iloc[0]) 

# Um texto é recochecido como um objeto.
# Com esses dois prints vc vai comparando a linha 1 com linha 1, linha 2 com linha 2 de cada tabela, para ver se a atribuição
# do dtype está de acordo com o tipo de dado.

#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '') # -> retirou $ do preço
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '') # -> retirou a vírgula do preço
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False) # -> transformando p float. np de numpy

# extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)
#verificando os tipos
print(base_airbnb.dtypes)


## Análise Exploratória e Tratar Outliers 

# - Ver correlação
# Essa análise é para vermos se existe colunas com correlação mto forte, próxima a 1. Como não existe colunas com correlação
# mto forte, não excluiremos nenhuma.

plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')

# -  Excluir outliers
# Vamos definir algumas funções para ajudar na análise de outliers das colunas.

def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df,  linhas_removidas

def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1) 
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.histplot(coluna, element='bars')
    
def grafico_barra(coluna):  
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))

    
## a. Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário).
# Esses são os valores numéricos contínuos.  

# price

diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])
print(base_airbnb.shape)

# Exclusão de linhas com valores acima do limite superior (outliers).
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{} linhas removidas'.format(linhas_removidas))

# coluna extra_people

diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])

# Excluindo os outliers da coluna extra-people.
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print('{} linhas removidas'.format(linhas_removidas))


## b. Vamos agora analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)

# host_listings_count (qntos imovéis a pessoa tem no Airbnb)
# A partir dessa coluna host_listings_count, foi criada e utilizada a função gráfico_barra, para que possamos
# ao plotar o gráfico termos uma visualização mais precisa da quantidade de pessoas por tipo de coluna avaliada.

diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{} linhas removidas'.format(linhas_removidas))

# Accommodates

diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{} linhas removidas'.format(linhas_removidas))

# Bathrooms

# No caso da coluna dos banheiros, o gráfico pela função grafico_barra, retornava valores sem sentido.
# Resolvi traçar o gráfico utilizando o sns.barplot.
diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{} linhas removidas'.format(linhas_removidas))

# Bedrooms

diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print('{} linhas removidas'.format(linhas_removidas))

# Beds

diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{} linhas removidas'.format(linhas_removidas))


# Guests_inclued

diagrama_caixa(base_airbnb['guests_included'])
grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))

# Vou remover essa feature da análise. Parece que os usuários do airbnb usam muito o valor padrão do airbnb como 1 guest
# included. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço,
# por isso, me parece melhor excluir a coluna da análise.
base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape

# minimum_nights

diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{} linhas removidas'.format(linhas_removidas))

# maximun_nighjts

diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])

# Essa coluna foi excluída pelas informações apresentadas nos gráficos não estarem fazendo sentido, como hospedagem de 1000,
# 2000 noites.
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape

# number_of_reviews

diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])

# Vou retirar a coluna de reviews, para não prejudicar as pessoas que não tem nenhum e quiserem saber o preco a ser cobrado.
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape

## c. Por fim, vou avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.

# property_type
# Todos os tipos de propriedades abaixo de 2000, colocaremos em Outros.

print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# room_type

print(base_airbnb['room_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# bed_type

print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de bed_type
tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# cancellation_policy
# Aqui agrupamos as linhas : strict, super_strict_60, super_strict_30

print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de cancellation_pollicy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# - amenities 
# Como temos uma diversidade muito grande de amenities e, às vezes, as mesmas amenities podem ser escritas de forma diferente,
# vamos avaliar a quantidade de amenities como o parâmetro para o nosso modelo.

print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)

# Vou excluir a coluna amenities e criarmos a coluna n_amenities.
print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape

# A coluna n_amenities que criamos é uma coluna numérica, logo temos que tratá-la como tratamos as colunas numéricas anteriomente,
# removendo os outliers.

diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))

## Visualização de Mapa das Propriedades
# Vou utilizar a biblioteca plotly
# O sample(n=) é uma amostra da sua base de dados que vc quer que ele trabalhe em cima. Pode ser mais ou menos.
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()


## Encoding

colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0
    
colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)
display(base_airbnb_cod.head())


## Modelo de Previsão
# Aqui iremos comparar as previsões segundo os dois métodos.
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)


## Análise do Melhor Modelo

# Separar os dados em treino e teste + Treino do Modelo
# TESTANDO OS TRÊS MODELOS!

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))
    
# Modelo escolhido como melhor: ExtraTressRegressor


## Ajustes e Melhorias no Melhor Modelo

# modelo_et.feature_importances é uma propriedade que ele tem para vc saber a relevância que ele deu p cada feature.
print(modelo_et.feature_importances_) 
print(X_train.columns)


# Aqui, para melhor visualização das relevâncias de cada features, iremos passar as infos para um Df e depois plotar num gráfico.
# Na criação do Df passamos apenas os dados (modelo_et.feature_importances_) e o índice (X_train.columns).
# Na segunda linha importancia, é a ordenação das relevância do menor para maior.

importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)    


## Ajustes Finais no Modelo

# is_business_travel ready não parece ter muito impacto no nosso modelo. Por isso, para chegar em um modelo mais simples,
# vou excluir essa feature e testar o modelo sem ela.

# Aqui percebemos que melhorou um pouco nosso modelo vencedor (modelo_et) com relação ao primeiro resultado qndo
# testamos os três modelos.

base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))

## Abaixo testei o modelo retirando as colunas que tem  bed type no nome.

# Testamos e tivemos os msm valores de resultados com relação ao primeiro teste com os três modelos, porém, deixamos nosso
# modelo vencedor(modelo_et) mais simples com menos features. Isso é bom tendo em vista que retornou o msm resultado
# praticamente.

base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


## Deploy do Projeto

# Aqui pegamos nossa base de dados tratada e salvamos num arquivo csv.

X['price'] = y
X.to_csv('dados.csv')

# Aqui iremos transformar nosso modelo em um arquivo para ser usado.
# O joblib é biblioteca para fazer o deploy, transforma em um arquivo python.
import joblib
joblib.dump(modelo_et, 'modelo.joblib')


# In[ ]:





# In[ ]:




