#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Para construirmos nosso deploy:
# O streamlit cuidará da parte visual do site. Só temos que passar o código que rodará nosso modelo de previão.

import pandas as pd
import streamlit as st
import joblib


x_numericos = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 'extra_people': 0,
               'minimum_nights': 0, 'ano': 0, 'mes': 0, 'n_amenities': 0, 'host_listings_count': 0}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_listas = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Outros', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'cancelation_policy': ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']
            }


# O dicionário abaixo foi criado para que possamos separar os valores que serão inseridos nas categorias de cada item (chaves 
# property_type, room_type, cancelation_policy). Teremos um dicionário para cada tipo de combinação, ou seja, tipo de
# propriedade p cada tipo de propriedade, tipo de quarto com cada tipo de quarto, política de cancelamento com cada tipo de
# politica de cancelamento.

dicionario = {}
for item in x_listas:
    for valor in x_listas[item]:
        dicionario[f'{item}_{valor}'] = 0
        
# Aqui, teremos que tratar os valores que serão números decimais, como: latitude, longitude e extra_people.
for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0) 
    else:
        valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor

    
for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor == "Sim":
        x_tf[item] = 1
    else:
        x_tf[item] = 0

botao = st.button('Prever Valor do Imóvel')

# o nosso modelo precisa de um Df para fazer a previsão.

if botao:
    dicionario.update(x_numericos)
    dicionario.update(x_tf)
    valores_x = pd.DataFrame(dicionario, index=[0])
    modelo = joblib.load('modelo.joblib')
    preco = modelo.predict(valores_x)
    st.write(preco[0])

