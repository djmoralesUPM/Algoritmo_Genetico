import gymnasium as gym
from tqdm import tqdm
import numpy as np
import random
import json
import os

#cruce
#como acelerar env
#revisar si algoritmo genetico bien hecho

VALOR_INICIAL = 0.25 #las redes se cargan con este valor inicial

class Car:
    def __init__(self, weights=None):
        if weights is None:
            # Matriz de pesos capa 1: (filas: entradas, columnas: salidas)
            self.W1 = np.random.uniform(-VALOR_INICIAL, VALOR_INICIAL, (96*96, 10))
            self.b1 = np.random.uniform(-VALOR_INICIAL, VALOR_INICIAL, (1, 10))
            # Matriz de pesos capa 2: (filas: entradas, columnas: salidas)
            self.W2 = np.random.uniform(-VALOR_INICIAL, VALOR_INICIAL, (10, 3))
            self.b2 = np.random.uniform(-VALOR_INICIAL, VALOR_INICIAL, (1, 3))
        else:
            # Matriz de pesos capa 1: (filas: entradas, columnas: salidas)
            self.W1 = np.reshape(a=np.array(weights[:96*96*10]), newshape=(96*96, 10))
            #print(self.W1.shape)
            weights = weights[96*96*10:]
            self.b1 = np.reshape(a=np.array(weights[:10]), newshape=(1, 10))
            #print(self.b1.shape)
            weights = weights[10:]
            # Matriz de pesos capa 2: (filas: entradas, columnas: salidas)
            self.W2 = np.reshape(np.array(weights[:10*3]), newshape=(10, 3))
            weights = weights[30:]
            self.b2 = np.reshape(np.array(weights[:3]), newshape=(1, 3))

    def activation(self, X):
        # Función de activación: función sigmoidal
        return 1 / (1 + np.exp(-X))
    
    def d_activation(self, X):
        # Derivada de la función de activación
        return X * (1 - X)
        
    def infer(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        # Calculamos la salida de la capa 1
        self.S1 = self.activation(X @ self.W1 + self.b1)
        # Calculamos la salida de la capa 2
        self.S2 = self.activation(self.S1 @ self.W2 + self.b2)
        return self.S2


 # , render_mode='human'
env = gym.make("CarRacing-v2", domain_randomize=True, render_mode='human') #render_mode='human'
env.action_space.seed(43)
observation, info = env.reset(seed=43)

def fitness(chromosome):
    # Cargo la red
    car = Car(chromosome)
    reward_list=[]

    
    # La simulo
    observation, _ = env.reset(seed=42)
    n=5_000

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    rec_acc = 0
    for _ in tqdm(range(n)):
        #print(observation.shape, type(observation))
        observation_byn = rgb2gray(observation).flatten()
        #print(observation_byn.shape, type(observation_byn), observation_byn)
        action = car.infer(observation_byn).flatten()
        #print(action)
        #action=[-.2, 1.2, 1.6]
        action[0] = (action[0]-.5) # el primer valor va de -1 a 1
        #action[0] = (action[0]*2-1)/4
        action[1] = action[1] * 2
        action[2] = action[2]/4 # para que no frene tanto el mamon
        #action[2] = 0
        observation, reward, terminated, truncated, _ = env.step(action)

        #Como calcula el fitness
        rec_acc += reward #si es al cuadrado puede ayudar a la selección a seleccionar mejores, en vez de usar regla de torneo
        steps = 100
        if(len(reward_list)>=steps):
            reward_list.pop(0)

        reward_list.append(rec_acc)

        #compara los elementos de la lista
        if(len(reward_list)==steps):
            if(reward_list[0]>reward_list[steps-1]):
                return rec_acc
            
        #print(rec_acc, a, reward_list)

        # if rec_acc<0:
        #     return rec_acc
        #print(rec_acc, type(rec_acc))

        if terminated or truncated:
            observation, info = env.reset()
    
    # Devuelvo el fitness
    return rec_acc

#metes el numero de la poblacion inicial
#devuelve diccionario con n_poblacion registros
def creacion_poblacion_inicial(n_poblacion):
    diccionario = {}
    print('INVENTADA DE PESOS')
    for i in range(n_poblacion):
        pesos = np.random.uniform(low=-VALOR_INICIAL, high=VALOR_INICIAL, size=(96*96*10+10+3*10+3))
        diccionario[f'red{i}'] = {'pesos':pesos, 'acc':None}
    return diccionario

def load_generacion(gen:int):
    for i in range(1,20):
        cadena1 = "pesos_"
        cadena2 = f"red{i}"
        cadena_carpeta = cadena1+cadena2
        chromosome = []
        with open(f'/ficheros/generacion{gen}/{cadena_carpeta}.json') as json_file:
            data = json.load(json_file)
        
            chromosome = data[f'{cadena2}'][0]
            rec = fitness(chromosome)
            print(rec)

load_generacion(26)



