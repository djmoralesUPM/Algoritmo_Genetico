import gymnasium as gym
from tqdm import tqdm
import numpy as np
import random
import json
import os

# Directorio de la carpeta donde se guardará el archivo
directorio = "/ficheros"

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
env = gym.make("CarRacing-v2", domain_randomize=True) #render_mode='human'
env.action_space.seed(43)
observation, info = env.reset(seed=43)

#funcion de calidad usada para calcular rewards
def fitness(chromosome):
    # Cargo la red
    car = Car(chromosome)
    reward_list=[]

    
    # La simulo
    observation, _ = env.reset(seed=42)
    n=10_000

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    rec_acc = 0
    for a in tqdm(range(n)):
        #print(observation.shape, type(observation))
        observation_byn = rgb2gray(observation).flatten()
        #print(observation_byn.shape, type(observation_byn), observation_byn)
        action = car.infer(observation_byn).flatten()
        #action=[-.2, 1.2, 1.6]
        #action[0] = (action[0]*2 -1) # el primer valor va de -1 a 1
        #action[0] = (action[0]*2-1)/4
        #action[1] = action[1] * 2
        #action[2] = action[2]/2 # para que no frene tanto el mamon
        #action[2] = 0
        observation, reward, terminated, truncated, _ = env.step(action)

        #Como calcula el fitness
        rec_acc += reward #si es al cuadrado puede ayudar a la selección a seleccionar mejores, en vez de usar regla de torneo
        steps = 35
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
        pesos = np.random.uniform(low=-1, high=1, size=(96*96*10+10+3*10+3))
        diccionario[f'red{i}'] = {'pesos':pesos, 'acc':None}
    return diccionario

#metes la poblacion (inicial o n-esima)
#devuelve chosen1 y chosen2
def seleccion_poblacion(poblacion, claves_aleatorias):

    def mejor_par(diccionarioo):
        diccionario = diccionarioo
        mejor_red = max(diccionario.items(), key=lambda x: x[1]['acc'])[0]
        child1 = diccionario[f'{mejor_red}']['pesos']
        del diccionario[mejor_red]
        mejor_red2 = max(diccionario.items(), key=lambda x: x[1]['acc'])[0]
        child2 = diccionario[f'{mejor_red2}']['pesos']
        del diccionario[mejor_red2]

        return child1, child2

    #coger 5 randoms del diccionario
    redes_aleatorias = random.sample(list(poblacion.keys()), k=claves_aleatorias)  # Seleccionamos 3 claves aleatorias del diccionario
    seleccionados = {}
    for red in redes_aleatorias:
         if poblacion[red]['acc'] is None:
            print(red)
            poblacion[red]['acc'] = fitness(poblacion[red]['pesos'])

            #PROBANDO CON ACC RANDOMS
            #poblacion[red]['acc'] = random.random()*100
         
         seleccionados[f'{red}'] = poblacion[red]

    #torneo
    try:
        chosen1, chosen2 = mejor_par(seleccionados)
    except TypeError:
        print("El objeto devuelto es NoneType, no se puede acceder a los atributos")
    

    return chosen1, chosen2
    

    #si no esta calculado, calcular fitness
    #si estan todos calculados, regla de torneo
    #devolver pareja de la poblacion elegidos por regla de torneo
    #return y cruzar
    
#metes la pareja elegida
#devuelve child1 y child2 (pueden ser los mismos de antes si no se cruzan)
def cruzar(chosen1, chosen2):
    #probabilidad entre 0.5 y 1
    prob_cruce = 0.8
    ran = random.random()

    #ESTA AQUI EL ERROR, PONE EL PIVOTE EN MITAD DEL CROMOSOMA, NO EN MITAD DE LA CAPA DE ENTRADA
    def one_pivot_recombination(parent1, parent2):
        parent1 = parent1
        parent2 = parent2
        pivot = len(parent1)+1 // 2
        child1 = np.concatenate((parent2[:pivot],parent1[pivot+1:]))
        child2 = np.concatenate((parent1[:pivot],parent2[pivot+1:]))
        return child1, child2
    
    if prob_cruce>ran: #si ran [0.8, 1]
        child1, child2 = one_pivot_recombination(chosen1, chosen2)
        print('CRUCE')
    else: 
        child1, child2 = chosen1, chosen2
        print('NO CRUCE')
        
    return child1, child2

#metes la pareja cruzada (o no)
#devuelve child1 y child2 mutados (o no)
def mutacion(child1, child2):
    #prob_mutacion = 2/(96*96) 
    prob_mutacion = .1/(96*96) 

    child1 = child1
    child2 = child2
    #recorrer valores de los vectores y si random<probabilidad_mutacion muta el gen
    for i in range(len(child1)):
        if random.random()<prob_mutacion:
            print(f'MUTA HIJO1 EN LA POSICION {i}')
            child1[i] = random.uniform(-VALOR_INICIAL, VALOR_INICIAL)
    for i in range(len(child2)):
        if random.random()<prob_mutacion:
            print(f'MUTA HIJO2 EN LA POSICION {i}')
            child2[i] = random.uniform(-VALOR_INICIAL, VALOR_INICIAL)
    return child1, child2

NUMERO_DE_ELECCIONES_ALEATORIAS = 2 #para el metodo crear_poblacion_n (conviene valores como 1/10 o 2/10 de la poblacion (10 o 20 de 100) para que salgan los mejores)

#metes la población inicial y el numero de individuos de las nuevas poblaciones
#devuelve generacion siguiente modificada

def crear_poblacion_n(diccionario, n_poblacion):
    poblacion_vieja=diccionario
    # def crear_pareja(poblacion):
    #     chosen1, chosen2 = seleccion_poblacion(poblacion, NUMERO_DE_ELECCIONES_ALEATORIAS)
    #     child1, child2 = cruzar(chosen1, chosen2)
    #     child1M, child2M = mutacion(child1, child2)
    #     return child1M, child2M
    
    n=0
    poblacion_nueva=[]
    while (n<n_poblacion):
        #meter pareja en diccionario
        child1, child2 = seleccion_poblacion(poblacion_vieja, NUMERO_DE_ELECCIONES_ALEATORIAS)
        poblacion_nueva.append(child1)
        poblacion_nueva.append(child2)
        n=n+2

    n=0
    poblacion_nueva_cruzada=[]
    while (n<n_poblacion):
        child1, child2 = cruzar(poblacion_nueva[n], poblacion_nueva[n+1])
        poblacion_nueva_cruzada.append(child1)
        poblacion_nueva_cruzada.append(child2)
        n=n+2
        print(f'Mete las posiciones {n}/{n_poblacion} y {n-1}/{n_poblacion} de ')
        

    n=0
    poblacion_nueva_mutada=[]
    while (n<n_poblacion):
        child1M, child2M = mutacion(poblacion_nueva_cruzada[n], poblacion_nueva_cruzada[n+1])
        poblacion_nueva_mutada.append(child1M)
        poblacion_nueva_mutada.append(child2M)
        n=n+2

    diccionario = {}
    for i, pesos in enumerate(poblacion_nueva_mutada):
        red = f"red{i+1}"
        diccionario[red] = {"pesos": pesos, "acc": None}
        
    return diccionario

#metes el tamaño de la poblacion inicial, el tamaño de las poblaciones siguientes y el numero de poblaciones siguientes
def algoritmo_gen(n_inicial, n_siguientes, n_poblaciones):
    print('GENERACION 1 /// '*4)
    poblacion = creacion_poblacion_inicial(n_inicial)
    for i in range(n_poblaciones-1):
        print(f'GENERACION {i+2} /// '*4)
        poblacion = crear_poblacion_n(poblacion, n_siguientes)
        # Nombre del archivo a guardar

def guardar(diccionario):
    for clave in diccionario:
        nombre_archivo = f"pesos_{clave}.json"

        # Combinar el directorio y el nombre del archivo
        ruta_completa = os.path.join(directorio, nombre_archivo)

        # Guardar el diccionario en el archivo
        with open(ruta_completa, "w") as archivo:
            dicc = {
                k: (v['acc'], v['pesos'].tolist())
                for k, v in diccionario.items()
                }
            archivo.write(json.dumps(dicc))


algoritmo_gen(n_inicial=30, n_siguientes=30, n_poblaciones=10)

env.close()