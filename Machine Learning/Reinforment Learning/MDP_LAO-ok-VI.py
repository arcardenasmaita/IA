#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Autor: ARCM 
@Tipo: Proceso de Decision Markoviano MDP
@Definicion:
 * Proceso de Decision Markoviano *
Algoritmo de Iteracion Valor (Value Iteration)
"""

#------------------------------------------
# Librerias
#------------------------------------------
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import csv
import sys
import os
import matplotlib
import seaborn as sb
#------------------------------------------
# Variáveis globais
#------------------------------------------
A = 4           # nro de acoes posiveis        
S = 0           # nro total de estados
So = 0          # estado inicial
G = 0           # estado objetivo
X = 0           # número de colunas da matriz do mundo
Y = 0           # número de filas da matriz do mundo
Enviroment = '' # ambiente para rodar os experimentos
T_norte = pd.DataFrame() # Probabiliade de transicao para a acao Norte
T_sul = pd.DataFrame()   # Probabiliade de transicao para a acao Sur
T_leste = pd.DataFrame() # Probabiliade de transicao para a acao Leste 
T_oeste = pd.DataFrame() # Probabiliade de transicao para a acao Oeste
C_matrix = pd.DataFrame()# matriz de custo        
G_So_I = set()           # conjunto de todos os nodos internos do hipergrafo G_So
G_So_F = set()           # conjunto de todos os nodos fronteira do hipergrafo G_So
G_So_v_I = set()         # conjunto de todos os nodos internos do hipergrafo G_So com a função V(s)
G_So_v_F = set()         # conjunto de todos os nodos fronteira do hipergrafo G_So com a função V(s)
V = pd.DataFrame()       # diccionario de valores para cada tupla (s_n, Vs_n)
Epsilon = 0.000000001
Ganma = 1
Lao_t = 0
Vi_t = 0
Vi_converg = pd.DataFrame(columns=['it','error'])
Vi_calc = 0
#------------------------------------------
# Definição de classes
#------------------------------------------
class MDP(object):
    def __init__(self, So, S, G):        
        self.G = G
        self.S = S
        self.So = So
        self.i = 0
        self.T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
        self.T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
        self.T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
        self.T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
        self.T_norte =np.array(T_norte)
        self.T_sul =np.array(T_sul)
        self.T_leste =np.array(T_leste)
        self.T_oeste =np.array(T_oeste)
        self.C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
        self.C_matrix = np.array(self.C_matrix)*(-1)
        # self.T_norte.columns=('s_in', 's_out', 'prob')
        # self.T_sul.columns=('s_in', 's_out', 'prob')
        # self.T_leste.columns=('s_in', 's_out', 'prob')
        # self.T_oeste.columns=('s_in', 's_out', 'prob')   
    def startState(self):        
        return So
    def isEnd(self, state):        
        return state == self.G-1
    def states(self):
        return range(0, self.S)
    def actions(self):
        return range(0, 4) # 0:acima, 1:abaixo, 2:direita, 3:izquerda
    def R(self, state, accion):               
        #return float(self.C_matrix[state])
        return 0. if self.isEnd(state) else -1.
    def succesorProbReward(self, state, action):        
        # retorna uma lista da tripla (novoEstado, probabilidade, recompensa)
        # estado = s, accion = a, nuevoEstado = s'
        # probabilidad = T(s, a, s'), recompensa = Reward(s, a, s')        
        result = [] 
        #if self.isEnd(state):            
        #    result.append((state, state, 0.))
        if action == 0:
            for item in self.T_norte:
                if item[0] == state:
                    result.append((item[1], item[2], -1))
        if action == 1:
            for item in self.T_sul:
                if item[0] == state:
                    result.append((item[1], item[2], -1))
        if action == 2:
            for item in self.T_leste:
                if item[0] == state:
                    result.append((item[1], item[2], -1))                
        if action == 3:
            for item in self.T_oeste:
                if item[0] == state:
                    result.append((item[1], item[2], -1))
        #print(result)
        return result
    
    
    def T(self, s_in, s_out, action):       
        # retorna a probabilidade de ir do state a stateNext
        # estado = s, accion = a, nuevoEstado = s'
        # probabilidad = T(s, a, s'), recompensa = Reward(s, a, s')  
        prob = 0.
        if action == 0:
            for item in self.T_norte:
                if item[0] == s_in and item[1]== s_out:
                    prob = item[2]
                    return prob
        elif action == 1:
            for item in self.T_sul:
                if item[0] == s_in and item[1]== s_out:
                    prob = item[2]
                    return prob
        elif action == 2:
            for item in self.T_leste:
                if item[0] == s_in and item[1]== s_out:
                    prob = item[2]              
                    return prob
        elif action == 3:
            for item in self.T_oeste:
                if item[0] == s_in and item[1]== s_out:
                    prob = item[2]
                    return prob
        return prob
        # if action == 0:
        #     return float(self.T_norte[(self.T_norte['s_in'] == s_in) & (self.T_norte['s_out'] == s_out)]["prob"]) if not self.T_norte[(self.T_norte['s_in'] == s_in) & (self.T_norte['s_out'] == s_out)].empty else 0
        # elif action == 1:
        #     return float(self.T_sul[(self.T_sul['s_in'] == s_in) & (self.T_sul['s_out'] == s_out)]["prob"]) if not self.T_sul[(self.T_sul['s_in'] == s_in) & (self.T_sul['s_out'] == s_out)].empty else 0    
        # elif action == 2:
        #     return float(self.T_leste[(self.T_leste['s_in'] == s_in) & (self.T_leste['s_out'] == s_out)]["prob"]) if not self.T_leste[(self.T_leste['s_in'] == s_in) & (self.T_leste['s_out'] == s_out)].empty else 0
        # elif action == 3:
        #     return float(self.T_oeste[(self.T_oeste['s_in'] == s_in) & (self.T_oeste['s_out'] == s_out)]["prob"]) if not self.T_oeste[(self.T_oeste['s_in'] == s_in) & (self.T_oeste['s_out'] == s_out)].empty else 0
        # else:
        #     return 0
        # if self.isEnd(state):
        #     return state
        # else:            
        # if action == 0: # N':
        #     for item in self.T_norte:                                        
        #         return float(item[2]) if item[0] == state and item[1] == stateNext else 0.
        # elif action == 1: #'S':
        #     for item in self.T_sul:
        #         return float(item[2]) if item[0] == state and item[1] == stateNext else 0.
        # elif action == 2: #'L':
        #     for item in self.T_leste:
        #         return float(item[2]) if item[0] == state and item[1] == stateNext else 0.
        # elif action == 3: #'O':
        #     for item in self.T_oeste:
        #         return float(item[2]) if item[0] == state and item[1] == stateNext else 0.

    
#------------------------------------------
# Métodos para definir o mundo
#------------------------------------------
def loadData(enviroment, x, y, a, so, g):   
    global A, G, So, S, X, Y, Enviroment
    global T_norte, T_sul, T_leste, T_oeste, C_matrix, T
    os.system('clear')
    Enviroment = enviroment
    S = x*y
    X = x
    Y = y
    A = a 
    So = so
    G = g
    T_norte = pd.read_csv(Enviroment+'/Action_Norte.txt', header = None, sep = "   ", engine='python')        
    T_sul = pd.read_csv(Enviroment+'/Action_Sul.txt', header = None, sep = "   ", engine='python')        
    T_leste = pd.read_csv(Enviroment+'/Action_Leste.txt', header = None, sep = "   ", engine='python')        
    T_oeste = pd.read_csv(Enviroment+'/Action_Oeste.txt', header = None, sep = "   ",engine='python')        
    C_matrix = pd.read_csv(Enviroment+'/Cost.txt', header = None, engine='python')                
    # T_norte =np.array(T_norte)
    # T_sul =np.array(T_sul)
    # T_leste =np.array(T_leste)
    # T_oeste =np.array(T_oeste)
    T_norte.columns=('s_in', 's_out', 'prob')
    T_sul.columns=('s_in', 's_out', 'prob')
    T_leste.columns=('s_in', 's_out', 'prob')
    T_oeste.columns=('s_in', 's_out', 'prob')        

    
def printSolution(V, Pi, nr_exp):        
    # resultado dos parámetros
    print('*********************************************')
    print('  Ambiente: ' + Enviroment)
    print('  Experimento: ' + str(nr_exp))
    print('  Ganma: ' + str(Ganma))
    print('  Epsilon: '+ str(Epsilon))
    print('  Tempo VI: '+ str(Vi_t))
    #print('  Nro iteracoes: ' + str(Vi_converg.loc['it']))
    #print('  Erro converg: ' + str(Vi_converg.loc['error']))
    print('  Nro calculos de Q: ' + str(Vi_calc))
    print('*********************************************')        
    Vi_converg.to_csv(os.path.join("VI_%s_converg_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 

    # resultados dos valores e politicas 
    Vm = pd.DataFrame(V)
    Pm = pd.DataFrame(Pi)
    Vm.to_csv(os.path.join("VI_%s_Val_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 
    Pm.to_csv(os.path.join("VI_%s_Pol_amb_%s_ep_%s_g_%s_calc_%s" % (nr_exp,Enviroment,Epsilon,Ganma,str(Vi_calc))), index=False) 


    Vm = Vm.to_numpy()
    Vm = Vm.reshape(X,Y)
    
    heat_mapV = sb.heatmap(Vm)
    figure1 = heat_mapV.get_figure()
    figure1.show()
    figure1.savefig(os.path.join("VI_%s_img_V_amb_%s_ep_%s_g_%s.png" % (nr_exp,Enviroment,Epsilon,Ganma)))
    
    # plot convergencia
    # x = Vi_converg.loc[:,'it']
    # y = Vi_converg.loc[:,'error']
    Vi_converg.plot(kind='line',x='it',y='error', color='blue')
    plt.show()

    # plt.title("Experimento en ambiente %s, epsilon %s e ganma %s" % (Enviroment,Epsilon,Ganma))
    # plt.ylabel('Error')
    # plt.xlabel('Iteracoes')
    # plt.plot(x,y, lw=2)        
    # plt.savefig(fname = "VI_%s_img_converg_amb_%s_ep_%s_g_%s.png"%(nr_exp,Enviroment,Epsilon,Ganma), bbox_inches='tight')
    print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
    print(V)
    print(Pi)
    
    

#------------------------------------------
#------------------------------------------
# ******* ALGORITMOS *********
#------------------------------------------
#------------------------------------------

def valueIteration(mdp, epsilon, ganma):
    global Vi_t, Vi_calc, Vi_converg
    Vi_t_ini = time()    
    V = np.zeros([mdp.S]) # almacenar o Voptimo[estado]    
    V_old = np.zeros([mdp.S])
    V[G-1] = 0.
    pi = np.zeros([mdp.S]) # politicas otimas
    res = np.inf
    Vi_it = 0
    Vi_calc = 0
    Q = np.zeros([S,A])
    #print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))

    while res > epsilon:
        np.copyto(V_old, V) 
        for state in mdp.states():
            for action in mdp.actions():
                Q[state,action] = mdp.R(state,action)
                for sNext in mdp.states():
                    Q[state,action] = Q[state,action] + ganma*mdp.T(state,sNext,action)*V_old[sNext]
                    Vi_calc = Vi_calc+1
                #print ('*{:1} {:15}'.format(state, Q[state-1,action-1]))
            V[state-1] = np.max(Q, axis=1)[state-1] 
            pi[state-1] = np.argmax(Q, axis=1)[state-1]            

        # verificar a convergencia
        res = 0
        for s in mdp.states():
            dif = abs(V_old[s-1]-V[s-1])
            if dif > res:
                res = dif
                
        Vi_converg.loc[len(Vi_converg)] = [Vi_it,res]
        Vi_it = Vi_it+1
        Vm = V.reshape(X,Y)
        heat_map = sb.heatmap(Vm)
        plt.show()         
        print('res', res)
        print('calc', Vi_calc)
    return V,pi
    
#------------------------------------------
# Main()
#------------------------------------------

def main():    
    #loadData(enviroment ='Ambiente0v', x = 2, y = 5, a = 4, so = 6, g = 10)        
    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 125)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.9, epsilon = 0.001)    
    printSolution(V,pi,1)
    
    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 125)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.9, epsilon = 0.001)    
    printSolution(V,pi,1)

    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 125)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.7, epsilon = 0.001)    
    printSolution(V,pi,2)

    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 125)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.7, epsilon = 0.00001)    
    printSolution(V,pi,3)
    
    #loadData(enviroment ='Ambiente0v', x = 2, y = 5, a = 4, so = 6, g = 10)        
    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 101)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.9, epsilon = 0.001)    
    printSolution(V,pi,4)
    
    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 101)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.9, epsilon = 0.001)    
    printSolution(V,pi,5)

    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 101)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.7, epsilon = 0.001)    
    printSolution(V,pi,6)

    loadData(enviroment ='Ambiente1', x = 5, y = 25, a = 4, so = 1, g = 101)    
    mdp = MDP(So, S, G)    
    V,pi = valueIteration(mdp, ganma = 0.7, epsilon = 0.00001)    
    printSolution(V,pi,7)
    
    #LAO(ganma_ =  1, epsilon_ = 0.001)
    #executarExperimentosLAO(mdpObject = lao, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=125, X=5, Y=25, enviroment='Ambiente1')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente1', nro_exp = 3)
    
    #mdp = LAO(So=1, G=2000, X=20, Y=100, enviroment='Ambiente2')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente2', nro_exp = 3)
    
    #mdp = LAO(So=1, G=12500, X=50, Y=250, enviroment='Ambiente3')
    #executarExperimentosVI(mdpObject = mdp, enviroment = 'Ambiente3', nro_exp = 3)

if __name__ == "__main__":
    main()