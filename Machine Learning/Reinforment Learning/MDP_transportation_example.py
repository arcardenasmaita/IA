#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creado el: 04/05/2020
@Actualizado el:
@Autor: ARCM 
@Fuente: https://www.youtube.com/watch?v=9g32v7bK3Co&t=905s
@Tipo: Proceso de Decision Markoviano MDP
@Definicion:
 * Proceso de Decision Markoviano *
Algoritmo de Iteracion Valor (Value Iteration)
"""
import os
import sys
sys.setrecursionlimit(10000)
### Modelo (problema de busqueda)
i = 0
class TransportationProblem(object):
    def __init__(self, N):
        # N = numero de bloques de calles
        self.N = N
    def startState(self):
        # estado inicial
        return 1
    def isEnd(self, state):
        # estado final
        return state == self.N
    def actions(self, state):
        # devuelve una lista de aciones validas
        result = [] 
        if state+1 <= self.N:
            result.append('walk')
        if state*2 <= self.N:
            result.append('tram')
        return result

    def succesorProbReward(self, state,action):
        global i
        # devuelve una lista de la tripla (nuevoEstado, probabilidad, recompensa)
        # estado = s, accion = a, nuevoEstado = s'
        # probabilidad = T(s, a, s'), recompensa = Reward(s, a, s')        
        result = []
        if action == 'walk':
            result.append((state+1, 1., -1.))
        if action == 'tram':
            result.append((state*2, 0.5, -2.))
            result.append((state, 0.5, -2.))
        i = i+1
        #print(i)
        return result
    def discount(self):
        return 1.
    def states(self):
        return range(1, self.N+1)
    
### Algoritmos

def valueIteration(mdp):
    # inicializar valores
    V = {} # estado ->Voptimo[estado]
    for state in mdp.states():
        V[state] = 0.
    
    def Q(state, action):
        return sum(prob*(reward + mdp.discount()*V[newState]) \
                   for newState, prob, reward in mdp.succesorProbReward(state, action))
    
    while True:
        # calcular los nuevos valores 'newV' dados los anteriores valores 'V'
        newV = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state] = 0.
            else:
                newV[state] = max(Q(state,action) for action in mdp.actions(state))        
        # verificar la convergencia
        if max(abs(V[state]-newV[state]) for state in mdp.states()) < 1e-10:
            break
        V = newV
        
        # leer la politica
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state] = 'none'
            else:
                # recuperar argumentos que maximizan la función Q (argmax)
                pi[state] = max((Q(state, action), action) for action in mdp.actions(state)) 
        
        # escribir los resultados
        os.system('clear')
        
        print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
        for state in mdp.states():
            print ('{:15} {:15} {:15}'.format(state, V[state], pi[state]), sep="  ")
        input()

    
### Main

# ejemplo básico para 10 bloques
mdp = TransportationProblem(N=10)

# prueba simple, cuantas acciones tengo desde el estado 3?
#print(mdp.actions(3))

# prueba simple, que pasa si estoy en el estado 3 y tomo la accion 'walk'
print(mdp.succesorProbReward(3,'walk'))
print(mdp.succesorProbReward(3,'tram'))

# valueIteration(mdp)
 
