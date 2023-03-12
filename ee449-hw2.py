#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:01:30 2022

@author: bahadir
"""
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# path
path = r'/Users/bahadir/Desktop/4-2/EE449/HW2-2304426/painting.png'

#defining the parameters
num_inds = 5 
num_genes = 10
num_generations = 10000
tm_size = 2
frac_elites = 0.05
frac_parents = 0.2
mutation_prob = 0.1
mutation_type = 'guided'

   
# Reading an image in default mode
source_image = cv2.imread(path)
ref = np.array(source_image)

height = ref.shape [1]
width = ref.shape [0]

#Initialize the image with white
image = np.full(source_image.shape, 255).astype(np.uint8)
copy_image = np.full(source_image.shape, 255).astype(np.uint8)



#Initialize the population randomly
class POPULATION:
    def __init__(self):
        pop = []
        for i in range (0,num_inds):
            ind = []
            for j in range (0,num_genes):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(0,150)
                r = random.randint(0,255)
                g = random.randint(0,255)                
                b = random.randint(0,255)    
                a = random.uniform(0.0,1.0)
                gene = [x,y,radius,r,g,b,a]
                ind.append(gene)
            pop.append(ind)        
        self.pop = pop         
        
        
#Evaluation

def image(adn):
    im = np.full(source_image.shape, 255, dtype = np.uint8)
    overlay = np.array(im , dtype = np.uint8)
    radius = []
    queue = []
    genes = adn
    
    #finding the gene with lowest radius
    for i in range (0,num_genes):
        radius.append((genes[i][2]))
        
    for i in range (0,len(radius)):
        a = radius.index(max(radius))
        queue.append(a)
        radius[a] = 0
        
    for i in queue:
        x = adn[i][0]
        y = adn[i][1]
        r = adn[i][2]
        red = adn[i][3]
        green = adn[i][4]
        blue = adn[i][5]
        alpha = adn[i][6]
        
        cv2.circle(overlay, (int(x), int(y)), r, (red, green, blue), -1)
        i += 1
    im = cv2.addWeighted(overlay, alpha, im, 1.0 - alpha, 0)

    return im

#Fitness functionpop
def fitness(indi):
    black = np.array(image(indi))
    fit = 0
    for i in range(width):
        for j in range(height):
            for k in range(3):
                fit = fit-(int(ref[i, j, k]) - int(black[i, j, k]))**2

    return fit

#Selection Function

def select(pop):
    fit = []
    fit_after = []
    new_samples = []
    crossover_ind = []
    mutation_samples = []
    num_elites = math.ceil(len(pop)*frac_elites)
    num_crossover = math.ceil(len(pop)*frac_parents/2)*2
    #fitness calculation
    for i in range (0,len(pop)):
        fit.append(fitness(pop[i]))
        
        
    # finding elite indivuduals and appending in new list and deleting 
    # from fitness list and from sample
    for i in range(0,num_elites):
        elite_index = fit.index(max(fit))
        new_samples.append(pop[elite_index])
        best_fitness = fit[elite_index]
        del(fit[elite_index])
        del(pop[elite_index])
        
        
    # finding the best indivuduals from rest to apply crossover
    for i in range(0,num_crossover):
         crossover_index = fit.index(max(fit))
         crossover_ind.append(pop[crossover_index])
         fit[crossover_index] = -99999999999

    # applying crosover and mutation
    cross = crossover(crossover_ind)
    for i in range (0,len(cross)):
        new_samples.append(cross[i])
        
    mutation_samples = mutate(pop)
   

    #finding new fitness value for indivuduals after mutation
    for i in range (0,len(mutation_samples)):
        fit_after.append(fitness(mutation_samples[i]))     
   
    #Tournement Selection
    for i in range(0, tm_size):
         tournement_index = fit_after.index(max(fit_after))
         new_samples.append(mutation_samples[tournement_index])
         fit_after[tournement_index] = -99999999999

    
    return best_fitness , new_samples



#Crossover function
def crossover(samples):
    new_samples = []
    child1 = []
    child2 = []
    for i in range (0,len(samples),2):
        parent1 = samples[i]
        parent2 = samples[i+1]
    
        for k in range (0,num_genes): 
            prob = random.uniform(0.0, 1.0)
            if prob >= 0.5:
                child1.append(parent1[k])
                child2.append(parent2[k])
            if prob < 0.5:
                child1.append(parent2[k])
                child2.append(parent1[k])
    
                #appending sons to next generation
        new_samples.append(child1)
        new_samples.append(child2)

    return new_samples


#Mutation Function
def mutate(samples):
    new_samples = []
    mutation_index = []
    #starting from one indivudual in sample
    for k in range (0,len(samples)):
        ind = samples[k]
        #find the genes to apply mutation 
        for i in range(0,num_genes):
            p = random.uniform(0.0, 1.0)
            if(p < mutation_prob):
                mutation_index.append(i)
                
                
        #apply mutation wrt ind selected        
        if mutation_type == 'guided':
            for i in mutation_index:
                #x
                x = random.randint(ind[i][0]-width/4, ind[i][0]+width/4)
                if x > width:
                        x = random.randint(ind[i][0]-width/4, 180)
                if x<0:
                        x = random.randint(0, ind[i][0]+width/4)
                #y       
                y = random.randint(ind[i][1]-height/4, ind[i][1]+height/4)
                if y > height:
                        y = random.randint(ind[i][1]-height/4, 180)
                if y<0:
                        y = random.randint(0, ind[i][1]+height/4)
                #radius
                radius = random.randint(ind[i][2]-10,ind[i][2]+10)
                if radius > 150:
                    radius = random.randint(ind[i][2]-10, 150) 
                if radius<0:
                    radius = random.randint(0,ind[i][2]+10)
                #r
                r = random.randint(ind[i][3]-64,ind[i][3]+64)
                if r > 255:
                    r = random.randint(ind[i][3]-64,255)
                if r<0:
                    r = random.randint(0,ind[i][3]+64)
                #g
                g = random.randint(ind[i][4]-64,ind[i][4]+64)     
                if g > 255:
                    g = random.randint(ind[i][4]-64,255) 
                if g<0:
                    g = random.randint(0,ind[i][4]+64) 
                #b
                b = random.randint(ind[i][5]-64,ind[i][5]+64) 
                if b > 255:
                    b = random.randint(ind[i][5]-64,255) 
                if b<0:
                    b = random.randint(0,ind[i][5]+64) 
                #a
                a = random.uniform(ind[i][6]-0.25,ind[i][6]+0.25)
                if a > 1:
                    a = random.uniform(ind[i][6]-0.25,1)
                if a<0:
                    a = random.uniform(0,ind[i][6]+0.25)
                mutated_gene = [x,y,radius,r,g,b,a]
                ind[i] = mutated_gene
            new_samples.append(ind)    
        if mutation_type == 'unguided':
            for i in mutation_index:
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(0,150)
                r = random.randint(0,255)
                g = random.randint(0,255)                
                b = random.randint(0,255)    
                a = random.uniform(0.0,1.0)
                mutated_gene = [x,y,radius,r,g,b,a]
                ind[i] = mutated_gene
            new_samples.append(ind)
        
    return new_samples



samples = POPULATION()
generation = samples.pop
best_fitness = []
for i in range (0,num_generations):
  elite_fit , new_generation = select(generation)
  best_fitness.append(elite_fit)
  generation =  new_generation
  if i%10 == 0:
      print(i)
  if i%1000 == 0:
      im = image(generation[0])
      name = str(i)+'. Generation_Best_Ind'
      status = cv2.imwrite('/Users/bahadir/Desktop/4-2/EE449/HW2-2304426/'+name+'.png',im)
      print("Image written to file-system : ",status)
      




          

