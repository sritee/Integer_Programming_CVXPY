#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:25:33 2018

@author: sritee
"""

import cvxpy as cvx
import numpy as np

#np.random.seed(1)

num_nodes=10#number of nodes in the TSP
max_cost=10 #maximum cost per edge, for sampling.
cost_matrix=np.random.randint(1,max_cost,size=(num_nodes,num_nodes)) #this will generate a random integer cost matrix
cost_matrix=cost_matrix + cost_matrix.T #ensure symmetry of the matrix

#sample matrix is given below

#cost_matrix=np.array([[1,5,4,3],[3,1,8,2],[5,3,1,9],[6,4,3,4]]) #the true least cost is 12, this is used as a check.
np.fill_diagonal(cost_matrix,100000) #make sure we don't travel from node to same node, by having high cost.


x=cvx.Variable((num_nodes,num_nodes),boolean=True) #x_ij is 1, if we travel from i to j in the tour.

u=cvx.Variable(num_nodes) #variables in subtour elimination constraints


cost=cvx.trace(cvx.matmul(cost_matrix.T,x)) #total cost of the tour



ones_arr=np.ones([num_nodes]) #array for ones
  
constraints=[]

#now, let us make sure each node is visited only once, and we leave only once from that node.

constraints.append(cvx.matmul(x.T,ones_arr)==1) 
constraints.append(cvx.matmul(x,ones_arr)==1)

#Let us add the subtour elimination constraints (Miller-Tucker-Zemlin similar formulation)

for i in range(1,num_nodes):
    for j in range(1,num_nodes):
        if i!=j:
            constraints.append((u[i]-u[j]+num_nodes*x[i,j]-num_nodes+1<=0))
        else:
            continue

prob=cvx.Problem(cvx.Minimize(cost),constraints)

prob.solve()

#print(x.value.astype('int32'))
#print(cost_matrix)



tour=[0] #the final tour we have found will be stored here. Initialize with start node.
verified_cost=0 #builds up the cost of the tour, independently from the gurobi solver. We use this as a sanity check.

now_node=0 #Initialize at start node

for k in range(num_nodes):
     
    cur=np.argmax(x.value[now_node,:]) #where  we go from node i
    verified_cost=verified_cost+cost_matrix[now_node,cur] #build up the cost
    tour.append(cur) #for 1 based indexing
    now_node=cur

print('The least cost tour found is ',end=" ")
for idx,k in enumerate(tour):
    if idx!=len(tour)-1:
        print(k+1,end=" -> ")
    else:
        print(k+1)
        
print('Cost of tour found by the solver is {} and cost computed by us for verification is {} '.format(round(prob.value,2),round(verified_cost,2)))
print('Time taken to solve the problem instance after formulation is {} seconds'.format(round(prob.solver_stats.solve_time,3)))