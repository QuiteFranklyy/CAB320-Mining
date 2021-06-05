#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:46:18 2021

@author: frederic

***** Perfect Score!! *****
    Q = [25,10,2,9,8,7]
    target 449 , tree value 449
    
    -
    |+
    ||*
    |||*
    ||||2
    |||
    ||||9
    ||
    |||25
    |
    ||7
    
    |8
    
**** Perfect Score!! *****
    Q = [50,75,9,10,2,2]
    target 533 , tree value 533
    
    +
    |+
    ||75
    |
    ||-
    |||*
    ||||9
    |||
    ||||50
    ||
    |||2
    
    |10

"""

import numpy as np

from number_game import pick_numbers, eval_tree, display_tree

from genetic_algorithm import  evolve_pop

import time

import matplotlib.pyplot as plt


Q = pick_numbers()
target = np.random.randint(1,1000)






Q = [50,75,9,10,2,2]
target = 533

# Q = [100,25,7,5,3,1]
# target = 69
Q.sort()

population_sizes = [7,10,25,50,100,150,200,400,500,750,1000,1500,2000,3000,4000,5000,6000,7000,8000,15000]

times = []

print('List of drawn numbers is ',Q)


# for pop_size in population_sizes:
#     print(pop_size)
#     tic = time.perf_counter()
#     v, T = evolve_pop(Q, target, 
#                   max_num_iteration = 1,
#                   population_size = pop_size,
#                   parents_portion = 0.3)
#     toc = time.perf_counter()
#     times.append(toc-tic)
#     print(f"time taken: {toc - tic:0.4f} seconds")
#     print('----------------------------')
#     if v==0:
#         print("\n***** Perfect Score!! *****")
#     print(f'\ntarget {target} , tree value {eval_tree(T)}\n')
#     display_tree(T)

# print(population_sizes)
# print(times)


recorded_times = [0.002132299996446818, 0.0024537999997846782, 0.005743000001530163, 0.00952829999732785, 0.016281600001093466, 0.01894489999540383, 0.027886299998499453, 0.05084550000174204, 0.058003199999802746, 0.11079919999610865, 0.13849650000338443, 0.2448593000008259, 0.22865379999711877, 0.3508832000006805, 0.44809710000117775, 0.604681900003925, 0.6531142000021646, 0.8305818000008003, 0.8954402000017581, 1.9173002999959863]

calculated_max_iterations = []

for timee in recorded_times:
    calculated_max_iterations.append(2/timee)
    
print(calculated_max_iterations)

calculated_max_iterations_static = [937.9543231875075, 815.0623523414706, 348.2500434384678, 209.90103172243602, 122.83805030621565, 105.56930891613123, 71.71980506942904, 39.33484772362308, 34.48085622873913, 18.05067184664006, 14.440798142560471, 8.167956046567372, 8.746847854814579, 5.699902417659555, 4.463318329877036, 3.307524170951732, 3.062251593968362, 2.407950667830758, 2.233538320030833, 1.0431334100371168]


plt.plot(population_sizes,calculated_max_iterations_static,'-r')
plt.ylabel('Num Iterations')
plt.xlabel('Population Size')

plt.title('Population Size vs Num Iterations')
plt.show()


# how long it takes, final cost, num iterations, 
average_time = []
average_final_cost = []
average_success_rate = []
average_num = []

for population_size_counter in range(len(population_sizes)):
    # array of 30 times, used to average after
    times = []
    final_costs = []
    success_rates = 0
    for x in range(30):
        tic = time.perf_counter()
        v, T = evolve_pop(Q, target, 
                      max_num_iteration =  int(calculated_max_iterations_static[population_size_counter]),
                      population_size = population_sizes[population_size_counter],
                      parents_portion = 0.3)
        toc = time.perf_counter()
        times.append(toc-tic)
        final_costs.append(v)
        if v == 0:
            success_rates += 1
        
        print(f"time taken: {toc - tic:0.4f} seconds")
        print('----------------------------')
        if v==0:
            print("\n***** Perfect Score!! *****")
        print(f'\ntarget {target} , tree value {eval_tree(T)}\n')
        display_tree(T)
    
    # calculate averages and put them into the total arrays
    average_time.append(sum(times)/len(times))
    average_final_cost.append(sum(final_costs)/len(final_costs))
    average_success_rate.append([success_rates])
    
print('Average Times:',average_time)
print('Average Final Cost:',average_final_cost)
print('Average Success Rate:', average_success_rate)



# average time, average final cost, success rate, average iterations, 
# Average Times: [0.6628967766662147, 0.6531969033334463, 0.5700741300010123, 0.6377944933325731, 0.48135464666726574, 0.3262110533340698, 0.3023150233326305, 0.18517460666674498, 0.17564634666584122, 0.27401182666726526, 0.23630423333330933, 0.313867426666305, 0.4102685666667336, 0.5140585366653492, 0.594718189999791, 0.724039553332841, 0.7716039433328358, 0.8955319199997273, 1.050181796666584, 1.8733364733333777]
# Average Final Cost: [92.3, 10.3, 6.7, 1.7333333333333334, 1.0333333333333334, 0.4666666666666667, 0.23333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666666666666667, 0.0, 0.0]
# Average Success Rate: [[5], [10], [11], [17], [19], [26], [27], [30], [30], [30], [30], [30], [30], [30], [30], [30], [30], [29], [30], [30]]