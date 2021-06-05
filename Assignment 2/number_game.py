'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''



# from genetic_algorithm import evolve_pop

import numpy as np
import random

import copy # for deepcopy

import collections

import time

import matplotlib.pyplot as plt


SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10272224, 'Michael', 'Novak'), (9952438, 'Sebastian', 'Young'), (10012320, 'Mathew', 'Haywood')]


# ----------------------------------------------------------------------------

def pick_numbers():
    '''    
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q


# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree

    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U
  
# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)
   
# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))
   
     
# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T

    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'
    

# ----------------------------------------------------------------------------

def polish_str_2_expr_tree(pn_str):
    
    
    '''
    
    Convert a polish notation string of an expression tree
    into an expression tree T.

    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression

    Returns
    -------
    T

    '''
    # raise NotImplementedError()
    
    def find_match(i):
        '''
        Starting at position i where pn_str[i] == '['
        Return the index j of the matching ']'
        That is, pn_str[j] == ']' and the substring pn_str[i:j+1]
        is balanced
        '''
        # Defining Binary While Condition:
        condition = 1
        #Defining index to look at
        counter = i
        #Defining variable to track inner brackets found
        bracket_count = 0
        
        while(condition):
            #Defining current character
            current = pn_str[counter]
            
            #If inner bracket found increment variable by 1
            if (current == '['):
                bracket_count += 1
            
            #If outter bracket found and bracket count is greater than 1
            # Decrease bracket count
            elif (current == ']') & (bracket_count > 1):
                bracket_count -= 1
                
            #Else the bracket is the matching bracket 
            
            elif (current == ']'):
                #End while loop condition
                condition = 0
                return counter
            #Increment while loop counter
            counter += 1
     # .................................................................  
    #Checking if matching amount of brackets in string
    left_brackets = pn_str.count('[')
    right_brackets = pn_str.count(']')
    assert left_brackets == right_brackets , "Bracketing error"
    
    
    # Defining variables
    Final = []
    operators = ['+','-','*']
    increment = 0
    #Finding left and right bracket of string
    
    left_p = pn_str.find('[')
    right_p = find_match(left_p)
    distance = right_p - left_p
    
    #For loop accross all variables inbetween brackets
    for i in range(distance):
        #If i is outside the string's length break 
        if (i + left_p + 1 + increment) > (distance - 1):
            break
        #Set index and value of the character at that index of the string
        value = i + left_p + 1 + increment
        item = pn_str[value]
        
        #If item is a comma pass
        if item == ',':
            pass
        
        #If item is a inside bracket
        elif item == '[':
            #Find matching braket
            a = find_match(value)
            
            #Recur function across values inside those brackets
            b = polish_str_2_expr_tree(pn_str[value:a+1])
            
            #Append finalized recurred function
            Final.append(b)
            
            #Increment the for loop by the distance of the inside brackets -1 (dont want to read twice)
            increment += a - value - 1
        
        #If item is a number
        elif item.isdigit():
            
            #Define counting variables
            acounter = 0
            condition = 1
            
            # While loop to find how big the number is
            while condition:
                #If value is a number incrememnt the while loop
                if pn_str[value + acounter].isdigit():
                    acounter += 1
                
                #Else end while loop
                else:
                    condition = 0
            #Use counter generated through while loop to cast a the string into a integer
            number = int(pn_str[value:(value + acounter)])
            
            #Append that number to the list
            Final.append(number)
            
            #Increment the for loop by the distance of that number - 1 (Dont want to read the number twice)
            increment += acounter-1
        
        # If item is a operator in the list of operators (*,+,-)
        elif item in operators:
            
            #Append to list
            Final.append(item)
            
        #raise NotImplementedError()
    #Return finalized list
    return Final
 
   
# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []
        
    L = [prefix.copy()+[0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L


# ----------------------------------------------------------------------------

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum

    '''
    if prefix is None:
        prefix = []

    if isinstance(T, int):
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum

    Aop = [prefix.copy()+[0]]
    Lop = [T[0]]
    Anum = []
    Lnum = []
    
    # Where recursion happens for the left part of the tree
    left_Aop, left_Lop, left_Anum, left_Lnum = decompose(T[1], prefix.copy()+[1])
    
    # Adding the index location of the opperator to the Aop list
    Aop.extend(left_Aop)
    
    # Adding Opperator from left sub-tree
    Lop.extend(left_Lop)
    
    # Adding the positions of the Numbers from the left tree
    Anum.extend(left_Anum)
    
    # Adding Number from left tree
    Lnum += left_Lnum
    
    
    # Where recursion happens for the right part of the tree
    right_Aop, right_Lop, right_Anum, right_Lnum = decompose(T[2], prefix.copy()+[2])
    
    # Adding the index lovation of the opperator to the Aop list
    Aop.extend(right_Aop)

    # Adding Opperator from right sub-tree
    Lop.extend(right_Lop)
    
    # Adding the positions of the Numbers from the right tree
    Anum.extend(right_Anum)
    
    # Adding Number from left tree
    Lnum += right_Lnum
    
    return Aop, Lop, Anum, Lnum


# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a

    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])
        
# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''    
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T


# ----------------------------------------------------------------------------

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T
    
    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T

    '''
    
    # print("Q",Q)
    # print("T",T)
    
    Aop, Lop, Anum, Lnum = decompose(T)    
    # print('Aop:',Aop,'Lop:',Lop,'Anum:',Anum,'Lnum:',Lnum)
    mutant_T = copy.deepcopy(T)
    
    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    # create a counter for the current count that is in the tree
    counter_Lnum = collections.Counter(Lnum)
    
    # get how many numbers are in the tree
    num_numbers = len(Anum)
    # choose the address of one of the numbers
    mutate_choice = random.randint(0,num_numbers-1)
    # print(mutate_choice)
    # get the tree address of this number
    chosen_address = Anum[mutate_choice]
    old_label = Lnum[mutate_choice]
    # print(chosen_address)    
    
    possible_values = []
    is_possible = False
    # make a list of available choices, if no possible choices then just return the tree\
    for key in Q:
        if counter_Q[key] - counter_Lnum[key] > 0:
            possible_values.append(key)
            is_possible = True
    
    if not is_possible:
        return T
    
        
    # print('possible_values:',possible_values)
    
    # choose a new random number to assign the variable to
    chosen_new_number = random.choice(possible_values)
    
    def set_element(lst, index, new_num):
        # Check if the randomly chosen coordinate is the first element of the 
        # nested list.
        if len(index) == 1:
            lst[index[0]] = new_num     # Set the first element of the nested list to 
            return lst
        
        elif isinstance(lst,int):
            lst = new_num
        
        else:
            set_element(lst[index[0]], index[1:], new_num)  
                       
        return lst


    # print("replace", old_label, "with", chosen_new_number)
    # print(mutant_T)
    new_tree = set_element(mutant_T, chosen_address, chosen_new_number)
    # print(new_tree)
    return new_tree
    
    
    
    
    
    
    

    # need to randomly choose a number address, then try to mutate it. Only mutate if the new tree doesn't have more than the count of each number. ie if 2 is in Q twice, then it won't mutate if it tries to put another 2 in
    
    
# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T

    '''
    if isinstance(T, int):
        return T
    
    # La is set to the indices of each operator.
    La = op_address_list(T)
    
    # A randomly chosen index coordinates are chosen from the list of operator
    # coordinates, La.
    a = random.choice(La)
    # print("random coordinate", a)
    
    # op_c is the operator at the randomly chosen coordinates within T
    op_c = get_item(T, a)
    # print("\noperator at random coordinate", op_c)
    # mutant_c : a different op
    if op_c == '+':
        new_op = random.choice(['*','-'])
    if op_c == '-':
        new_op = random.choice(['*','+'])
    if op_c == '*':
        new_op = random.choice(['-','+'])

    # print("\nrandomly chosen new operator", new_op)
    
    # a is the index of the operator we are changing
    
    # Need to put the new operator into the same place as 'a'
    def set_element(lst, index, new_op):
        
        
        # Check if the randomly chosen coordinate is the first element of the 
        # nested list.
        if len(index) == 1:
            lst[0] = new_op     # Set the first element of the nested list to 
            return lst
        
        # Check if the current list is a leaf
        elif isinstance(lst[0], str) and isinstance(lst[1], int) and isinstance(lst[2], int):
            
            lst[0] = new_op     # Set the first element of the nested list to
            
        # If not a leaf then move to the next node
        else:
            if len(index) == 1:
                # Where recursion happens
                set_element(lst[index[0]], index, new_op)
            else:
                set_element(lst[index[0]], index[1:], new_op)  
                       
        return lst

    new_T = copy.deepcopy(T)
    new_T = set_element(new_T, a, new_op)
    return new_T
    

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):    
    '''
    Perform crossover on two non trivial parents
    
    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        DESCRIPTION
        
    Q : list of the available numbers
        Q may contain repeated small numbers    
        

    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''
    
    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree 
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval
        
        '''
        d = len(aop)-1  # depth of the operator. 
                        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d] # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d]==p ]
        return K[0], K[-1]+1
        # .........................................................
        
    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)
    
    i1 = np.random.randint(0,len(Lop_1)) # pick a subtree in C1 by selecting the index
                                         # of an op
    i2 = np.random.randint(0,len(Lop_2)) # Select a subtree in C2 in a similar way
 
    # i1, i2 = 4, 0 # DEBUG    
 
    # Try to swap in C1 and C2 the sub-trees S1 and S2 
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!
    
    # Compute some auxiliary number lists
    
    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)     # indices of the numbers in S1 
                                                # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)   # same for S2 wrt C2
    
    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2
    
    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1]+Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2]+Lnum_2[b2:]
    
    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    
    d1 = len(Aop_1[i1])-1
    aS1 = Aop_1[i1][:d1] # address of the subtree S1 
    S1 = get_item(C1, aS1)

    # ABOUT 3 LINES DELETED
    d2 = len(Aop_2[i2])-1
    aS2 = Aop_2[i2][:d2]
    S2 = get_item(C2, aS2)

    # print(' \nDEBUG -------- S1 and S2 ----------') # DEBUG
    # print(S1)
    # print(S2,'\n')


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2]+nums_C1mS1)
    
    # Test whether child C1 is ok
    if all(counter_Q[v]>=counter_1[v]  for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
            )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)
        
    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1]+nums_C2mS2)
    
    # Test whether child C2 is ok
    
    # ABOUT 10 LINES DELETED    
    if all(counter_Q[v] >= counter_2[v] for v in counter_Q):
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C2mS2)
            )
        R2, _ = bottom_up_creator(list(available_nums.elements()))
        C2 = replace_subtree(C2, aS2, R2)
    
    
    
    return C1, C2



## --------------- Evolve Function ------------------
default_GA_params = {
    'max_num_iteration': 50,
    'population_size':100,
    'mutation_probability':0.1,
    'elit_ratio': 0.05,
    'parents_portion': 0.3}


def evolve_pop(Q, target, **ga_params):
    '''
    
    Evolve a population of expression trees for the game
    Letters and Numbers given a target value and a set of numbers.
    

    Parameters
    ----------
    Q : list of integers
        Integers that were drawn by the game host
    
    target: integer
           target value of the game
        
    params : dictionary, optional
        The default is GA_params.
        Dictionary of parameters for the genetic algorithm

    Returns
    -------
    v, T: the best expression tree found and its value

    '''
    
    params = default_GA_params.copy()
    params.update(ga_params)
    
    # print('GA Parameters ', params)
    
    mutation_probability = params['mutation_probability']
    pop_size = params['population_size']
    
    # ------------- Initialize Population ------------------------
    
    pop = [] # list of pairs (cost, individuals)
    
    for _ in range(pop_size):
        T, _ = bottom_up_creator(Q)
        cost = abs(target-eval_tree(T))
        pop.append((cost,T))
    
    # Sort the initial population
    # print(pop) # debug
    pop.sort(key=lambda x:x[0])
    
    # Report
    # print('\n'+'-'*40+'\n')
    # print("The best individual of the initial population has a cost of {}".format(pop[0][0]))
    # print("The best individual is \n")
    # display_tree(pop[0][1])
    # print('\n')
    # ------------- Loop on generations ------------------------
    
    # Rank of last individual in the current population
    # allowed to breed.
    rank_parent = int(params['parents_portion'] * 
                                      params['population_size'])
    
    # Rank of the last elite individual. The elite is copied unchanged 
    # into the next generation.
    rank_elite = max(1, int(params['elit_ratio'] *
                                      params['population_size']))
 
    for g in range(params['max_num_iteration']):
        
        # Generate children
        children = []
        while len(children) < pop_size:
            # pick two parents
            (_, P1), (_, P2) = random.sample(pop[:rank_parent], 2)
            # skip cases where one of the parents is trivial (a number)
            if isinstance(P1, list) and isinstance(P2, list):
                C1, C2 = cross_over(P1, P2, Q)
            else:
                # if one of the parents is trivial, just compute mutants
                C1 = mutate_num(P1,Q)
                C2 = mutate_num(P2,Q)
            # Compute the costs of the children
            cost_1 =  abs(target-eval_tree(C1))
            cost_2 =  abs(target-eval_tree(C2))
            children.extend([ (cost_1,C1), (cost_2,C2) ])
             
        new_pop = pop[rank_elite:]+children 
        
        # Mutate some individuals (keep aside the elite for now)
        # Pick randomly the indices of the mutants
        mutant_indices = random.sample(range(len(new_pop)), 
                                       int(mutation_probability*pop_size))      
        # i: index of a mutant in new_pop
        for i in mutant_indices:
            # Choose a mutation by flipping a coin
            Ti = new_pop[i][1]  #  new_pop[i][0]  is the cost of Ti
            # Flip a coin to decide whether to mutate an op or a number
            # If Ti is trivial, we can only use mutate_num
            if isinstance(Ti, int) or random.choice((False, True)): 
                Mi = mutate_num(Ti, Q)
            else:
                Mi = mutate_op(Ti)
            # update the mutated entry
            new_pop[i] = (abs(target-eval_tree(Mi)), Mi)
                
        # add without any chance of mutation the elite
        new_pop.extend(pop[:rank_elite])
        
        # sort
        new_pop.sort(key=lambda x:x[0])
        
        # keep only pop_size individuals
        pop = new_pop[:pop_size]
        
        # Report some stats
        # print(f"\nAfter {g+1} generations, the best individual has a cost of {pop[0][0]}\n")
        
        if pop[0][0] == 0:
            # found a solution!
            break

      # return best found
    return pop[0]



## ---------------------------- MAIN BLOCK -------------------------------------
def run_experiments():
    # generate the random game requirements
    Q = []
    target = []
    population_sizes = [7,10,25,50,100,150,200,400,500,750,1000,1500,2000,3000,4000,5000,6000,7000,8000,15000]
    print('generating games',end="")
    for x in range(30):
        print('.',end = "")
        # these are our chosen population sizes
        Q.append(pick_numbers())
        # sort the game numbers
        Q[x].sort()
        target.append(np.random.randint(1,1000))
        
    print('\nGames:')
    for x in range(30):
        print('Game:',x,'Target:',target[x],'Numbers:',Q[x])
        
        
        
    ## calculate iterations vs population size graph    
    times = []
    calculated_max_iterations = []
    print('\ncalculating iterations',end="")
    for pop_size in population_sizes:
        print('.',end = "")
        tic = time.perf_counter()
        v, T = evolve_pop(Q[0], target[0], 
                      max_num_iteration = 1,
                      population_size = pop_size,
                      parents_portion = 0.3)
        toc = time.perf_counter()
        times.append(toc-tic)
        
        calculated_max_iterations.append(2/(toc-tic))
        
    print('\nTimes for one generation:',times)
    print('\nIterations for each population:', calculated_max_iterations)
    
    # our calculated values that were used in the report
    calculated_max_iterations_static = [937, 815, 348, 209, 122, 105, 71, 39, 34, 18, 14, 8, 8, 5, 4, 3, 3, 2, 2, 1]
    
    plt.plot(population_sizes,calculated_max_iterations,'-r')
    plt.ylabel('Num Iterations')
    plt.xlabel('Population Size')
    
    plt.title('Random Game: Population Size vs Num Iterations')
    plt.show()
    
    
    
    ## Run 30 different games for each population size, using the calculated_max_iterations and population_sizes
    # how long it takes, final cost, num iterations, 
    average_time = []
    average_final_cost = []
    average_success_rate = []
    
    for population_size_counter in range(len(population_sizes)):
        # array of 30 times, used to average after
        times = []
        final_costs = []
        success_rates = 0
        for x in range(30):
            tic = time.perf_counter()
            v, T = evolve_pop(Q[x], target[x], 
                          max_num_iteration =  int(calculated_max_iterations[population_size_counter]),
                          population_size = population_sizes[population_size_counter],
                          parents_portion = 0.3)
            toc = time.perf_counter()
            times.append(toc-tic)
            final_costs.append(v)
            if v == 0:
                success_rates += 1
            
            # print(f"time taken: {toc - tic:0.4f} seconds")
            # print('----------------------------')
            # if v==0:
            #     print("\n***** Perfect Score!! *****")
            # print(f'\ntarget {target} , tree value {eval_tree(T)}\n')
            # display_tree(T)
        
        # calculate averages and put them into the total arrays
        average_time.append(sum(times)/len(times))
        average_final_cost.append(sum(final_costs)/len(final_costs))
        average_success_rate.append(success_rates)
        
    print('Average Times:',average_time)
    print('Average Final Cost:',average_final_cost)
    print('Average Success Rate:', average_success_rate)
    
    average_success_rate_percent = [x / 30 * 100 for x in average_success_rate]
    
    
    # Solution Times
    plt.plot(population_sizes,average_time,'-r')
    plt.plot(population_sizes,average_time,'.b')
    plt.ylabel('Average Solution Time (sec)')
    plt.xlabel('Population Size')
    plt.title('Population Size vs Solution Time')
    plt.show()
    
    # Costs
    plt.plot(population_sizes,average_final_cost,'-r')
    plt.plot(population_sizes,average_final_cost,'.b')
    plt.ylabel('Average Final Cost')
    plt.xlabel('Population Size')
    plt.title('Population Size vs Final Cost')
    plt.show()
    
    # Success Rate
    plt.plot(population_sizes, average_success_rate_percent,'-r')
    plt.plot(population_sizes,average_success_rate_percent,'.b')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Population Size')
    plt.title('Population Size vs Success Rate')
    plt.show()
    
## UNCOMMENT 
# run_experiments()
