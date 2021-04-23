#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

from functools import lru_cache

from numbers import Number

import time

import search


    
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''
        # super().__init__() # call to parent class constructor not needed
        
        self.underground = underground
        # self.underground  should be considered as a 'read-only' variable!
        assert dig_tolerance > 0
        self.dig_tolerance = dig_tolerance
        dimensions = underground.ndim
        assert dimensions in (2,3)
        shape = underground.shape
        
        #3D
        if dimensions == 3:
            self.len_x = shape[0]
            self.len_y = shape[1]
            self.len_z = shape[2]
            self.initial = tuple(map(tuple,np.zeros(tuple((self.len_x,self.len_y)),dtype=int)))
            self.cumsum_mine = np.cumsum(self.underground, axis=2)
            self.padded_sum = np.insert(self.cumsum_mine,0,0,axis=2)
            
        else: #2D
            self.len_x = shape[0]
            self.len_z = shape[1]
            self.len_y = 0
            self.initial = tuple(np.zeros((self.len_x,), dtype=int))
            self.cumsum_mine = np.cumsum(self.underground, axis=1)
            self.padded_sum = np.insert(self.cumsum_mine,0,0,axis=1)
        
        
        
        
        self.goal = None
        
        self.num_actions = 0
        self.biggestPayoff = 0
        self.counter = 0
        self.bfs = []
        self.actionList = []
        self.bestActionList = []



    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy <= self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
    
    
    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''       
        state = np.array(state, dtype = "int")
        # 2D Generator
        if (self.len_y == 0):
           def mygen(state):
                for x in range(self.len_x):
                    new_state = np.array(state)
                    new_state[x] = new_state[x] + 1
                    if (not self.is_dangerous(new_state) and new_state[x] <= self.len_z):
                        yield (x)
           
        # 3D Generator
        else:
            def mygen(state):
                for x in range(self.len_x):
                    for y in range(self.len_y):
                        new_state = np.array(state)
                        new_state[x,y] = new_state[x,y] + 1
                        if (not self.is_dangerous(new_state) and new_state[x,y] <= self.len_z):
                            yield (x,y)
        
        # # uncomment the following to see the actions working
        # print("State:\n", state)
        # test = list(mygen(state))
        # print("Actions:\n",np.array(test, dtype="int"),"\n")
        # self.num_actions += 1
        # if (self.num_actions == 5):
        #     quit()
        
                        
        return mygen(state)
                
  
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        if (self.len_y > 0):
            action = tuple(action) 
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)
                
    
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[...,z]) for z in range(self.len_z))
                    
                        
                
            return self.underground[loc[0], loc[1],:]
        
    
    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        
        No loops needed in the implementation!        
        '''
        state = np.array(state) # convert to np.array in order to use tuple addressing
                                # state[loc]   where loc is a tuple
        comparison = state == self.initial
        if comparison.all():
            return 0
        '''
        2D
        '''
        if state.ndim == 1:
            #Find the x, z coordinates
            where = np.where(state>0)
            x = where[0]
            z = state[x]
            
            # Calculate the cumalative sum
            # cumsum = np.cumsum(self.underground, axis=1)
            
            # Pad out the mine for indexing
            # padded_sum = np.insert(self.cumsum_mine,0,0,axis=1)
            # Return Payoff
            return sum(self.padded_sum[x, z])
                  
        '''
        3D
        '''
        if state.ndim == 2:
            # Find the x, y, z coordinates
            where = np.where(state>0)
            x = np.array(where[0])
            y = np.array(where[1])         
            z = state[x,y]
            
            # Do the cumalative sum
            # cumsum = np.cumsum(self.underground, axis=2)
            
            # Pad out the mine for indexing
            # padded_sum = np.insert(self.cumsum_mine,0,0,axis=2)
            
            # Return payoff

            return sum(self.padded_sum[x, y, z])
            
        


    def is_dangerous(self, state):
        '''
        Return True iff the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)         
        # 2D implementation
        if self.len_y == 0:
            # create an array with all of the differences in height in adjacent columns
            difference_array = np.absolute(np.diff(state))
            # if there exists a difference that is greater than the dig tolerance
            if (self.dig_tolerance + 1) in difference_array:
                return True
            else:
                return False
        
        # 3D implementation
        # orthogonal down
        dif1 = np.absolute(state[:-1,:] - state[1:,:])
        if (self.dig_tolerance + 1) in dif1:
            return True
        
        # orthoganol left
        dif2 = np.absolute(state[:,:-1] - state[:,1:])
        if (self.dig_tolerance + 1) in dif2:
            return True
        
        # diagonal left down
        dif3 = np.absolute(state[:-1,:-1] - state[1:,1:])
        if (self.dig_tolerance + 1) in dif3:
            return True
        
        # diagonal right down
        dif4 = np.absolute(state[1:,:-1] - state[:-1,1:])
        if (self.dig_tolerance + 1) in dif4:
            return True        
        
        return False
    
    
    def b(self, s):
        # using the cumsum find the max possible payoff
        '''
        2D
        '''
        state = np.array(s)
        if (self.len_y) == 0:
            max_payoff = 0
            for x in range (self.len_x):
                # max_payoff =
                pass
                
                
    
    
    

            
           


    
    # ========================  Class Mine  ==================================
    
    
def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    
    initial_state = mine.initial
    
    t0 = time.time()
    @lru_cache(maxsize=None)
    def search_recs(state):
        # print("state")
        # print(state)

        # print("payoff")
        # print(payoff)
        a = mine.actions(state)
        
        
        for action in a:
            # mine.actionList.append(action)
            s2 = mine.result(state, action)
            mine.counter += 1
            search_recs(s2)
            
        payoff = mine.payoff(state)
        if (payoff > mine.biggestPayoff):
            mine.biggestPayoff = payoff
            mine.bfs = state
            # print(mine.initial,state)
            mine.bestActionList = find_action_sequence(mine.initial, state)
            
        # del mine.actionList[-1]
        return payoff
        
    #     # calculate the max of the payoffs
    
    #search_recs(initial_state)

    
    # cached_search_recs = lru_cache(maxsize=None)(search_recs)
    
    
    # return best_payoff, best_action_list, best_final_state
    
    #search_recs_mem = memoize(search_recs)
    
    #search_recs_mem(initial_state)
    search_recs(initial_state)
    print("biggest cum")
    print(mine.biggestPayoff)
    print("big count")
    print(mine.counter)
    print("big state")
    print(mine.bfs)
    print("best action list")
    print(mine.bestActionList)
    t1 = time.time()
    print(t1-t0)
    


    
    
def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    
    # we want to use uniform cost search where the heuristic is the max payoff
    
    
    
    initial_state = mine.initial
    
    t0 = time.time()
    def search_recs(state):
        a = mine.actions(state)
        
        
        for action in a:
            s2 = mine.result(state, action)
            mine.counter += 1
            search_recs(s2)
            
        payoff = mine.payoff(state)
        if (payoff > mine.biggestPayoff):
            mine.biggestPayoff = payoff
            mine.bfs = state
            mine.bestActionList = find_action_sequence(mine.initial, state)
        return payoff
    
    search_recs(initial_state)
    print("biggest cum")
    print(mine.biggestPayoff)
    print("big count")
    print(mine.counter)
    print("big state")
    print(mine.bfs)
    print("best action list")
    print(mine.bestActionList)
    t1 = time.time()
    print(t1-t0)



def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''    
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]
    
    action_list = []
    s0 = np.array(s0)
    s1 = np.array(s1)
    
    '''
    2D
    '''
    if (s0.ndim == 1):
        while (not (s0 == s1).all()):
            # want to increment over each column, while each column in s0 is less than s1
            for x in range(s0.shape[0]):
                if (s0[x] < s1[x]):
                    s0[x] = s0[x] + 1
                    # add the action to the action list
                    action_list.append(x)
                    
        return action_list
    
    '''
    3D
    '''
    while (not (s0 == s1).all()):
        # want to increment over each column, while each column in s0 is less than s1
        for x in range(s0.shape[0]):
            for y in range(s0.shape[1]):
                if (s0[x,y] < s1[x,y] ):
                    s0[x,y] = s0[x,y] + 1
                    # add the action to the action list
                    action_list.append([x,y])
                    
    return action_list               
    
    
    
def my_team():    
     '''Return the list of the team members of this assignment submission 
     as a list of triplet of the form (student_number, first_name, last_name)''' 
     return [ (10272224, 'Michael', 'Novak'), (9952438, 'Sebastian', 'Young'), (10012320, 'Mathew', 'Haywood') ]
        
        
        # mat is a homo-sapien
        
if __name__ == "__main__":
    # underground = a = np.arange(27).reshape(3,3,3)
    # m = Mine(underground)
    underground = a = np.array([[[ 0.455,  0.579, -0.54 , -0.995, -0.771],
                                    [ 0.049,  1.311, -0.061,  0.185, -1.959],
                                    [ 2.38 , -1.404,  1.518, -0.856,  0.658],
                                    [ 0.515, -0.236, -0.466, -1.241, -0.354]],
                                    [[ 0.801,  0.072, -2.183,  0.858, -1.504],
                                    [-0.09 , -1.191, -1.083,  0.78 , -0.763],
                                    [-1.815, -0.839,  0.457, -1.029,  0.915],
                                    [ 0.708, -0.227,  0.874,  1.563, -2.284]],
                                    [[ -0.857,  0.309, -1.623,  0.364,  0.097],
                                    [-0.876,  1.188, -0.16 ,  0.888, -0.546],
                                    [-1.936, -3.055, -0.535, -1.561, -1.992],
                                    [ 0.316,  0.97 ,  1.097,  0.234, -0.296]]])

    # underground = a =np.array([
    #    [-0.814,  0.637, 1.824, -0.563],
    #    [ 0.559, -0.234, -0.366,  0.07 ],
    #    [ 0.175, -0.284,  0.026, -0.316],
    #    [ 0.212,  0.088,  0.304,  0.604],
    #    [-1.231, 1.558, -0.467, -0.371]])
    m = Mine(np.array(underground))
    print("underground:")
    print(underground)
    print("cumsum:")
    print(m.cumsum_mine)
    print("dig tolerance:")
    print(m.dig_tolerance)
    print("initial:")
    print(m.initial)
    print("len x y z:")
    print(m.len_x)
    print(m.len_y)
    print(m.len_z)
    
    sol_ts = search_dp_dig_plan(m)

    