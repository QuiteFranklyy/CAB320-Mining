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

from search import best_first_graph_search, Node, PriorityQueue


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
    if len(a) == 0:
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
    if len(a) == 0:
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

    def __init__(self, underground, dig_tolerance=1):
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

        # self.underground  should be considered as a 'read-only' variable!
        # Checking if dig_tolerance > 0 and a letter
        assert isinstance(dig_tolerance,int)
        assert underground.dtype == int or underground.dtype == float
        assert dig_tolerance > 0
        # Assigning variables
        self.dig_tolerance = dig_tolerance
        self.underground = underground
        dimensions = underground.ndim
        assert dimensions in (2, 3)
        shape = underground.shape

        # 3D
        if dimensions == 3:
            self.len_x = shape[0]
            self.len_y = shape[1]
            self.len_z = shape[2]
            self.initial = tuple(
                map(tuple, np.zeros(tuple((self.len_x, self.len_y)), dtype=int)))
            self.cumsum_mine = np.cumsum(self.underground, axis=2)
            self.padded_sum = np.insert(self.cumsum_mine, 0, 0, axis=2)

        else:  # 2D
            self.len_x = shape[0]
            self.len_z = shape[1]
            self.len_y = 0
            self.initial = tuple(np.zeros((self.len_x,), dtype=int))
            self.cumsum_mine = np.cumsum(self.underground, axis=1)
            self.padded_sum = np.insert(self.cumsum_mine, 0, 0, axis=1)

        # assigning carrier variables for Actions,Biggest Payoff and Best State
        self.num_actions = 0
        self.biggestPayoff = 0
        self.counter = 0
        self.bestState = []
        self.actionList = []
        self.bestActionList = []
        self.calc_time = 0


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
        L = []
        assert len(loc) in (1, 2)
        if len(loc) == 1:
            if loc[0]-1 >= 0:
                L.append((loc[0]-1,))
            if loc[0]+1 < self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx, dy in ((-1, -1), (-1, 0), (-1, +1),
                           (0, -1), (0, +1),
                           (+1, -1), (+1, 0), (+1, +1)):
                if (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy <= self.len_y):
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
        state = np.array(state, dtype="int")
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
                        new_state[x, y] = new_state[x, y] + 1
                        if (not self.is_dangerous(new_state) and new_state[x, y] <= self.len_z):
                            yield (x, y)

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
        new_state = np.array(state)  # Make a copy
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
                             + str(self.underground[..., z]) for z in range(self.len_z))

    @staticmethod
    def plot_state(state):
        if state.ndim == 1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]),
                   state
                   )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim == 2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x)  # cols, rows
            x, y = _xx.ravel(), _yy.ravel()
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3, 3))
            ax1 = fig.add_subplot(111, projection='3d')
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

        # if it's a node, get the state
        if isinstance(state, Node):
            state = state.state
        # convert to np.array in order to use tuple addressing
        state = np.array(state)
        # state[loc]   where loc is a tuple
        comparison = state == self.initial
        if comparison.all():
            return 0
        '''
        2D
        '''
        if state.ndim == 1:
            # Find the x, z coordinates
            where = np.where(state > 0)
            # Take the x coordinates from where function
            x = where[0]
            # Extract column depths using the x coordinates
            z = state[x]
            # Calculate and return the sum of the current state
            return sum(self.padded_sum[x, z])

        '''
        3D
        '''
        if state.ndim == 2:
            # Find the x, y, z coordinates
            where = np.where(state > 0)
            # Split x and y coordinates out of where function into 2 seperate arrays
            x = np.array(where[0])
            y = np.array(where[1])
            # Extract the depths of each column using the x and y coordinates of the state
            z = state[x, y]
            # Calculate and return the sum of the current state
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
        dif1 = np.absolute(state[:-1, :] - state[1:, :])
        if (self.dig_tolerance + 1) in dif1:
            return True

        # orthoganol left
        dif2 = np.absolute(state[:, :-1] - state[:, 1:])
        if (self.dig_tolerance + 1) in dif2:
            return True

        # diagonal left down
        dif3 = np.absolute(state[:-1, :-1] - state[1:, 1:])
        if (self.dig_tolerance + 1) in dif3:
            return True

        # diagonal right down
        dif4 = np.absolute(state[1:, :-1] - state[:-1, 1:])
        if (self.dig_tolerance + 1) in dif4:
            return True

        # if none of the states have a dangerous difference, then the state is safe
        return False



    def max_theoretical_payoff(self, s):
        # using the cumsum find the max possible payoff
        
        max_payoff = 0
        # if it's a node, get the state
        if isinstance(s, Node):
            s = s.state
        '''
        2D
        '''
        state = np.array(s)
        if (self.len_y) == 0:
            # for each column, calculate the maximum possible cumulative sum
            for x in range(self.len_x):
                # get the depth of the column
                depth = state[x]
                # add the maximum cumulative sum to the maximum payoff
                max_payoff += max(self.padded_sum[x, depth:])
            return max_payoff

        '''
        3D
        '''
        # for each column, calculate the maximum possible cumulative sum
        for x in range(self.len_x):
            for y in range(self.len_y):
                # get the depth of the column
                depth = state[x, y]
                # add the maximum cumulative sum to the maximum payoff
                max_payoff += max(self.padded_sum[x, y, depth:])
        return max_payoff
    
    


    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""

        # checks if the max payoff is equal to the current payoff of the state
        return self.b(state) == self.payoff(state)

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
    # Initialize the mine
    initial_state = mine.initial

    @lru_cache(maxsize=None)
    def search_recs(state):

        # Initialize the list of actions for the current block
        a = mine.actions(state)

        # For each action...
        for action in a:
            # Find the resulting state of that action
            s2 = mine.result(state, action)
            # Counter used for tracking amount of computation
            mine.counter += 1
            # Recursion with new state
            search_recs(s2)

        # Find the payoff of the deepest state (DFS)
        payoff = mine.payoff(state)

        # If the payoff is bigger than the stored payoff
        if (payoff > mine.biggestPayoff):
            # Assign the Best variables respectively
            mine.biggestPayoff = payoff
            mine.bestState = state

        # del mine.actionList[-1]
        return payoff

    # Run function on intitial state of array
    search_recs(initial_state)
    #Find the best action list for resulting state
    mine.bestActionList = find_action_sequence(mine.initial, mine.bestState)
    
    #Return resulting variables
    return mine.biggestPayoff, mine.bestState, mine.bestActionList


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

    # initialise the first node, the frontier, and the set of explored states

    initial_node = Node(mine.initial)
    # frontier is a Priority Queue, ordered in descedning order by the payoff of each state
    frontier = PriorityQueue(order="max", f=mine.payoff)
    frontier.append(initial_node)
    explored = set()  # set of explored states
    
    # while there are nodes in the frontier, continue searching
    while frontier:
        
        # get the first node off the frontier
        current_node = frontier.pop()
        # add it to the explored nodes
        explored.add(current_node.state)
        
        # if the payoff of the current node is bigger than our current best, update the biggest payoff, best state, and best action list
        if mine.payoff(current_node.state) > mine.biggestPayoff:
            mine.biggestPayoff = mine.payoff(current_node.state)
            mine.bestState = current_node.state
            mine.bestActionList = find_action_sequence(mine.initial,current_node.state)
            
            mine.counter += 1
            
        # expand the child nodes 
        for child_node in current_node.expand(mine):
            # if the node is not in explored, and it is not in the frontier, AND it's best possible payoff is NOT less than our current best payoff
            if child_node.state not in explored and child_node not in frontier and (not (mine.max_theoretical_payoff(child_node) < mine.biggestPayoff)):
                    frontier.append(child_node)
            
            # otherwise if it is already in the frontier, delete it because it is not
            elif child_node in frontier:                
                del frontier[child_node]
                
    print("Number of Explored Nodes:",mine.counter)
                    
    return mine.biggestPayoff, mine.bestState, mine.bestActionList

 
    


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
                if (s0[x, y] < s1[x, y]):
                    s0[x, y] = s0[x, y] + 1
                    # add the action to the action list
                    action_list.append([x, y])

    return action_list


def my_team():
    '''Return the list of the team members of this assignment submission 
    as a list of triplet of the form (student_number, first_name, last_name)'''
    return [(10272224, 'Michael', 'Novak'), (9952438, 'Sebastian', 'Young'), (10012320, 'Mathew', 'Haywood')]

    # mat is a homo-sapien


if __name__ == "__main__":
    # underground = a = np.arange(27).reshape(3,3,3)
    # m = Mine(underground)
    # underground = a = np.array([[[0.455,  0.579, -0.54, -0.995, -0.771],
    #                               [0.049,  1.311, -0.061,  0.185, -1.959],
    #                               [2.38, -1.404,  1.518, -0.856,  0.658],
    #                               [0.515, -0.236, -0.466, -1.241, -0.354]],
    #                             [[0.801,  0.072, -2.183,  0.858, -1.504],
    #                               [-0.09, -1.191, -1.083,  0.78, -0.763],
    #                               [-1.815, -0.839,  0.457, -1.029,  0.915],
    #                               [0.708, -0.227,  0.874,  1.563, -2.284]],
    #                             [[-0.857,  0.309, -1.623,  0.364,  0.097],
    #                               [-0.876,  1.188, -0.16,  0.888, -0.546],
    #                               [-1.936, -3.055, -0.535, -1.561, -1.992],
    #                               [0.316,  0.97,  1.097,  0.234, -0.296]]])
    
    # underground = a = np.array([[[0.455,  0.579, -0.54, -0.995, -0.771, -0.696],
    #                               [0.049,  1.311, -0.061,  0.185, -1.959, 0.421],
    #                               [2.38, -1.404,  1.518, -0.856,  0.658, 0.703],
    #                               [0.515, -0.236, -0.466, -1.241, -0.354, 0.703]],
    #                             [[0.801,  0.072, -2.183,  0.858, -1.504, 0.314],
    #                               [-0.09, -1.191, -1.083,  0.78, -0.763, -0.833],
    #                               [-1.815, -0.839,  0.457, -1.029,  0.915, 0.187],
    #                               [0.708, -0.227,  0.874,  1.563, -2.284, 0.261]],
    #                             [[-0.857,  0.309, -1.623,  0.364,  0.097, -0.261],
    #                               [-0.876,  1.188, -0.16,  0.888, -0.546, -0.845],
    #                               [-1.936, -3.055, -0.535, -1.561, -1.992, 0.262],
    #                               [0.316,  0.97,  1.097,  0.234, -0.296, 0.248]],
    #                             [[-0.857,  0.309, -1.623,  0.364,  0.097, -0.261],
    #                               [-0.876,  1.188, -0.16,  0.888, -0.546, -0.845],
    #                               [-1.936, -3.055, -0.535, -1.561, -1.992, 0.262],
    #                               [0.316,  0.97,  1.097,  0.234, -0.296, 0.248]]])

    # underground = a =np.array([
    #     [-0.814,  0.637, 1.824, -0.563],
    #     [ 0.559, -0.234, -0.366,  0.07 ],
    #     [ 0.175, -0.284,  0.026, -0.316],
    #     [ 0.212,  0.088,  0.304,  0.604],
    #     [-1.231, 1.558, -0.467, -0.371]])
    
    underground = a =np.array([
        [-0.814,  0.637, 1.824, -0.563, 0.683],
        [ 0.559, -0.234, -0.366,  0.07, -0.030],
        [ 0.175, -0.284,  0.026, -0.316, 0.443],
        [ 0.212,  0.088,  0.304,  0.604, 0.001],
        [-1.231, 1.558, -0.467, -0.371, -0.998],
        [0.450, -0.042, 0.147, -0.097, 0.885],
        [0.427, -0.922, 0.893, 0.849, -0.075]])
    
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
    
    print("dp")
    
    t0 = time.time()
    sol_ts = search_dp_dig_plan(m)
    t1 = time.time()
    
    print("Biggest Payoff")
    print(sol_ts[0])
    
    print("Best State")
    print(sol_ts[1])
    
    print("Best Action List")
    print(sol_ts[2])
    
    print("Time")
    print(t1-t0)
    
    print("Iterations")
    print(m.counter)

    print("bb")
    t0 = time.time()
    sol_ts = search_bb_dig_plan(m)
    t1 = time.time()
    
    print("Biggest Payoff")
    print(sol_ts[0])
    
    print("Best State")
    print(sol_ts[1])
    
    print("Best Action List")
    print(sol_ts[2])
    
    print("Time")
    print(t1-t0)

    # print(sol_ts.state)
