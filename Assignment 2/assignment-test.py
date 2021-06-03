# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:03:17 2021

@author: n10012320
"""

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


def main():
    polish = "[*,[-,[*,10,9],7],[+,10,[-,9,8]]]"
    a = polish_str_2_expr_tree(polish)
    print(a)
    
if __name__ == "__main__":
    main()

    