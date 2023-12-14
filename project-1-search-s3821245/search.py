# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # initialize the LIFO queue
    stack = util.Stack()                       
    path = []
    visited = []    
    
    startState = problem.getStartState()

    # mark the start state as visited 
    visited.append(startState)                 

    # push the start state into stack
    stack.push((startState,[]))                

    # loop until stack is empty or goal state is reached
    while(not stack.isEmpty()):
        
        # pop the first element and its path from stack
        top, current_path = stack.pop()                   

        # check if the top element is goal state
        if(problem.isGoalState(top)):                  
            path = current_path
            break
        
        # push the unvisited successors of top element into stack
        for ele in problem.getSuccessors(top):
            if(ele[0] not in visited):
                stack.push((ele[0], current_path + [ele[1]]))  
                visited.append(ele[0])

    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    # initialize the FIFO queue
    queue = util.Queue()                                    
    
    path = []
    visited = []    

    # push the start state into queue
    queue.push((problem.getStartState(), []))

    # loop until queue is empty or goal state is reached
    while(not queue.isEmpty()):
        # pop the first element and its path from queue
        first, current_path = queue.pop()                   

        # check if the first element is goal state
        if(problem.isGoalState(first)):                  
            path = current_path
            break
        
        # check if the first element is visited
        if(first not in visited):              
            visited.append(first)
            # push the successors of first element into queue          
            for ele in problem.getSuccessors(first):
                queue.push((ele[0], current_path + [ele[1]]))


    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Initialize the priority queue
    queue = util.PriorityQueue()                                
    # Initialize the visited list  
    visited = []
    # Initialize the path list
    path = []

    startCost = 0
    # Push the start state into queue
    queue.push((problem.getStartState(),[],startCost),startCost)

    # Loop until queue is empty or goal state is reached
    while(not queue.isEmpty()):
        # Pop the first element, its path and cost from queue
        top, curr_path, cost = queue.pop()
        # Check if the top element is goal state
        if(problem.isGoalState(top)):
            path = curr_path
            break
        # Check if the top element is visited
        if(top not in visited):        
            visited.append(top)
            # Push the successors, their path and cost into queue
            for ele in problem.getSuccessors(top):
                    new_path = curr_path + [ele[1]]
                    new_cost = cost + ele[2]
                    queue.update((ele[0],new_path,new_cost),new_cost)

    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # Initialize the priority queue
    queue = util.PriorityQueue()                                
    visited = []
    path = []

    startCost = 0
    # Push the start state, its path and cost consisting of start cost and heuristic into queue
    queue.push((problem.getStartState(),[],startCost),startCost + heuristic(problem.getStartState(),problem))

    while(not queue.isEmpty()):
        top, curr_path, cost = queue.pop()

        if(problem.isGoalState(top)):
            path = curr_path
            break

        if(top not in visited):        
            for ele in problem.getSuccessors(top):
                    new_path = curr_path + [ele[1]]
                    ele_gx = cost + ele[2]
                    # Push the successors, their path and cost consisting of its g(x) cost and heuristic into queue
                    queue.update((ele[0],new_path,ele_gx),ele_gx + heuristic(ele[0],problem))
            visited.append(top)        
    return path


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"

    depth = 0
    # Loop until result is None
    while True:
        # Call depthLimitedSearch with depth
        result = depthLimitedSearch(problem, depth)
        # Check if result is not None
        if result is not None:
            return result            
        # Increment depth if result is None        
        depth += 1

def depthLimitedSearch(problem, l):
    
    visited = []
    # Initialize the result action list
    result = []
    # initialize the LIFO queue
    frontier = util.Stack()
    # Push the start state, its path and cost consisting of start cost into queue
    frontier.push((problem.getStartState(),[],0))
    # Loop until queue is empty or goal state is reached
    while not frontier.isEmpty():
        # Pop the first element, its path and cost from queue
        node, curr_path, curr_depth = frontier.pop()
        # Check if the top element is goal state
        if problem.isGoalState(node):
            result = curr_path
            break
        # result is None if the depth is greater than the limit
        if curr_depth > l:
            result = None
        # Check if the top element is visited
        elif node not in visited:
            visited.append(node)
            # Push the successors, their path and depth into queue
            for child in problem.getSuccessors(node):
                if child[0] not in visited:
                    frontier.push((child[0],curr_path+[child[1]],curr_depth+1))

    return result

#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
