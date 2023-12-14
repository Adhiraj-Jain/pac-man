# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # The default score
        score = successorGameState.getScore()
        # Getting a food list including food and capsules
        foodList = newFood.asList() + successorGameState.getCapsules()
        # Getting the manhattanDistance distance to the closest food
        closestfood = min([manhattanDistance(newPos, food) for food in foodList]) if len(foodList) > 0 else 0
        # Getting the manhattanDistance distance to all the ghosts from the new position
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates] 
        # Getting the manhattanDistance distance to the closest ghost
        closestghost = min(ghostDistances)
        # Adding the reciprocal of the distance to the closest food and the reciprocal of the number of food left multiplied by constant weights to the score 
        score += (20 / (closestfood + 1)) + (50 / (len(foodList) + 1))
        # Subtracting the reciprocal of the distance to the closest ghost multiplied by constant weights if the ghost is not scared else 
        # adding the reciprocal of the distance to the closest ghost multiplied by constant weights to the score
        score += (-20 / (closestghost + 1)) if newScaredTimes[ghostDistances.index(closestghost)] <= 0 else (80 / (closestghost + 1))

        return score    # successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):  
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Getting the best action from minimax algorithm
        depth = 1
        bestAction = self.max_value(gameState,depth)[0]
        return bestAction
    
            
    def max_value(self, gameState, depth):
        bestAction = None
        # If the game is over or the depth is reached, return the best action and the score
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        # The score is set to -infinity
        score = -float("Inf")
        # Getting the pacman index
        PacmanIndex = 0
        # Getting the legal actions of the pacman
        actions = gameState.getLegalActions(PacmanIndex)
        # For each action, get the score and the best action
        for action in actions:
            temp_score = self.min_value(gameState.generateSuccessor(PacmanIndex, action), 1, depth)[1]
            # If the score is higher than the current score, update the score and the best action
            if temp_score > score:
                score, bestAction = temp_score, action
        # Return the best action and the score
        return bestAction, score
    
    def min_value(self, gameState, agentIndex, depth):
        bestAction = None
        # If the game is over or the depth is reached, return the best action and the score
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        # The score is set to infinity
        score = float("Inf")
        # Getting the legal actions of the agent (Ghost)
        actions = gameState.getLegalActions(agentIndex)
        # For each action, get the score and the best action
        for action in actions:
            # Getting the new game state
            newGameState = gameState.generateSuccessor(agentIndex, action)
            # If the agent is the last agent, call the max_value function
            if agentIndex == (gameState.getNumAgents() - 1):
                temp_score = self.max_value(newGameState,depth+1)[1]
            # Else, call the min_value function with next agent
            else:
                temp_score = self.min_value(newGameState, agentIndex + 1, depth)[1]
            # If the score is lower than the current score, update the score and the best action
            if temp_score < score:
                score, bestAction = temp_score, action
        # Return the best action and the score
        return bestAction, score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        depth = 1
        # Getting the best action from alpha-beta pruning algorithm
        bestAction = self.max_value(gameState, -float("Inf"), float("Inf"),depth)[0]
        return bestAction
    
    def max_value(self, gameState, alpha, beta, depth):
        bestAction = None
        # If the game is over or the depth is reached, return the best action and the score
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        
        score = -float("Inf")
        PacmanIndex = 0
        actions = gameState.getLegalActions(PacmanIndex)
        for action in actions:
            # Getting the score and the best action
            temp_score = self.min_value(gameState.generateSuccessor(PacmanIndex, action), 1,alpha, beta, depth)[1]
            # If the score is higher than the current score, update the score and the best action
            if temp_score > score:
                score, bestAction = temp_score, action
            # If the score is higher than beta, return the best action and the score
            if score > beta:
                return action, score
            # Update alpha with maximum value of alpha and score
            alpha = max(alpha, score)

        return bestAction, score
    
    def min_value(self, gameState, agentIndex, alpha, beta, depth):
        
        bestAction = None
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        
        score = float("Inf")
        # Getting the legal actions of the agent (Ghost)
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            # Getting the new game state
            newGameState = gameState.generateSuccessor(agentIndex, action)
            # If the agent is the last agent, call the max_value function 
            if agentIndex == (gameState.getNumAgents() - 1):
                temp_score = self.max_value(newGameState, alpha, beta,depth+1)[1]
            # Else, call the min_value function with next agent
            else:
                temp_score = self.min_value(newGameState, agentIndex + 1, alpha, beta, depth)[1]
            # If the score is lower than the current score, update the score and the best action
            if temp_score < score:
                score, bestAction = temp_score, action
            # If the score is lower than alpha, return the best action and the score
            if score < alpha:
                return action, score
            # Update beta with minimum value of beta and score
            beta = min(beta, score)
        
        return bestAction, score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        depth = 1
        bestAction = self.max_value(gameState,depth)[0]
        return bestAction

    def max_value(self, gameState, depth):
        bestAction = None
        # If the game is over or the depth is reached, return the best action and the score
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        # The score is set to -infinity
        score = -float("Inf")
        # Setting the pacman index
        PacmanIndex = 0
        # Getting the legal actions of the pacman
        actions = gameState.getLegalActions(PacmanIndex)
        # For each action, get the score and the best action
        for action in actions:
            temp_score = self.expect_value(gameState.generateSuccessor(PacmanIndex, action), 1, depth)[1]
            # If the score is higher than the current score, update the score and the best action
            if temp_score > score:
                score, bestAction = temp_score, action

        return bestAction, score
    
    def expect_value(self, gameState, agentIndex, depth):
        bestAction = None
        # If the game is over or the depth is reached, return the best action and the score
        if gameState.isWin() or gameState.isLose() or depth > self.depth: 
            return bestAction, self.evaluationFunction(gameState)
        # The scores array is used to store the scores of all the actions
        scores = []
        # The average score is used to store the average score of all the actions
        avg = 0
        # Getting the legal actions of the agent (Ghost)
        actions = gameState.getLegalActions(agentIndex)
        # For each action, get the score and the best action
        for action in actions:
            newGameState = gameState.generateSuccessor(agentIndex, action)
            # If the agent is the last agent, call the max_value function
            if agentIndex == (gameState.getNumAgents() - 1):
                temp_score = self.max_value(newGameState,depth+1)[1]
            # Else, call the expect_value function with next agent
            else:
                temp_score = self.expect_value(newGameState, agentIndex + 1, depth)[1]
            # Add the score to the scores array
            scores.append(temp_score)
        # Calculate the average score
        avg = sum(scores) / len(scores)
        # Diff is used to store the minimum difference between the average score and the score of the action
        diff = float("Inf")
        # For each score, get the minimum difference between the average score and the score of the action
        for i in range(1,len(scores)): 
            # If the difference is lower than the current difference, update the difference and the best action
            if abs(scores[i] - avg) < diff:
                diff = abs(scores[i] - avg)
                bestAction = action
        # Return the best action and the average score
        return bestAction, avg

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:    This evaluation function is based on the distance to the closest food, 
                    the number of food left, the distance to the closest ghost, and the 
                    scared time of the closest ghost. 
                    The reciprocal of all the variables mentioned above are used to calculate the score.
                    And, have been added to the original/default score.

                    The closer the food is, the higher the score is. 
                    The less food left, the higher the score is. 
                    The closer the ghost is, the lower the score is. 
                    The higher the scared time of the ghost is, the higher the score is.

                    Getting a capsule will increase the score.
                    After getting a capsule, pacman will chase the ghost and eat it.
    """
    "*** YOUR CODE HERE ***"
    

    currPacmanPos = currentGameState.getPacmanPosition()
    foodMatrix = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()
    foodList = foodMatrix.asList() + capsules
    closestfood = 0
    # Getting the manhattanDistance distance from the current pacman position to the closest ghost
    ghostDistances = [manhattanDistance(currPacmanPos, ghost.getPosition()) for ghost in newGhostStates]
    # Getting the ghost with the closest distance
    closestghost = min(ghostDistances)
    # Getting the index of the closest food in the food list else set it to 0
    closestfood = min([manhattanDistance(currPacmanPos, food) for food in foodList]) if len(foodList) > 0 else 0
    # Getting the default score
    score = currentGameState.getScore()
    # Adding the reciprocal of the distance to the closest food and the reciprocal of the number of food left multiplied by constant weights to the score
    score += (20 / (closestfood + 1)) + (60 / (len(foodList) + 1))
    # Subtracting the reciprocal of the distance to the closest ghost multiplied by constant weights if the ghost is not scared else 
    # adding the reciprocal of the distance to the closest ghost multiplied by constant weights to the score
    score += (-20 / (closestghost + 1)) if newScaredTimes[ghostDistances.index(closestghost)] <= 0 else (80 / (closestghost + 1))

    return score

# Abbreviation
better = betterEvaluationFunction
