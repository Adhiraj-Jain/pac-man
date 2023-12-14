# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Initialize values
        self.values = util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Return values for state, action pair
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Initialize empty list for q values
        qValues = []
        # Initialize result to 0.0
        result = 0.0
        # Get all legal actions
        actions = self.getLegalActions(state)
        # Loop through all actions
        for action in actions:
            # Append q value for state, action pair
            qValues.append(self.getQValue(state, action))
        # Check if q values list is not empty
        if len(qValues) > 0:
          # Set result to max q value
          result = max(qValues)
        # Return result
        return result

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialize result to default None
        result = None
        # Get all legal actions
        actions = self.getLegalActions(state)
        # Check if actions list is not empty
        if actions:
          # Get max q-value for the state
          maxValue = self.computeValueFromQValues(state)
          # Initialize empty list for best actions
          bestActions = []
          # Loop through all legal actions
          for action in actions:
            # Check if q-value for state, action pair is equal to max q-value
            if self.getQValue(state, action) == maxValue:
              # Append action to best actions list
              bestActions.append(action)
          # Choose random action from best actions list
          result = random.choice(bestActions)
        # Return result
        return result

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Flip a coin with epsilon probability 
        if (util.flipCoin(self.epsilon)):
            # Choose random action from legal actions
            action = random.choice(legalActions)
        else:
            # Else choose best policy action 
            action = self.getPolicy(state)
        # Return action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Calculating sample value by adding reward and discounted value of next state 
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        # Updating value for state, action pair by adding alpha times the difference between sample and current value to current value
        self.values[(state, action)] += self.alpha * (sample - self.values[(state, action)])

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Initialize q value to default 0.0
        qValue = 0.0
        # Get features for state, action pair
        features = self.featExtractor.getFeatures(state,action)
        # Loop through all features
        for feature in features:
          # Calculate q value by adding feature times weight to q value
            qValue += features[feature] * self.weights[feature]
        # Return q value
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Calculating difference between reward and discounted value of next state minus q value for state, action pair
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        # Get features for state, action pair
        features = self.featExtractor.getFeatures(state, action)
        # Loop through all features
        for feature in features:
          # Update weight for feature by adding alpha times difference times feature to weight for feature
          self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
