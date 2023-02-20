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
import math

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
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodDists = [util.manhattanDistance(f, newPos) for f in newFood.asList()]
        closestFood = 1 if len(foodDists) == 0 else min(foodDists)

        ghostDists = [util.manhattanDistance(g.getPosition(), newPos) for g in newGhostStates]
        closestGhost = 1 if len(ghostDists) == 0 else min(ghostDists)

        if closestGhost == 0:
            return -10000
        
        #punish stopping
        stop = -2 if action == "Stop" else 0

        return successorGameState.getScore() - (1/closestGhost) + (1/closestFood) + stop

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
        return self.calcAction(gameState, 0, 0)[0] #action, score
    
    def calcAction(self, gameState, curDepth, agent):
        numAgents = gameState.getNumAgents()
        #base case: game ended or we've hit depth
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * numAgents:
            return (None, self.evaluationFunction(gameState))
        
        #increment agent index
        if curDepth != 0:
            agent = 0 if agent == numAgents - 1 else agent + 1  

        scores = []
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.calcAction(successor, curDepth + 1, agent)
            scores.append(res_score)
        
        bestScore, actInd = self.getBestScore(agent, scores)
        return (gameState.getLegalActions()[actInd], bestScore)

    def getBestScore(self, agent, scores):
        if agent == 0: #pacman
            return max(scores), scores.index(max(scores)) #should return the index of the action taken, since they're always the same order
        return min(scores), scores.index(min(scores)) #ghosts

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #init alpha, beta array
            
        return self.calcAction(gameState, 0, 0, -math.inf, math.inf)[0]
    
    def calcAction(self, gameState, curDepth, agent, alpha, beta):
        numAgents = gameState.getNumAgents()
        #base case: game ended or we've hit depth
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * numAgents:
            return (None, self.evaluationFunction(gameState))

        #increment agent index 
        if curDepth != 0:
            agent = 0 if agent == numAgents - 1 else agent + 1            

        t_max =  -math.inf
        t_min =  math.inf
        best_act = ""
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.calcAction(successor, curDepth + 1, agent, alpha, beta)
            
            if agent == 0: #pacman
                v = max(res_score, t_max)
                if v != t_max:
                    t_max = v
                    best_act = action
                if v > beta:
                    return best_act, t_max
                alpha = max(alpha, v)
            
            else: #ghosts
                v = min(res_score, t_min)
                if v != t_min:
                    t_min = v
                    best_act = action
                if v < alpha:
                    return best_act, t_min
                beta = min(beta, v)

        if agent == 0: return best_act, t_max
        else: return best_act, t_min


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
        return self.calcAction(gameState, 0, 0)[0] #action, score
    
    def calcAction(self, gameState, curDepth, agent):
        numAgents = gameState.getNumAgents()
        #base case: game ended or we've hit depth
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * numAgents:
            return (None, self.evaluationFunction(gameState))
        
        #increment agent index
        if curDepth != 0:
            agent = 0 if agent == numAgents - 1 else agent + 1  

        scores = []
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.calcAction(successor, curDepth + 1, agent)
            scores.append(res_score)
        
        bestScore, actInd = self.getBestScore(agent, scores)
        return (gameState.getLegalActions()[actInd], bestScore)

    def getBestScore(self, agent, scores):
        if agent == 0: #pacman
            return max(scores), scores.index(max(scores)) #should return the index of the action taken, since they're always the same order
        else: #ghosts
            score = sum(scores) * (1/len(scores)) #score should be sum(prob_i * score_i)
            return score, random.randint(0,len(scores) - 1) #return a random action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newCapsules = successorGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    foodDists = [util.manhattanDistance(f, newPos) for f in newFood.asList()]
    closestFood = 1 if len(foodDists) == 0 else min(foodDists)

    capsuleDists = [util.manhattanDistance(f, newPos) for f in newCapsules]
    closestCapsule = 1 if len(capsuleDists) == 0 else min(capsuleDists)

    ghostDists = [util.manhattanDistance(g.getPosition(), newPos) for g in newGhostStates]
    badGhosts = []
    for ind, gd in enumerate(ghostDists):
        if gd > newScaredTimes[ind]: #if we have enough scared time to eat it, we're good
            badGhosts.append(gd)
    closestGhost = 1 if len(badGhosts) == 0 else min(badGhosts)

    if closestGhost == 0:
        return -10000

    return successorGameState.getScore() - (1/closestGhost) + (1/closestFood) + (1/closestCapsule)


# Abbreviation
better = betterEvaluationFunction
