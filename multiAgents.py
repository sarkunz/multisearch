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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("GHOSTS", newGhostStates)
        # print("SCARED", newScaredTimes)
        # print("FOOD", newFood)

        foodDists = [util.manhattanDistance(f, newPos) for f in newFood.asList()]
        closestFood = 0 if len(foodDists) == 0 else min(foodDists)

        ghostDists = [util.manhattanDistance(g.getPosition(), newPos) for g in newGhostStates]
        closestGhost = 0 if len(ghostDists) == 0 else min(ghostDists)

        if closestGhost == 0:
            return -10000
        
        #punish stopping
        stop = -5 if action == "Stop" else 0

        return successorGameState.getScore() - (10 - closestGhost) + (10 - closestFood) + stop

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
        #base case: game ended or we've hit depth
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * gameState.getNumAgents():
            return (None, self.evaluationFunction(gameState))
        
        #increment agent index
        next_agent = 0 if agent == gameState.getNumAgents() - 1 else agent + 1

        scores = []
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.calcAction(successor, curDepth + 1, next_agent)
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
        #base case: game ended or we've hit depth
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * gameState.getNumAgents():
            return (None, self.evaluationFunction(gameState))
        
        #increment agent index
        next_agent = 0 if agent == gameState.getNumAgents() - 1 else agent + 1

        scores = []
        bestAct = gameState.getLegalActions()[0]
        bestScore = 0
        t_alpha = -math.inf
        t_beta = math.inf
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.calcAction(successor, curDepth + 1, next_agent, alpha, beta)
            
            #eval score against alpha beta
            if agent == 0: #pacman
                if res_score > t_alpha:
                    t_alpha = res_score
                    bestAct = action
                if t_alpha > beta: return (bestAct, res_score) #prune
                if t_alpha > alpha:
                    alpha = t_alpha
                    bestAct = res_act
                    bestScore = t_alpha
            else: #ghosts
                if res_score < t_beta:
                    t_beta = res_score
                    bestAct = action
                if t_beta < alpha: return (bestAct, res_score) #prune
                if t_beta < beta:
                    beta = t_beta
                    bestAct = res_act
                    bestScore = t_beta

            # scores.append(res_score)
        #bestScore, actInd = self.getBestScore(agent, scores)
        # print(bestAct, bestScore)
        return (bestAct, bestScore)
    
        

    # def getBestScore(self, agent, scores):
        if agent == 0: #pacman
            return max(scores), scores.index(max(scores)) #should return the index of the action taken, since they're always the same order
        return min(scores), scores.index(min(scores)) #ghosts


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction