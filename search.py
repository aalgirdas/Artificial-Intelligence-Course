"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque

from utils import *

import math
import decimal

import random

import maps

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________


class SimpleProblemSolvingAgentProgram:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError


# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None







def random_search(problem, step_limits = -1):

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    step_num = 0

    while True:
        step_num = step_num + 1
        if step_num >= step_limits > 0:  # its for debug
            return node

        nodes_list = node.expand(problem)
        child = random.choice(nodes_list)
        if problem.goal_test(child.state):
            return child

        node = child

    return None




def breadth_first_graph_search(problem, step_limits = -1):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    step_num = 0
    while frontier:
        step_num = step_num + 1
        node = frontier.popleft()
        if step_limits > 0 and step_num >= step_limits:  # its for debug
            return node

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""

    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Bidirectional Search
# Pseudocode from https://webdocs.cs.ualberta.ca/%7Eholte/Publications/MM-AAAI2016.pdf

def bidirectional_search(problem):
    e = 0
    if isinstance(problem, GraphProblem):
        e = problem.find_min_edge()
    gF, gB = {Node(problem.initial): 0}, {Node(problem.goal): 0}
    openF, openB = [Node(problem.initial)], [Node(problem.goal)]
    closedF, closedB = [], []
    U = np.inf

    def extend(U, open_dir, open_other, g_dir, g_other, closed_dir):
        """Extend search in given direction"""
        n = find_key(C, open_dir, g_dir)

        open_dir.remove(n)
        closed_dir.append(n)

        for c in n.expand(problem):
            if c in open_dir or c in closed_dir:
                if g_dir[c] <= problem.path_cost(g_dir[n], n.state, None, c.state):
                    continue

                open_dir.remove(c)

            g_dir[c] = problem.path_cost(g_dir[n], n.state, None, c.state)
            open_dir.append(c)

            if c in open_other:
                U = min(U, g_dir[c] + g_other[c])

        return U, open_dir, closed_dir, g_dir

    def find_min(open_dir, g):
        """Finds minimum priority, g and f values in open_dir"""
        # pr_min_f isn't forward pr_min instead it's the f-value
        # of node with priority pr_min.
        pr_min, pr_min_f = np.inf, np.inf
        for n in open_dir:
            f = g[n] + problem.h(n)
            pr = max(f, 2 * g[n])
            pr_min = min(pr_min, pr)
            pr_min_f = min(pr_min_f, f)

        return pr_min, pr_min_f, min(g.values())

    def find_key(pr_min, open_dir, g):
        """Finds key in open_dir with value equal to pr_min
        and minimum g value."""
        m = np.inf
        node = Node(-1)
        for n in open_dir:
            pr = max(g[n] + problem.h(n), 2 * g[n])
            if pr == pr_min:
                if g[n] < m:
                    m = g[n]
                    node = n

        return node

    while openF and openB:
        pr_min_f, f_min_f, g_min_f = find_min(openF, gF)
        pr_min_b, f_min_b, g_min_b = find_min(openB, gB)
        C = min(pr_min_f, pr_min_b)

        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + e):
            return U

        if C == pr_min_f:
            # Extend forward
            U, openF, closedF, gF = extend(U, openF, openB, gF, gB, closedF)
        else:
            # Extend backward
            U, openB, closedB, gB = extend(U, openB, openF, gB, gF, closedB)

    return np.inf


# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics




class Knuth_conject_4(Problem):
    def __init__(self, start_num, goal):
        """ Define goal state and initialize a problem """
        super().__init__(('Start', start_num), goal)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only 3 possible actions
        in any given state of the environment """

        possible_actions = ['!', 'F', 'S']
        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        new_state = state

        if state[1] in [float("-inf"), float("inf")]:
            new_state = ('nan', state[1])
            return new_state

        if action == '!' and state[1] < 2147483647 and state[1] % 1 == 0:
            new_state = ('!', math.factorial(state[1]))
        if action == 'F':
            new_state = ('F', math.floor(state[1]))
        if action == 'S':
            new_state = ('S', math.sqrt( decimal.Decimal( state[1]) ))

        return new_state


    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state[1] == self.goal[1]

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return 1


# ______________________________________________________________________________













class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal
    '''
    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0
    '''
    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))


# ______________________________________________________________________________


class PlanRoute(Problem):
    """ The problem of moving the Hybrid Wumpus Agent from one place to other """

    def __init__(self, initial, goal, allowed, dimrow):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)
        self.dimrow = dimrow
        self.goal = goal
        self.allowed = allowed

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only three possible actions
        in any given state of the environment """

        possible_actions = ['Forward', 'TurnLeft', 'TurnRight']
        x, y = state.get_location()
        orientation = state.get_orientation()

        # Prevent Bumps
        if x == 1 and orientation == 'LEFT':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if y == 1 and orientation == 'DOWN':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if x == self.dimrow and orientation == 'RIGHT':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if y == self.dimrow and orientation == 'UP':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        x, y = state.get_location()
        proposed_loc = list()

        # Move Forward
        if action == 'Forward':
            if state.get_orientation() == 'UP':
                proposed_loc = [x, y + 1]
            elif state.get_orientation() == 'DOWN':
                proposed_loc = [x, y - 1]
            elif state.get_orientation() == 'LEFT':
                proposed_loc = [x - 1, y]
            elif state.get_orientation() == 'RIGHT':
                proposed_loc = [x + 1, y]
            else:
                raise Exception('InvalidOrientation')

        # Rotate counter-clockwise
        elif action == 'TurnLeft':
            if state.get_orientation() == 'UP':
                state.set_orientation('LEFT')
            elif state.get_orientation() == 'DOWN':
                state.set_orientation('RIGHT')
            elif state.get_orientation() == 'LEFT':
                state.set_orientation('DOWN')
            elif state.get_orientation() == 'RIGHT':
                state.set_orientation('UP')
            else:
                raise Exception('InvalidOrientation')

        # Rotate clockwise
        elif action == 'TurnRight':
            if state.get_orientation() == 'UP':
                state.set_orientation('RIGHT')
            elif state.get_orientation() == 'DOWN':
                state.set_orientation('LEFT')
            elif state.get_orientation() == 'LEFT':
                state.set_orientation('UP')
            elif state.get_orientation() == 'RIGHT':
                state.set_orientation('DOWN')
            else:
                raise Exception('InvalidOrientation')

        if proposed_loc in self.allowed:
            state.set_location(proposed_loc[0], [proposed_loc[1]])

        return state

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state.get_location() == tuple(self.goal)

    def h(self, node):
        """ Return the heuristic value for a given state."""

        # Manhattan Heuristic Function
        x1, y1 = node.state.get_location()
        x2, y2 = self.goal

        return abs(x2 - x1) + abs(y2 - y1)


# ______________________________________________________________________________
# Other search algorithms


def recursive_best_first_search(problem, h=None):
    """[Figure 3.26]"""
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (The second value is immaterial)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, np.inf
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = np.inf
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, np.inf)
    return result


def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice


def simulated_annealing_full(problem, schedule=exp_schedule()):
    """ This version returns all the states encountered in reaching
    the goal state."""
    states = []
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        states.append(current.state)
        T = schedule(t)
        if T == 0:
            return states
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice


def and_or_graph_search(problem):
    """[Figure 4.11]Used when the environment is nondeterministic and completely observable.
    Contains OR nodes where the agent is free to choose any action.
    After every action there is an AND node which contains all possible states
    the agent may reach due to stochastic nature of environment.
    The agent must be able to handle all possible states of the AND node (as it
    may end up in any of them).
    Returns a conditional plan to reach goal state,
    or failure if the former is not possible."""

    # functions used by and_or_search
    def or_search(state, problem, path):
        """returns a plan as a list of actions"""
        if problem.goal_test(state):
            return []
        if state in path:
            return None
        for action in problem.actions(state):
            plan = and_search(problem.result(state, action),
                              problem, path + [state, ])
            if plan is not None:
                return [action, plan]

    def and_search(states, problem, path):
        """Returns plan in form of dictionary where we take action plan[s] if we reach state s."""
        plan = {}
        for s in states:
            plan[s] = or_search(s, problem, path)
            if plan[s] is None:
                return None
        return plan

    # body of and or search
    return or_search(problem.initial, problem, [])


# Pre-defined actions for PeakFindingProblem
directions4 = {'W': (-1, 0), 'N': (0, 1), 'E': (1, 0), 'S': (0, -1)}
directions8 = dict(directions4)
directions8.update({'NW': (-1, 1), 'NE': (1, 1), 'SE': (1, -1), 'SW': (-1, -1)})


class PeakFindingProblem(Problem):
    """Problem of finding the highest peak in a limited grid"""

    def __init__(self, initial, grid, defined_actions=directions4):
        """The grid is a 2 dimensional array/list whose state is specified by tuple of indices"""
        super().__init__(initial)
        self.grid = grid
        self.defined_actions = defined_actions
        self.n = len(grid)
        assert self.n > 0
        self.m = len(grid[0])
        assert self.m > 0

    def actions(self, state):
        """Returns the list of actions which are allowed to be taken from the given state"""
        allowed_actions = []
        for action in self.defined_actions:
            next_state = vector_add(state, self.defined_actions[action])
            if 0 <= next_state[0] <= self.n - 1 and 0 <= next_state[1] <= self.m - 1:
                allowed_actions.append(action)

        return allowed_actions

    def result(self, state, action):
        """Moves in the direction specified by action"""
        return vector_add(state, self.defined_actions[action])

    def value(self, state):
        """Value of a state is the value it is the index to"""
        x, y = state
        assert 0 <= x < self.n
        assert 0 <= y < self.m
        return self.grid[x][y]


class OnlineDFSAgent:
    """
    [Figure 4.21]
    The abstract class for an OnlineDFSAgent. Override
    update_state method to convert percept to state. While initializing
    the subclass a problem needs to be provided which is an instance of
    a subclass of the Problem class.
    """

    def __init__(self, problem):
        self.problem = problem
        self.s = None
        self.a = None
        self.untried = dict()
        self.unbacktracked = dict()
        self.result = {}

    def __call__(self, percept):
        s1 = self.update_state(percept)
        if self.problem.goal_test(s1):
            self.a = None
        else:
            if s1 not in self.untried.keys():
                self.untried[s1] = self.problem.actions(s1)
            if self.s is not None:
                if s1 != self.result[(self.s, self.a)]:
                    self.result[(self.s, self.a)] = s1
                    self.unbacktracked[s1].insert(0, self.s)
            if len(self.untried[s1]) == 0:
                if len(self.unbacktracked[s1]) == 0:
                    self.a = None
                else:
                    # else a <- an action b such that result[s', b] = POP(unbacktracked[s'])
                    unbacktracked_pop = self.unbacktracked.pop(s1)
                    for (s, b) in self.result.keys():
                        if self.result[(s, b)] == unbacktracked_pop:
                            self.a = b
                            break
            else:
                self.a = self.untried.pop(s1)
        self.s = s1
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type state."""
        return percept


# ______________________________________________________________________________


class OnlineSearchProblem(Problem):
    """
    A problem which is solved by an agent executing
    actions, rather than by just computation.
    Carried in a deterministic and a fully observable environment."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, state):
        return self.graph.graph_dict[state].keys()

    def output(self, state, action):
        return self.graph.graph_dict[state][action]

    def h(self, state):
        """Returns least possible cost to reach a goal for the given state."""
        return self.graph.least_costs[state]

    def c(self, s, a, s1):
        """Returns a cost estimate for an agent to move from state 's' to state 's1'."""
        return 1

    def update_state(self, percept):
        raise NotImplementedError

    def goal_test(self, state):
        if state == self.goal:
            return True
        return False


class LRTAStarAgent:
    """ [Figure 4.24]
    Abstract class for LRTA*-Agent. A problem needs to be
    provided which is an instance of a subclass of Problem Class.

    Takes a OnlineSearchProblem [Figure 4.23] as a problem.
    """

    def __init__(self, problem):
        self.problem = problem
        # self.result = {}      # no need as we are using problem.result
        self.H = {}
        self.s = None
        self.a = None

    def __call__(self, s1):  # as of now s1 is a state rather than a percept
        if self.problem.goal_test(s1):
            self.a = None
            return self.a
        else:
            if s1 not in self.H:
                self.H[s1] = self.problem.h(s1)
            if self.s is not None:
                # self.result[(self.s, self.a)] = s1    # no need as we are using problem.output

                # minimum cost for action b in problem.actions(s)
                self.H[self.s] = min(self.LRTA_cost(self.s, b, self.problem.output(self.s, b),
                                                    self.H) for b in self.problem.actions(self.s))

            # an action b in problem.actions(s1) that minimizes costs
            self.a = min(self.problem.actions(s1),
                         key=lambda b: self.LRTA_cost(s1, b, self.problem.output(s1, b), self.H))

            self.s = s1
            return self.a

    def LRTA_cost(self, s, a, s1, H):
        """Returns cost to move from state 's' to state 's1' plus
        estimated cost to get to goal from s1."""
        print(s, a, s1)
        if s1 is None:
            return self.problem.h(s)
        else:
            # sometimes we need to get H[s1] which we haven't yet added to H
            # to replace this try, except: we can initialize H with values from problem.h
            try:
                return self.problem.c(s, a, s1) + self.H[s1]
            except:
                return self.problem.c(s, a, s1) + self.problem.h(s1)


# ______________________________________________________________________________
# Genetic Algorithm


def genetic_search(problem, ngen=1000, pmut=0.1, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""

    # NOTE: This is not tested and might not work.
    # TODO: Use this function to make Problems work with genetic_algorithm.

    s = problem.initial_state
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)
    return genetic_algorithm(states[:n], problem.value, ngen, pmut)


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """[Figure 4.8]"""
    for i in range(ngen):
        population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
                      for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            return fittest_individual

    return max(population, key=fitness_fn)


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population


def select(r, population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


def recombine(x, y):
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]


def recombine_uniform(x, y):
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]


# _____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.

# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
                curvature=lambda: random.uniform(1.1, 1.5)):
    """Construct a random graph, with the specified nodes, and random links.
    The nodes are laid out randomly on a (width x height) rectangle.
    Then each node is connected to the min_links nearest neighbors.
    Because inverse links are added, some nodes will have more connections.
    The distance between nodes is the hypotenuse times curvature(),
    where curvature() defaults to a random number between 1.1 and 1.5."""
    g = UndirectedGraph()
    g.locations = {}
    # Build the cities
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    # Build roads from each city to at least min_links nearest neighbors.
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]

                def distance_to_node(n):
                    if n is node or g.get(node, n):
                        return np.inf
                    return distance(g.locations[n], here)

                neighbor = min(nodes, key=distance_to_node)
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d))
    return g




def switch_country_map(country = "Lithuania"):
    global romania_map , romania_map_start , romania_map_goal


    if maps.romania_map_start is None:
        if country == "Lithuania":
            maps.romania_map_start = "Druskininkai"
        if country == "Romania":
            maps.romania_map_start = "Arad"
        if country == "India":
            maps.romania_map_start = "Mumbai"
        if country == "Russia":
            maps.romania_map_start = "Moscow"
        if country == "Ukraine":
            maps.romania_map_start = "Kyiv"

    if maps.romania_map_goal is None:
        if country == "Lithuania":
            maps.romania_map_goal = "Nida"
        if country == "Romania":
            maps.romania_map_goal = "Bucharest"
        if country == "India":
            maps.romania_map_goal = "Delhi"
        if country == "Russia":
            maps.romania_map_goal = "Vladivostok"
        if country == "Ukraine":
            maps.romania_map_goal = "Dnipro"

    if maps.romania_map is None:
        if country == "Lithuania":

            maps.romania_map = UndirectedGraph(dict(
                Druskininkai=dict(Lazdijai=37, Alytus=42, Varėna=44),
                Lazdijai=dict(Kalvarija=27, Marijampolė=37),
                Nida=dict(Klaipėda=45, Palanga=68, Šilutė=300),
                Kalvarija=dict(Marijampolė=17, Lazdijai=27),
                Kelmė=dict(Telšiai=57, Kuršėnai=41, Tauragė=58, Šilalė=50, Jurbarkas=62, Raseiniai=30, Šiauliai=40,
                           Radviliškis=42),
                Telšiai=dict(Šilalė=54, Kelmė=57, Kuršėnai=42, Mažeikiai=36, Rietavas=35, Plungė=25),
                Šilalė=dict(Kelmė=50, Tauragė=27, Rietavas=30, Telšiai=54),
                Pagėgiai=dict(Jurbarkas=57, Tauragė=29, Šilalė=44),
                Vilkaviškis=dict(Kalvarija=30, Marijampolė=24, Šakiai=32),
                Jurbarkas=dict(Tauragė=36, Kelmė=62, Raseiniai=40, Šakiai=21),
                Klaipėda=dict(Palanga=23, Kretinga=21, Gargždai=16),  # Kaunas=220  ,Kaunas=100
                Palanga=dict(Kretinga=11, Klaipėda=23),
                Ignalina=dict(Švenčionys=23, Pabradė=46, Visaginas=33, Zarasai=43, Utena=39),
                Švenčionys=dict(Visaginas=54, Pabradė=29, Ignalina=23),
                Visaginas=dict(Ignalina=33, Zarasai=18),
                Pabradė=dict(Švenčionys=29, Vilnius=45, Molėtai=35, Ignalina=46, Utena=58),
                Šalčininkai=dict(Pabradė=78, Vilnius=41, Varėna=53),
                Kuršėnai=dict(Telšiai=42, Naujoji_Akmenė=35, Mažeikiai=50, Kelmė=41, Šiauliai=25),
                Mažeikiai=dict(Kuršėnai=50, Telšiai=36, Skuodas=49, Plungė=53),
                Naujoji_Akmenė=dict(Mažeikiai=33, Joniškis=46, Šiauliai=51, Kuršėnai=35),
                Tauragė=dict(Šilalė=27, Jurbarkas=36, Kelmė=58),
                Šilutė=dict(Klaipėda=45, Gargždai=39, Šilalė=46, Rietavas=50),
                Gargždai=dict(Klaipėda=16, Plungė=37, Kretinga=23, Rietavas=33),
                Zarasai=dict(Ignalina=43, Rokiškis=48, Utena=47),
                Raseiniai=dict(Kelmė=30, Jurbarkas=40, Kėdainiai=54, Radviliškis=53, Domeikava=68, Šakiai=47),
                Plungė=dict(Mažeikiai=53, Telšiai=25, Skuodas=43, Kretinga=37, Gargždai=37, Rietavas=21),
                Skuodas=dict(Plungė=43, Palanga=48, Kretinga=45),
                Kretinga=dict(Skuodas=45, Plungė=37, Gargždai=23, Palanga=11, Klaipėda=21),
                Marijampolė=dict(Kazlų_Rūda=23, Kalvarija=17, Lazdijai=37, Prienai=38),
                Kazlų_Rūda=dict(Šakiai=36, Marijampolė=23, Prienai=31, Garliava=25, Domeikava=36),
                Šakiai=dict(Raseiniai=47, Jurbarkas=21, Kazlų_Rūda=36, Domeikava=55),
                Radviliškis=dict(Raseiniai=53, Kelmė=42, Kėdainiai=63, Šiauliai=20, Panevėžys=51, Pakruojis=27),
                Kėdainiai=dict(Radviliškis=63, Panevėžys=55, Ukmergė=49, Jonava=30, Domeikava=35, Raseiniai=54),
                Pasvalys=dict(Biržai=26, Pakruojis=35, Joniškis=52, Kupiškis=43, Panevėžys=37),
                Biržai=dict(Joniškis=70, Kupiškis=42, Pasvalys=26),
                Joniškis=dict(Pasvalys=52, Biržai=70, Naujoji_Akmenė=46, Pakruojis=32, Šiauliai=39),
                Vilnius=dict(Lentvaris=15, Pabradė=45, Molėtai=61, Širvintos=45),  # , Kaunas=110, Kaunas=70
                Lentvaris=dict(Vilnius=15, Trakai=7),
                Molėtai=dict(Vilnius=61, Utena=31, Pabradė=35, Anykščiai=38, Ukmergė=42, Širvintos=36),
                Rietavas=dict(Plungė=21, Gargždai=33, Šilalė=30, Telšiai=35),
                Likiškiai=dict(Alytus=36, Prienai=26, Birštonas=23),
                Panevėžys=dict(Radviliškis=51, Ukmergė=56, Kėdainiai=55, Pasvalys=37, Pakruojis=42, Anykščiai=51, Kupiškis=39),
                Šiauliai=dict(Kuršėnai=25, Kelmė=40, Joniškis=39, Naujoji_Akmenė=51, Pakruojis=34, Radviliškis=20),
                Anykščiai=dict(Utena=31, Molėtai=38, Ukmergė=37, Panevėžys=51, Kupiškis=34, Rokiškis=55),
                Utena=dict(Rokiškis=51, Anykščiai=31, Zarasai=47, Ignalina=39, Pabradė=58, Molėtai=31),
                Rokiškis=dict(Anykščiai=55, Utena=51, Biržai=57, Kupiškis=40),
                Alytus=dict(Birštonas=22, Elektrėnai=58, Varėna=39),
                Birštonas=dict(Prienai=6, Kaišiadorys=40, Garliava=25, Kaunas=33, Elektrėnai=45),
                Prienai=dict(Marijampolė=38, Kazlų_Rūda=31, Birštonas=6, Garliava=20),
                Grigiškės=dict(Trakai=11, Vilnius=12, Elektrėnai=29),
                Ukmergė=dict(Molėtai=42, Anykščiai=37, Panevėžys=56, Širvintos=27, Kaišiadorys=48, Kėdainiai=49, Jonava=36),
                Pakruojis=dict(Panevėžys=42, Radviliškis=27, Pasvalys=35, Joniškis=32, Šiauliai=34),
                Kupiškis=dict(Panevėžys=39, Rokiškis=40, Anykščiai=34, Biržai=42, Pasvalys=43),
                Trakai=dict(Lentvaris=7, Varėna=52, Elektrėnai=24),
                Varėna=dict(Trakai=52, Elektrėnai=64),
                Kaišiadorys=dict(Kaunas=34, Širvintos=36, Jonava=25, Ukmergė=48, Elektrėnai=15, Birštonas=40),
                Kaunas=dict(Birštonas=33, Garliava=10, Kaišiadorys=34, Jonava=29, Domeikava=7),  # , Klaipėda=200
                Širvintos=dict(Molėtai=36, Vilnius=45, Ukmergė=27, Kaišiadorys=36, Elektrėnai=33),
                Garliava=dict(Prienai=20, Birštonas=25, Kazlų_Rūda=25, Domeikava=16, Kaunas=10),
                Jonava=dict(Kaunas=29, Ukmergė=36, Kaišiadorys=25, Kėdainiai=30, Domeikava=25),
                Elektrėnai=dict(Širvintos=33, Varėna=64, Trakai=24, Kaišiadorys=15, Birštonas=45),
                Domeikava=dict(Jonava=25, Kaunas=7, Kėdainiai=35, Kazlų_Rūda=36, Garliava=16, Šakiai=55, Raseiniai=68)
            ))

            maps.romania_map.locations = dict(
                Vilnius=(1261, 258),
                Kaunas=(863, 343),
                Klaipėda=(40, 658),
                Šiauliai=(681, 744),
                Panevėžys=(990, 665),
                Alytus=(897, 148),
                Marijampolė=(692, 209),
                Mažeikiai=(392, 894),
                Utena=(1355, 577),
                Jonava=(965, 410),
                Kėdainiai=(872, 493),
                Telšiai=(366, 766),
                Tauragė=(378, 480),
                Ukmergė=(1103, 486),
                Visaginas=(1600, 616),
                Plungė=(248, 740),
                Kretinga=(71, 731),
                Šilutė=(140, 519),
                Palanga=(17, 740),
                Radviliškis=(750, 694),
                Rokiškis=(1348, 758),
                Druskininkai=(874, 0),
                Elektrėnai=(1077, 299),
                Biržai=(1103, 851),
                Jurbarkas=(521, 412),
                Vilkaviškis=(597, 252),
                Joniškis=(770, 868),
                Anykščiai=(1209, 591),
                Varėna=(1051, 74),
                Prienai=(865, 239),
                Kelmė=(568, 629),
                Naujoji_Akmenė=(553, 900),
                Šalčininkai=(1289, 113),
                Pasvalys=(1001, 796),
                Zarasai=(1545, 667),
                Kupiškis=(1167, 707),
                Kazlų_Rūda=(731, 284),
                Molėtai=(1300, 473),
                Skuodas=(155, 877),
                Šakiai=(601, 365),
                Šilalė=(345, 574),
                Trakai=(1157, 239),
                Pakruojis=(840, 765),
                Švenčionys=(1518, 434),
                Kalvarija=(655, 153),
                Lazdijai=(740, 82),
                Rietavas=(271, 665),
                Nida=(0, 501),
                Birštonas=(891, 228),
                Širvintos=(1160, 400),
                Pagėgiai=(257, 434),
                Kaišiadorys=(1020, 329),
                Raseiniai=(624, 530),
                Ignalina=(1519, 515),
                Gargždai=(116, 655),
                Kuršėnai=(569, 774),
                Grigiškės=(1202, 258),
                Likiškiai=(881, 146),
                Lentvaris=(1192, 243),
                Garliava=(843, 310),
                Pabradė=(1403, 375),
                Domeikava=(858, 369)
            )
        if country == "Romania":
            maps.romania_map = UndirectedGraph(dict(
                Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
                Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
                Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
                Drobeta=dict(Mehadia=75),
                Eforie=dict(Hirsova=86),
                Fagaras=dict(Sibiu=99),
                Hirsova=dict(Urziceni=98),
                Iasi=dict(Vaslui=92, Neamt=87),
                Lugoj=dict(Timisoara=111, Mehadia=70),
                Oradea=dict(Zerind=71, Sibiu=151),
                Pitesti=dict(Rimnicu=97),
                Rimnicu=dict(Sibiu=80),
                Urziceni=dict(Vaslui=142)))
            maps.romania_map.locations = dict(
                Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
                Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
                Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
                Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
                Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
                Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
                Vaslui=(509, 444), Zerind=(108, 531))

        if country == "India":
            maps.romania_map = UndirectedGraph(dict(
                Quilon=dict(Mangalore=480, Calicut=277, Kochi=124, Thiruvananthapuram=54, Coimbatore=238, Rājapālaiyam=124),
                Mumbai=dict(Jūnāgadh=369, Bhāvnagar=308, Sūrat=232, Kolhāpur=300, Mangalore=722, Thāne=14, Pune=120, Ulhāsnagar=32),
                Mangalore=dict(Mumbai=722, Calicut=204, Quilon=480, Shimoga=138, Hubli=278, Kolhāpur=431, Belgaum=335),
                Jāmnagar=dict(Rājkot=77, Jūnāgadh=113),
                Rājkot=dict(Bhuj=146, Jāmnagar=77, Jodhpur=498, Ahmedabad=200, Bhāvnagar=150, Jūnāgadh=92),
                Bhuj=dict(Jāmnagar=90, Rājkot=146, Bīkaner=636, Jodhpur=469),
                Jūnāgadh=dict(Rājkot=92, Bhāvnagar=174, Mumbai=369),
                Āīzawl=dict(Brahmapur=954, Imphāl=173, Guwāhāti=288, Agartala=147, Bhubaneshwar=807, Kolkāta=463),
                Brahmapur=dict(Chennai=845, Raurkela=325, Sambalpur=254, Vishākhapatnam=235, Bhubaneshwar=150, Āīzawl=954),
                Guwāhāti=dict(Āīzawl=288, Muzaffarpur=635, Agartala=264, Purnea=429),
                Imphāl=dict(Guwāhāti=267, Āīzawl=173),
                Chennai=dict(Tuticorin=532, Brahmapur=845, Vishākhapatnam=611, Kākināda=476, Trichinopoly=304, Cuddapah=220, Bezwāda=383, Rājahmundry=462, Guntūr=358, Nellore=153),
                Handwāra=dict(Srīnagar=59, Jammu=192, Amritsar=313, Bīkaner=715),
                Jodhpur=dict(Bhuj=469, Rājkot=498, Ahmedabad=365, Udaipur=201, Ajmer=161, Bīkaner=193),
                Ahmedabad=dict(Jodhpur=365, Rājkot=200, Udaipur=205, Vadodara=103, Bhāvnagar=147),
                Bhāvnagar=dict(Ahmedabad=147, Rājkot=150, Jūnāgadh=174, Mumbai=308, Vadodara=124, Sūrat=97),
                Muzaffarpur=dict(Purnea=212, Guwāhāti=635, Dehra_Dūn=858, Gorakhpur=213, Patna=61, Begusarai=107),
                Purnea=dict(Guwāhāti=429, Agartala=440, Bhātpāra=337, Muzaffarpur=212, Barddhamān=283, Durgāpur=248, Begusarai=140, Bhāgalpur=74),
                Gorakhpur=dict(Muzaffarpur=213, Shāhjānpur=363, Dehra_Dūn=654, Lucknow=240, Vārānasi=164, Gaya=278, Patna=218, Allahābād=210, Mirzāpur=195),
                Dehra_Dūn=dict(Gorakhpur=654, Muzaffarpur=858, Shāhjānpur=326, Bareilly=255, Srīnagar=517, Meerut=151, Sahāranpur=60, Chandīgarh=127, Rāmpur=195, Morādābād=178),
                Srīnagar=dict(Dehra_Dūn=517, Handwāra=59, Jammu=151, Chandīgarh=417, Jalandhar=315),
                Shāhjānpur=dict(Lucknow=154, Gorakhpur=363, Cawnpore=162, Dehra_Dūn=326, Bareilly=72, Etāwah=150),
                Lucknow=dict(Gorakhpur=240, Shāhjānpur=154, Allahābād=179, Cawnpore=74, Sannai=299),
                Cawnpore=dict(Lucknow=74, Jhānsi=209, Jabalpur=369, Sannai=261, Shāhjānpur=162, Etāwah=135),
                Allahābād=dict(Gorakhpur=210, Bilāspur=367, Mirzāpur=80, Lucknow=179, Sannai=176),
                Agartala=dict(Āīzawl=147, Guwāhāti=264, Purnea=440, Kolkāta=328, Bārāsat=310, Bhātpāra=310),
                Bhātpāra=dict(Agartala=310, Purnea=337, Bārāsat=17, Barddhamān=71, Kāmārhāti=22, Shrīrāmpur=15),
                Kolkāta=dict(Āīzawl=463, Agartala=328, Bhubaneshwar=367, Cuttack=347, Hāora=3, Bārāsat=20, Kāmārhāti=10),
                Raurkela=dict(Sambalpur=127, Gaya=278, Cuttack=223, Rānchi=131, Jamshedpur=146, Brahmapur=325, Bhubaneshwar=241),
                Sambalpur=dict(Brahmapur=254, Raurkela=127, Vishākhapatnam=421, Vārānasi=438, Gaya=380, Bilāspur=215, Raipur=243),
                Gaya=dict(Sambalpur=380, Gorakhpur=278, Vārānasi=211, Patna=96, Begusarai=134, Raurkela=278, Rānchi=158),
                Vishākhapatnam=dict(Brahmapur=235, Chennai=611, Kākināda=146, Raipur=427, Sambalpur=421),
                Kākināda=dict(Chennai=476, Rājahmundry=46, Vishākhapatnam=146, Raipur=482),
                Vārānasi=dict(Bilāspur=366, Gaya=211, Sambalpur=438, Mirzāpur=47, Gorakhpur=164),
                Bilāspur=dict(Sambalpur=215, Raipur=108, Mirzāpur=338, Vārānasi=366, Allahābād=367, Sannai=254, Drug=131, Bhilai=119),
                Gulbarga=dict(Bijāpur=131, Lātūr=121, Solāpur=104, Hospet=234, Rāichūr=138, Nānded=210, Nizāmābād=200),
                Bijāpur=dict(Hospet=188, Gulbarga=131, Solāpur=96, Hubli=174, Sāngli=122),
                Hospet=dict(Gulbarga=234, Rāichūr=146, Bellary=56, Bijāpur=188, Hubli=137, Davangere=103),
                Tuticorin=dict(Tinnevelly=47, Chennai=532, Rājapālaiyam=93, Trichinopoly=235, Madurai=126),
                Tinnevelly=dict(Thiruvananthapuram=92, Tuticorin=47, Rājapālaiyam=78),
                Thiruvananthapuram=dict(Tuticorin=139, Rājapālaiyam=126, Tinnevelly=92),
                Bharatpur=dict(Bhopāl=441, Alwar=94, Kota=280, Farīdābād=136, Gwalior=131, Mathura=36, Āgra=51),
                Bhopāl=dict(Gwalior=338, Bharatpur=441, Kota=268, Akola=283, Jhānsi=270, Ujjain=167, Indore=171, Saugor=146, Amrāvati=259),
                Gwalior=dict(Bharatpur=131, Bhopāl=338, Āgra=107, Jhānsi=93, Fīrozābād=105, Etāwah=103),
                Udaipur=dict(Jodhpur=201, Bhīlwāra=128, Ajmer=230, Ratlām=198, Vadodara=258, Ahmedabad=205),
                Kota=dict(Bharatpur=280, Bhopāl=268, Ujjain=222, Ajmer=186, Alwar=276, Jaipur=193, Bhīlwāra=121, Ratlām=221),
                Alwar=dict(Kota=276, Bharatpur=94, Jaipur=103, Hisar=195, Rohtak=148, Farīdābād=118, New_Delhi=130),
                Ujjain=dict(Bhopāl=167, Kota=222, Indore=51, Ratlām=74),
                Jammu=dict(Srīnagar=151, Jalandhar=170, Amritsar=123, Handwāra=192),
                Jalandhar=dict(Srīnagar=315, Chandīgarh=132, Jammu=170, Amritsar=74, Ludhiāna=53, Bhatinda=137),
                Sūrat=dict(Bhāvnagar=97, Nāsik=163, Vadodara=131, Bhiwandi=209, Mumbai=232, Thāne=221),
                Rājahmundry=dict(Chennai=462, Kākināda=46, Bezwāda=134, Drug=470, Bhilai=472, Raipur=473),
                Cuddapah=dict(Hyderābād=323, Guntūr=268, Warangal=399, Trichinopoly=404, Chennai=220, Nellore=123, Kurnool=172, Anantapur=133, Bangalore=211, Salem=321),
                Hyderābād=dict(Kurnool=177, Cuddapah=323, Rāichūr=174, Warangal=137, Karīmnagar=138, Nizāmābād=150),
                Kurnool=dict(Cuddapah=172, Hyderābād=177, Anantapur=134, Rāichūr=82),
                Patna=dict(Gorakhpur=218, Muzaffarpur=61, Gaya=96, Begusarai=101),
                Begusarai=dict(Patna=101, Muzaffarpur=107, Purnea=140, Gaya=134, Rānchi=243, Bhāgalpur=91, Dhanbād=183),
                Bareilly=dict(Shāhjānpur=72, Etāwah=179, Rāmpur=60, Dehra_Dūn=255),
                Barddhamān=dict(Durgāpur=63, Bhātpāra=71, Purnea=283, Jamshedpur=177, Shrīrāmpur=74, Cuttack=370, Bāli=83, Hāora=89),
                Durgāpur=dict(Jamshedpur=143, Barddhamān=63, Āsansol=38, Purnea=248, Bhāgalpur=191),
                Jamshedpur=dict(Barddhamān=177, Raurkela=146, Cuttack=261, Rānchi=106, Durgāpur=143, Dhanbād=113, Āsansol=126),
                Bārāsat=dict(Kolkāta=20, Agartala=310, Bhātpāra=17, Kāmārhāti=12),
                Ajmer=dict(Udaipur=230, Jodhpur=161, Bhīlwāra=124, Jaipur=131, Kota=186, Bīkaner=216, Sīkar=136),
                Bhīlwāra=dict(Ajmer=124, Kota=121, Ratlām=230, Udaipur=128),
                Cuttack=dict(Jamshedpur=261, Barddhamān=370, Raurkela=223, Bhubaneshwar=22, Kolkāta=347, Hāora=344),
                Rānchi=dict(Gaya=158, Raurkela=131, Begusarai=243, Jamshedpur=106, Dhanbād=122),
                Raipur=dict(Kākināda=482, Rājahmundry=473, Sambalpur=243, Vishākhapatnam=427, Bhilai=21, Bilāspur=108),
                Mirzāpur=dict(Allahābād=80, Bilāspur=338, Gorakhpur=195, Vārānasi=47),
                Warangal=dict(Cuddapah=399, Hyderābād=137, Drug=398, Rāmgundam=93, Karīmnagar=69, Guntūr=207, Bezwāda=194),
                Guntūr=dict(Warangal=207, Chennai=358, Bezwāda=29, Nellore=213, Cuddapah=268),
                Drug=dict(Rāmgundam=327, Bilāspur=131, Sannai=333, Jabalpur=259, Bezwāda=524, Warangal=398, Rājahmundry=470, Bhilai=16, Chānda=247, Nāgpur=227),
                Rāmgundam=dict(Warangal=93, Karīmnagar=51, Chānda=128, Drug=327),
                Solāpur=dict(Gulbarga=104, Bijāpur=96, Lātūr=106, Aurangābād=253, Sāngli=168, Ahmadnagar=200),
                Lātūr=dict(Solāpur=106, Aurangābād=210, Parbhani=98, Nānded=115, Gulbarga=121),
                Aurangābād=dict(Solāpur=253, Ahmadnagar=107, Lātūr=210, Dhūlia=126, Mālegaon=109, Parbhani=165, Jalgaon=128),
                Rāichūr=dict(Kurnool=82, Gulbarga=138, Hospet=146, Anantapur=170, Bellary=126, Hyderābād=174, Nizāmābād=285),
                Bellary=dict(Rāichūr=126, Anantapur=90, Tumkūr=202, Hospet=56, Davangere=131),
                Trichinopoly=dict(Tuticorin=235, Chennai=304, Madurai=118, Salem=108, Cuddapah=404),
                Calicut=dict(Shimoga=297, Mangalore=204, Quilon=277, Mysore=152, Kochi=153, Trichūr=94),
                Shimoga=dict(Mangalore=138, Hubli=167, Calicut=297, Davangere=71, Mysore=214),
                Kochi=dict(Calicut=153, Trichūr=62, Coimbatore=137, Quilon=124),
                Farīdābād=dict(Alwar=118, Bharatpur=136, Mathura=109, Hāpur=55, Ghāziābād=27, New_Delhi=22, Delhi=27),
                Chandīgarh=dict(Dehra_Dūn=127, Srīnagar=417, Sahāranpur=112, Pānīpat=151, Patiāla=60, Jalandhar=132, Ludhiāna=92),
                Ahmadnagar=dict(Solāpur=200, Nāsik=142, Sāngli=247, Pune=111, Mālegaon=164, Aurangābād=107),
                Jaipur=dict(Kota=193, Sīkar=105, Ajmer=131, Alwar=103, Hisar=248),
                Sīkar=dict(Ajmer=136, Bhatinda=289, Bīkaner=184, Jaipur=105, Hisar=180),
                Ratlām=dict(Kota=221, Ujjain=74, Bhīlwāra=230, Vadodara=222, Udaipur=198, Indore=103, Dhūlia=270, Jalgaon=260),
                Parbhani=dict(Aurangābād=165, Nānded=57, Lātūr=98, Jalgaon=230, Akola=164),
                Akola=dict(Amrāvati=81, Bhopāl=283, Nānded=176, Parbhani=164, Indore=251, Jalgaon=152),
                Amrāvati=dict(Bhopāl=259, Saugor=336, Nāgpur=140, Nānded=201, Akola=81, Chānda=195, Nizāmābād=254),
                Karīmnagar=dict(Warangal=69, Hyderābād=138, Nizāmābād=113, Chānda=169, Rāmgundam=51),
                Jhānsi=dict(Jabalpur=288, Gwalior=93, Bhopāl=270, Saugor=180, Cawnpore=209, Etāwah=155),
                Jabalpur=dict(Cawnpore=369, Sannai=143, Drug=259, Nāgpur=240, Jhānsi=288, Saugor=144),
                Sannai=dict(Cawnpore=261, Lucknow=299, Allahābād=176, Bilāspur=254, Drug=333, Jabalpur=143),
                Nāgpur=dict(Drug=227, Jabalpur=240, Saugor=300, Amrāvati=140, Chānda=135),
                Nāsik=dict(Pune=164, Vadodara=262, Sūrat=163, Ahmadnagar=142, Mālegaon=100, Bhiwandi=108, Kalyān=105),
                Pune=dict(Ahmadnagar=111, Sāngli=198, Kolhāpur=206, Mumbai=120, Ulhāsnagar=107, Nāsik=164, Kalyān=109),
                Vadodara=dict(Dhūlia=225, Ratlām=222, Sūrat=131, Udaipur=258, Ahmedabad=103, Bhāvnagar=124, Mālegaon=239, Nāsik=262),
                Dhūlia=dict(Ratlām=270, Mālegaon=45, Vadodara=225, Aurangābād=126, Jalgaon=82),
                Bezwāda=dict(Guntūr=29, Chennai=383, Warangal=194, Rājahmundry=134, Drug=524),
                Nellore=dict(Chennai=153, Cuddapah=123, Guntūr=213),
                Anantapur=dict(Cuddapah=133, Bangalore=189, Kurnool=134, Rāichūr=170, Tumkūr=158, Bellary=90),
                Bangalore=dict(Cuddapah=211, Anantapur=189, Tumkūr=66, Salem=160, Mysore=126, Tiruppūr=209),
                Āsansol=dict(Jamshedpur=126, Dhanbād=55, Durgāpur=38, Bhāgalpur=174),
                Shrīrāmpur=dict(Bhātpāra=15, Barddhamān=74, Kāmārhāti=9, Bāli=11),
                Bhatinda=dict(Bīkaner=291, Amritsar=157, Jalandhar=137, Sīkar=289, Ludhiāna=115, Hisar=140),
                Bīkaner=dict(Sīkar=184, Jodhpur=193, Ajmer=216, Handwāra=715, Bhuj=636, Bhatinda=291, Amritsar=428),
                Bhubaneshwar=dict(Raurkela=241, Cuttack=22, Brahmapur=150, Āīzawl=807, Kolkāta=367),
                Dhanbād=dict(Rānchi=122, Begusarai=183, Jamshedpur=113, Bhāgalpur=172, Āsansol=55),
                Bhilai=dict(Drug=16, Rājahmundry=472, Bilāspur=119, Raipur=21),
                Tumkūr=dict(Bangalore=66, Anantapur=158, Mysore=124, Bellary=202, Davangere=178),
                Rājapālaiyam=dict(Tinnevelly=78, Quilon=124, Thiruvananthapuram=126, Tuticorin=93, Coimbatore=188, Madurai=81, Tiruppūr=189),
                Āgra=dict(Bharatpur=51, Mathura=49, Alīgarh=78, Fīrozābād=37, Gwalior=107),
                Alīgarh=dict(Morādābād=125, Mathura=58, Hāpur=100, Āgra=78, Fīrozābād=86),
                Morādābād=dict(Hāpur=97, Dehra_Dūn=178, Meerut=103, Alīgarh=125, Rāmpur=26, Fīrozābād=191),
                Hāpur=dict(Alīgarh=100, Morādābād=97, Mathura=138, Farīdābād=55, Meerut=28, Ghāziābād=34),
                Meerut=dict(Morādābād=103, Hāpur=28, Dehra_Dūn=151, Sahāranpur=109, Pānīpat=83, Ghāziābād=45),
                Mathura=dict(Farīdābād=109, Bharatpur=36, Āgra=49, Alīgarh=58, Hāpur=138),
                Sahāranpur=dict(Meerut=109, Dehra_Dūn=60, Pānīpat=85, Chandīgarh=112),
                Sāngli=dict(Bijāpur=122, Solāpur=168, Ahmadnagar=247, Hubli=177, Belgaum=111, Pune=198, Ichalkaranji=21, Kolhāpur=40),
                Hubli=dict(Hospet=137, Mangalore=278, Belgaum=87, Bijāpur=174, Sāngli=177, Davangere=131, Shimoga=167),
                Belgaum=dict(Mangalore=335, Hubli=87, Kolhāpur=96, Sāngli=111, Ichalkaranji=92),
                Nizāmābād=dict(Hyderābād=150, Rāichūr=285, Gulbarga=200, Amrāvati=254, Nānded=100, Karīmnagar=113, Chānda=190),
                Indore=dict(Bhopāl=171, Akola=251, Ujjain=51, Jalgaon=191, Ratlām=103),
                Saugor=dict(Jhānsi=180, Jabalpur=144, Nāgpur=300, Bhopāl=146, Amrāvati=336),
                Etāwah=dict(Shāhjānpur=150, Cawnpore=135, Jhānsi=155, Gwalior=103, Fīrozābād=73, Bareilly=179, Rāmpur=222),
                Bhiwandi=dict(Nāsik=108, Sūrat=209, Thāne=17, Kalyān=11, Ulhāsnagar=12),
                Mālegaon=dict(Nāsik=100, Aurangābād=109, Ahmadnagar=164, Dhūlia=45, Vadodara=239),
                Bhāgalpur=dict(Purnea=74, Durgāpur=191, Āsansol=174, Begusarai=91, Dhanbād=172),
                Hāora=dict(Cuttack=344, Barddhamān=89, Bāli=8, Kāmārhāti=11, Kolkāta=3),
                Bāli=dict(Shrīrāmpur=11, Barddhamān=83, Kāmārhāti=3, Hāora=8),
                Kāmārhāti=dict(Bārāsat=12, Bhātpāra=22, Shrīrāmpur=9, Kolkāta=10, Bāli=3, Hāora=11),
                Amritsar=dict(Bīkaner=428, Jalandhar=74, Bhatinda=157, Jammu=123, Handwāra=313),
                Mysore=dict(Shimoga=214, Bangalore=126, Davangere=252, Tumkūr=124, Calicut=152, Trichūr=204, Tiruppūr=153, Coimbatore=149),
                Davangere=dict(Hospet=103, Hubli=131, Bellary=131, Tumkūr=178, Shimoga=71, Mysore=252),
                Madurai=dict(Tuticorin=126, Rājapālaiyam=81, Trichinopoly=118, Salem=192, Tiruppūr=157),
                Pānīpat=dict(Meerut=83, Sahāranpur=85, Chandīgarh=151, Ghāziābād=90, Sonīpat=43, Patiāla=118, Rohtak=66),
                Patiāla=dict(Pānīpat=118, Rohtak=158, Hisar=144, Ludhiāna=83, Chandīgarh=60),
                Hisar=dict(Jaipur=248, Sīkar=180, Bhatinda=140, Rohtak=85, Alwar=195, Patiāla=144, Ludhiāna=195),
                Kolhāpur=dict(Sāngli=40, Pune=206, Mumbai=300, Mangalore=431, Belgaum=96, Ichalkaranji=25),
                Nānded=dict(Nizāmābād=100, Amrāvati=201, Akola=176, Parbhani=57, Lātūr=115, Gulbarga=210),
                Chānda=dict(Nizāmābād=190, Karīmnagar=169, Nāgpur=135, Amrāvati=195, Rāmgundam=128, Drug=247),
                Jalgaon=dict(Akola=152, Aurangābād=128, Parbhani=230, Dhūlia=82, Ratlām=260, Indore=191),
                Fīrozābād=dict(Alīgarh=86, Morādābād=191, Āgra=37, Gwalior=105, Rāmpur=192, Etāwah=73),
                Rāmpur=dict(Etāwah=222, Bareilly=60, Dehra_Dūn=195, Morādābād=26, Fīrozābād=192),
                Thāne=dict(Sūrat=221, Bhiwandi=17, Ulhāsnagar=20, Mumbai=14),
                Ulhāsnagar=dict(Bhiwandi=12, Mumbai=32, Thāne=20, Kalyān=3, Pune=107),
                Trichūr=dict(Calicut=94, Mysore=204, Coimbatore=98, Kochi=62),
                Salem=dict(Madurai=192, Cuddapah=321, Trichinopoly=108, Bangalore=160, Tiruppūr=108),
                Ghāziābād=dict(Meerut=45, Hāpur=34, Farīdābād=27, Pānīpat=90, Delhi=19, Sonīpat=53),
                Sonīpat=dict(Ghāziābād=53, Pānīpat=43, Delhi=41, New_Delhi=46, Rohtak=44),
                Rohtak=dict(Pānīpat=66, Patiāla=158, Sonīpat=44, Hisar=85, Alwar=148, New_Delhi=70),
                Ludhiāna=dict(Hisar=195, Chandīgarh=92, Patiāla=83, Jalandhar=53, Bhatinda=115),
                Ichalkaranji=dict(Belgaum=92, Kolhāpur=25, Sāngli=21),
                Kalyān=dict(Nāsik=105, Bhiwandi=11, Pune=109, Ulhāsnagar=3),
                Coimbatore=dict(Mysore=149, Trichūr=98, Kochi=137, Quilon=238, Tiruppūr=42, Rājapālaiyam=188),
                Delhi=dict(Farīdābād=27, Ghāziābād=19, New_Delhi=5, Sonīpat=41),
                Tiruppūr=dict(Bangalore=209, Mysore=153, Salem=108, Madurai=157, Rājapālaiyam=189, Coimbatore=42),
                New_Delhi=dict(Alwar=130, Farīdābād=22, Delhi=5, Rohtak=70, Sonīpat=46)
            ))


            maps.romania_map.locations = dict(
                Delhi=(491,703),
                Mumbai=(237,375),
                Kolkāta=(1143,494),
                Bangalore=(512,167),
                Chennai=(669,170),
                Hyderābād=(564,316),
                Pune=(294,356),
                Ahmedabad=(219,510),
                Allahābād=(762,593),
                Sūrat=(234,447),
                Lucknow=(709,641),
                Jaipur=(411,643),
                Cawnpore=(673,628),
                Mirzāpur=(804,583),
                Nāgpur=(600,446),
                Ghāziābād=(502,703),
                Vadodara=(255,485),
                Vishākhapatnam=(847,329),
                Indore=(410,500),
                Thāne=(242,379),
                Bhopāl=(502,518),
                Patna=(954,598),
                Bilāspur=(771,480),
                Ludhiāna=(410,779),
                Āgra=(537,652),
                Kalyān=(253,381),
                Madurai=(543,62),
                Jamshedpur=(1015,502),
                Nāsik=(290,407),
                Farīdābād=(496,695),
                Aurangābād=(380,402),
                Rājkot=(115,485),
                Meerut=(519,714),
                Jabalpur=(649,515),
                Dhanbād=(1030,536),
                Vārānasi=(830,588),
                Srīnagar=(349,888),
                Amritsar=(354,804),
                Alīgarh=(541,676),
                Guwāhāti=(1341,618),
                Bhilai=(737,448),
                Hāora=(1141,495),
                Rānchi=(966,521),
                Gwalior=(548,619),
                Bezwāda=(689,287),
                Chandīgarh=(466,773),
                Jodhpur=(245,622),
                Mysore=(458,144),
                Raipur=(749,449),
                Kota=(409,584),
                New_Delhi=(490,701),
                Bareilly=(619,692),
                Coimbatore=(476,99),
                Solāpur=(414,327),
                Trichinopoly=(577,93),
                Hubli=(368,248),
                Jalandhar=(395,794),
                Bhubaneshwar=(994,416),
                Morādābād=(581,709),
                Kolhāpur=(316,294),
                Thiruvananthapuram=(472,13),
                Bhiwandi=(248,383),
                Sahāranpur=(510,747),
                Warangal=(630,337),
                Salem=(546,121),
                Mālegaon=(334,425),
                Kochi=(436,64),
                Gorakhpur=(850,638),
                Shimoga=(394,199),
                Guntūr=(680,280),
                Tiruppūr=(498,103),
                Raurkela=(939,483),
                Mangalore=(354,163),
                Nānded=(495,378),
                Cuttack=(997,422),
                Tumkūr=(484,179),
                Chānda=(612,405),
                Dehra_Dūn=(538,759),
                Durgāpur=(1082,528),
                Āsansol=(1061,532),
                Bhāvnagar=(193,467),
                Nellore=(651,216),
                Amrāvati=(522,438),
                Ajmer=(340,628),
                Tinnevelly=(519,21),
                Bīkaner=(262,681),
                Agartala=(1313,538),
                Ujjain=(406,515),
                Ulhāsnagar=(253,380),
                Jhānsi=(570,593),
                Davangere=(414,217),
                Jammu=(352,842),
                Belgaum=(332,265),
                Gulbarga=(468,315),
                Jāmnagar=(72,491),
                Dhūlia=(348,437),
                Gaya=(947,569),
                Jalgaon=(394,441),
                Kurnool=(538,264),
                Udaipur=(284,563),
                Bellary=(473,241),
                Sāngli=(335,299),
                Tuticorin=(544,23),
                Calicut=(406,107),
                Akola=(478,432),
                Bhāgalpur=(1064,586),
                Quilon=(454,27),
                Sīkar=(369,667),
                Bhātpāra=(1146,505),
                Kākināda=(783,302),
                Bhīlwāra=(339,589),
                Nizāmābād=(542,361),
                Pānihāti=(1143,498),
                Parbhani=(464,382),
                Rohtak=(452,711),
                Lātūr=(453,352),
                Rājapālaiyam=(512,45),
                Ahmadnagar=(345,375),
                Rājahmundry=(758,303),
                Cuddapah=(584,217),
                Muzaffarpur=(968,616),
                Alwar=(455,665),
                Brahmapur=(934,383),
                Kāmārhāti=(1143,498),
                Mathura=(517,663),
                Patiāla=(442,759),
                Saugor=(578,537),
                Bijāpur=(403,298),
                Shāhjānpur=(648,676),
                Jūnāgadh=(96,458),
                Trichūr=(432,83),
                Barddhamān=(1113,518),
                Purnea=(1091,604),
                Sambalpur=(886,457),
                Fīrozābād=(559,651),
                Hisar=(404,719),
                Rāmpur=(596,707),
                Bāli=(1141,497),
                Pānīpat=(476,727),
                Āīzawl=(1398,534),
                Karīmnagar=(604,353),
                Bhuj=(57,518),
                Ichalkaranji=(330,294),
                Hospet=(443,245),
                Bhatinda=(358,756),
                Sannai=(702,549),
                Bārāsat=(1150,500),
                Ratlām=(365,520),
                Drug=(728,447),
                Handwāra=(319,899),
                Imphāl=(1470,571),
                Anantapur=(513,225),
                Etāwah=(596,638),
                Rāichūr=(499,277),
                Bharatpur=(506,653),
                Begusarai=(1012,592),
                Sonīpat=(478,714),
                Shrīrāmpur=(1141,501),
                Hāpur=(522,705),
                Rāmgundam=(621,366)
            )

        if country == "Russia":
            maps.romania_map = UndirectedGraph(dict(
                Yuzhno_Sakhalinsk=dict(Nakhodka=900, Ussuriysk=911, Khabarovsk=598, Komsomol_sk_na_Amure=580),
                Norilsk=dict(Noginsk=554, Novyy_Urengoy=604),
                Novyy_Urengoy=dict(Norilsk=604, Noginsk=697, Surgut=560),
                Noginsk=dict(Norilsk=554, Nizhnevartovsk=841, Novyy_Urengoy=697, Surgut=967),
                Nizhnevartovsk=dict(Seversk=676, Noginsk=841, Tyumen=758, Surgut=173, Nefteyugansk=215),
                Murmansk=dict(Arkhangelsk=589, Petrozavodsk=800, Severodvinsk=571),
                Arkhangelsk=dict(Syktyvkar=607, Petrozavodsk=436, Murmansk=589, Severodvinsk=32),
                Petrozavodsk=dict(Syktyvkar=865, Severodvinsk=414, Arkhangelsk=436, Murmansk=800, Vologda=416, Saint_Petersburg=298),
                Syktyvkar=dict(Arkhangelsk=607, Vologda=656, Petrozavodsk=865, Berezniki=410, Kirov=347),
                Makhachkala=dict(Kaspiysk=16, Vladikavkaz=228, Derbent=121, Astrakhan=374, Khasavyurt=78),
                Kaspiysk=dict(Derbent=105, Vladikavkaz=241, Makhachkala=16),
                Derbent=dict(Makhachkala=121, Vladikavkaz=315, Kaspiysk=105),
                Novorossiysk=dict(Taganrog=293, Krasnodar=102, Sochi=200, Maykop=183),
                Taganrog=dict(Belgorod=410, Novoshakhtinsk=96, Shakhty=112, Novocherkassk=92, Bataysk=65, Rostov=61, Novorossiysk=293, Krasnodar=244),
                Surgut=dict(Novyy_Urengoy=560, Noginsk=967, Nizhnevartovsk=173, Nefteyugansk=47),
                Smolensk=dict(Pskov=407, Tver=334, Kaluga=272, Kaliningrad=740, Bryansk=228, Dolgoprudnyy=367, Obninsk=294, Odintsovo=345),
                Pskov=dict(Kaliningrad=593, Saint_Petersburg=263, Smolensk=407, Kolpino=251, Velikiy_Novgorod=189, Tver=467, Rybinsk=619),
                Kaliningrad=dict(Smolensk=740, Saint_Petersburg=826, Pskov=593, Bryansk=919),
                Severodvinsk=dict(Murmansk=571, Arkhangelsk=32, Petrozavodsk=414),
                Kyzyl=dict(Rubtsovsk=913, Biysk=639, Abakan=301, Angarsk=652, Irkutsk=676, Ulan_Ude=904),
                Ussuriysk=dict(Nakhodka=132, Yuzhno_Sakhalinsk=911, Artëm=53, Vladivostok=76, Khabarovsk=573, Blagoveshchensk=791),
                Nakhodka=dict(Yuzhno_Sakhalinsk=900, Ussuriysk=132, Vladivostok=86, Artëm=82),
                Khabarovsk=dict(Ussuriysk=573, Yuzhno_Sakhalinsk=598, Komsomol_sk_na_Amure=268, Blagoveshchensk=579),
                Komsomol_sk_na_Amure=dict(Yuzhno_Sakhalinsk=580, Khabarovsk=268, Blagoveshchensk=671),
                Astrakhan=dict(Makhachkala=374, Volgodonsk=464, Khasavyurt=361, Pyatigorsk=465, Bataysk=635, Orsk=938, Nevinnomyssk=510, Stavropol=489, Volgograd=372, Volzhskiy=365),
                Krasnoyarsk=dict(Bratsk=543, Abakan=271, Kemerovo=433, Seversk=498, Tomsk=491),
                Seversk=dict(Krasnoyarsk=498, Nizhnevartovsk=676, Tomsk=13, Omsk=738),
                Abakan=dict(Bratsk=705, Krasnoyarsk=271, Kemerovo=389, Biysk=437, Angarsk=842, Kyzyl=301, Prokopyevsk=309, Novokuznetsk=282),
                Bratsk=dict(Krasnoyarsk=543, Abakan=705, Angarsk=425, Chita=897),
                Vladikavkaz=dict(Sochi=405, Derbent=315, Kaspiysk=241, Nalchik=99, Nazran=20, Makhachkala=228, Groznyy=88, Khasavyurt=156),
                Sochi=dict(Derbent=719, Vladikavkaz=405, Nalchik=314, Kislovodsk=243, Maykop=116, Cherkessk=200),
                Novoshakhtinsk=dict(Belgorod=396, Taganrog=96, Shakhty=24, Volgodonsk=169, Volgograd=356),
                Belgorod=dict(Taganrog=410, Kursk=129, Volzhskiy=620, Staryy_Oskol=116, Kamyshin=626, Novoshakhtinsk=396, Volgograd=606),
                Saint_Petersburg=dict(Petrozavodsk=298, Kaliningrad=826, Vologda=545, Cherepovets=437, Kolpino=30, Pskov=263),
                Vologda=dict(Kirov=563, Syktyvkar=656, Petrozavodsk=416, Saint_Petersburg=545, Kostroma=172, Cherepovets=113, Rybinsk=143),
                Kirov=dict(Syktyvkar=347, Yoshkar_Ola=242, Berezniki=418, Izhevsk=287, Perm=390, Kostroma=518, Vologda=563),
                Cherepovets=dict(Vologda=113, Saint_Petersburg=437, Kolpino=416, Velikiy_Novgorod=388, Rybinsk=131),
                Nefteyugansk=dict(Nizhnevartovsk=215, Tyumen=595, Nizhniy_Tagil=794, Surgut=47, Berezniki=889),
                Tyumen=dict(Nizhnevartovsk=758, Nefteyugansk=595, Nizhniy_Tagil=342, Kamensk_Ural_skiy=234, Omsk=544, Kurgan=190, Yekaterinburg=300, Pervouralsk=338),
                Nizhniy_Tagil=dict(Tyumen=342, Nefteyugansk=794, Berezniki=248, Pervouralsk=111, Perm=219),
                Yoshkar_Ola=dict(Izhevsk=325, Neftekamsk=396, Kirov=242, Kostroma=437, Nizhniy_Novgorod=241, Ivanovo=421, Cheboksary=68, Novocheboksarsk=62),
                Izhevsk=dict(Kirov=287, Perm=222, Pervouralsk=408, Neftekamsk=106, Yoshkar_Ola=325),
                Tver=dict(Pskov=467, Rybinsk=218, Smolensk=334, Yaroslavl=250, Ivanovo=308, Vladimir=286, Kovrov=334, Dolgoprudnyy=141, Mytishchi=152, Korolëv=156),
                Artëm=dict(Nakhodka=82, Ussuriysk=53, Vladivostok=34),
                Vladivostok=dict(Ussuriysk=76, Artëm=34, Nakhodka=86),
                Bataysk=dict(Astrakhan=635, Stavropol=288, Rostov=11, Volgodonsk=185, Krasnodar=240, Taganrog=65),
                Volgodonsk=dict(Bataysk=185, Novoshakhtinsk=169, Shakhty=145, Rostov=185, Novocherkassk=154, Astrakhan=464, Volgograd=219),
                Orsk=dict(Astrakhan=938, Magnitogorsk=242, Orenburg=240, Kamyshin=928),
                Magnitogorsk=dict(Orsk=242, Orenburg=320, Ufa=250, Omsk=948, Miass=197, Kopeysk=254, Sterlitamak=205, Salavat=205),
                Orenburg=dict(Orsk=240, Kamyshin=704, Engels=620, Balakovo=502, Samara=372, Magnitogorsk=320, Salavat=186),
                Kamyshin=dict(Orsk=928, Orenburg=704, Volzhskiy=151, Belgorod=626, Staryy_Oskol=549, Engels=161),
                Kemerovo=dict(Abakan=389, Prokopyevsk=166, Novosibirsk=202, Tomsk=146, Krasnoyarsk=433),
                Prokopyevsk=dict(Abakan=309, Kemerovo=166, Novokuznetsk=30, Barnaul=201, Novosibirsk=276),
                Novosibirsk=dict(Prokopyevsk=276, Kemerovo=202, Tomsk=205, Barnaul=194, Omsk=607),
                Biysk=dict(Rubtsovsk=293, Kyzyl=639, Abakan=437, Novokuznetsk=189, Barnaul=131),
                Rubtsovsk=dict(Kyzyl=913, Omsk=645, Biysk=293, Barnaul=269),
                Engels=dict(Kamyshin=161, Orenburg=620, Balakovo=131, Staryy_Oskol=574, Saratov=10, Voronezh=477),
                Balakovo=dict(Engels=131, Saratov=134, Penza=227, Tambov=437, Syzran=133, Orenburg=502, Samara=202),
                Saratov=dict(Engels=10, Balakovo=134, Tambov=338, Voronezh=468),
                Almetyevsk=dict(Arzamas=542, Ufa=234, Oktyabr_skiy=89, Kazan=224, Ulyanovsk=261, Dimitrovgrad=190, Naberezhnyye_Chelny=88, Nizhnekamsk=87),
                Arzamas=dict(Ulyanovsk=315, Saransk=160, Ryazan=273, Almetyevsk=542, Cheboksary=231, Kazan=336, Dzerzhinsk=97, Kolomna=319, Murom=112),
                Ulyanovsk=dict(Almetyevsk=261, Arzamas=315, Saransk=207, Dimitrovgrad=79, Tolyatti=112),
                Berezniki=dict(Nefteyugansk=889, Nizhniy_Tagil=248, Syktyvkar=410, Kirov=418, Perm=159),
                Kaluga=dict(Obninsk=66, Serpukhov=84, Smolensk=272, Bryansk=189, Tula=94, Ryazan=222),
                Obninsk=dict(Smolensk=294, Kaluga=66, Odintsovo=76, Serpukhov=53, Kolomna=137, Domodedovo=80, Podolsk=69),
                Saransk=dict(Novomoskovsk=451, Ulyanovsk=207, Penza=110, Tolyatti=287, Ryazan=357, Arzamas=160),
                Novomoskovsk=dict(Penza=453, Tambov=256, Tula=46, Saransk=451, Ryazan=114, Orël=187, Bryansk=271),
                Penza=dict(Saransk=110, Tambov=244, Novomoskovsk=453, Tolyatti=295, Balakovo=227, Syzran=231),
                Perm=dict(Kirov=390, Berezniki=159, Nizhniy_Tagil=219, Pervouralsk=252, Izhevsk=222),
                Pervouralsk=dict(Tyumen=338, Nizhniy_Tagil=111, Perm=252, Neftekamsk=361, Izhevsk=408, Kamensk_Ural_skiy=134, Yekaterinburg=41),
                Neftekamsk=dict(Kamensk_Ural_skiy=475, Pervouralsk=361, Izhevsk=106, Kazan=321, Naberezhnyye_Chelny=126, Zlatoust=354, Yoshkar_Ola=396, Novocheboksarsk=418),
                Kamensk_Ural_skiy=dict(Pervouralsk=134, Kurgan=237, Tyumen=234, Yekaterinburg=94, Neftekamsk=475, Zlatoust=197, Chelyabinsk=142),
                Kostroma=dict(Yoshkar_Ola=437, Kirov=518, Vologda=172, Rybinsk=127, Yaroslavl=66, Ivanovo=85),
                Angarsk=dict(Bratsk=425, Abakan=842, Kyzyl=652, Chita=653, Irkutsk=40, Ulan_Ude=264),
                Irkutsk=dict(Angarsk=40, Kyzyl=676, Ulan_Ude=231),
                Blagoveshchensk=dict(Khabarovsk=579, Ussuriysk=791, Komsomol_sk_na_Amure=671, Chita=997),
                Nalchik=dict(Sochi=314, Vladikavkaz=99, Kislovodsk=86, Pyatigorsk=76, Groznyy=169, Nazran=97),
                Nazran=dict(Nalchik=97, Groznyy=76, Vladikavkaz=20),
                Khasavyurt=dict(Makhachkala=78, Astrakhan=361, Vladikavkaz=156, Pyatigorsk=297, Groznyy=71),
                Pyatigorsk=dict(Khasavyurt=297, Kislovodsk=29, Nalchik=76, Groznyy=227, Astrakhan=465, Nevinnomyssk=111, Cherkessk=81, Yessentuki=14),
                Stavropol=dict(Astrakhan=489, Nevinnomyssk=46, Bataysk=288, Krasnodar=235, Armavir=68),
                Sterlitamak=dict(Magnitogorsk=205, Ufa=121, Oktyabr_skiy=186, Dimitrovgrad=421, Tolyatti=431, Salavat=29),
                Ufa=dict(Sterlitamak=121, Naberezhnyye_Chelny=253, Almetyevsk=234, Oktyabr_skiy=160, Magnitogorsk=250, Miass=267, Zlatoust=242),
                Volzhskiy=dict(Astrakhan=365, Kamyshin=151, Volgograd=20, Belgorod=620),
                Kurgan=dict(Tyumen=190, Omsk=512, Kopeysk=238, Kamensk_Ural_skiy=237, Chelyabinsk=251),
                Tomsk=dict(Novosibirsk=205, Krasnoyarsk=491, Kemerovo=146, Seversk=13, Omsk=742),
                Novokuznetsk=dict(Abakan=282, Biysk=189, Prokopyevsk=30, Barnaul=224),
                Omsk=dict(Magnitogorsk=948, Tomsk=742, Novosibirsk=607, Seversk=738, Tyumen=544, Kopeysk=749, Kurgan=512, Barnaul=699, Rubtsovsk=645),
                Staryy_Oskol=dict(Kursk=124, Belgorod=116, Kamyshin=549, Engels=574, Voronezh=103),
                Kursk=dict(Belgorod=129, Bryansk=208, Staryy_Oskol=124, Orël=137, Lipetsk=252, Voronezh=208),
                Tambov=dict(Balakovo=437, Saratov=338, Penza=244, Voronezh=190, Novomoskovsk=256, Orël=361, Lipetsk=124),
                Shakhty=dict(Novoshakhtinsk=24, Taganrog=112, Volgodonsk=145, Novocherkassk=32),
                Novocherkassk=dict(Shakhty=32, Volgodonsk=154, Taganrog=92, Rostov=35),
                Rostov=dict(Taganrog=61, Novocherkassk=35, Bataysk=11, Volgodonsk=185),
                Kolpino=dict(Cherepovets=416, Velikiy_Novgorod=139, Saint_Petersburg=30, Pskov=251),
                Velikiy_Novgorod=dict(Cherepovets=388, Kolpino=139, Pskov=189, Rybinsk=444),
                Serpukhov=dict(Obninsk=53, Kaluga=84, Ryazan=152, Kolomna=90),
                Yekaterinburg=dict(Kamensk_Ural_skiy=94, Pervouralsk=41, Tyumen=300),
                Naberezhnyye_Chelny=dict(Almetyevsk=88, Neftekamsk=126, Nizhnekamsk=33, Kazan=201, Ufa=253, Zlatoust=466),
                Rybinsk=dict(Pskov=619, Velikiy_Novgorod=444, Cherepovets=131, Vologda=143, Kostroma=127, Tver=218, Yaroslavl=77),
                Chita=dict(Blagoveshchensk=997, Bratsk=897, Ulan_Ude=404, Angarsk=653),
                Ulan_Ude=dict(Kyzyl=904, Irkutsk=231, Angarsk=264, Chita=404),
                Kislovodsk=dict(Sochi=243, Nalchik=86, Cherkessk=63, Pyatigorsk=29, Yessentuki=18),
                Cherkessk=dict(Sochi=200, Pyatigorsk=81, Maykop=162, Nevinnomyssk=46, Yessentuki=67, Kislovodsk=63),
                Groznyy=dict(Pyatigorsk=227, Khasavyurt=71, Nalchik=169, Nazran=76, Vladikavkaz=88),
                Nevinnomyssk=dict(Pyatigorsk=111, Astrakhan=510, Cherkessk=46, Maykop=146, Stavropol=46, Armavir=76),
                Krasnodar=dict(Stavropol=235, Taganrog=244, Bataysk=240, Novorossiysk=102, Armavir=167, Maykop=99),
                Maykop=dict(Novorossiysk=183, Sochi=116, Cherkessk=162, Krasnodar=99, Armavir=92, Nevinnomyssk=146),
                Oktyabr_skiy=dict(Ufa=160, Sterlitamak=186, Almetyevsk=89, Dimitrovgrad=254),
                Tolyatti=dict(Saransk=287, Ulyanovsk=112, Dimitrovgrad=80, Sterlitamak=431, Penza=295, Syzran=74, Salavat=431, Samara=59),
                Volgograd=dict(Volgodonsk=219, Astrakhan=372, Novoshakhtinsk=356, Belgorod=606, Volzhskiy=20),
                Kopeysk=dict(Magnitogorsk=254, Omsk=749, Miass=96, Chelyabinsk=14, Kurgan=238),
                Barnaul=dict(Novokuznetsk=224, Biysk=131, Prokopyevsk=201, Novosibirsk=194, Rubtsovsk=269, Omsk=699),
                Tula=dict(Bryansk=238, Kaluga=94, Novomoskovsk=46, Ryazan=143),
                Bryansk=dict(Novomoskovsk=271, Smolensk=228, Kaliningrad=919, Kaluga=189, Tula=238, Kursk=208, Orël=117),
                Voronezh=dict(Saratov=468, Engels=477, Staryy_Oskol=103, Kursk=208, Lipetsk=108, Tambov=190),
                Dolgoprudnyy=dict(Tver=141, Smolensk=367, Odintsovo=32, Khimki=5, Mytishchi=14),
                Yaroslavl=dict(Rybinsk=77, Kostroma=66, Tver=250, Ivanovo=97),
                Ryazan=dict(Novomoskovsk=114, Tula=143, Saransk=357, Kaluga=222, Serpukhov=152, Arzamas=273, Kolomna=79),
                Kazan=dict(Arzamas=336, Naberezhnyye_Chelny=201, Almetyevsk=224, Nizhnekamsk=170, Cheboksary=122, Novocheboksarsk=106, Neftekamsk=321),
                Yessentuki=dict(Pyatigorsk=14, Kislovodsk=18, Cherkessk=67),
                Armavir=dict(Stavropol=68, Krasnodar=167, Nevinnomyssk=76, Maykop=92),
                Dimitrovgrad=dict(Almetyevsk=190, Oktyabr_skiy=254, Ulyanovsk=79, Sterlitamak=421, Tolyatti=80),
                Syzran=dict(Penza=231, Tolyatti=74, Samara=109, Balakovo=133),
                Miass=dict(Ufa=267, Magnitogorsk=197, Zlatoust=30, Kopeysk=96, Chelyabinsk=83),
                Orël=dict(Bryansk=117, Tambov=361, Novomoskovsk=187, Kursk=137, Lipetsk=240),
                Ivanovo=dict(Yaroslavl=97, Kostroma=85, Yoshkar_Ola=421, Tver=308, Nizhniy_Novgorod=198, Kovrov=73),
                Nizhnekamsk=dict(Almetyevsk=87, Kazan=170, Naberezhnyye_Chelny=33),
                Cheboksary=dict(Dzerzhinsk=235, Arzamas=231, Yoshkar_Ola=68, Nizhniy_Novgorod=201, Novocheboksarsk=15, Kazan=122),
                Dzerzhinsk=dict(Arzamas=97, Murom=114, Vladimir=188, Kovrov=132, Cheboksary=235, Nizhniy_Novgorod=35),
                Samara=dict(Tolyatti=59, Balakovo=202, Syzran=109, Salavat=387, Orenburg=372),
                Zlatoust=dict(Ufa=242, Naberezhnyye_Chelny=466, Neftekamsk=354, Kamensk_Ural_skiy=197, Miass=30, Chelyabinsk=110),
                Lipetsk=dict(Orël=240, Tambov=124, Kursk=252, Voronezh=108),
                Murom=dict(Arzamas=112, Dzerzhinsk=114, Vladimir=119, Orekhovo_Zuyevo=194, Kolomna=213, Ramenskoye=240),
                Vladimir=dict(Murom=119, Orekhovo_Zuyevo=96, Dzerzhinsk=188, Kovrov=62, Shchelkovo=151, Tver=286, Korolëv=162),
                Orekhovo_Zuyevo=dict(Murom=194, Ramenskoye=52, Vladimir=96, Elektrostal=32, Shchelkovo=61),
                Ramenskoye=dict(Murom=240, Kolomna=64, Podolsk=45, Orekhovo_Zuyevo=52, Domodedovo=33, Elektrostal=29, Zhukovskiy=8),
                Odintsovo=dict(Smolensk=345, Dolgoprudnyy=32, Khimki=26, Krasnogorsk=16, Domodedovo=40, Obninsk=76, Orekhovo_Borisovo_Yuzhnoye=29, Moscow=23, Lyubertsy=38),
                Kovrov=dict(Vladimir=62, Tver=334, Ivanovo=73, Dzerzhinsk=132, Nizhniy_Novgorod=165),
                Nizhniy_Novgorod=dict(Cheboksary=201, Dzerzhinsk=35, Yoshkar_Ola=241, Kovrov=165, Ivanovo=198),
                Novocheboksarsk=dict(Yoshkar_Ola=62, Cheboksary=15, Neftekamsk=418, Kazan=106),
                Salavat=dict(Sterlitamak=29, Tolyatti=431, Magnitogorsk=205, Orenburg=186, Samara=387),
                Chelyabinsk=dict(Miass=83, Zlatoust=110, Kamensk_Ural_skiy=142, Kurgan=251, Kopeysk=14),
                Kolomna=dict(Ryazan=79, Arzamas=319, Serpukhov=90, Murom=213, Obninsk=137, Podolsk=87, Ramenskoye=64),
                Podolsk=dict(Obninsk=69, Kolomna=87, Domodedovo=12, Ramenskoye=45),
                Khimki=dict(Dolgoprudnyy=5, Odintsovo=26, Mytishchi=18, Korolëv=23, Krasnogorsk=10),
                Shchelkovo=dict(Orekhovo_Zuyevo=61, Elektrostal=30, Balashikha=13, Korolëv=11, Vladimir=151),
                Elektrostal=dict(Ramenskoye=29, Orekhovo_Zuyevo=32, Zhukovskiy=30, Reutov=36, Lyubertsy=37, Balashikha=31, Shchelkovo=30),
                Krasnogorsk=dict(Khimki=10, Odintsovo=16, Moscow=19, Balashikha=38, Korolëv=32),
                Domodedovo=dict(Orekhovo_Borisovo_Yuzhnoye=18, Odintsovo=40, Obninsk=80, Podolsk=12, Zhukovskiy=29, Ramenskoye=33),
                Orekhovo_Borisovo_Yuzhnoye=dict(Odintsovo=29, Lyubertsy=13, Zhukovskiy=24, Domodedovo=18),
                Zhukovskiy=dict(Ramenskoye=8, Lyubertsy=16, Elektrostal=30, Orekhovo_Borisovo_Yuzhnoye=24, Domodedovo=29),
                Lyubertsy=dict(Odintsovo=38, Orekhovo_Borisovo_Yuzhnoye=13, Elektrostal=37, Zhukovskiy=16, Moscow=19, Reutov=9),
                Reutov=dict(Lyubertsy=9, Moscow=15, Elektrostal=36, Balashikha=6),
                Mytishchi=dict(Dolgoprudnyy=14, Tver=152, Khimki=18, Korolëv=5),
                Moscow=dict(Krasnogorsk=19, Odintsovo=23, Lyubertsy=19, Reutov=15, Balashikha=21),
                Balashikha=dict(Reutov=6, Elektrostal=31, Moscow=21, Krasnogorsk=38, Shchelkovo=13, Korolëv=15),
                Korolëv=dict(Mytishchi=5, Balashikha=15, Tver=156, Vladimir=162, Shchelkovo=11, Krasnogorsk=32, Khimki=23)
            ))


            maps.romania_map.locations = dict(
                Moscow=(968,384),
                Saint_Petersburg=(935,497),
                Novosibirsk=(1170,365),
                Yekaterinburg=(1070,413),
                Nizhniy_Novgorod=(996,400),
                Kazan=(1019,385),
                Chelyabinsk=(1074,368),
                Omsk=(1127,363),
                Samara=(1024,315),
                Rostov=(977,155),
                Ufa=(1050,356),
                Krasnoyarsk=(1214,391),
                Voronezh=(975,274),
                Perm=(1051,445),
                Volgograd=(999,194),
                Krasnodar=(974,96),
                Saratov=(1005,271),
                Tyumen=(1092,422),
                Tolyatti=(1020,324),
                Izhevsk=(1037,414),
                Barnaul=(1174,320),
                Ulyanovsk=(1016,345),
                Irkutsk=(1265,291),
                Khabarovsk=(1402,188),
                Yaroslavl=(978,434),
                Vladivostok=(1388,44),
                Makhachkala=(1012,40),
                Tomsk=(1179,404),
                Orenburg=(1046,277),
                Kemerovo=(1184,373),
                Novokuznetsk=(1189,330),
                Ryazan=(977,354),
                Astrakhan=(1014,131),
                Naberezhnyye_Chelny=(1033,383),
                Penza=(1001,315),
                Lipetsk=(977,300),
                Kirov=(1021,461),
                Cheboksary=(1011,394),
                Tula=(968,342),
                Kaliningrad=(892,356),
                Balashikha=(969,385),
                Kursk=(961,276),
                Stavropol=(987,96),
                Ulan_Ude=(1280,278),
                Tver=(960,414),
                Magnitogorsk=(1063,320),
                Sochi=(977,57),
                Ivanovo=(983,418),
                Bryansk=(953,317),
                Belgorod=(963,245),
                Surgut=(1128,532),
                Vladimir=(980,394),
                Nizhniy_Tagil=(1067,442),
                Arkhangelsk=(981,621),
                Chita=(1306,284),
                Kaluga=(962,351),
                Smolensk=(943,358),
                Volzhskiy=(1000,197),
                Cherepovets=(969,475),
                Vologda=(978,477),
                Saransk=(1002,342),
                Orël=(961,309),
                Yakutsk=(1378,553),
                Kurgan=(1091,376),
                Vladikavkaz=(999,42),
                Podolsk=(968,375),
                Murmansk=(948,740),
                Tambov=(985,302),
                Groznyy=(1004,49),
                Sterlitamak=(1050,327),
                Petrozavodsk=(953,546),
                Kostroma=(983,438),
                Nizhnevartovsk=(1142,524),
                Novorossiysk=(968,87),
                Yoshkar_Ola=(1014,408),
                Nalchik=(995,54),
                Engels=(1006,269),
                Taganrog=(974,155),
                Komsomol_sk_na_Amure=(1411,244),
                Syktyvkar=(1027,543),
                Khimki=(967,388),
                Nizhnekamsk=(1031,381),
                Shakhty=(979,167),
                Dzerzhinsk=(994,397),
                Bratsk=(1253,395),
                Orsk=(1061,262),
                Noginsk=(1207,619),
                Kolpino=(937,491),
                Angarsk=(1263,298),
                Korolëv=(969,389),
                Blagoveshchensk=(1369,236),
                Velikiy_Novgorod=(940,459),
                Staryy_Oskol=(969,264),
                Pskov=(926,440),
                Mytishchi=(968,389),
                Biysk=(1180,297),
                Lyubertsy=(969,382),
                Prokopyevsk=(1187,334),
                Yuzhno_Sakhalinsk=(1436,147),
                Armavir=(983,95),
                Balakovo=(1013,284),
                Rybinsk=(973,446),
                Severodvinsk=(978,621),
                Abakan=(1208,329),
                Petropavlovsk_Kamchatskiy=(1507,310),
                Norilsk=(1193,750),
                Orekhovo_Borisovo_Yuzhnoye=(968,380),
                Syzran=(1016,315),
                Krasnogorsk=(967,386),
                Volgodonsk=(988,162),
                Kamensk_Ural_skiy=(1076,402),
                Ussuriysk=(1388,62),
                Novocherkassk=(979,160),
                Zlatoust=(1066,368),
                Elektrostal=(972,385),
                Salavat=(1049,320),
                Almetyevsk=(1033,361),
                Miass=(1068,365),
                Nakhodka=(1393,36),
                Kopeysk=(1075,367),
                Pyatigorsk=(992,69),
                Rubtsovsk=(1162,270),
                Odintsovo=(966,382),
                Kolomna=(973,366),
                Berezniki=(1053,483),
                Khasavyurt=(1008,48),
                Maykop=(979,84),
                Kovrov=(984,400),
                Kislovodsk=(991,65),
                Serpukhov=(967,362),
                Novocheboksarsk=(1012,394),
                Bataysk=(977,152),
                Domodedovo=(968,376),
                Kaspiysk=(1012,38),
                Neftekamsk=(1042,393),
                Nefteyugansk=(1124,528),
                Shchelkovo=(970,389),
                Novomoskovsk=(971,338),
                Pervouralsk=(1067,415),
                Cherkessk=(988,74),
                Derbent=(1015,16),
                Orekhovo_Zuyevo=(974,385),
                Nazran=(1000,47),
                Nevinnomyssk=(987,85),
                Reutov=(969,384),
                Dimitrovgrad=(1021,343),
                Obninsk=(963,367),
                Kyzyl=(1221,275),
                Ramenskoye=(971,379),
                Oktyabr_skiy=(1039,350),
                Novyy_Urengoy=(1142,662),
                Kamyshin=(1003,232),
                Dolgoprudnyy=(967,389),
                Zhukovskiy=(970,380),
                Murom=(988,379),
                Yessentuki=(991,69),
                Novoshakhtinsk=(978,169),
                Seversk=(1178,407),
                Arzamas=(995,374),
                Artëm=(1389,50)
            )

        if country == "Ukraine":
            maps.romania_map = UndirectedGraph(dict(
                Kerch=dict(Alushta=179, Dzhankoi=166, Melitopol=186, Berdiansk=159, Dovzhansk=389, Mariupol=216),
                Alushta=dict(Yalta=26, Bakhchysarai=43, Kerch=179, Simferopol=39, Dzhankoi=115),
                Yalta=dict(Kerch=204, Alushta=26, Sevastopol=50, Bakhchysarai=36),
                Bakhchysarai=dict(Yalta=36, Sevastopol=30, Alushta=43, Simferopol=28, Saky=47),
                Sevastopol=dict(Izmail=378, Yalta=50, Bilhorod_Dnistrovskyi=305, Yevpatoriia=68, Bakhchysarai=30, Saky=59),
                Izmail=dict(Yalta=429, Podilsk=271, Bilhorod_Dnistrovskyi=148, Sevastopol=378, Khust=525, Mohyliv_Podilskyi=353, Chernivtsi=394),
                Mohyliv_Podilskyi=dict(Izmail=353, Chernivtsi=137, Zhmerynka=69, Podilsk=152, Haisyn=124, Kamianets_Podilskyi=92, Khmelnytskyi=122),
                Podilsk=dict(Mohyliv_Podilskyi=152, Izmail=271, Voznesensk=134, Uman=122, Haisyn=119, Odesa=167, Bilhorod_Dnistrovskyi=183),
                Luhansk=dict(Hlukhiv=517, Rovenky=55, Antratsyt=55, Sievierodonetsk=74, Rubizhne=85, Perevalsk=41, Holubivske=51, Alchevsk=41),
                Hlukhiv=dict(Rubizhne=432, Shostka=36, Romny=107, Konotop=69, Chuhuiv=282, Sumy=105),
                Rubizhne=dict(Luhansk=85, Sievierodonetsk=11, Lysychansk=12, Bakhmut=52, Sloviansk=56, Balakliia=121, Chuhuiv=152, Hlukhiv=432),
                Bilhorod_Dnistrovskyi=dict(Podilsk=183, Izmail=148, Sevastopol=305, Yevpatoriia=259, Odesa=44, Chornomorsk=28),
                Khust=dict(Vynohradiv=20, Izmail=525, Chernivtsi=195, Mukacheve=51, Kolomyia=133, Kalush=123, Stryi=125, Boryslav=123, Truskavets=123),
                Vynohradiv=dict(Izmail=539, Mukacheve=41, Khust=20),
                Korosten=dict(Slavutych=159, Kovel=276, Kostopil=154, Malyn=47, Novohrad_Volynskyi=82, Korostyshiv=75, Zhytomyr=77),
                Slavutych=dict(Kovel=420, Brovary=112, Korosten=159, Chernihiv=37, Shostka=191, Vyshhorod=105, Hostomel=111, Malyn=133),
                Kovel=dict(Korosten=276, Kostopil=126, Slavutych=420, Lutsk=67, Novovolynsk=66),
                Brovary=dict(Chernihiv=114, Slavutych=112, Nizhyn=98, Boryspil=21, Kyiv=20, Vyshhorod=22),
                Chernihiv=dict(Slavutych=37, Shostka=156, Brovary=114, Nizhyn=64),
                Kostopil=dict(Korosten=154, Novohrad_Volynskyi=90, Kovel=126, Lutsk=80, Rivne=32, Netishyn=63),
                Novohrad_Volynskyi=dict(Korosten=82, Zhytomyr=81, Berdychiv=103, Khmilnyk=117, Shepetivka=59, Kostopil=90, Netishyn=75),
                Malyn=dict(Slavutych=133, Fastiv=89, Korostyshiv=52, Korosten=47, Hostomel=73, Bucha=72),
                Chernivtsi=dict(Izmail=394, Khust=195, Kolomyia=71, Chortkiv=81, Kamianets_Podilskyi=64, Mohyliv_Podilskyi=137),
                Kolomyia=dict(Khust=133, Kalush=76, Ivano_Frankivsk=50, Chernivtsi=71, Chortkiv=79),
                Rovenky=dict(Dovzhansk=21, Krasnodon=36, Sukhodilsk=39, Antratsyt=21, Luhansk=55, Mariupol=171, Snizhne=45),
                Dovzhansk=dict(Krasnodon=24, Mariupol=189, Rovenky=21),
                Krasnodon=dict(Rovenky=36, Sukhodilsk=5),
                Shostka=dict(Slavutych=191, Kovel=609, Konotop=72, Chernihiv=156, Nizhyn=141),
                Sukhodilsk=dict(Rovenky=39, Luhansk=38),
                Mukacheve=dict(Uzhhorod=36, Vynohradiv=41, Boryslav=107, Khust=51),
                Uzhhorod=dict(Vynohradiv=76, Sambir=119, Mukacheve=36, Boryslav=110),
                Lutsk=dict(Kostopil=80, Kovel=67, Novovolynsk=81, Chervonohrad=87, Dubno=48, Rivne=66),
                Novovolynsk=dict(Lutsk=81, Chervonohrad=39, Novoyavorovskoye=98, Uzhhorod=270, Sambir=151),
                Vyshhorod=dict(Brovary=22, Slavutych=105, Kyiv=14, Kotsyubyns_ke=15, Hostomel=16),
                Antratsyt=dict(Snizhne=26, Rovenky=21, Khrustalnyi=10, Luhansk=55, Perevalsk=40),
                Snizhne=dict(Rovenky=45, Khrestivka=33, Antratsyt=26, Khrustalnyi=17, Perevalsk=46, Mariupol=133, Shakhtarsk=21),
                Khartsyzk=dict(Khrestivka=20, Shakhtarsk=24, Mariupol=109, Yenakiieve=22, Donetsk=25, Makiivka=14),
                Khrestivka=dict(Yenakiieve=14, Khartsyzk=20, Brianka=45, Shakhtarsk=15, Snizhne=33, Perevalsk=45),
                Yenakiieve=dict(Khartsyzk=22, Khrestivka=14, Makiivka=26, Horlivka=14, Brianka=45),
                Shakhtarsk=dict(Khrestivka=15, Snizhne=21, Mariupol=121, Khartsyzk=24),
                Brianka=dict(Yenakiieve=45, Stakhanov=3, Khrestivka=45, Alchevsk=12, Perevalsk=15, Horlivka=45, Pervomaisk=16),
                Chervonohrad=dict(Lutsk=87, Novovolynsk=39, Dubno=106, Ternopil=133, Vynnyky=63, Novoyavorovskoye=68, Lviv=61),
                Novoyavorovskoye=dict(Chervonohrad=68, Sambir=53, Novovolynsk=98, Lviv=34, Drohobych=64),
                Dubno=dict(Lutsk=48, Chervonohrad=106, Rivne=44, Ternopil=92, Netishyn=64),
                Zhytomyr=dict(Korosten=77, Novohrad_Volynskyi=81, Berdychiv=40, Korostyshiv=29),
                Berdychiv=dict(Zhytomyr=40, Novohrad_Volynskyi=103, Korostyshiv=57, Fastiv=96, Vinnytsia=76, Khmilnyk=59, Bila_Tserkva=109, Haisyn=133),
                Zhmerynka=dict(Khmelnytskyi=90, Khmilnyk=57, Mohyliv_Podilskyi=69, Haisyn=97, Vinnytsia=31),
                Khmelnytskyi=dict(Mohyliv_Podilskyi=122, Khmilnyk=71, Zhmerynka=90, Shepetivka=85, Netishyn=104, Ternopil=102, Chortkiv=98, Kamianets_Podilskyi=87),
                Ivano_Frankivsk=dict(Ternopil=96, Novyi_Rozdil=74, Kalush=28, Kolomyia=50, Chortkiv=80),
                Ternopil=dict(Novyi_Rozdil=106, Dubno=92, Vynnyky=109, Chervonohrad=133, Khmelnytskyi=102, Netishyn=112, Chortkiv=62, Ivano_Frankivsk=96),
                Novyi_Rozdil=dict(Ivano_Frankivsk=74, Kalush=50, Ternopil=106, Drohobych=47, Stryi=31, Vynnyky=38, Lviv=41),
                Khmilnyk=dict(Shepetivka=95, Khmelnytskyi=71, Berdychiv=59, Novohrad_Volynskyi=117, Vinnytsia=50, Zhmerynka=57),
                Shepetivka=dict(Khmelnytskyi=85, Khmilnyk=95, Netishyn=34, Novohrad_Volynskyi=59),
                Rivne=dict(Lutsk=66, Kostopil=32, Dubno=44, Netishyn=42),
                Holubivske=dict(Lysychansk=34, Pervomaisk=9, Luhansk=51, Sievierodonetsk=36, Alchevsk=22, Stakhanov=10),
                Lysychansk=dict(Pervomaisk=32, Holubivske=34, Sievierodonetsk=5, Bakhmut=46, Rubizhne=12),
                Pervomaisk=dict(Holubivske=9, Brianka=16, Stakhanov=12, Lysychansk=32, Bakhmut=37, Horlivka=45),
                Khrustalnyi=dict(Snizhne=17, Antratsyt=10, Perevalsk=35),
                Voznesensk=dict(Uman=154, Kropyvnytskyi=126, Mykolaiv=85, Yuzhne=106, Podilsk=134, Odesa=129),
                Uman=dict(Podilsk=122, Haisyn=60, Bila_Tserkva=116, Voznesensk=154, Kropyvnytskyi=152, Smila=131, Kaniv=142),
                Yevpatoriia=dict(Bilhorod_Dnistrovskyi=259, Saky=18, Sevastopol=68, Yany_Kapu=92, Chornomorsk=242, Kherson=169, Komyshany=173),
                Dzhankoi=dict(Alushta=115, Kerch=166, Melitopol=145, Simferopol=87, Yany_Kapu=54, Saky=90),
                Melitopol=dict(Dzhankoi=145, Yany_Kapu=153, Kerch=186, Kakhovka=144, Enerhodar=91, Tokmak=53, Berdiansk=108),
                Yany_Kapu=dict(Dzhankoi=54, Yevpatoriia=92, Saky=94, Kherson=118, Melitopol=153, Nova_Kakhovka=94, Kakhovka=96),
                Berdiansk=dict(Melitopol=108, Kerch=159, Mariupol=72, Tokmak=98, Hirnyk=150),
                Kalush=dict(Ivano_Frankivsk=28, Khust=123, Kolomyia=76, Novyi_Rozdil=50, Stryi=43),
                Sambir=dict(Drohobych=28, Novovolynsk=151, Novoyavorovskoye=53, Boryslav=30, Uzhhorod=119),
                Drohobych=dict(Novoyavorovskoye=64, Stryi=27, Lviv=66, Novyi_Rozdil=47, Truskavets=7, Sambir=28, Boryslav=8),
                Vynnyky=dict(Novyi_Rozdil=38, Ternopil=109, Chervonohrad=63, Lviv=7),
                Haisyn=dict(Podilsk=119, Mohyliv_Podilskyi=124, Zhmerynka=97, Berdychiv=133, Vinnytsia=83, Uman=60, Bila_Tserkva=121),
                Fastiv=dict(Bucha=57, Korostyshiv=66, Malyn=89, Berdychiv=96, Bila_Tserkva=34, Vasylkiv=30, Irpin=54, Boyarka=38),
                Bucha=dict(Malyn=72, Hostomel=3, Fastiv=57, Irpin=3),
                Korostyshiv=dict(Zhytomyr=29, Berdychiv=57, Fastiv=66, Malyn=52, Korosten=75),
                Netishyn=dict(Khmelnytskyi=104, Ternopil=112, Dubno=64, Rivne=42, Kostopil=63, Novohrad_Volynskyi=75, Shepetivka=34),
                Chortkiv=dict(Khmelnytskyi=98, Kolomyia=79, Ivano_Frankivsk=80, Ternopil=62, Chernivtsi=81, Kamianets_Podilskyi=68),
                Stakhanov=dict(Pervomaisk=12, Holubivske=10, Alchevsk=14, Brianka=3),
                Sievierodonetsk=dict(Holubivske=36, Lysychansk=5, Luhansk=74, Rubizhne=11),
                Perevalsk=dict(Antratsyt=40, Khrustalnyi=35, Snizhne=46, Khrestivka=45, Brianka=15, Luhansk=41, Alchevsk=4),
                Konotop=dict(Hlukhiv=69, Nizhyn=93, Shostka=72, Romny=58, Pryluky=90),
                Nizhyn=dict(Pryluky=61, Chernihiv=64, Brovary=98, Shostka=141, Konotop=93, Pereyaslav_Khmel_nyts_kyy=114, Boryspil=102),
                Pryluky=dict(Konotop=90, Zolotonosha=105, Lubny=76, Romny=78, Pereyaslav_Khmel_nyts_kyy=90, Nizhyn=61),
                Boryspil=dict(Nizhyn=102, Pereyaslav_Khmel_nyts_kyy=47, Obukhiv=36, Brovary=21, Kyiv=32),
                Zolotonosha=dict(Lubny=77, Pryluky=105, Cherkasy=26, Pereyaslav_Khmel_nyts_kyy=59, Kaniv=42),
                Lubny=dict(Pryluky=76, Myrhorod=45, Romny=88, Svitlovodsk=105, Zolotonosha=77, Cherkasy=92),
                Simferopol=dict(Bakhchysarai=28, Alushta=39, Dzhankoi=87, Saky=45),
                Saky=dict(Yany_Kapu=94, Dzhankoi=90, Simferopol=45, Bakhchysarai=47, Sevastopol=59, Yevpatoriia=18),
                Mariupol=dict(Kerch=216, Berdiansk=72, Dovzhansk=189, Rovenky=171, Snizhne=133, Shakhtarsk=121, Hirnyk=103, Donetsk=99, Khartsyzk=109),
                Kherson=dict(Yevpatoriia=169, Mykolaiv=58, Komyshany=6, Yany_Kapu=118, Kryvyi_Rih=152, Nova_Kakhovka=60),
                Nikopol=dict(Enerhodar=24, Kakhovka=109, Pokrov=22, Marhanets=20),
                Enerhodar=dict(Marhanets=16, Nikopol=24, Kakhovka=118, Melitopol=91, Zaporizhzhia=52, Tokmak=83),
                Marhanets=dict(Nikopol=20, Karnaukhivka=91, Zaporizhzhia=44, Enerhodar=16, Pokrov=40, Kamianske=95),
                Stryi=dict(Novyi_Rozdil=31, Kalush=43, Khust=125, Truskavets=25, Drohobych=27),
                Lviv=dict(Chervonohrad=61, Vynnyky=7, Novyi_Rozdil=41, Novoyavorovskoye=34, Drohobych=66),
                Vinnytsia=dict(Haisyn=83, Zhmerynka=31, Berdychiv=76, Khmilnyk=50),
                Bila_Tserkva=dict(Haisyn=121, Fastiv=34, Berdychiv=109, Vasylkiv=44, Obukhiv=49, Kaniv=96, Uman=116),
                Hostomel=dict(Vyshhorod=16, Slavutych=111, Malyn=73, Bucha=3, Kotsyubyns_ke=10, Irpin=5),
                Kamianets_Podilskyi=dict(Chortkiv=68, Khmelnytskyi=87, Chernivtsi=64, Mohyliv_Podilskyi=92),
                Alchevsk=dict(Luhansk=41, Perevalsk=4, Holubivske=22, Stakhanov=14, Brianka=12),
                Sumy=dict(Myrhorod=134, Poltava=149, Hlukhiv=105, Romny=93, Chuhuiv=178, Kharkiv=142),
                Myrhorod=dict(Poltava=81, Horishni_Plavni=106, Sumy=134, Romny=87, Lubny=45, Kremenchuk=99, Svitlovodsk=101),
                Poltava=dict(Sumy=149, Myrhorod=81, Kharkiv=128, Horishni_Plavni=91, Kamianske=118, Pervomaiskyi=120, Novomoskovsk=115),
                Horishni_Plavni=dict(Poltava=91, Kamianske=89, Zhovti_Vody=74, Oleksandriia=54, Myrhorod=106, Kremenchuk=18),
                Kropyvnytskyi=dict(Uman=152, Smila=83, Svitlovodsk=96, Voznesensk=126, Oleksandriia=64, Mykolaiv=172, Kryvyi_Rih=104),
                Smila=dict(Uman=131, Kropyvnytskyi=83, Svitlovodsk=101, Cherkasy=28, Kaniv=65),
                Kharkiv=dict(Sumy=142, Poltava=128, Pervomaiskyi=68, Chuhuiv=36),
                Romny=dict(Sumy=93, Hlukhiv=107, Myrhorod=87, Lubny=88, Pryluky=78, Konotop=58),
                Chornomorsk=dict(Bilhorod_Dnistrovskyi=28, Yuzhne=49, Odesa=20, Komyshany=146, Yevpatoriia=242),
                Vasylkiv=dict(Fastiv=30, Bila_Tserkva=44, Boyarka=17, Obukhiv=22, Vyshneve=23),
                Obukhiv=dict(Vasylkiv=22, Pereyaslav_Khmel_nyts_kyy=59, Kaniv=71, Bila_Tserkva=49, Vyshneve=36, Kyiv=39, Boryspil=36),
                Svitlovodsk=dict(Smila=101, Myrhorod=101, Cherkasy=95, Lubny=105, Kropyvnytskyi=96, Kremenchuk=12, Oleksandriia=47),
                Pereyaslav_Khmel_nyts_kyy=dict(Zolotonosha=59, Pryluky=90, Nizhyn=114, Boryspil=47, Kaniv=35, Obukhiv=59),
                Hirnyk=dict(Berdiansk=150, Tokmak=152, Pershotravensk=78, Pokrovsk=29, Myrnohrad=29, Avdiivka=29, Donetsk=32, Mariupol=103),
                Kakhovka=dict(Enerhodar=118, Yany_Kapu=96, Melitopol=144, Nova_Kakhovka=8, Nikopol=109, Kryvyi_Rih=123, Pokrov=106),
                Tokmak=dict(Enerhodar=83, Melitopol=53, Berdiansk=98, Zaporizhzhia=77, Synelnykove=119, Hirnyk=152, Pershotravensk=132),
                Truskavets=dict(Khust=123, Stryi=25, Drohobych=7, Boryslav=5),
                Mykolaiv=dict(Kropyvnytskyi=172, Voznesensk=85, Yuzhne=78, Komyshany=53, Kherson=58, Kryvyi_Rih=145),
                Yuzhne=dict(Mykolaiv=78, Odesa=32, Voznesensk=106, Chornomorsk=49, Komyshany=107),
                Kamianske=dict(Poltava=118, Horishni_Plavni=89, Marhanets=95, Novomoskovsk=47, Karnaukhivka=10, Zhovti_Vody=82, Pokrov=101),
                Zhovti_Vody=dict(Kamianske=82, Horishni_Plavni=74, Pokrov=87, Oleksandriia=45, Kryvyi_Rih=50),
                Karnaukhivka=dict(Kamianske=10, Marhanets=91, Novomoskovsk=40, Dnipro=22, Zaporizhzhia=76),
                Zaporizhzhia=dict(Karnaukhivka=76, Marhanets=44, Enerhodar=52, Tokmak=77, Dnipro=70, Synelnykove=60),
                Novomoskovsk=dict(Poltava=115, Kamianske=47, Pervomaiskyi=110, Lozova=84, Karnaukhivka=40, Dnipro=22, Pavlohrad=49, Synelnykove=40),
                Bakhmut=dict(Pervomaisk=37, Lysychansk=46, Rubizhne=52, Sloviansk=40, Toretsk=26, Horlivka=30, Kostiantynivka=22, Kramatorsk=35),
                Odesa=dict(Chornomorsk=20, Voznesensk=129, Yuzhne=32, Podilsk=167, Bilhorod_Dnistrovskyi=44),
                Komyshany=dict(Yuzhne=107, Mykolaiv=53, Kherson=6, Yevpatoriia=173, Chornomorsk=146),
                Boyarka=dict(Fastiv=38, Irpin=21, Vasylkiv=17, Kotsyubyns_ke=17, Vyshneve=8, Sofiyivs_ka_Borshchahivka=10),
                Cherkasy=dict(Smila=28, Lubny=92, Svitlovodsk=95, Kaniv=54, Zolotonosha=26),
                Oleksandriia=dict(Svitlovodsk=47, Zhovti_Vody=45, Horishni_Plavni=54, Kremenchuk=50, Kryvyi_Rih=85, Kropyvnytskyi=64),
                Kremenchuk=dict(Horishni_Plavni=18, Myrhorod=99, Oleksandriia=50, Svitlovodsk=12),
                Kaniv=dict(Zolotonosha=42, Smila=65, Cherkasy=54, Pereyaslav_Khmel_nyts_kyy=35, Obukhiv=71, Uman=142, Bila_Tserkva=96),
                Nova_Kakhovka=dict(Kherson=60, Yany_Kapu=94, Kryvyi_Rih=126, Kakhovka=8),
                Pokrov=dict(Kakhovka=106, Kamianske=101, Nikopol=22, Marhanets=40, Zhovti_Vody=87, Kryvyi_Rih=61),
                Boryslav=dict(Drohobych=8, Truskavets=5, Uzhhorod=110, Sambir=30, Mukacheve=107, Khust=123),
                Pervomaiskyi=dict(Kharkiv=68, Poltava=120, Chuhuiv=59, Balakliia=45, Novomoskovsk=110, Lozova=55),
                Synelnykove=dict(Zaporizhzhia=60, Tokmak=119, Novomoskovsk=40, Dnipro=38, Pavlohrad=34, Pershotravensk=66),
                Makiivka=dict(Khartsyzk=14, Yenakiieve=26, Donetsk=12, Horlivka=32, Yasynuvata=11),
                Sloviansk=dict(Bakhmut=40, Kramatorsk=17, Rubizhne=56, Balakliia=86),
                Kramatorsk=dict(Bakhmut=35, Sloviansk=17, Balakliia=97, Lozova=92, Kostiantynivka=23, Druzhkivka=11),
                Balakliia=dict(Sloviansk=86, Kramatorsk=97, Rubizhne=121, Chuhuiv=43, Pervomaiskyi=45, Lozova=73),
                Chuhuiv=dict(Balakliia=43, Rubizhne=152, Kharkiv=36, Pervomaiskyi=59, Hlukhiv=282, Sumy=178),
                Irpin=dict(Bucha=3, Fastiv=54, Hostomel=5, Boyarka=21, Kotsyubyns_ke=6),
                Vyshneve=dict(Vasylkiv=23, Boyarka=8, Obukhiv=36, Sofiyivs_ka_Borshchahivka=3, Kyiv=13),
                Pershotravensk=dict(Tokmak=132, Hirnyk=78, Synelnykove=66, Lozova=60, Pokrovsk=58, Ternivka=30, Pavlohrad=43),
                Pokrovsk=dict(Pershotravensk=58, Druzhkivka=45, Lozova=92, Myrnohrad=6, Hirnyk=29),
                Kryvyi_Rih=dict(Mykolaiv=145, Kherson=152, Kropyvnytskyi=104, Nova_Kakhovka=126, Zhovti_Vody=50, Oleksandriia=85, Pokrov=61, Kakhovka=123),
                Lozova=dict(Pervomaiskyi=55, Balakliia=73, Ternivka=44, Kramatorsk=92, Druzhkivka=93, Pokrovsk=92, Pershotravensk=60, Pavlohrad=52, Novomoskovsk=84),
                Ternivka=dict(Pershotravensk=30, Lozova=44, Pavlohrad=15),
                Druzhkivka=dict(Kramatorsk=11, Lozova=93, Pokrovsk=45, Myrnohrad=40, Kostiantynivka=16),
                Dnipro=dict(Novomoskovsk=22, Synelnykove=38, Karnaukhivka=22, Zaporizhzhia=70),
                Kyiv=dict(Vyshneve=13, Boryspil=32, Obukhiv=39, Brovary=20, Vyshhorod=14, Sofiyivs_ka_Borshchahivka=11, Kotsyubyns_ke=14),
                Kotsyubyns_ke=dict(Irpin=6, Kyiv=14, Vyshhorod=15, Hostomel=10, Boyarka=17, Sofiyivs_ka_Borshchahivka=8),
                Pavlohrad=dict(Ternivka=15, Lozova=52, Novomoskovsk=49, Pershotravensk=43, Synelnykove=34),
                Sofiyivs_ka_Borshchahivka=dict(Boyarka=10, Kotsyubyns_ke=8, Vyshneve=3, Kyiv=11),
                Myrnohrad=dict(Druzhkivka=40, Avdiivka=40, Pokrovsk=6, Hirnyk=29, Kostiantynivka=42, Toretsk=43),
                Avdiivka=dict(Toretsk=30, Myrnohrad=40, Hirnyk=29, Yasynuvata=6, Donetsk=14),
                Toretsk=dict(Myrnohrad=43, Horlivka=20, Kostiantynivka=17, Bakhmut=26, Avdiivka=30, Yasynuvata=31),
                Horlivka=dict(Makiivka=32, Bakhmut=30, Pervomaisk=45, Yenakiieve=14, Brianka=45, Toretsk=20, Yasynuvata=30),
                Donetsk=dict(Avdiivka=14, Hirnyk=32, Mariupol=99, Khartsyzk=25, Makiivka=12, Yasynuvata=12),
                Kostiantynivka=dict(Druzhkivka=16, Myrnohrad=42, Toretsk=17, Bakhmut=22, Kramatorsk=23),
                Yasynuvata=dict(Toretsk=31, Horlivka=30, Makiivka=11, Donetsk=12, Avdiivka=6)
            ))


            maps.romania_map.locations = dict(
                Kyiv=(759,699),
                Kharkiv=(1280,646),
                Odesa=(778,237),
                Dnipro=(1171,468),
                Donetsk=(1423,415),
                Zaporizhzhia=(1180,395),
                Lviv=(167,628),
                Kryvyi_Rih=(1016,403),
                Sevastopol=(1034,19),
                Mykolaiv=(894,294),
                Mariupol=(1402,313),
                Luhansk=(1563,482),
                Vinnytsia=(568,555),
                Simferopol=(1085,59),
                Makiivka=(1438,420),
                Poltava=(1128,597),
                Chernihiv=(829,820),
                Kherson=(949,255),
                Cherkasy=(899,582),
                Khmelnytskyi=(438,579),
                Zhytomyr=(589,676),
                Chernivtsi=(340,448),
                Sumy=(1149,752),
                Horlivka=(1450,453),
                Rivne=(369,718),
                Ivano_Frankivsk=(229,521),
                Kamianske=(1132,473),
                Kropyvnytskyi=(918,473),
                Ternopil=(310,596),
                Lutsk=(285,733),
                Kremenchuk=(1024,539),
                Bila_Tserkva=(722,623),
                Kramatorsk=(1401,498),
                Melitopol=(1201,278),
                Sievierodonetsk=(1485,524),
                Kerch=(1302,105),
                Drohobych=(118,571),
                Khrustalnyi=(1528,428),
                Uzhhorod=(8,487),
                Berdiansk=(1330,270),
                Sloviansk=(1407,515),
                Nikopol=(1109,365),
                Pavlohrad=(1247,474),
                Yevpatoriia=(1018,89),
                Alchevsk=(1514,468),
                Brovary=(783,706),
                Konotop=(1004,790),
                Kamianets_Podilskyi=(399,493),
                Mukacheve=(46,465),
                Uman=(731,501),
                Chervonohrad=(185,691),
                Khartsyzk=(1455,418),
                Stryi=(150,559),
                Chornomorsk=(771,217),
                Novohrad_Volynskyi=(495,714),
                Myrnohrad=(1374,449),
                Rovenky=(1566,424),
                Myrhorod=(1041,642),
                Bucha=(733,710),
                Chuhuiv=(1320,627),
                Sukhodilsk=(1598,455),
                Kotsyubyns_ke=(742,703),
                Hostomel=(735,713),
                Vynnyky=(176,625),
                Hirnyk=(1384,420),
                Komyshany=(940,256),
                Karnaukhivka=(1144,469),
                Lysychansk=(1481,521),
                Dovzhansk=(1592,424),
                Oleksandriia=(996,492),
                Yalta=(1090,7),
                Yenakiieve=(1460,441),
                Bakhmut=(1442,484),
                Shostka=(1029,863),
                Stakhanov=(1499,478),
                Berdychiv=(584,634),
                Nizhyn=(885,768),
                Kostiantynivka=(1415,476),
                Izmail=(605,106),
                Novomoskovsk=(1188,488),
                Kovel=(229,788),
                Smila=(882,555),
                Kalush=(197,535),
                Korosten=(588,757),
                Pokrovsk=(1367,447),
                Kolomyia=(258,474),
                Boryspil=(798,687),
                Rubizhne=(1475,532),
                Druzhkivka=(1398,486),
                Bilhorod_Dnistrovskyi=(742,203),
                Irpin=(734,707),
                Antratsyt=(1540,428),
                Lozova=(1288,517),
                Enerhodar=(1136,356),
                Pryluky=(930,716),
                Horishni_Plavni=(1045,531),
                Novovolynsk=(179,732),
                Shakhtarsk=(1485,418),
                Brianka=(1500,474),
                Snizhne=(1511,416),
                Marhanets=(1133,373),
                Fastiv=(704,655),
                Lubny=(984,649),
                Svitlovodsk=(1008,540),
                Nova_Kakhovka=(1019,271),
                Zhovti_Vody=(1032,455),
                Krasnodon=(1600,449),
                Shepetivka=(444,668),
                Podilsk=(669,384),
                Romny=(1030,733),
                Vyshneve=(745,691),
                Pokrov=(1084,375),
                Vasylkiv=(741,667),
                Dubno=(322,692),
                Pervomaisk=(1488,488),
                Dzhankoi=(1112,148),
                Boryslav=(111,564),
                Holubivske=(1500,488),
                Yasynuvata=(1426,428),
                Netishyn=(405,685),
                Zhmerynka=(538,535),
                Avdiivka=(1419,430),
                Kakhovka=(1028,275),
                Toretsk=(1426,461),
                Sambir=(91,590),
                Boyarka=(738,685),
                Voznesensk=(831,364),
                Hlukhiv=(1068,841),
                Obukhiv=(768,658),
                Yuzhne=(812,254),
                Mohyliv_Podilskyi=(509,466),
                Kostopil=(387,749),
                Tokmak=(1232,327),
                Novoyavorovskoye=(125,638),
                Synelnykove=(1214,451),
                Pervomaiskyi=(1278,575),
                Alushta=(1113,27),
                Khmilnyk=(526,594),
                Truskavets=(119,563),
                Balakliia=(1335,583),
                Novyi_Rozdil=(176,585),
                Chortkiv=(328,532),
                Pershotravensk=(1296,454),
                Khust=(100,435),
                Khrestivka=(1475,431),
                Ternivka=(1266,475),
                Vyshhorod=(757,714),
                Zolotonosha=(897,610),
                Bakhchysarai=(1064,37),
                Pereyaslav_Khmel_nyts_kyy=(843,654),
                Sofiyivs_ka_Borshchahivka=(745,694),
                Perevalsk=(1516,464),
                Malyn=(645,736),
                Yany_Kapu=(1058,178),
                Haisyn=(656,508),
                Vynohradiv=(76,430),
                Korostyshiv=(625,683),
                Saky=(1038,81),
                Slavutych=(780,823),
                Kaniv=(844,617)
            )




""" [Figure 4.9]
Eight possible states of the vacumm world
Each state is represented as
   *       "State of the left room"      "State of the right room"   "Room in which the agent
                                                                      is present"
1 - DDL     Dirty                         Dirty                       Left
2 - DDR     Dirty                         Dirty                       Right
3 - DCL     Dirty                         Clean                       Left
4 - DCR     Dirty                         Clean                       Right
5 - CDL     Clean                         Dirty                       Left
6 - CDR     Clean                         Dirty                       Right
7 - CCL     Clean                         Clean                       Left
8 - CCR     Clean                         Clean                       Right
"""
vacuum_world = Graph(dict(
    State_1=dict(Suck=['State_7', 'State_5'], Right=['State_2']),
    State_2=dict(Suck=['State_8', 'State_4'], Left=['State_2']),
    State_3=dict(Suck=['State_7'], Right=['State_4']),
    State_4=dict(Suck=['State_4', 'State_2'], Left=['State_3']),
    State_5=dict(Suck=['State_5', 'State_1'], Right=['State_6']),
    State_6=dict(Suck=['State_8'], Left=['State_5']),
    State_7=dict(Suck=['State_7', 'State_3'], Right=['State_8']),
    State_8=dict(Suck=['State_8', 'State_6'], Left=['State_7'])
))

""" [Figure 4.23]
One-dimensional state space Graph
"""
one_dim_state_space = Graph(dict(
    State_1=dict(Right='State_2'),
    State_2=dict(Right='State_3', Left='State_1'),
    State_3=dict(Right='State_4', Left='State_2'),
    State_4=dict(Right='State_5', Left='State_3'),
    State_5=dict(Right='State_6', Left='State_4'),
    State_6=dict(Left='State_5')
))
one_dim_state_space.least_costs = dict(
    State_1=8,
    State_2=9,
    State_3=2,
    State_4=2,
    State_5=4,
    State_6=3)

""" [Figure 6.1]
Principal states and territories of Australia
"""
australia_map = UndirectedGraph(dict(
    T=dict(),
    SA=dict(WA=1, NT=1, Q=1, NSW=1, V=1),
    NT=dict(WA=1, Q=1),
    NSW=dict(Q=1, V=1)))
australia_map.locations = dict(WA=(120, 24), NT=(135, 20), SA=(135, 30),
                               Q=(145, 20), NSW=(145, 32), T=(145, 42),
                               V=(145, 37))


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


class GraphProblemStochastic(GraphProblem):
    """
    A version of GraphProblem where an action can lead to
    nondeterministic output i.e. multiple possible states.

    Define the graph as dict(A = dict(Action = [[<Result 1>, <Result 2>, ...], <cost>], ...), ...)
    A the dictionary format is different, make sure the graph is created as a directed graph.
    """

    def result(self, state, action):
        return self.graph.get(state, action)

    def path_cost(self):
        raise NotImplementedError


# ______________________________________________________________________________


class NQueensProblem(Problem):
    """The problem of placing N queens on an NxN board with none attacking
    each other. A state is represented as an N-element array, where
    a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been
    filled in yet. We fill in columns left to right.
    >>> depth_first_tree_search(NQueensProblem(8))
    <Node (7, 3, 0, 2, 5, 1, 6, 4)>
    """

    def __init__(self, N):
        super().__init__(tuple([-1] * N))
        self.N = N

    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            return []  # All columns filled; no successors
        else:
            col = state.index(-1)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        if state[-1] == -1:
            return False
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def h(self, node):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1, c1) in enumerate(node.state):
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicts += self.conflict(r1, c1, r2, c2)

        return num_conflicts


# ______________________________________________________________________________
# Inverse Boggle: Search for a high-scoring Boggle board. A good domain for
# iterative-repair and related search techniques, as suggested by Justin Boyan.


ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cubes16 = ['FORIXB', 'MOQABJ', 'GURILW', 'SETUPL',
           'CMPDAE', 'ACITAO', 'SLCRAE', 'ROMASH',
           'NODESW', 'HEFIYE', 'ONUDTK', 'TEVIGN',
           'ANEDVZ', 'PINESH', 'ABILYT', 'GKYLEU']


def random_boggle(n=4):
    """Return a random Boggle board of size n x n.
    We represent a board as a linear list of letters."""
    cubes = [cubes16[i % 16] for i in range(n * n)]
    random.shuffle(cubes)
    return list(map(random.choice, cubes))


# The best 5x5 board found by Boyan, with our word list this board scores
# 2274 words, for a score of 9837


boyan_best = list('RSTCSDEIAEGNLRPEATESMSSID')


def print_boggle(board):
    """Print the board in a 2-d array."""
    n2 = len(board)
    n = exact_sqrt(n2)
    for i in range(n2):

        if i % n == 0 and i > 0:
            print()
        if board[i] == 'Q':
            print('Qu', end=' ')
        else:
            print(str(board[i]) + ' ', end=' ')
    print()


def boggle_neighbors(n2, cache={}):
    """Return a list of lists, where the i-th element is the list of indexes
    for the neighbors of square i."""
    if cache.get(n2):
        return cache.get(n2)
    n = exact_sqrt(n2)
    neighbors = [None] * n2
    for i in range(n2):
        neighbors[i] = []
        on_top = i < n
        on_bottom = i >= n2 - n
        on_left = i % n == 0
        on_right = (i + 1) % n == 0
        if not on_top:
            neighbors[i].append(i - n)
            if not on_left:
                neighbors[i].append(i - n - 1)
            if not on_right:
                neighbors[i].append(i - n + 1)
        if not on_bottom:
            neighbors[i].append(i + n)
            if not on_left:
                neighbors[i].append(i + n - 1)
            if not on_right:
                neighbors[i].append(i + n + 1)
        if not on_left:
            neighbors[i].append(i - 1)
        if not on_right:
            neighbors[i].append(i + 1)
    cache[n2] = neighbors
    return neighbors


def exact_sqrt(n2):
    """If n2 is a perfect square, return its square root, else raise error."""
    n = int(np.sqrt(n2))
    assert n * n == n2
    return n


# _____________________________________________________________________________


class Wordlist:
    """This class holds a list of words. You can use (word in wordlist)
    to check if a word is in the list, or wordlist.lookup(prefix)
    to see if prefix starts any of the words in the list."""

    def __init__(self, file, min_len=3):
        lines = file.read().upper().split()
        self.words = [word for word in lines if len(word) >= min_len]
        self.words.sort()
        self.bounds = {}
        for c in ALPHABET:
            c2 = chr(ord(c) + 1)
            self.bounds[c] = (bisect.bisect(self.words, c),
                              bisect.bisect(self.words, c2))

    def lookup(self, prefix, lo=0, hi=None):
        """See if prefix is in dictionary, as a full word or as a prefix.
        Return two values: the first is the lowest i such that
        words[i].startswith(prefix), or is None; the second is
        True iff prefix itself is in the Wordlist."""
        words = self.words
        if hi is None:
            hi = len(words)
        i = bisect.bisect_left(words, prefix, lo, hi)
        if i < len(words) and words[i].startswith(prefix):
            return i, (words[i] == prefix)
        else:
            return None, False

    def __contains__(self, word):
        return self.lookup(word)[1]

    def __len__(self):
        return len(self.words)


# _____________________________________________________________________________


class BoggleFinder:
    """A class that allows you to find all the words in a Boggle board."""

    wordlist = None  # A class variable, holding a wordlist

    def __init__(self, board=None):
        if BoggleFinder.wordlist is None:
            BoggleFinder.wordlist = Wordlist(open_data("EN-text/wordlist.txt"))
        self.found = {}
        if board:
            self.set_board(board)

    def set_board(self, board=None):
        """Set the board, and find all the words in it."""
        if board is None:
            board = random_boggle()
        self.board = board
        self.neighbors = boggle_neighbors(len(board))
        self.found = {}
        for i in range(len(board)):
            lo, hi = self.wordlist.bounds[board[i]]
            self.find(lo, hi, i, [], '')
        return self

    def find(self, lo, hi, i, visited, prefix):
        """Looking in square i, find the words that continue the prefix,
        considering the entries in self.wordlist.words[lo:hi], and not
        revisiting the squares in visited."""
        if i in visited:
            return
        wordpos, is_word = self.wordlist.lookup(prefix, lo, hi)
        if wordpos is not None:
            if is_word:
                self.found[prefix] = True
            visited.append(i)
            c = self.board[i]
            if c == 'Q':
                c = 'QU'
            prefix += c
            for j in self.neighbors[i]:
                self.find(wordpos, hi, j, visited, prefix)
            visited.pop()

    def words(self):
        """The words found."""
        return list(self.found.keys())

    scores = [0, 0, 0, 0, 1, 2, 3, 5] + [11] * 100

    def score(self):
        """The total score for the words found, according to the rules."""
        return sum([self.scores[len(w)] for w in self.words()])

    def __len__(self):
        """The number of words found."""
        return len(self.found)


# _____________________________________________________________________________


def boggle_hill_climbing(board=None, ntimes=100, verbose=True):
    """Solve inverse Boggle by hill-climbing: find a high-scoring board by
    starting with a random one and changing it."""
    finder = BoggleFinder()
    if board is None:
        board = random_boggle()
    best = len(finder.set_board(board))
    for _ in range(ntimes):
        i, oldc = mutate_boggle(board)
        new = len(finder.set_board(board))
        if new > best:
            best = new
            if verbose:
                print(best, _, board)
        else:
            board[i] = oldc  # Change back
    if verbose:
        print_boggle(board)
    return board, best


def mutate_boggle(board):
    i = random.randrange(len(board))
    oldc = board[i]
    # random.choice(boyan_best)
    board[i] = random.choice(random.choice(cubes16))
    return i, oldc


# ______________________________________________________________________________

# Code to compare searchers on various problems.


class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __repr__(self):
        return '<{:4d}/{:4d}/{:4d}/{}>'.format(self.succs, self.goal_tests,
                                               self.states, str(self.found)[:4])


def compare_searchers(problems, header,
                      searchers=[breadth_first_tree_search,
                                 breadth_first_graph_search,
                                 depth_first_graph_search,
                                 iterative_deepening_search,
                                 depth_limited_search,
                                 recursive_best_first_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p)
        return p

    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)


def compare_graph_searchers():
    """Prints a table of search results."""
    compare_searchers(problems=[GraphProblem('Arad', 'Bucharest', romania_map),
                                GraphProblem('Oradea', 'Neamt', romania_map),
                                GraphProblem('Q', 'WA', australia_map)],
                      header=['Searcher', 'romania_map(Arad, Bucharest)',
                              'romania_map(Oradea, Neamt)', 'australia_map'])