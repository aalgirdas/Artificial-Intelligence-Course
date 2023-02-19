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
            maps.romania_map_goal = "Belgorod"
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
                Kaunas=dict(Birštonas=33, Garliava=10, Kaišiadorys=34, Jonava=29, Domeikava=7, Klaipėda=150),  # , Klaipėda=200
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
                Gorakhpur=dict(Guwāhāti=836, Lucknow=240, Patna=218, Vārānasi=164, Bareilly=428, Allahābād=210,
                               Mirzāpur=195),
                Guwāhāti=dict(Kolkāta=526, Dhanbād=597, Hāora=529, Patna=663, Gorakhpur=836),
                Chennai=dict(Madurai=422, Vishākhapatnam=611, Trichinopoly=304, Hyderābād=513, Bhubaneshwar=993,
                             Bangalore=290, Salem=278, Bezwāda=383, Guntūr=358),
                Kolhāpur=dict(Hubli=176, Shimoga=340, Pune=206, Solāpur=208, Kochi=780, Mumbai=300),
                Hubli=dict(Solāpur=270, Shimoga=167, Kolhāpur=176),
                Solāpur=dict(Kolhāpur=208, Hubli=270, Shimoga=419, Pune=236, Bangalore=552, Aurangābād=253, Bhopāl=639,
                             Hyderābād=274),
                Shimoga=dict(Solāpur=419, Hubli=167, Bangalore=242, Mysore=214, Kochi=446, Kolhāpur=340),
                Bangalore=dict(Solāpur=552, Shimoga=242, Hyderābād=496, Chennai=290, Salem=160, Mysore=126,
                               Coimbatore=230),
                Dhanbād=dict(Hāora=236, Patna=240, Guwāhāti=597, Rānchi=122, Jamshedpur=113),
                Hāora=dict(Guwāhāti=529, Kolkāta=3, Bhubaneshwar=364, Dhanbād=236, Jamshedpur=221),
                Kolkāta=dict(Guwāhāti=526, Hāora=3, Bhubaneshwar=367),
                Bezwāda=dict(Chennai=383, Vishākhapatnam=317, Raipur=535, Bhilai=529, Guntūr=29, Warangal=194),
                Vishākhapatnam=dict(Bezwāda=317, Raipur=427, Chennai=611, Bilāspur=509, Bhubaneshwar=385),
                Raipur=dict(Bezwāda=535, Bhilai=21, Vishākhapatnam=427, Bilāspur=108),
                Lucknow=dict(Allahābād=179, Cawnpore=74, Gorakhpur=240, Bareilly=226),
                Allahābād=dict(Gorakhpur=210, Cawnpore=189, Lucknow=179, Mirzāpur=80, Bilāspur=367, Jabalpur=319),
                Pune=dict(Mumbai=120, Kolhāpur=206, Kalyān=109, Nāsik=164, Solāpur=236, Mālegaon=237, Aurangābād=215),
                Mumbai=dict(Kolhāpur=300, Pune=120, Rājkot=418, Sūrat=232, Kalyān=35, Thāne=14),
                Trichinopoly=dict(Madurai=118, Salem=108, Chennai=304),
                Madurai=dict(Chennai=422, Trichinopoly=118, Salem=192, Coimbatore=174),
                Mysore=dict(Bangalore=126, Shimoga=214, Coimbatore=149, Kochi=263),
                Bhilai=dict(Jabalpur=265, Bezwāda=529, Nāgpur=243, Warangal=408, Raipur=21, Bilāspur=119),
                Jabalpur=dict(Nāgpur=240, Allahābād=319, Cawnpore=369, Bilāspur=241, Bhilai=265, Bhopāl=257,
                              Gwalior=381),
                Nāgpur=dict(Bhilai=243, Warangal=357, Hyderābād=426, Jabalpur=240, Bhopāl=289),
                Hyderābād=dict(Solāpur=274, Bangalore=496, Warangal=137, Chennai=513, Guntūr=241, Bhopāl=663,
                               Nāgpur=426),
                Patna=dict(Rānchi=251, Dhanbād=240, Guwāhāti=663, Gorakhpur=218, Vārānasi=216),
                Rānchi=dict(Dhanbād=122, Patna=251, Jamshedpur=106, Bhubaneshwar=347, Vārānasi=320, Bilāspur=365),
                Vārānasi=dict(Patna=216, Rānchi=320, Gorakhpur=164, Mirzāpur=47, Bilāspur=366),
                Bhubaneshwar=dict(Kolkāta=367, Vishākhapatnam=385, Chennai=993, Jamshedpur=284, Hāora=364, Bilāspur=447,
                                  Rānchi=347),
                Cawnpore=dict(Jabalpur=369, Allahābād=189, Lucknow=74, Gwalior=214, Bareilly=228),
                Bareilly=dict(Gorakhpur=428, Srīnagar=772, Lucknow=226, Cawnpore=228, Gwalior=267, Morādābād=83,
                              Āgra=191, Alīgarh=141),
                Srīnagar=dict(Bareilly=772, Morādābād=693, Sahāranpur=526, Chandīgarh=417, Amritsar=274, Jalandhar=315,
                              Jodhpur=883),
                Morādābād=dict(Bareilly=83, Sahāranpur=170, Srīnagar=693, Meerut=103, Alīgarh=125),
                Kalyān=dict(Mumbai=35, Pune=109, Nāsik=105, Thāne=22, Bhiwandi=11),
                Nāsik=dict(Kalyān=105, Bhiwandi=108, Sūrat=163, Pune=164, Vadodara=262, Mālegaon=100),
                Aurangābād=dict(Pune=215, Indore=320, Mālegaon=109, Bhopāl=432, Solāpur=253),
                Salem=dict(Madurai=192, Chennai=278, Trichinopoly=108, Bangalore=160, Coimbatore=149),
                Kochi=dict(Mysore=263, Shimoga=446, Kolhāpur=780, Coimbatore=137, Thiruvananthapuram=176),
                Thiruvananthapuram=dict(Kochi=176, Madurai=206, Coimbatore=278),
                Warangal=dict(Bhilai=408, Bezwāda=194, Nāgpur=357, Guntūr=207, Hyderābād=137),
                Guntūr=dict(Chennai=358, Hyderābād=241, Bezwāda=29, Warangal=207),
                Mirzāpur=dict(Gorakhpur=195, Vārānasi=47, Bilāspur=338, Allahābād=80),
                Jamshedpur=dict(Dhanbād=113, Rānchi=106, Hāora=221, Bhubaneshwar=284),
                Sahāranpur=dict(Meerut=109, Morādābād=170, Srīnagar=526, Ghāziābād=144, Chandīgarh=112, Delhi=147),
                Meerut=dict(Morādābād=103, Alīgarh=128, Ghāziābād=45, Sahāranpur=109),
                Chandīgarh=dict(Sahāranpur=112, Srīnagar=417, Jalandhar=132, Ludhiāna=92, Delhi=233, New_Delhi=239),
                Amritsar=dict(Jalandhar=74, Ludhiāna=122, Jaipur=532, Jodhpur=619, Srīnagar=274),
                Jalandhar=dict(Srīnagar=315, Chandīgarh=132, Ludhiāna=53, Amritsar=74),
                Bhiwandi=dict(Kalyān=11, Nāsik=108, Thāne=17, Sūrat=209),
                Gwalior=dict(Jabalpur=381, Cawnpore=214, Jaipur=244, Kota=263, Bhopāl=338, Āgra=107, Bareilly=267),
                Jaipur=dict(Kota=193, Ludhiāna=443, Jodhpur=290, Amritsar=532, Gwalior=244, Āgra=214, Farīdābād=221,
                            New_Delhi=230),
                Kota=dict(Gwalior=263, Indore=273, Bhopāl=268, Jaipur=193, Vadodara=417, Jodhpur=306),
                Sūrat=dict(Bhiwandi=209, Rājkot=244, Mumbai=232, Thāne=221, Nāsik=163, Ahmedabad=208, Vadodara=131),
                Rājkot=dict(Mumbai=418, Jodhpur=498, Ahmedabad=200, Sūrat=244),
                Mālegaon=dict(Aurangābād=109, Nāsik=100, Pune=237, Indore=276, Vadodara=239),
                Indore=dict(Mālegaon=276, Bhopāl=171, Aurangābād=320, Kota=273, Vadodara=275),
                Coimbatore=dict(Bangalore=230, Salem=149, Madurai=174, Mysore=149, Thiruvananthapuram=278, Kochi=137),
                Bilāspur=dict(Vārānasi=366, Rānchi=365, Mirzāpur=338, Raipur=108, Bhilai=119, Vishākhapatnam=509,
                              Bhubaneshwar=447, Allahābād=367, Jabalpur=241),
                Thāne=dict(Mumbai=14, Kalyān=22, Sūrat=221, Bhiwandi=17),
                Bhopāl=dict(Kota=268, Gwalior=338, Indore=171, Nāgpur=289, Jabalpur=257, Aurangābād=432, Solāpur=639,
                            Hyderābād=663),
                Ludhiāna=dict(Chandīgarh=92, Jalandhar=53, Amritsar=122, Jaipur=443, New_Delhi=286),
                Jodhpur=dict(Kota=306, Jaipur=290, Srīnagar=883, Amritsar=619, Vadodara=444, Rājkot=498, Ahmedabad=365),
                Alīgarh=dict(Morādābād=125, Bareilly=141, Farīdābād=97, Āgra=78, Meerut=128, Ghāziābād=108),
                Āgra=dict(Jaipur=214, Farīdābād=156, Alīgarh=78, Gwalior=107, Bareilly=191),
                Farīdābād=dict(Jaipur=221, Āgra=156, New_Delhi=22, Alīgarh=97, Delhi=27, Ghāziābād=27),
                New_Delhi=dict(Jaipur=230, Ludhiāna=286, Chandīgarh=239, Delhi=5, Farīdābād=22),
                Vadodara=dict(Indore=275, Kota=417, Mālegaon=239, Sūrat=131, Nāsik=262, Jodhpur=444, Ahmedabad=103),
                Ghāziābād=dict(Alīgarh=108, Farīdābād=27, Meerut=45, Sahāranpur=144, Delhi=19),
                Delhi=dict(Sahāranpur=147, Chandīgarh=233, Ghāziābād=19, New_Delhi=5, Farīdābād=27),
                Ahmedabad=dict(Jodhpur=365, Vadodara=103, Rājkot=200, Sūrat=208)
            ))

            maps.romania_map.locations = dict(
                Delhi=(589, 843),
                Mumbai=(284, 450),
                Kolkāta=(1372, 593),
                Bangalore=(615, 200),
                Chennai=(803, 204),
                Hyderābād=(677, 380),
                Pune=(353, 427),
                Ahmedabad=(263, 612),
                Allahābād=(914, 711),
                Sūrat=(281, 536),
                Lucknow=(851, 769),
                Jaipur=(494, 772),
                Cawnpore=(807, 753),
                Mirzāpur=(965, 699),
                Nāgpur=(720, 535),
                Ghāziābād=(603, 843),
                Vadodara=(307, 582),
                Vishākhapatnam=(1017, 395),
                Indore=(493, 600),
                Thāne=(290, 454),
                Bhopāl=(603, 621),
                Patna=(1145, 718),
                Bilāspur=(926, 576),
                Ludhiāna=(493, 935),
                Āgra=(645, 782),
                Kalyān=(304, 457),
                Madurai=(652, 75),
                Jamshedpur=(1218, 603),
                Nāsik=(348, 488),
                Farīdābād=(596, 834),
                Aurangābād=(456, 483),
                Rājkot=(138, 582),
                Meerut=(623, 857),
                Jabalpur=(779, 618),
                Dhanbād=(1236, 644),
                Vārānasi=(996, 706),
                Srīnagar=(419, 1066),
                Amritsar=(424, 965),
                Alīgarh=(649, 811),
                Guwāhāti=(1609, 741),
                Bhilai=(885, 538),
                Hāora=(1369, 594),
                Rānchi=(1159, 626),
                Gwalior=(657, 743),
                Bezwāda=(827, 345),
                Chandīgarh=(559, 928),
                Jodhpur=(295, 746),
                Mysore=(549, 172),
                Raipur=(899, 539),
                Kota=(491, 700),
                New_Delhi=(588, 841),
                Bareilly=(743, 831),
                Coimbatore=(571, 119),
                Solāpur=(497, 392),
                Trichinopoly=(692, 112),
                Hubli=(442, 298),
                Jalandhar=(474, 952),
                Bhubaneshwar=(1193, 499),
                Morādābād=(697, 851),
                Kolhāpur=(379, 353),
                Thiruvananthapuram=(566, 16),
                Bhiwandi=(297, 459),
                Sahāranpur=(612, 897),
                Warangal=(756, 405),
                Salem=(655, 145),
                Mālegaon=(401, 511),
                Kochi=(523, 76),
                Gorakhpur=(1021, 765),
                Shimoga=(473, 238),
                Guntūr=(816, 336)
            )
        if country == "Russia":
            maps.romania_map = UndirectedGraph(dict(
                Arkhangelsk=dict(Kirov=817, Vologda=593, Saint_Petersburg=734),
                Kirov=dict(Perm=390, Arkhangelsk=817, Yaroslavl=585, Vologda=563, Izhevsk=287, Ivanovo=542),
                Vladivostok=dict(Khabarovsk=645),
                Perm=dict(Nizhniy_Tagil=219, Kirov=390, Yekaterinburg=292, Izhevsk=222),
                Nizhniy_Tagil=dict(Surgut=841, Tyumen=342, Yekaterinburg=126, Perm=219),
                Surgut=dict(Tomsk=845, Tyumen=639, Nizhniy_Tagil=841),
                Tomsk=dict(Krasnoyarsk=491, Kemerovo=146, Novosibirsk=205, Omsk=742, Surgut=845),
                Astrakhan=dict(Makhachkala=374, Orenburg=793, Volzhskiy=365, Volgograd=372, Rostov=639, Stavropol=489),
                Krasnoyarsk=dict(Novokuznetsk=445, Irkutsk=850, Tomsk=491, Kemerovo=433),
                Irkutsk=dict(Krasnoyarsk=850, Ulan_Ude=231, Chita=627),
                Ulan_Ude=dict(Chita=404, Irkutsk=231),
                Chita=dict(Irkutsk=627, Ulan_Ude=404,Khabarovsk=1100),
                Orenburg=dict(Astrakhan=793, Volzhskiy=804, Saratov=627, Samara=372, Magnitogorsk=320),
                Novokuznetsk=dict(Barnaul=224, Krasnoyarsk=445, Novosibirsk=307, Kemerovo=188),
                Volzhskiy=dict(Astrakhan=365, Volgograd=20, Belgorod=620, Saratov=318, Orenburg=804),
                Volgograd=dict(Astrakhan=372, Volzhskiy=20, Rostov=392, Belgorod=606),
                Belgorod=dict(Volgograd=606, Rostov=438, Volzhskiy=620, Saratov=664, Kursk=129, Voronezh=217),
                Rostov=dict(Astrakhan=639, Volgograd=392, Belgorod=438, Stavropol=299, Krasnodar=251),
                Kemerovo=dict(Krasnoyarsk=433, Novokuznetsk=188, Tomsk=146, Novosibirsk=202),
                Vologda=dict(Kirov=563, Cherepovets=113, Yaroslavl=177, Arkhangelsk=593, Saint_Petersburg=545),
                Yaroslavl=dict(Tver=250, Cherepovets=202, Vologda=177, Ivanovo=97, Kirov=585),
                Tver=dict(Ivanovo=308, Yaroslavl=250, Saint_Petersburg=473, Cherepovets=278, Kaliningrad=990,
                          Vladimir=286, Smolensk=334, Balashikha=171, Moscow=161),
                Ivanovo=dict(Yaroslavl=97, Kirov=542, Tver=308, Vladimir=103, Izhevsk=740, Nizhniy_Novgorod=198),
                Cherepovets=dict(Tver=278, Yaroslavl=202, Saint_Petersburg=437, Vologda=113),
                Saint_Petersburg=dict(Kaliningrad=826, Tver=473, Vologda=545, Cherepovets=437),
                Kaliningrad=dict(Tver=990, Smolensk=740, Bryansk=919),
                Sochi=dict(Krasnodar=171, Makhachkala=631, Stavropol=242),
                Krasnodar=dict(Rostov=251, Stavropol=235, Sochi=171),
                Novosibirsk=dict(Kemerovo=202, Barnaul=194, Novokuznetsk=307, Tomsk=205, Omsk=607),
                Barnaul=dict(Novokuznetsk=224, Novosibirsk=194, Omsk=699),
                Saratov=dict(Belgorod=664, Voronezh=468, Volzhskiy=318, Lipetsk=453, Penza=197, Samara=334,
                             Orenburg=627),
                Voronezh=dict(Belgorod=217, Saratov=468, Lipetsk=108, Kursk=208),
                Lipetsk=dict(Voronezh=108, Kursk=252, Saratov=453, Bryansk=357, Penza=367, Tula=219),
                Kursk=dict(Voronezh=208, Belgorod=129, Lipetsk=252, Bryansk=208),
                Smolensk=dict(Tver=334, Bryansk=228, Kaliningrad=740, Kaluga=272, Moscow=369),
                Vladimir=dict(Ivanovo=103, Ryazan=173, Nizhniy_Novgorod=223, Cheboksary=423, Balashikha=157, Tver=286),
                Penza=dict(Lipetsk=367, Samara=340, Saratov=197, Tolyatti=295, Ulyanovsk=253, Ryazan=380, Tula=498),
                Ryazan=dict(Cheboksary=504, Moscow=183, Penza=380, Tula=143, Kaluga=222, Vladimir=173, Balashikha=172,
                            Kazan=609, Ulyanovsk=559),
                Cheboksary=dict(Vladimir=423, Nizhniy_Novgorod=201, Izhevsk=374, Kazan=122, Ryazan=504),
                Stavropol=dict(Astrakhan=489, Rostov=299, Makhachkala=496, Sochi=242, Krasnodar=235),
                Makhachkala=dict(Astrakhan=374, Stavropol=496),
                Omsk=dict(Novosibirsk=607, Barnaul=699, Tomsk=742, Magnitogorsk=948, Chelyabinsk=762, Tyumen=544),
                Bryansk=dict(Kursk=208, Kaluga=189, Kaliningrad=919, Smolensk=228, Lipetsk=357, Tula=238),
                Kaluga=dict(Smolensk=272, Moscow=160, Ryazan=222, Tula=94, Bryansk=189),
                Nizhniy_Novgorod=dict(Ivanovo=198, Vladimir=223, Cheboksary=201, Izhevsk=566),
                Moscow=dict(Smolensk=369, Kaluga=160, Tver=161, Ryazan=183, Balashikha=21),
                Samara=dict(Tolyatti=59, Penza=340, Saratov=334, Magnitogorsk=592, Orenburg=372),
                Tolyatti=dict(Penza=295, Ulyanovsk=112, Samara=59, Ufa=446, Magnitogorsk=636),
                Tyumen=dict(Omsk=544, Surgut=639, Nizhniy_Tagil=342, Chelyabinsk=339, Yekaterinburg=300),
                Ulyanovsk=dict(Penza=253, Ryazan=559, Tolyatti=112, Kazan=170, Ufa=491, Naberezhnyye_Chelny=295),
                Tula=dict(Penza=498, Lipetsk=219, Bryansk=238, Ryazan=143, Kaluga=94),
                Balashikha=dict(Ryazan=172, Moscow=21, Vladimir=157, Tver=171),
                Magnitogorsk=dict(Tolyatti=636, Samara=592, Orenburg=320, Ufa=250, Omsk=948, Chelyabinsk=249),
                Ufa=dict(Ulyanovsk=491, Tolyatti=446, Naberezhnyye_Chelny=253, Chelyabinsk=351, Magnitogorsk=250),
                Izhevsk=dict(Perm=222, Nizhniy_Novgorod=566, Kirov=287, Ivanovo=740, Yekaterinburg=449, Cheboksary=374,
                             Naberezhnyye_Chelny=139, Kazan=278),
                Yekaterinburg=dict(Tyumen=300, Nizhniy_Tagil=126, Perm=292, Izhevsk=449, Naberezhnyye_Chelny=526,
                                   Chelyabinsk=193),
                Kazan=dict(Izhevsk=278, Cheboksary=122, Ryazan=609, Naberezhnyye_Chelny=201, Ulyanovsk=170),
                Naberezhnyye_Chelny=dict(Yekaterinburg=526, Izhevsk=139, Ulyanovsk=295, Kazan=201, Ufa=253,
                                         Chelyabinsk=574),
                Chelyabinsk=dict(Naberezhnyye_Chelny=574, Yekaterinburg=193, Magnitogorsk=249, Ufa=351, Omsk=762,
                                 Tyumen=339)
            ))

            maps.romania_map.locations = dict(
                Moscow=(968, 384),
                Saint_Petersburg=(935, 497),
                Novosibirsk=(1170, 365),
                Yekaterinburg=(1070, 413),
                Nizhniy_Novgorod=(996, 400),
                Kazan=(1019, 385),
                Chelyabinsk=(1074, 368),
                Omsk=(1127, 363),
                Samara=(1024, 315),
                Rostov=(977, 155),
                Ufa=(1050, 356),
                Krasnoyarsk=(1214, 391),
                Voronezh=(975, 274),
                Perm=(1051, 445),
                Volgograd=(999, 194),
                Krasnodar=(974, 96),
                Saratov=(1005, 271),
                Tyumen=(1092, 422),
                Tolyatti=(1020, 324),
                Izhevsk=(1037, 414),
                Barnaul=(1174, 320),
                Ulyanovsk=(1016, 345),
                Irkutsk=(1265, 291),
                Khabarovsk=(1402, 188),
                Yaroslavl=(978, 434),
                Vladivostok=(1388, 44),
                Makhachkala=(1012, 40),
                Tomsk=(1179, 404),
                Orenburg=(1046, 277),
                Kemerovo=(1184, 373),
                Novokuznetsk=(1189, 330),
                Ryazan=(977, 354),
                Astrakhan=(1014, 131),
                Naberezhnyye_Chelny=(1033, 383),
                Penza=(1001, 315),
                Lipetsk=(977, 300),
                Kirov=(1021, 461),
                Cheboksary=(1011, 394),
                Tula=(968, 342),
                Kaliningrad=(892, 356),
                Balashikha=(969, 385),
                Kursk=(961, 276),
                Stavropol=(987, 96),
                Ulan_Ude=(1280, 278),
                Tver=(960, 414),
                Magnitogorsk=(1063, 320),
                Sochi=(977, 57),
                Ivanovo=(983, 418),
                Bryansk=(953, 317),
                Belgorod=(963, 245),
                Surgut=(1128, 532),
                Vladimir=(980, 394),
                Nizhniy_Tagil=(1067, 442),
                Arkhangelsk=(981, 621),
                Chita=(1306, 284),
                Kaluga=(962, 351),
                Smolensk=(943, 358),
                Volzhskiy=(1000, 197),
                Cherepovets=(969, 475),
                Vologda=(978, 477)
            )

        if country == "Ukraine":
            maps.romania_map = UndirectedGraph(dict(
                Luhansk=dict(Khrustalnyi=58, Sievierodonetsk=74, Alchevsk=41, Chuhuiv=237, Sumy=415),
                Khrustalnyi=dict(Rovenky=31, Luhansk=58, Alchevsk=39, Mariupol=151, Khartsyzk=59),
                Rovenky=dict(Luhansk=55, Khrustalnyi=31, Mariupol=171),
                Chernivtsi=dict(Chornomorsk=419, Kamianets_Podilskyi=64, Mukacheve=238, Ternopil=143,
                                Ivano_Frankivsk=114),
                Chornomorsk=dict(Kamianets_Podilskyi=404, Chernivtsi=419, Mukacheve=643, Sevastopol=293,
                                 Yevpatoriia=242, Kherson=153, Odesa=20),
                Kamianets_Podilskyi=dict(Chernivtsi=64, Ternopil=121, Chornomorsk=404, Odesa=396, Vinnytsia=147,
                                         Khmelnytskyi=87),
                Ternopil=dict(Chernivtsi=143, Rivne=125, Ivano_Frankivsk=96, Lviv=116, Lutsk=132, Chervonohrad=133,
                              Kamianets_Podilskyi=121, Khmelnytskyi=102),
                Alchevsk=dict(Luhansk=41, Khrustalnyi=39, Khartsyzk=68, Sievierodonetsk=58, Horlivka=54),
                Sievierodonetsk=dict(Alchevsk=58, Luhansk=74, Horlivka=74, Sloviansk=63, Chuhuiv=163),
                Mukacheve=dict(Chornomorsk=643, Drohobych=116, Sevastopol=930, Chernivtsi=238, Stryi=122,
                               Ivano_Frankivsk=156),
                Drohobych=dict(Uzhhorod=119, Chervonohrad=126, Lviv=66, Stryi=27, Mukacheve=116),
                Uzhhorod=dict(Mukacheve=36, Drohobych=119),
                Sevastopol=dict(Chornomorsk=293, Yevpatoriia=68, Kerch=245, Simferopol=59),
                Rivne=dict(Lutsk=66, Chernihiv=365, Novohrad_Volynskyi=97, Khmelnytskyi=143, Ternopil=125),
                Lutsk=dict(Ternopil=132, Rivne=66, Chervonohrad=87),
                Chernihiv=dict(Lutsk=424, Rivne=365, Novohrad_Volynskyi=275, Brovary=114, Bucha=128),
                Novohrad_Volynskyi=dict(Chernihiv=275, Vinnytsia=162, Khmelnytskyi=137, Rivne=97, Bucha=183,
                                        Zhytomyr=81),
                Ivano_Frankivsk=dict(Chernivtsi=114, Mukacheve=156, Ternopil=96, Lviv=113, Stryi=72),
                Lviv=dict(Ivano_Frankivsk=113, Ternopil=116, Chervonohrad=61, Drohobych=66, Stryi=67),
                Kherson=dict(Yevpatoriia=169, Chornomorsk=153, Nikopol=169, Melitopol=212, Kryvyi_Rih=152, Odesa=143,
                             Mykolaiv=58),
                Yevpatoriia=dict(Melitopol=238, Chornomorsk=242, Kherson=169, Simferopol=64, Sevastopol=68),
                Melitopol=dict(Kherson=212, Nikopol=112, Simferopol=231, Yevpatoriia=238, Zaporizhzhia=113,
                               Berdiansk=108, Kerch=186),
                Nikopol=dict(Kryvyi_Rih=84, Kherson=169, Melitopol=112, Kamianske=105, Zaporizhzhia=65),
                Kryvyi_Rih=dict(Kherson=152, Kremenchuk=130, Mykolaiv=145, Kropyvnytskyi=104, Kamianske=115,
                                Nikopol=84),
                Konotop=dict(Myrhorod=144, Cherkasy=215, Chernihiv=135, Brovary=187),
                Myrhorod=dict(Sumy=134, Poltava=81, Kremenchuk=99, Konotop=144, Cherkasy=125),
                Sumy=dict(Konotop=117, Myrhorod=134, Chuhuiv=178, Poltava=149, Kharkiv=142),
                Chervonohrad=dict(Ternopil=133, Lviv=61, Uzhhorod=240, Drohobych=126),
                Stryi=dict(Lviv=67, Ivano_Frankivsk=72, Drohobych=27, Mukacheve=122),
                Poltava=dict(Sumy=149, Myrhorod=81, Kamianske=118, Kremenchuk=99, Kharkiv=128, Dnipro=127,
                             Pavlohrad=150),
                Khartsyzk=dict(Khrustalnyi=59, Alchevsk=68, Mariupol=109, Donetsk=25, Horlivka=33, Makiivka=14),
                Simferopol=dict(Kerch=190, Melitopol=231, Sevastopol=59, Yevpatoriia=64),
                Kerch=dict(Melitopol=186, Simferopol=190, Rovenky=376, Berdiansk=159, Mariupol=216),
                Zaporizhzhia=dict(Nikopol=65, Dnipro=70, Kamianske=84, Pavlohrad=93, Berdiansk=172, Melitopol=113),
                Kremenchuk=dict(Kropyvnytskyi=105, Poltava=99, Myrhorod=99, Cherkasy=107, Kamianske=107,
                                Kryvyi_Rih=130),
                Kropyvnytskyi=dict(Kryvyi_Rih=104, Uman=152, Mykolaiv=172, Cherkasy=104, Kremenchuk=105),
                Mykolaiv=dict(Kropyvnytskyi=172, Kherson=58, Kryvyi_Rih=145, Uman=238, Odesa=110),
                Uman=dict(Mykolaiv=238, Kropyvnytskyi=152, Odesa=255, Vinnytsia=140, Cherkasy=154, Bila_Tserkva=116),
                Cherkasy=dict(Myrhorod=125, Uman=154, Kremenchuk=107, Kropyvnytskyi=104, Bila_Tserkva=145, Konotop=215,
                              Brovary=149),
                Vinnytsia=dict(Zhytomyr=116, Odesa=349, Kamianets_Podilskyi=147, Khmelnytskyi=106,
                               Novohrad_Volynskyi=162, Uman=140, Bila_Tserkva=137),
                Zhytomyr=dict(Novohrad_Volynskyi=81, Bucha=116, Bila_Tserkva=116, Vinnytsia=116),
                Chuhuiv=dict(Sievierodonetsk=163, Luhansk=237, Sumy=178, Kharkiv=36, Sloviansk=127, Pavlohrad=157,
                             Kramatorsk=139),
                Horlivka=dict(Alchevsk=54, Khartsyzk=33, Sievierodonetsk=74, Myrnohrad=61, Kramatorsk=58, Sloviansk=68,
                              Makiivka=32, Donetsk=41),
                Mariupol=dict(Rovenky=171, Khrustalnyi=151, Kerch=216, Donetsk=99, Khartsyzk=109, Myrnohrad=132,
                              Berdiansk=72),
                Kamianske=dict(Dnipro=31, Poltava=118, Kremenchuk=107, Kryvyi_Rih=115, Zaporizhzhia=84, Nikopol=105),
                Dnipro=dict(Poltava=127, Kamianske=31, Zaporizhzhia=70, Pavlohrad=61),
                Odesa=dict(Mykolaiv=110, Uman=255, Chornomorsk=20, Kherson=143, Kamianets_Podilskyi=396, Vinnytsia=349),
                Khmelnytskyi=dict(Kamianets_Podilskyi=87, Vinnytsia=106, Ternopil=102, Novohrad_Volynskyi=137,
                                  Rivne=143),
                Kharkiv=dict(Sumy=142, Chuhuiv=36, Pavlohrad=166, Poltava=128),
                Pavlohrad=dict(Dnipro=61, Chuhuiv=157, Poltava=150, Kharkiv=166, Berdiansk=207, Zaporizhzhia=93,
                               Myrnohrad=105, Kramatorsk=125),
                Sloviansk=dict(Horlivka=68, Sievierodonetsk=63, Kramatorsk=17, Chuhuiv=127),
                Donetsk=dict(Horlivka=41, Myrnohrad=51, Mariupol=99, Khartsyzk=25, Makiivka=12),
                Myrnohrad=dict(Donetsk=51, Mariupol=132, Berdiansk=175, Pavlohrad=105, Horlivka=61, Kramatorsk=51),
                Berdiansk=dict(Mariupol=72, Myrnohrad=175, Pavlohrad=207, Zaporizhzhia=172, Melitopol=108, Kerch=159),
                Bucha=dict(Chernihiv=128, Novohrad_Volynskyi=183, Bila_Tserkva=83, Zhytomyr=116, Brovary=39, Kyiv=23),
                Bila_Tserkva=dict(Uman=116, Kyiv=78, Bucha=83, Vinnytsia=137, Zhytomyr=116, Cherkasy=145, Brovary=92),
                Kyiv=dict(Bucha=23, Brovary=20, Bila_Tserkva=78),
                Brovary=dict(Konotop=187, Cherkasy=149, Chernihiv=114, Bucha=39, Bila_Tserkva=92, Kyiv=20),
                Kramatorsk=dict(Myrnohrad=51, Horlivka=58, Pavlohrad=125, Chuhuiv=139, Sloviansk=17),
                Makiivka=dict(Khartsyzk=14, Donetsk=12, Horlivka=32)
            ))

            maps.romania_map.locations = dict(
                Kyiv=(759, 699),
                Kharkiv=(1280, 646),
                Odesa=(778, 237),
                Dnipro=(1171, 468),
                Donetsk=(1423, 415),
                Zaporizhzhia=(1180, 395),
                Lviv=(167, 628),
                Kryvyi_Rih=(1016, 403),
                Sevastopol=(1034, 19),
                Mykolaiv=(894, 294),
                Mariupol=(1402, 313),
                Luhansk=(1563, 482),
                Vinnytsia=(568, 555),
                Simferopol=(1085, 59),
                Makiivka=(1438, 420),
                Poltava=(1128, 597),
                Chernihiv=(829, 820),
                Kherson=(949, 255),
                Cherkasy=(899, 582),
                Khmelnytskyi=(438, 579),
                Zhytomyr=(589, 676),
                Chernivtsi=(340, 448),
                Sumy=(1149, 752),
                Horlivka=(1450, 453),
                Rivne=(369, 718),
                Ivano_Frankivsk=(229, 521),
                Kamianske=(1132, 473),
                Kropyvnytskyi=(918, 473),
                Ternopil=(310, 596),
                Lutsk=(285, 733),
                Kremenchuk=(1024, 539),
                Bila_Tserkva=(722, 623),
                Kramatorsk=(1401, 498),
                Melitopol=(1201, 278),
                Sievierodonetsk=(1485, 524),
                Kerch=(1302, 105),
                Drohobych=(118, 571),
                Khrustalnyi=(1528, 428),
                Uzhhorod=(8, 487),
                Berdiansk=(1330, 270),
                Sloviansk=(1407, 515),
                Nikopol=(1109, 365),
                Pavlohrad=(1247, 474),
                Yevpatoriia=(1018, 89),
                Alchevsk=(1514, 468),
                Brovary=(783, 706),
                Konotop=(1004, 790),
                Kamianets_Podilskyi=(399, 493),
                Mukacheve=(46, 465),
                Uman=(731, 501),
                Chervonohrad=(185, 691),
                Khartsyzk=(1455, 418),
                Stryi=(150, 559),
                Chornomorsk=(771, 217),
                Novohrad_Volynskyi=(495, 714),
                Myrnohrad=(1374, 449),
                Rovenky=(1566, 424),
                Myrhorod=(1041, 642),
                Bucha=(733, 710),
                Chuhuiv=(1320, 627)
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