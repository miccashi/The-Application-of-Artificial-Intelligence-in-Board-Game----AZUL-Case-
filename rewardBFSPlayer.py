import copy
import time
from collections import deque

from Gui import Gui
from reward import Reward
from rewardBasedPlayer import RewardBasedPlayer, get_max_difference
from state_monitor import set_search_time

FIRST_ACTION_NO_PRUNE = True
FIRST_ACTION_SELECT_NUM = 4
SELECT_NUM = 2
NO_PRUNE_THRESHOLD = 10
GAMMA = 0.9
USING_GUI = False

class BFS_Player(RewardBasedPlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.search_agent = BFS_search(_id, self)

    def SelectMove(self, moves, game_state):
        player_order = self.get_player_order(game_state)
        moves = self.filtering_moves(game_state.players[self.id], moves)
        move = self.search_agent.search(moves, game_state, player_order)
        return move


class BFS_search:
    def __init__(self, _id, agent):
        self.max_num = 1
        self.id = _id
        self.queue = deque()
        self.num = 0
        self.round = 0
        self.agent = agent

    def search(self, moves, game_state, player_order):
        self.tree = []
        self.init_moves = moves
        self.init_game_state = game_state
        root_node = Node(self.init_game_state, None, None, self.init_moves, self.id, (), 0, 0, self.tree)
        self.queue.append(root_node)
        self.player_order = player_order

        start = time.time()
        while len(self.queue) != 0:
            node = self.queue.popleft()
            children = self.get_successors(node, player_order)
            for c in children:
                if not c.state.TilesRemaining():
                    self.update_node(c)
                else:
                    self.queue.append(c)

        set_search_time('nodesBFS', len(self.tree))

        set_search_time('timeBFS', time.time() - start)

        print('search nodes:', len(self.tree))
        print('BFS search finished')
        print('search duration:', time.time() - start)

        print()

        children = sorted(root_node.children, key=lambda x:get_max_difference(x.value, self.id), reverse=True)
        # print('attacking')
        # for c in children:
        #     print(c.from_move[1], c.from_move[2], c.value, c.instant_reward)
        track = self.get_predict_track(root_node)
        if USING_GUI:
            Gui(self.tree)
        # print('track:')
        # for t in track:
        #     print(t)
        return children[0].from_move


    # get all the successors from a state which are regarded as the most valuable choices
    # the other successors are pruned by some certain mechanisms

    def get_successors(self, node, player_order, max_num = 2):
        if not node.state.TilesRemaining():
            return []
        moves = node.moves
        state = node.state
        act_id = node.act_id
        children = self.prune(state, moves, act_id, len(self.init_moves), node.layer, player_order)
        #print(children)
        nodes = []
        for c in children:
            gs_copy = copy.deepcopy(state)
            new_act_id = act_id+1 if act_id<len(player_order)-1 else 0
            gs_copy.ExecuteMove(act_id, c[0])
            new_moves = gs_copy.players[new_act_id].GetAvailableMoves(gs_copy)
            new_moves = self.agent.filtering_moves(gs_copy.players[new_act_id], new_moves)
            nodes.append(Node(gs_copy, node, c[0], new_moves, new_act_id, c[0], node.layer+1, c[1], self.tree))
        return nodes


    # the prune function to select the moves that are regarded as more valuable

    def prune(self, game_state, moves, act_id, init_moves_num, layer, player_order):

        # Threshold
        # if the number of the initial moves is less than a threshold, no pruning and expand all the state
        if init_moves_num <= NO_PRUNE_THRESHOLD:
            moves_data = [(move, self.get_place_reward(game_state, move, act_id, player_order)) for move in moves]
            return moves_data
        children = {}
        for move in moves:
            reward, score_list = self.get_place_reward(game_state, move, act_id, player_order)
            children[move] = reward, score_list

        if (FIRST_ACTION_NO_PRUNE and layer == 0):
            #print('init@@@@@@@@@@@@@@')
            children = sorted(children.items(), key=lambda x: x[1][0], reverse=True)[:FIRST_ACTION_SELECT_NUM] if len(
                children) > SELECT_NUM else children.items()
            return children
        else:
            children = sorted(children.items(), key=lambda x: x[1][0], reverse=True)[:SELECT_NUM] if len(
                children) > SELECT_NUM else children.items()
            return children


    def get_place_reward(self, game_state, move, act_id, player_order):
        reward, score_list = eval(self.agent.using_reward)(game_state, act_id, player_order).estimate(move)
        return reward, score_list


    def get_predict_track(self, root_node):
        track = []
        node = root_node
        while True:
            act_id = node.act_id
            node.mark = True
            node = max(node.children, key=lambda x:get_max_difference(x.value, act_id))
            track.append((node.from_move[1], str(node.from_move[2]), str(act_id)))
            if len(node.children) == 0:
                break
        return track

    def update_node(self, node):
        node.state.ExecuteEndOfRound()
        reward = [0] * len(node.value)

        for i, plr in enumerate(node.state.players):
            reward[i] = node.state.players[i].score

        game_continuing = True
        for i in range(len(node.value)):
            plr_state = node.state.players[i]
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows > 0:
                game_continuing = False
                break

        if not game_continuing:
            for i in range(len(node.value)):
                node.state.players[i].EndOfGameScore()
                reward[i] = node.state.players[i].score
        else:
            for i, plr in enumerate(node.state.players):
                expect_score = eval(self.agent.using_reward)(node.state, i, self.player_order).get_round_expection()
                reward[i] = node.state.players[i].score + expect_score
        node.value = reward
        update_node = node

        while True:
            update_node = update_node.parent
            if update_node.parent is None:
                break
            value_list = []
            for c in update_node.children:
                value = c.value.copy()
                value[update_node.act_id] = c.value[update_node.act_id]
                value_list.append(value)
            value_list = sorted(value_list, key=lambda x: get_max_difference(x, update_node.act_id), reverse=True)
            update_node.value = value_list[0]

class Node:
    def __init__(self, game_state, parent, move, moves, _id, edge, layer, instant_reward, tree):
        self.state = game_state
        self.parent = parent
        self.children = []
        if self.parent is not None:
            self.parent.children.append(self)
        self.moves = moves
        self.from_move = move
        self.act_id = _id
        self.value = [0]*len(game_state.players)
        self.edge = edge
        self.layer = layer
        self.instant_reward = instant_reward
        self.mark = False
        tree.append(self)

    def get_children(self):
        return self.children


    def is_end(self):
        return not self.state.TilesRemaining()


    def info(self):
        info = '{:2},{}\n{} {}'.format(
            self.from_move[1], self.from_move[2],
            str(round(self.value[0],2)),str(round(self.value[1],2))) if self.from_move is not None else ''
        return info

