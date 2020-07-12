import math
import time


# from graphTree import TreeGraph
from collections import defaultdict

import numpy as np

from Gui import Gui
from model import *
from naive_player import NaivePlayer
from reward import Reward
from rewardBasedPlayer import RewardBasedPlayer, get_max_difference, randomMax

FIRST_SEARCH = 5
FOE_SEARCH = 3
SEARCH_TIME = 0.1
GAMMA = 0.9
MAX = 10000
USING_GUI =True


class MI_PlayerSimuSave(RewardBasedPlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.search_agent = Mcts_search(_id, False, self)
        #self.using_reward = 'RewardPro'


    def SelectMove(self, moves, game_state):
        player_order = self.get_player_order(game_state)
        moves = self.filtering_moves(game_state.players[self.id], moves)
        move = self.search_agent.search(moves, game_state, player_order)
        return move


class Mcts_search:
    def __init__(self, _id, log, agent):
        self.id = _id
        self.log = log
        self.agent = agent

    def search(self, moves, game_state, player_order):
        self.tree = []
        self.init_game_state = game_state
        self.init_moves = moves
        self.player_order = player_order

        state = self.init_game_state
        parent = None
        f_move = None
        act_id = self.id
        root_node = Node(state, parent, f_move, act_id, self.tree)
        self.root_node = root_node

        self.time_monitor = defaultdict(float)


        start = time.time()
        n = 0
        # while n<= 4:
        # while True:
        nodes_len = (2**(len(moves)**(1/2)))

        while time.time()-start<len(moves)*SEARCH_TIME:
        #while n<nodes_len:
        #while time.time() - start < 0.2474:
            #a = input('input')
            n += 1
            self.one_search(root_node)
        print('searched times', n)
        print('nodes:', len(self.tree))
        print('{} finished'.format(str(self.agent.__class__)))
        print('seach duration', time.time() - start)
        print('distribute', self.time_monitor)
        print()

        dict = {}
        for m, (c, p) in root_node.moves.items():
            Q = get_max_difference(c.value, self.id) if c is not None else -1000
            dict[m] = Q
        move = randomMax(dict)
        track = self.get_predict_track(root_node, move)
        if USING_GUI:
            Gui(self.tree, 'mcts save')
        return move

    def get_predict_track(self, root_node, move):
        track = [move]
        node = root_node.moves[move][0]
        while True:
            node.mark = True
            id = node.act_id
            children = [c for m, (c,p) in node.moves.items() if c is not None] if node.moves is not None else []
            if len(children) == 0:
                break
            node = max(children, key=lambda x:get_max_difference(x.value, id))

            track.append((node.from_move[1], str(node.from_move[2]), str(id), node, node.value))

        return track

    def one_search(self, root_node):
        start = time.time()
        select_node, move = self.select(root_node)
        self.time_monitor['select']+=(time.time()-start)


        if self.log:
            print('select')
            print(select_node, move)

        start = time.time()
        node = self.expand(select_node, move)
        self.time_monitor['expand'] += (time.time() - start)

        if self.log:
            print('expand')
            print(node)


        start = time.time()
        node, result = self.simulate(node)
        self.time_monitor['simulate'] += (time.time() - start)

        if self.log:
            print(result)

        start = time.time()
        self.backup(node, result)
        self.time_monitor['back'] += (time.time() - start)


    def select(self, root_node):
        node = root_node
        while not node.is_end():
            if node.moves is None:
                moves = self.get_pre_prob(node.state,
                                          node.state.players[node.act_id].GetAvailableMoves(node.state),
                                          node.act_id, self.player_order)
                node.give_moves(moves)
            if not node.is_fully_expanded():
                return node, node.get_unexpanded_move()
            else:
                node = self.best_child(node)
        return node, None

    def best_child(self, p_node):
        if self.log:
            print('jump')
            print(p_node)
        node_v_para = 2 * math.log(p_node.visited)
        uct_dict = {}
        for m, (c, p) in p_node.moves.items():
            Q = get_max_difference(c.value, self.id) / max(c.value) if max(c.value) != 0 else 0
            #N = 1 / c.visited if c.visited != 0 else MAX
            N = ((node_v_para/c.visited)**(1/2)) if c.visited!=0 else MAX
            uct_value = Q + p + N
            uct_dict[c] = uct_value
        node = randomMax(uct_dict)
        return node


    def expand(self, p_node, move):
        if move is None:
            return p_node
        state = copy.deepcopy((p_node.state))
        state.ExecuteMove(p_node.act_id, move)
        parent = p_node
        f_move = move
        act_id = p_node.act_id + 1 if p_node.act_id < len(self.player_order) - 1 else 0
        return Node(state, parent, f_move, act_id, self.tree)

    def simulate(self, s_node):

        player_count = len(self.player_order)
        players = [Simu_Player(i, self.agent.using_reward) for i in range(player_count)]
        act_id = s_node.act_id

        node = s_node
        moves = self.get_pre_prob(node.state,
                                  node.state.players[node.act_id].GetAvailableMoves(node.state),
                                  node.act_id, self.player_order)
        node.give_moves(moves)
        while node.state.TilesRemaining():

            state = copy.deepcopy(node.state)
            #print(state)
            move = players[act_id].SelectMove(node.moves, state)
            #print(move)
            #print(move[1],str(move[2]))

            state.ExecuteMove(act_id, move)
            act_id = act_id + 1 if act_id + 1 < player_count else 0
            new_node = Node(state, node, move, act_id, self.tree)

            moves = self.get_pre_prob(new_node.state,
                                      new_node.state.players[new_node.act_id].GetAvailableMoves(new_node.state),
                                      new_node.act_id, self.player_order)
            new_node.give_moves(moves)

            node =  new_node

            if self.log:
                print(act_id)
                print('id', act_id)
                print('before')
                print(state.detail_str())

        state = copy.deepcopy(node.state)
        if self.log:
            print('simulate over')
        state.ExecuteEndOfRound()
        reward = [0] * player_count
        for i, plr in enumerate(state.players):
            reward[i] = state.players[i].score
        # print(state.detail_str())
        # print(reward)
        game_continuing = True
        for i in range(player_count):
            plr_state = state.players[i]
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows > 0:
                game_continuing = False
                break
        if not game_continuing:
            for i in range(player_count):
                state.players[i].EndOfGameScore()
                reward[i] = state.players[i].score
        else:
            for i, plr in enumerate(state.players):
                expect_score = eval(self.agent.using_reward)(state, i, self.player_order).get_round_expection()
                reward[i] = state.players[i].score + expect_score
        # print(reward)
        return node, reward

    def backup(self, node, result):
        update_node = node
        update_node.update(self.id, result)

        while True:
            update_node = update_node.parent
            if update_node is None:break
            update_node.update(self.id)

    def get_pre_prob(self, game_state, moves, act_id, player_order):
        threshold_most = FOE_SEARCH if act_id!= self.id else FIRST_SEARCH
        #threshold_impo = 4

        ft_moves = self.agent.filtering_moves(game_state.players[act_id], moves)
        move_dict = {}
        for move in ft_moves:
            reward, score_list = self.agent.get_place_reward(game_state, move, act_id, player_order)
            move_dict[move] = reward, score_list

        move_tuple = sorted(move_dict.items(), key=lambda x: x[1][0], reverse=True)[:threshold_most] if len(
            move_dict) > threshold_most else move_dict.items()

        move_prob_dict = {}
        sum_reward = sum([math.e**m[1][0] for m in move_tuple])
        for i, m in enumerate(move_tuple):
            move_prob_dict[m[0]] = None, (math.e**m[1][0])/sum_reward
        return move_prob_dict

class Node:
    def __init__(self, game_state, parent, from_move, act_id, tree):
        self.state = game_state
        self.parent = parent
        self.from_move = from_move
        if self.parent is not None:
            #print( self.parent.moves[from_move])
            self.parent.moves[from_move] = (self, self.parent.moves[from_move][1])
            peers = [c for m, (c, p) in self.parent.moves.items()]
            assert self in peers
        self.act_id = act_id
        self.value = [0] * len(game_state.players)
        tree.append(self)
        self.visited = 0
        self.name = 'n'+str(len(tree))
        self.mark = False
        self.moves = None


    def is_fully_expanded(self):
        for m, (c, p) in self.moves.items():
            if c is None:
                return False
        return True

    def give_moves(self, moves):
        self.moves = moves


    def get_unexpanded_move(self):
        unexp_dict = {}
        for m, (c, p) in self.moves.items():
            if c is None:
                unexp_dict[m] = p
        unexp_prob = sum(unexp_dict.values())
        assert len(unexp_dict) > 0
        # print(unexp_dict.values())
        for m, p in unexp_dict.items():
            unexp_dict[m] = p/unexp_prob
        # print(sum(unexp_dict.values()))
        unexp_m_list = [(k, v) for k, v in unexp_dict.items()]
        p = np.array([v for k, v in unexp_m_list])
        # print(p)
        index = np.random.choice([i for i in range(len(p))], p=p.ravel())
        m, _ = unexp_m_list[index]
        return m

    def is_end(self):
        return not self.state.TilesRemaining()


    def update(self, agent_id, result=None):
        self.visited += 1
        if result is not None:
            for i in range(len(self.value)):
                self.value[i] = (self.value[i]*(self.visited-1)+result[i])/self.visited
            return

        value_list = []
        for m, (c, p) in self.moves.items():
            if c is None or c.visited == 0:
                continue
            value = c.value.copy()
            value_list.append(value)
        value_list = sorted(value_list, key=lambda x: get_max_difference(x, self.act_id), reverse=True)
        self.value = value_list[0]


    def get_children(self):
        return [c for m, (c, p) in self.moves.items()]



    def info(self):
        info = '{:2},{}\nr:{}\np:{}\nv:{}\n{} {}\n{}'.format(
            self.from_move[1], self.from_move[2], ' ', ' ', self.visited,
            str(round(self.value[0], 2)), str(round(self.value[1], 2)),
            self.act_id) if self.from_move is not None else ''
        return info


class Simu_Player(RewardBasedPlayer):
    def __init__(self, _id, using_reward):
        super().__init__(_id)
        self.using_reward = using_reward

    def SelectMove(self, moves, game_state):
        move= max(moves.items(), key=lambda x:x[1][1])[0]
        return move

