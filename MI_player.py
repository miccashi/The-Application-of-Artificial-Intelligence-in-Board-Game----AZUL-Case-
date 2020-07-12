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
SEARCH_TIME = 0.4
GAMMA = 0.9
MAX = 10000
USING_GUI = False


class MI_Player(RewardBasedPlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.search_agent = Mcts_search(_id, False, self)
        self.using_reward = 'RewardPro'


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
        moves_dict = self.get_pre_prob(state, self.init_moves, self.id, self.player_order)
        i_r = Instant_reward()
        root_node = Node(state, parent, f_move, moves_dict, act_id, i_r, self.tree)
        self.root_node = root_node

        self.time_monitor = defaultdict(float)


        start = time.time()
        n = 0
        # while n<= 4:
        # while True:
        nodes_len = 5*(2**(len(moves)**(1/2)))

        while n<nodes_len:
            #a = input('input')
            n += 1
            self.one_search(root_node)
        print('searched times', n)
        print('nodes:', len(self.tree))
        print('MCTS search finished')
        print('seach duration', time.time() - start)
        print('distribute', self.time_monitor)
        print()


        # for m,(c, p) in root_node.moves.items():
        #     print(m[1], m[2], p, (c.value, get_max_difference(c.value, self.id)) if c is not None else ())

        dict = {}
        for m, (c, p) in root_node.moves.items():
            Q = get_max_difference(c.value, self.id) if c is not None else -1000
            dict[m] = Q

        move = randomMax(dict)
        track = self.get_predict_track(root_node, move)
        # print('track:')
        # for t in track:
        #     print(t)

        # for i, plr in enumerate(state.players):
        #     row_score, column_score, set_score, left_score = eval(self.agent.using_reward)(state, i,
        #                                                                                    self.player_order).get_round_expection()
        #     print(row_score, column_score, set_score, left_score)

        if USING_GUI:
            Gui(self.tree)
        return move


    def get_predict_track(self, root_node, move):
        track = [move]
        node = root_node.moves[move][0]
        while True:
            node.mark = True
            id = node.act_id
            children = [c for m, (c,p) in node.moves.items() if c is not None]
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
        node_dict = self.expand(select_node, move)
        self.time_monitor['expand'] += (time.time() - start)

        if self.log:
            print('expand')
            print(node_dict)

        start = time.time()
        choose_node = self.choose(node_dict)
        self.time_monitor['choose'] += (time.time() - start)

        if self.log:
            print('choose')
            print(choose_node.state, choose_node.act_id)

        start = time.time()
        result = self.simulate(choose_node)
        self.time_monitor['simulate'] += (time.time() - start)

        if self.log:
            print(result)

        start = time.time()
        self.backup(choose_node, result)
        self.time_monitor['back'] += (time.time() - start)

    def select(self, root_node):
        c_node = root_node
        while True:
            if c_node.is_end():
                return c_node, None
            if not c_node.is_fully_expanded():
                return c_node, c_node.get_unexpanded_move()
            node = self.jump(c_node)

            if node.act_id != self.id:
                return node, None
            else:
                c_node = node

    def jump(self, node):
        if self.log:
            print('jump')
            print(node)
        node_v_para = 2 * math.log(node.visited)
        uct_dict = {}
        for m, (c, p) in node.moves.items():
            Q = get_max_difference(c.value, self.id) / max(c.value) if max(c.value) != 0 else 0
            N = 1 / c.visited if c.visited != 0 else MAX
            # N = ((node_v_para/c.visited)**(1/2)) if c.visited!=0 else MAX
            uct_value = Q + p + N
            uct_dict[c] = uct_value
        uc_node = randomMax(uct_dict)
        uc_node_v_para = 2 * math.log(uc_node.visited) if uc_node.visited != 0 else 1

        uct_dict = {}
        for m, (c, p) in uc_node.moves.items():
            Q = get_max_difference(c.value, self.id) / max(c.value) if max(c.value) != 0 else 0
            N = 1 / c.visited if c.visited != 0 else MAX
            # N = ((uc_node_v_para/c.visited))**(1/2) if c.visited!=0 else MAX
            uct_value = Q + p + N
            uct_dict[c] = uct_value

        if len(uct_dict) == 0:
            if self.log:
                print('reach the end, jump to the uc_node')
                print(uc_node)
            return uc_node
        jump_node = randomMax(uct_dict)
        if self.log:
            print('normal jump to the node')
            print(jump_node)
        return jump_node




    def generate_node(self, p_node, move):
        state = copy.deepcopy((p_node.state))
        state.ExecuteMove(p_node.act_id, move)
        parent = p_node
        f_move = move
        act_id = p_node.act_id + 1 if p_node.act_id < len(self.player_order) - 1 else 0
        start = time.time()
        moves = self.get_pre_prob(state, state.players[act_id].GetAvailableMoves(state), act_id, self.player_order)
        self.time_monitor['g']+=(time.time()-start)
        i_r = Instant_reward()
        return Node(state, parent, f_move, moves, act_id, i_r, self.tree)


    def expand(self, node, move):
        default = {}
        default[node] = (node, 1)
        if move is None:
            return default
        uc_node = self.generate_node(node, move)
        moves = uc_node.moves
        if self.log:
            print('expanding')
            print('uc_node')
            print(uc_node.state)
        node_dict = {}
        for m, (c, p) in moves.items():
            c_node = self.generate_node(uc_node, m)
            node_dict[c_node] = (c, p)
            if self.log:
                print('c node')
                print(c_node.state)

        if len(node_dict) == 0:
            default = {}
            default[uc_node] = (uc_node, 1)
            return default
        return node_dict

    def choose(self, nodes_prob_dict):
        nodes_list = [(k,v) for k,v in nodes_prob_dict.items()]
        p = np.array([v[1] for k,v in nodes_list])
        index = np.random.choice([i for i in range(len(p))], p=p.ravel())
        node, _ = nodes_list[index]
        return node

    def simulate(self, node):
        state = copy.deepcopy(node.state)
        player_count = len(self.player_order)
        players = [Simu_NaivePlayer(i, self.agent.using_reward) for i in range(player_count)]
        act_id = node.act_id
        while state.TilesRemaining():
            if self.log:
                print(act_id)
                print('id', act_id)
                print('before')
                print(state.detail_str())
            move = players[act_id].SelectMove(None, state)
            state.ExecuteMove(act_id, move)
            act_id = act_id + 1 if act_id+1 < player_count else 0

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
                expect_score  = eval(self.agent.using_reward)(state, i, self.player_order).get_round_expection()
                reward[i] = state.players[i].score + expect_score
        # print(reward)
        return reward



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
        #sum_reward = sum([math.e**m[1][0] for m in move_tuple])
        sum_reward = sum([100+m[1][0] for m in move_tuple])
        for i, m in enumerate(move_tuple):
            move_prob_dict[m[0]] = None, (100+m[1][0])/sum_reward
        return move_prob_dict

class Instant_reward:
    def __init__(self, reward = 0, info=None):
        if info is None:
            info = {}
        self.reward = reward
        self.info = info

    def to_tuple(self):
        return self.reward, self.info

class Node:
    def __init__(self, game_state, parent, from_move, moves, act_id, instant_reward, tree):
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
        self.instant_reward = instant_reward
        tree.append(self)
        self.moves = moves
        self.visited = 0
        self.name = 'n'+str(len(tree))
        self.mark = False

    def is_fully_expanded(self):
        for m, (c, p) in self.moves.items():
            if c is None:
                return False
        return True

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
        info = '{:2},{}\nr:{}\np:{}\nv:{}\n{} {}'.format(
            self.from_move[1], self.from_move[2], ' ', ' ', self.visited,
            str(round(self.value[0],2)),str(round(self.value[1],2))) if self.from_move is not None else ''
        return info

class Simu_Player(RewardBasedPlayer):
    def __init__(self, _id, using_reward):
        super().__init__(_id)
        self.using_reward = using_reward

    def SelectMove(self, moves, game_state):
        player_order = []
        for i in range(self.id + 1, len(game_state.players)):
            player_order.append(i)
        for i in range(0, self.id + 1):
            player_order.append(i)

        i_moves = game_state.players[self.id].GetAvailableMoves(game_state)

        ft_moves = self.filtering_moves(game_state.players[self.id], i_moves)
        move_dict = {}
        for m in ft_moves:
            r = self.get_place_reward(game_state, m, self.id, player_order)
            move_dict[m] = r
            #print(m, r[0])
        move = max(move_dict.items(), key=lambda x:x[1][0])[0]
        #print(move)
        return move


class Simu_NaivePlayer(RewardBasedPlayer):
    def __init__(self, _id, using_reward):
        super().__init__(_id)
        self.using_reward = using_reward

    def SelectMove(self, moves, game_state):
        player_order = []
        for i in range(self.id + 1, len(game_state.players)):
            player_order.append(i)
        for i in range(0, self.id + 1):
            player_order.append(i)
        i_moves = game_state.players[self.id].GetAvailableMoves(game_state)
        move = NaivePlayer(self.id).SelectMove(i_moves, game_state)
        #print(move)
        return move

