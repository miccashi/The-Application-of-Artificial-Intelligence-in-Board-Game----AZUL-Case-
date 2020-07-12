import math
import time

# from graphTree import TreeGraph
from collections import defaultdict

import numpy as np

from Gui import Gui
from model import *
from naive_player import NaivePlayer
from randomPlayer import RandomPlayer
from reward import Reward
from rewardBasedPlayer import RewardBasedPlayer, get_max_difference, randomMax
from rewardBFSPlayer import BFS_search
from state_monitor import increase_actions, set_search_time


SEARCH_TIME = 3
GAMMA = 0.9
MAX = 10000
USING_GUI = False
USING_OPPONENT = False
NO_PRUNE_THRESHOLD = 20

class MI_PlayerNew(RewardBasedPlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.search_agent = Mcts_search(_id, False, self)
        # self.using_reward = 'RewardPro'

    def SelectMove(self, moves, game_state):
        increase_actions('moves')
        original_moves = moves
        player_order = self.get_player_order(game_state)
        moves = self.filtering_moves(game_state.players[self.id], moves)


        # if len(moves)<=NO_PRUNE_THRESHOLD:
        #     move = BFS_search(self.id, self).search(moves, game_state, player_order)
        # else:
        move = self.search_agent.search(moves, game_state, player_order, original_moves)
        return move


class Mcts_search:
    def __init__(self, _id, log, agent):
        self.id = _id
        self.log = log
        self.agent = agent

    def search(self, moves, game_state, player_order, original_moves):
        # print('!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(BoardToString(game_state))
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
        nodes_len = 5 * (2 ** (len(moves) ** (1 / 2)))
        move_len = len(moves)
        print('@@@@@@@@@@@@@@@@', move_len)
        # self.FIRST_SEARCH = 6
        # self.FOE_SEARCH = 2
        self.FIRST_SEARCH = 500 // move_len
        self.FOE_SEARCH = 2

        while time.time() - start < SEARCH_TIME:
        #while time.time()-start<move_len*0.2:
            # while n<nodes_len:
            # a = input('input')
            n += 1
            self.one_search(root_node)

        set_search_time('nodesnew', len(self.tree))

        set_search_time('timenew', time.time() - start)


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
            #print('{:2}{}: {:5} {:5}, {:5}'.format(m[1],str(m[2]), str(round(c.value[0],2)), str(round(c.value[1],2)), round(Q,2)))
        move = randomMax(dict)
        dict = {}
        for m, (c, p) in root_node.moves.items():
            Q = get_max_difference(c.value, self.id) if c is not None else -1000
            r = self.agent.get_place_reward(game_state, m, self.id, player_order)[0]
            dict[m] = Q, r
            print(
                '{:2}{}: {:5} {:5}, {:5} r:{:5}'.format(m[1], str(m[2]), round(c.value[0], 2), round(c.value[1], 2),
                                                        round(Q, 2), round(r, 2)))




        print()
        # self.track(game_state, root_node, player_order, move)
        # self.track_lose_mark(game_state, root_node, player_order, move)

        if USING_GUI:
            track = self.get_predict_track(root_node, move)
            Gui(self.tree, 'mcts')
        return move


    def track(self, game_state, root_node, player_order, move):

        dict = {}
        for m, (c, p) in root_node.moves.items():
            Q = get_max_difference(c.value, self.id) if c is not None else -1000
            r = self.agent.get_place_reward(game_state, m, self.id, player_order)[0]
            dict[m] = Q, r
            print(
                '{:2}{}: {:5} {:5}, {:5} r:{:5}'.format(m[1], str(m[2]), round(c.value[0], 2), round(c.value[1], 2),
                                                 round(Q, 2), round(r,2)))

        print()
        print(BoardToString(game_state))
        print(PlayerToString(self.id, game_state.players[self.id]))

        max_tuple = max(dict.items(), key=lambda x:x[1][1])
        max_move, max_r = max_tuple[0], max_tuple[1][1]


        naive_player_id = self.id + 1 if self.id + 1 < len(player_order) else 0
        print(BoardToString(game_state))
        print(PlayerToString(naive_player_id, game_state.players[naive_player_id]))
        print('Naive may choose:')
        moves = game_state.players[naive_player_id].GetAvailableMoves(game_state)
        naive_move = NaivePlayer(naive_player_id).SelectMove(moves, game_state)

        print()
        print('{:2}{}'.format(naive_move[1], str(naive_move[2])))

        if naive_move[1] == move[1] and move[1] != -1 and dict[move][1] != max_r and max_move[1] != move[1]:
            increase_actions('interfer')


    def track_lose_mark(self, game_state, root_node, player_order, move):
        gs_copy = copy.deepcopy(game_state)
        gs_copy.ExecuteMove(self.id, move)
        naive_player_id = self.id + 1 if self.id + 1 < len(player_order) else 0
        moves = gs_copy.players[naive_player_id].GetAvailableMoves(gs_copy)
        print(BoardToString(gs_copy))
        print(PlayerToString(naive_player_id, game_state.players[naive_player_id]))
        SUCCEED = True if len(moves)>0 else False

        for m in moves:
            floor_tiles = [tile for tile in gs_copy.players[naive_player_id].floor if tile > 0]
            floor_penalty = gs_copy.players[naive_player_id].FLOOR_SCORES[
                            len(floor_tiles):(len(floor_tiles) + m[2].num_to_floor_line)]
            print(sum(floor_penalty))
            if sum(floor_penalty)<-3:
                continue
            else:
                SUCCEED = False
                break
        if SUCCEED:
            # print('attack succeed')
            # for m in moves:
            #     print(m[1],str(m[2]))
            #
            # time.sleep(1)
            # assert False
            increase_actions('attack')





    def get_predict_track(self, root_node, move):
        track = [move]
        node = root_node.moves[move][0]
        while True:
            node.mark = True
            id = node.act_id
            children = [c for m, (c, p) in node.moves.items() if c is not None] if node.moves is not None else []
            if len(children) == 0:
                break
            node = max(children, key=lambda x: get_max_difference(x.value, id))

            track.append((node.from_move[1], str(node.from_move[2]), str(id), node, node.value))

        return track

    def one_search(self, root_node):
        start = time.time()
        select_node, move = self.select(root_node)
        self.time_monitor['select'] += (time.time() - start)

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
        result = self.simulate(node)
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
            Q = get_max_difference(c.value, p_node.act_id) / max(c.value) if max(c.value) != 0 else 0
            # N = 1 / c.visited if c.visited != 0 else MAX
            N = ((node_v_para / c.visited) ** (1 / 2)) if c.visited != 0 else MAX
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

    def simulate(self, node):
        state = copy.deepcopy(node.state)
        player_count = len(self.player_order)
        #players = [Simu_Player(i, self.agent.using_reward) for i in range(player_count)]

        players = [RandomPlayer(i) for i in range(player_count)]
        act_id = node.act_id
        while state.TilesRemaining():
            if self.log:
                print(act_id)
                print('id', act_id)
                print('before')
                print(state.detail_str())
            move = players[act_id].SelectMove(None, state)
            state.ExecuteMove(act_id, move)
            act_id = act_id + 1 if act_id + 1 < player_count else 0


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
            start = time.time()
            for i in range(player_count):
                state.players[i].EndOfGameScore()
                reward[i] = state.players[i].score
            self.time_monitor['simulate p'] += (time.time() - start)
        else:

            for i, plr in enumerate(state.players):
                expect_score = eval(self.agent.using_reward)(state, i, self.player_order).get_round_expection()

                # start = time.time()
                # row_score = eval(self.agent.using_reward)(state, i, self.player_order).get_score(2, is_row=True)
                # self.time_monitor['row'] += (time.time() - start)
                # start = time.time()
                # column_score = eval(self.agent.using_reward)(state, i, self.player_order).get_score(7, is_column=True)
                # self.time_monitor['c'] += (time.time() - start)
                # start = time.time()
                # set_score = eval(self.agent.using_reward)(state, i, self.player_order).get_score(10)
                # self.time_monitor['s'] += (time.time() - start)
                # start = time.time()
                # left_score = eval(self.agent.using_reward)(state, i, self.player_order).get_left_score()
                # self.time_monitor['l'] += (time.time() - start)

                reward[i] = state.players[i].score+expect_score


        return reward

    def backup(self, node, result):
        update_node = node
        update_node.update(self.id, result)

        while True:
            update_node = update_node.parent
            if update_node is None: break
            update_node.update(self.id)

    def get_pre_prob(self, game_state, moves, act_id, player_order):
        #threshold_most = self.FOE_SEARCH if act_id != self.id else self.FIRST_SEARCH
        threshold_most = self.FOE_SEARCH if len(self.tree) != 1 else self.FIRST_SEARCH
        # threshold_impo = 4

        ft_moves = self.agent.filtering_moves(game_state.players[act_id], moves)
        move_dict = {}
        for move in ft_moves:
            reward, score_list = self.agent.get_place_reward(game_state, move, act_id, player_order)
            move_dict[move] = reward, score_list

        move_tuple = sorted(move_dict.items(), key=lambda x: x[1][0], reverse=True)[:threshold_most] if len(
            move_dict) > threshold_most else move_dict.items()

        move_prob_dict = {}
        sum_reward = sum([1.3 ** m[1][0] for m in move_tuple])
        for i, m in enumerate(move_tuple):
            move_prob_dict[m[0]] = None, (1.3 ** m[1][0]) / sum_reward
        return move_prob_dict


class Node:
    def __init__(self, game_state, parent, from_move, act_id, tree):
        self.state = game_state
        self.parent = parent
        self.from_move = from_move
        if self.parent is not None:
            # print( self.parent.moves[from_move])
            self.parent.moves[from_move] = (self, self.parent.moves[from_move][1])
            peers = [c for m, (c, p) in self.parent.moves.items()]
            assert self in peers
        self.act_id = act_id
        self.value = [0] * len(game_state.players)
        tree.append(self)
        self.visited = 0
        self.name = 'n' + str(len(tree))
        self.mark = False
        self.moves = None
        self.tree = tree

    def is_fully_expanded(self):
        for m, (c, p) in self.moves.items():
            if c is None:
                return False
        return True

    def give_moves(self, moves):
        self.moves = moves

    def get_unexpanded_move(self):

        if USING_OPPONENT and self.act_id != self.tree[0].act_id:
            dice = random.random()
            if dice <= 0.7:
                moves = [m for m, _ in self.moves.items()]
                return NaivePlayer(self.act_id).SelectMove(moves, self.state)
        unexp_dict = {}
        for m, (c, p) in self.moves.items():
            if c is None:
                unexp_dict[m] = p
        # unexp_prob = sum(unexp_dict.values())
        # assert len(unexp_dict) > 0
        # # print(unexp_dict.values())
        # for m, p in unexp_dict.items():
        #     unexp_dict[m] = p / unexp_prob
        # # print(sum(unexp_dict.values()))
        # unexp_m_list = [(k, v) for k, v in unexp_dict.items()]
        # p = np.array([v for k, v in unexp_m_list])
        # # print(p)
        # index = np.random.choice([i for i in range(len(p))], p=p.ravel())
        # m, _ = unexp_m_list[index]

        m = max(unexp_dict, key=unexp_dict.get)

        return m

    def is_end(self):
        return not self.state.TilesRemaining()

    def update(self, agent_id, result=None):
        self.visited += 1
        if result is not None:
            for i in range(len(self.value)):
                self.value[i] = (self.value[i] * (self.visited - 1) + result[i]) / self.visited
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
        return [c for m, (c, p) in self.moves.items()] if self.moves is not None else []

    def info(self):
        info = '{:2},{}\np:{}\nv:{}\n{} {}\n{}'.format(
            self.from_move[1], self.from_move[2], round(self.parent.moves[self.from_move][1],2), self.visited,
            str(round(self.value[0], 2)), str(round(self.value[1], 2)),
            self.act_id) if self.from_move is not None else ''
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
            move_dict[m] = (None, r[0])
        move = max(move_dict.items(), key=lambda x: x[1][1])[0]
        return move
