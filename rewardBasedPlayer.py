import random

from model import Player
from reward import Reward
from state_monitor import set_state


def randomMax(dict):
    maxValue = dict[max(dict, key=dict.get)]
    maxGroup = [k for k, v in dict.items() if v == maxValue]
    return random.choice(maxGroup)

def get_max_difference(value_list, act_id):
    id_value = value_list[act_id]
    max_other = max([v for i, v in enumerate(value_list) if i!= act_id])
    return id_value-max_other



class RewardBasedPlayer(Player):
    def __init__(self, _id):
        super().__init__(_id)
        self.using_reward = 'Reward'

    def SelectMove(self, moves, game_state):
        player_order = self.get_player_order(game_state)
        moves = self.filtering_moves(game_state.players[self.id], moves)
        return random.choice(moves)

    def filtering_moves(self, player_state, moves):
        remove_list = []
        for index, move in enumerate(moves):
            tile_type = move[2].tile_type
            pattern_line_dest = move[2].pattern_line_dest
            if pattern_line_dest > 0 and player_state.lines_tile[pattern_line_dest] == tile_type and \
                    player_state.lines_number[pattern_line_dest] == pattern_line_dest + 1:
                remove_list.append(index)
        moves = [moves[i] for i in range(len(moves)) if i not in remove_list]
        return moves

    def get_player_order(self, game_state):
        player_order = []
        for i in range(self.id + 1, len(game_state.players)):
            player_order.append(i)
        for i in range(0, self.id + 1):
            player_order.append(i)
        return player_order

    def get_place_reward(self, game_state, move, act_id, player_order):
        reward, score_list = eval(self.using_reward)(game_state, act_id, player_order).estimate(move)
        return reward, score_list

class ImmediatePlayer(RewardBasedPlayer):
    def __init__(self, _id):
        super().__init__(_id)
        self.using_reward = 'Reward'
        self.state_cut = False

    def SelectMove(self, moves, game_state):

        if 7<len(moves) < 30 and not self.state_cut:
            set_state(game_state)
            self.state_cut = True

        player_order = self.get_player_order(game_state)
        moves = self.filtering_moves(game_state.players[self.id], moves)
        move_dict = {}
        for m in moves:
            move_dict[m] = self.get_place_reward(game_state, m, self.id, player_order)[0]
        return randomMax(move_dict)