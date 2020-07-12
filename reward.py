import numpy
import copy
from collections import defaultdict
from functools import reduce

class Reward:
    def __init__(self, game_state, act_id, player_order):

        self.GRID_SIZE = 5

        self.game_state = game_state
        self.act_id = act_id
        self.player_state = game_state.players[act_id]
        self.round_id = len(self.player_state.player_trace.round_scores)

        self.grid_state = self.player_state.grid_state
        self.grid_scheme = self.player_state.grid_scheme
        self.player_order = player_order

        # bag dic / factory and centre pool
        bag = self.game_state.bag
        self.factories = self.game_state.factories
        self.centre_pool = self.game_state.centre_pool
        self.bag_dic = defaultdict(int)
        self.fac_and_ctr = defaultdict(int)

        for tile in bag:
            self.bag_dic[tile] += 1
        for factory in self.factories:
            for tile in range(5):
                self.bag_dic[tile] += factory.tiles[tile]
                self.fac_and_ctr[tile] += factory.tiles[tile]
        for tile in range(5):
            self.bag_dic[tile] += self.centre_pool.tiles[tile]
            self.fac_and_ctr[tile] += self.centre_pool.tiles[tile]

        self.bag_dic = self.check_refill(self.bag_dic, self.game_state)


    def estimate(self, move):

        # all features about a move
        self.actual_reward = 0
        self.row_contri = 0
        self.column_contri = 0
        self.set_contri = 0
        self.linkness = 0
        self.hardness = 0
        self.completeness = 0
        self.completion_this_round = 0
        self.penalty = 0


        # assign some important parameters
        self.tile_type = move[2].tile_type
        self.number = move[2].number
        self.pattern_line_dest = move[2].pattern_line_dest
        self.num_to_pattern_line = move[2].num_to_pattern_line
        self.num_to_floor_line = move[2].num_to_floor_line

        floor_tiles = [tile for tile in self.player_state.floor if tile>0]
        floor_penalty = self.player_state.FLOOR_SCORES[
                                    len(floor_tiles):(len(floor_tiles) +self.num_to_floor_line)]
        # print(str(move[2]),len(self.player_state.floor_tiles),(len(self.player_state.floor_tiles) +
        #                                                                 self.num_to_floor_line), floor_penalty)
        first_token = 0
        if self.game_state.first_player_taken is False and move[1] == -1:
            first_token = -1

        move_to_floor_penalty = sum(floor_penalty)




        score_list = {}
        if self.pattern_line_dest == -1 or self.player_state.lines_number[self.pattern_line_dest] == self.pattern_line_dest + 1:
            score_list['move to floor'] = round(move_to_floor_penalty+first_token, 2)
            return move_to_floor_penalty+first_token , score_list


        self.column_index = int(self.grid_scheme[self.pattern_line_dest][self.tile_type])
        completed_tiles = self.num_to_pattern_line + self.player_state.lines_number[self.pattern_line_dest]




        # all features of a move
        self.actual_reward = self.get_current_reward(self.pattern_line_dest, self.column_index)
        self.row_contri, self.column_contri, self.set_contri = self.exploration(self.pattern_line_dest, self.column_index, self.tile_type)
        self.linkness = self.get_linkness(self.pattern_line_dest, self.column_index)


        # self.linkness = -(self.pattern_line_dest-self.column_index)/5
        self.hardness = self.pattern_line_dest/5
        self.completeness = completed_tiles/(self.pattern_line_dest+1)
        self.completion_this_round = self.get_achieveable_rate(completed_tiles)
        self.penalty = move_to_floor_penalty + first_token



        score_list['actrual_reward:'] = self.actual_reward
        score_list['row_contri:'] = self.row_contri
        score_list['column_contri:'] = self.column_contri
        score_list['set_contri:'] = self.set_contri
        score_list['linkness'] = self.linkness
        score_list['hardness'] = self.hardness
        score_list['completeness'] = self.completeness
        score_list['completion_this_round'] = self.completion_this_round
        score_list['move_to_floor'] = move_to_floor_penalty

        score = self.actual_reward + self.row_contri + self.column_contri + self.set_contri + self.linkness + self.hardness

        factor = (self.completeness + 0.9 * max(0, self.completion_this_round - self.completeness))
        if self.player_state.lines_number[self.pattern_line_dest] == 0 and self.completeness < 1:
            if self.bag_dic[self.tile_type] < self.pattern_line_dest + 1 - self.num_to_pattern_line:
                # print('bag:',self.bag_dic)
                score = self.get_other_expection(self.pattern_line_dest, self.column_index, self.num_to_floor_line)
                factor = 0.9 ** (max(0, 5-self.round_id))
            elif self.fac_and_ctr[self.tile_type] < self.pattern_line_dest + 1 - self.num_to_pattern_line:
                # print('fac:',self.fac_and_ctr)
                score/=2


        # consider completeness and whether this line can be finished this round

        complete_score =  factor * score + self.penalty
        return complete_score, score_list

    def get_features(self):

        return [self.actual_reward / 20, (self.row_contri + self.column_contri + self.set_contri)/3.8 ,
                self.linkness,
                self.hardness, self.completeness, self.penalty / 10]


    def get_current_reward(self, row_index, column_index):
        column_prefix, row_value = self.getreward(row_index, column_index)

        factor = 1
        column_suffix = 0
        new_row_index = row_index

        if row_index < 4:
            for i in range(row_index + 1, self.GRID_SIZE):

                # already has tile in grid_state
                if self.grid_state[i][column_index] == 1:
                    column_suffix += 1
                    new_row_index = i
                else:
                    break
        if new_row_index < 4:
            for i in range(new_row_index + 1, self.GRID_SIZE):
                # new tile preparing to move to grid_state __  h add
                if self.player_state.lines_tile[i] != -1 and self.player_state.lines_number[i] == i + 1:
                    tile_index = int(self.grid_scheme[i][self.player_state.lines_tile[i]])
                    if column_index == tile_index:
                        factor += 1
                    else:
                        break
        column_value = column_prefix * factor + column_suffix
        if column_value == 1 or row_value == 1:
            # print('current:', row_value, column_value)
            return column_value + row_value - 1
        # print('current2:', row_value, column_value)

        return column_value + row_value


    def getreward(self, row_index, column_index):

        row_value = 0
        if self.grid_state[row_index][column_index] == 1: return -1, -1
        # print('from',max(row_index-1, 0), 'to',str(-1))
        column_value = self.reward(max(row_index-1, 0), -1, -1, column=column_index)
        # print('c_prefix',column_value)

        # column_value += self.reward(line_number, 5, 1, column=self.column_index)
        # print('from',max(column_index-1, 0),'to -1')
        row_value += self.reward(max(column_index-1, 0), -1, -1, row=row_index)
        # print('r_prefix',row_value)

        # print('from',min(column_index+1, 4),'to 5')
        row_value += self.reward(min(column_index+1, 4), 5, 1, row = row_index)
        # print('r_suffix',row_value)
        return column_value + 1, row_value + 1


    def reward(self, low, high, grad, row=None, column=None):
        reward = 0
        r = row
        c = column
        for i in range(low, high, grad):
            if r is None:   row = i
            elif c is None:     column = i
            # print('index',(row,column),self.grid_state[row][column])
            if self.grid_state[row][column] == 1 \
                    or (self.player_state.lines_number[row] == row+1
                        and self.grid_scheme[row][self.player_state.lines_tile[row]] == column):
                reward += 1
            else: break
        return reward



    # ensure to finish a line
    def get_achieveable_rate(self, completed_tiles):
        if completed_tiles == self.pattern_line_dest + 1: return 1
        else:
            factories = self.game_state.factories
            centre_pool = self.game_state.centre_pool

            available_tiles = [factory.tiles[self.tile_type] for factory in factories if factory.tiles[self.tile_type] > 0]
            if centre_pool.tiles[self.tile_type] > 0: available_tiles.append(centre_pool.tiles[self.tile_type])

            available_tiles.remove(self.number)
            if len(available_tiles) == 0: return 0

            sorted_tiles = sorted(available_tiles,reverse=True)
            tile_set = sorted(list(set(sorted_tiles)),reverse=True)

            # print(sorted_tiles)
            # print(tile_set)
            idx = [sorted_tiles.index(i) for i in tile_set]

            tile_length = []
            for i in range(len(tile_set)-1):
                tile_length.append(idx[i+1]-idx[i])
            tile_length.append(len(sorted_tiles)-idx[-1])

            # print(tile_length)

            player_num = 0
            next_id = self.act_id
            for j in range(len(self.player_order) - 1):
                next_id = next_id + 1 if next_id < len(self.player_order) - 1 else 0
                next_player_state = self.game_state.players[next_id]
                next_grid_state = copy.deepcopy(next_player_state.grid_state)

                for i in range(self.GRID_SIZE):
                    if (next_player_state.lines_number[i] == 0 and next_grid_state[i][self.column_index] == 0) or \
                            (next_player_state.lines_tile[i] == self.tile_type and next_player_state.lines_number[i] < i + 1):
                        player_num += 1
                        break

            available_length = [length - player_num for length in tile_length]
            final_tiles = [tile_set[i] for i in range(len(tile_set)) if available_length[i] > 0]

            # print(final_tiles)

            for tile_num in final_tiles:
                if tile_num + completed_tiles == self.pattern_line_dest + 1: return 1

            return 0


    def get_other_expection(self, row_index, column_index,num_to_floor_line):

        vacant_index = []
        for i in range(5):
            if self.grid_state[row_index][i] == 0 and i != column_index:
                vacant_index.append(i)

        if len(vacant_index) > 0:
            vacant_value_list = []
            for i in vacant_index:
                tile = (row_index, i)
                tile_type = numpy.where(self.grid_scheme[tile[0]] == tile[1])[0]
                current_reward = self.get_current_reward(tile[0], tile[1])
                row_contri, column_contri, set_contri = self.exploration(tile[0], tile[1], tile_type)
                linkness = self.get_linkness(tile[0], tile[1])
                achieveness = self.bag_dic[int(tile_type)] / (tile[0] + 1) - 1
                vacant_value_list.append(min(achieveness,1) * (current_reward + row_contri + column_contri + set_contri + linkness))
                # print((tile[0], tile[1]), ': ',min(achieveness, 1), current_reward, row_contri, column_contri, set_contri, linkness, ' = ', vacant_value_list[-1])

            # print(self.bag_dic)
            # print(vacant_value_list)
            sorted_list = sorted(vacant_value_list)
            floor_tiles = [tile for tile in self.player_state.floor if tile > 0]
            move_to_floor_penalty = sum(self.player_state.FLOOR_SCORES[
                            len(floor_tiles):(len(floor_tiles) + num_to_floor_line)])

            if sorted_list[-1] > -move_to_floor_penalty:
                return -sorted_list[-1]
        return 0



    def exploration(self, row_index, column_index, tile_type):

        row_contri = self.get_row_tiles(self.bag_dic, row_index, column_index)
        column_contri = self.get_column_tiles(self.bag_dic, row_index, column_index)
        set_contri = self.get_set_tiles(self.bag_dic, row_index, column_index, tile_type)

        return row_contri, column_contri, set_contri



    def get_row_tiles(self, bag_dic, row_index, column_index):
        completed = 0
        vacant_dic = {}
        self.grid_state[row_index][column_index] = 1

        for j in range(self.GRID_SIZE):
            if self.grid_state[row_index][j] == 1:
                completed += 1
            else:
                tile_type = numpy.where(self.grid_scheme[row_index] == j)[0]
                vacant_dic[int(tile_type)] = row_index + 1

        self.grid_state[row_index][column_index] = 0

        if completed >= self.round_id and self.has_prob(bag_dic, vacant_dic):

            # return  2 * completed / 5 /5
            return 2/5 * 0.9 ** (5-completed)

        return 0

    def get_column_tiles(self, bag_dic, row_index, column_index):
        completed = 0
        vacant_dic = defaultdict(int)
        self.grid_state[row_index][column_index] = 1

        for row_i in range(self.GRID_SIZE):

            if self.grid_state[row_i][column_index] == 1:
                completed += 1
            elif self.player_state.lines_number[row_i] == row_i + 1 \
                    and self.grid_scheme[row_i][self.player_state.lines_tile[row_i]] == column_index:
                completed += 1
            else:
                tile_type = numpy.where(self.grid_scheme[row_i] == column_index)[0]
                vacant_dic[int(tile_type)] += row_i + 1
        self.grid_state[row_index][column_index] = 0
        if completed >= self.round_id and self.has_prob(bag_dic, vacant_dic):
            # return 7*completed/5/5
            return 7 / 5 * 0.9 ** (5 - completed)
        return 0



    def get_set_tiles(self, bag_dic, row_index, column_index, tile_type):

        completed = 0
        vacant_dic = defaultdict(int)
        self.grid_state[row_index][column_index] = 1



        for row_i in range(self.GRID_SIZE):
            tile_index = int(self.grid_scheme[row_i][tile_type])
            if self.grid_state[row_i][tile_index] == 1:
                completed += 1
            elif self.player_state.lines_number[row_i] == row_i + 1 \
                    and self.player_state.lines_tile[row_i] == tile_type:
                completed += 1
            else:
                vacant_dic[int(tile_type)] += (row_i + 1)

        self.grid_state[row_index][column_index] = 0

        if completed >= self.round_id and self.has_prob(bag_dic, vacant_dic):
            # return 10*completed/5/5
            return 10 / 5 * 0.9 ** (5 - completed)
        return 0



    def get_linkness(self, row_index, column_index):

        self.grid_state[row_index][column_index] = 1
        vacant_index = numpy.argwhere(self.grid_state == 0)
        self.grid_state[row_index][column_index] = 0

        linkness = 0
        for tile in vacant_index:
            tile_type = numpy.where(self.grid_scheme[tile[0]] == tile[1])[0]

            if self.has_prob(self.bag_dic, {int(tile_type):tile[0]+1}):
                power = abs(tile[0]-row_index) + abs(tile[1]-column_index)
                length = len(vacant_index)
                if power > 0:
                    if tile[0] == row_index or tile[1] == column_index:
                        linkness += 0.9 ** power * (power -1 )/ max(1, reduce(lambda x,y:x*y,range(length+1-power,length+1)))
                    else:
                        linkness += power * (0.9 ** power) * (power - 1) / max(1, reduce(lambda x,y:x*y,range(length+1-power,length+1)))

        return linkness

    def player_tiles(self, player_state):
        tile_num = 0
        for i in range(5):
            tile_num += player_state.lines_number[i]
            for j in range(5):
                tile_num += player_state.grid_state[i][j]

        floor_tiles = [tile for tile in self.player_state.floor if tile > 0]
        tile_num += len(floor_tiles)
        return tile_num


    def check_refill(self, bag_dic, game_state):
        player_num = len(game_state.players)
        NUM_FACTORIES = [5, 7, 9]
        NUM_PLAYERS = [2, 3, 4]

        round_id = len(game_state.players[0].player_trace.round_scores)

        if player_num > 2:
            if (player_num == 3 and round_id >3) or (player_num ==4 and round_id > 2):
                used_num = 0
                # print('before')
                # print(game_state)
                # print(bag_dic)
                # for i in range(player_num):
                #     used_num += self.player_tiles(game_state.players[i])
                # print(len(game_state.bag), 5-round_id, NUM_FACTORIES[NUM_PLAYERS.index(player_num)] * 4)
                for type in range(5):
                    bag_dic[type] -= (sum(bag_dic.values()) - (5-round_id) * NUM_FACTORIES[NUM_PLAYERS.index(player_num)] * 4)/5
                # print('after')
                # print(bag_dic)
            else:
                for type in range(5):
                    refill_num_per_type = NUM_FACTORIES[NUM_PLAYERS.index(player_num)] * 4 - 20
                    bag_dic[type] += refill_num_per_type


        return bag_dic
    # has probability to finish a row/ column/ set
    def has_prob(self, bag_dic, vacant_dic):
        for tile in vacant_dic.keys():
            if not tile in bag_dic.keys() or vacant_dic[tile] > bag_dic[tile]:
                return False
        return True



    def get_round_expection(self):
        row_score = self.get_score(2, is_row=True)
        column_score = self.get_score(7, is_column=True)
        set_score = self.get_score(10)

        left_score = self.get_left_score()
        # if left_score != 0:
        #     print(left_score)
        return row_score + column_score + set_score + left_score


    def get_score(object, factor, is_row=False, is_column=False):
        round_id = len(object.player_state.player_trace.round_scores)
        bag_dic = object.bag_dic

        expected_score = 0

        for i in range(5):
            each_unit = 0
            vacant_unit = defaultdict(int)
            for j in range(5):
                if is_row:
                    row_index = i
                    column_index = j
                    tile_type = numpy.where(object.grid_scheme[i] == j)[0]
                elif is_column:
                    row_index = j
                    column_index = i
                    tile_type = numpy.where(object.grid_scheme[j] == i)[0]
                else:
                    row_index = j
                    column_index = int(object.grid_scheme[j][i])
                    tile_type = i

                if object.grid_state[row_index][column_index] == 1:
                    each_unit += 1
                elif object.grid_state[row_index][column_index] == 0:
                    left = 0
                    if object.player_state.lines_tile[row_index] == tile_type:
                        left = object.player_state.lines_number[row_index]
                    vacant_unit[int(tile_type)] += row_index + 1 - left
            # print('***********************')
            # print(each_unit)
            # print(vacant_unit)
            # print(each_unit >= round_id)
            # print(object.has_prob(bag,vacant_unit))
            if each_unit >= round_id and object.has_prob(bag_dic, vacant_unit):
                expected_score += each_unit * factor / 5

        # print(round_id, round(expected_score, 2))
        expected_score = expected_score * 0.9 ** (max(0, (5-round_id)))
        return expected_score

    def get_left_score(self):
        final_score = 0
        for row in range(5):
            if 0 < self.player_state.lines_number[row] < row + 1:
                column_index = int(self.grid_scheme[row][self.player_state.lines_tile[row]])
                if self.bag_dic[self.player_state.lines_tile[row]] < row + 1 - self.player_state.lines_number[
                    row]:
                    #pass
                    final_score += self.get_other_expection(row, column_index, 0)

                else:

                    current_score = self.get_current_reward(row, column_index)
                    row_contri, column_contri, set_contri = self.exploration(row, column_index,
                                                                               self.player_state.lines_tile[row])
                    hardness = row / 5
                    completeness = self.player_state.lines_number[row] / (row + 1)
                    final_score += completeness * (
                                current_score + row_contri + column_contri + set_contri  + hardness) * 0.1

        return final_score
