import time
from collections import defaultdict

from model import *
from state_monitor import set_search_time
from utils import *

class RandomPlayer(Player):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectMove(self, moves, game_state):
        #print(game_state)
        start= time.time()
        move = self.select_randomMove(game_state)
        set_search_time('random', time.time()-start)
        return move



    def select_randomMove(self, game_state):
        player_state = game_state.players[self.id]
        tile_type_required = {}
        for i in range(player_state.GRID_SIZE):
            if player_state.lines_number[i] == i + 1:
                continue
            if player_state.lines_tile[i] != -1 and player_state.lines_number[i]<i+1:

                if tile_type_required.get(player_state.lines_tile[i]) is not None:
                    tile_type_required[player_state.lines_tile[i]].add(i)
                else:
                    tile_type_required[player_state.lines_tile[i]] = {i}

            else:
                for j in range(5):
                    if player_state.grid_state[i][j] == 0:
                        tile_type = int(numpy.where(player_state.grid_scheme[i] == j)[0])
                        if tile_type_required.get(tile_type) is not None:
                            tile_type_required[tile_type].add(i)
                        else:
                            tile_type_required[tile_type] = {i}
        tile_possess_dict ={}
        for fd in game_state.factories:
            for t,n in fd.tiles.items():
                if tile_possess_dict.get(t) is not None:
                    tile_possess_dict[t].append(n)
                else:
                    tile_possess_dict[t] = [n]
        for t, n in game_state.centre_pool.tiles.items():
            tile_possess_dict[t].append(n)

        available_tile = [t for t in tile_possess_dict.keys() if t in tile_type_required.keys() and sum(tile_possess_dict[t])>0]
        # print(available_tile)

        if len(available_tile)>0:


            #print('available')
            #print(available_tile)

            tile = random.choice(available_tile)
            #print('colordict')
            #print(tile_possess_dict[tile])

            #print('fid')
            f_id = random.choice([i for i in range(6) if tile_possess_dict[tile][i]>0])


            number = tile_possess_dict[tile][f_id]
            tg = TileGrab()
            tg.number = number
            tg.tile_type = tile
            tg.pattern_line_dest = random.choice(list(tile_type_required[tile]))
            #print('dest')
            #print(tg.pattern_line_dest)
            slots_free = (tg.pattern_line_dest + 1) - player_state.lines_number[tg.pattern_line_dest]
            tg.num_to_pattern_line = min(number, slots_free)
            tg.num_to_floor_line = tg.number - tg.num_to_pattern_line

            if f_id == 5:
                move=(Move.TAKE_FROM_CENTRE, -1, tg)
            else:
                move=(Move.TAKE_FROM_FACTORY, f_id, tg)
            return move
        else:
            available_type = [t for t in tile_possess_dict.keys() if sum(tile_possess_dict[t])>0]
            tile = random.choice(available_type)
            # print(tile_possess_dict[tile])
            f_id = random.choice([i for i in range(6) if tile_possess_dict[tile][i]>0])
            number = tile_possess_dict[tile][f_id]
            tg = TileGrab()
            tg.number = number
            tg.tile_type = tile
            tg.num_to_floor_line = tg.number
            if f_id == 5:
                move= (Move.TAKE_FROM_CENTRE, -1, tg)
            else:
                move=(Move.TAKE_FROM_FACTORY, f_id, tg)
            return move
