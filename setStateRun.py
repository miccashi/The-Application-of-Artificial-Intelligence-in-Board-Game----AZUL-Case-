import _thread
import random
import time
from multiprocessing import Process

from model import GameRunner,Player

from iplayer import InteractivePlayer

from naive_player import NaivePlayer

from rewardBFSPlayer import BFS_Player
from MI_PlayerPro import MI_Player
from MI_PlayerSimuSave import MI_PlayerSimuSave
from state_monitor import get_state #, get_time
from rewardBasedPlayer import ImmediatePlayer



if __name__ == '__main__':

    players_name=['ImmediatePlayer', 'ImmediatePlayer']
    SEED = 2000
    players = [eval(players_name[0])(0), eval(players_name[1])(1)]
    ps = players
    gr = GameRunner(ps, SEED)
    gr.Run(False)
    state = get_state()

    # BFS_PlayerMonitor(0).SelectMove(None, state)
    # MI_PlayerMonitor(0, get_time()).SelectMove(None, state)
    moves = state.players[0].GetAvailableMoves(state)

    Process(target=BFS_Player(0).SelectMove, args=(moves, state)).start()
    Process(target=MI_Player(0).SelectMove, args=(moves, state)).start()
    #Process(target=MI_PlayerSimuSave(0).SelectMove, args=(moves, state)).start()

    while True:
        time.sleep(1)

