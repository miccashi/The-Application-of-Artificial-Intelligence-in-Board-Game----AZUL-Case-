# Written by Michelle Blom, 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import random
import time


from model import GameRunner,Player

from iplayer import InteractivePlayer

from naive_player import NaivePlayer

from rewardBFSPlayer import BFS_Player
from MI_PlayerSimuSave import MI_PlayerSimuSave
from MI_PlayerPro import MI_Player
from newMct import MI_PlayerNew
from state_monitor import get_special_action, get_search_time
from randomPlayer import RandomPlayer
from simuallMCT import MI_PlayerAll
ROUND = 10
PLAYERS_NUM = 2

win = [0]*4
mark_sum = [0]*4
result = []

#players_name=['MI_Player', 'MI_PlayerSimuSave']
players_name=['MI_PlayerNew', 'BFS_Player']
#players_name=['BFS_Player', 'NaivePlayer']
#players_name=['BFS_Player', 'MI_Player']
#players_name=['MI_PlayerNew', 'NaivePlayer']
#players_name=['RandomPlayer', 'NaivePlayer']
#players_name=['MI_PlayerNew', 'MI_Player']
#players_name=['BFS_Player', 'MI_PlayerNew']
#players_name=['MI_Player', 'InteractivePlayer']
players = [eval(players_name[0])(0), eval(players_name[1])(1)]
players_reverse = [eval(players_name[1])(0), eval(players_name[0])(1)]
players.extend([NaivePlayer(i) for i in range(len(players), PLAYERS_NUM)])
players_reverse.extend([NaivePlayer(i) for i in range(len(players), PLAYERS_NUM)])

SEED = []
for i in range(ROUND//2+1):
    SEED.append(random.randrange(99999))

print('seeds:', SEED)
start = time.time()
for i in range(ROUND):
    print('NEW GAME')
    print('seed:', SEED[i//2])
    ps = players if i%2==0 else players_reverse
    gr = GameRunner(ps, SEED[i//2])
    activity = gr.Run(True)
    #print(activity[0][0], activity[1][0])
    resultList = [activity[0][0], activity[1][0]] if i%2==0 else [activity[1][0], activity[0][0]]
    resultList.extend([activity[i][0] for i in range(len(players), PLAYERS_NUM)])
    #print(resultList)

    for j in range(PLAYERS_NUM):
        print("{} score is {}".format(players_name[j], resultList[j]))

    result.append(resultList)
    print(result)
    print('time cost:', time.time()-start)
    print('games:', i+1)
    print()

for n in range(len(result)):
    for i in range(PLAYERS_NUM):
        if result[n][i] == max(result[n]):
            win[i]+=1
        mark_sum[i]+=result[n][i]

for i in range(PLAYERS_NUM):
    print("{} has win {} times for {} times. Winning rate is {}%.\nplayer {}'s average mark:{}".format(
        players_name[i], win[i], ROUND, round(win[i]/ROUND*100,2), i+1, mark_sum[i]/ROUND
    ))

print( get_search_time())

