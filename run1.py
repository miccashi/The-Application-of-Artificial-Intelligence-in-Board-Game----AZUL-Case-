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

from rewardBFSPlayer import BFS_Player
from MI_PlayerPro import MI_Player
# from mctsPlayer import MctsPlayer
from model import GameRunner,Player
from iplayer import InteractivePlayer
from naive_player import NaivePlayer
# from randomPlayer import RandomPlaye
from newMct import MI_PlayerNew
from simuallMCT import MI_PlayerAll
from utils import *

#players = [BFS_Player(0), NaivePlayer(1), NaivePlayer(2), NaivePlayer(3)]

start = time.time()
# players = [NaivePlayer(0), NaivePlayer(1)]
# players = [BFS_Player(0), NaivePlayer(1),]
# players = [BFS_Player(0), InteractivePlayer(1)]
# players = [BFS_Player(0), Test_Player(1)]
# players = [BFS_Player(0), InteractivePlayer(1)]
# players = [MctsPlayer(0), Player(1)]

#
# gr = GameRunner(players, random.randrange(122222222222222222222223))
# activity = gr.Run(True)
# print("Player 0 score is {}".format(activity[0][0]))
# print("Player 1 score is {}".format(activity[1][0]))


ROUND = 30
PLAYERS_NUM = 2


win = [0]*4
mark_sum = [0]*4

marksList= [[], [], [], []]
rowsList = [[], [], [], []]
columnsList = [[], [], [], []]
setsList = [[], [], [], []]
penaltiesList = [[], [], [], []]
placingList = [[], [], [], []]
bounsList = [[], [], [], []]

result = []

players = [NaivePlayer(0), MI_PlayerNew(1)]
players.extend([NaivePlayer(i) for i in range(len(players), PLAYERS_NUM)])

for i in range(ROUND):
    gr = GameRunner(players, random.randrange(122222222222222222222223))
    activity = gr.Run(True)



    resultList = [activity[i][0] for i in range(PLAYERS_NUM)]
    for i in range(PLAYERS_NUM):
        print("Player {} score is {}".format(i+1, activity[i][0]))
        mark_sum[i]+=activity[i][0]

        marksList[i].append(activity[i][0])
        placingList[i].append(activity[i][2])
        rowsList[i].append(activity[i][3])
        columnsList[i].append(activity[i][4])
        setsList[i].append(activity[i][5])
        penaltiesList[i].append(activity[i][6])
        bounsList[i].append(activity[i][7])

        if activity[i][0] == max(resultList):
            win[i]+=1


    # assert activity[0][0] > activity[1][0]
    result.append(resultList)
    print(result)
    print('time cost:', time.time()-start)
    print('games:', i+1)
    #assert False
for i in range(PLAYERS_NUM):
    print("***-------------------------------------------------------------------------------------------***")
    print("player {} | Overall | Win: {}/ {} | Winning rate: {}%.".format(
        i+1, win[i], ROUND, round(win[i]/ROUND*100,2)))
    print("player {} | Marks | Max Mark: {} | Min Mark: {}: | Avg Mark: {}".format(
        i+1, max(marksList[i]), min(marksList[i]), sum(marksList[i])/ROUND))
    print("player {} | Placing Tails | Max Mark: {} | Min Mark: {}: | Avg Mark: {}".format(
        i+1, max(placingList[i]), min(placingList[i]), sum(placingList[i])/ROUND))
    print("player {} | Finished Rows | Max Finished: {} | Min Finished: {}: | Avg Finished: {}".format(
        i+1, max(rowsList[i]), min(rowsList[i]), sum(rowsList[i])/ROUND))
    print("player {} | Finished Columns | Max Finished: {} | Min Finished: {}: | Avg Finished: {}".format(
        i+1, max(columnsList[i]), min(columnsList[i]), sum(columnsList[i])/ROUND))
    print("player {} | Finished Sets | Max Finsihed: {} | Min Finished: {}: | Avg Finished: {}".format(
        i+1, max(setsList[i]), min(setsList[i]), sum(setsList[i])/ROUND))
    print("player {} | Penalties | Max Penalties: {} | Min Penalties: {}: | Avg Penalties: {}".format(
        i+1, min(penaltiesList[i]), max(penaltiesList[i]), sum(penaltiesList[i])/ROUND))
    print("player {} | Evaluation | PSE: {} % | RPE: {}: | CPE: {} | SPE: {} | BSE: {} ".format(
        i+1, round(sum(placingList[i])/(ROUND*150) *100,2), sum(rowsList[i])/(ROUND*5), sum(columnsList[i])/(ROUND*5), sum(setsList[i])/(ROUND*5),
        round(sum(rowsList[i])/(ROUND*5*5) + sum(columnsList[i])*7/(ROUND*5*10) + sum(setsList[i])/(ROUND*5), 2)))



# print("Player 2 score is {}".format(activity[2][0]))
# print("Player 3 score is {}".format(activity[3][0]))


#print("Player 0 round-by-round activity")
#player_trace = activity[0][1]
#for r in range(len(player_trace.moves)):
#    print("ROUND {}".format(r+1))
#    for move in player_trace.moves[r]:
#        print(MoveToString(0, move))
#    print("Score change {}".format(player_trace.round_scores[r]))

#print("Bonus points {}".format(player_trace.bonuses))