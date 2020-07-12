# The Application of Artificial Intelligence in Board Game - AZUL Case


1. Introduction
AZUL was first released in 2017 as a scoring-based abstract strategic board game for 2 – 4 players and soon became famous around the world. The rules of AZUL are simple, a reasonable human could have the ability to play the game with only the basic knowledge of the rules. 
The board of AZUL includes four main areas: the factory zone, the pattern line area, the floor line, and the wall area, as shown in Figure 1. Firstly, the factory zone which includes five-tonine displays (the number changes with the number of players) and there are four tiles on each display, players will be required to choose tiles from the factory zone; Second, the pattern line, players use this area to store the tiles they pick from factory display; Third, the floor line where players use to place the tiles that cannot fit into the available slots in pattern lines and the firsttoken-player deduction of marks (The first player taking tiles from central area); Fourth, the wall area where players put reasonable tiles at the end of each round and calculate marks based on the situation of wall area


2. Domain Analysis
Complete information game: AZUL is a complete information and abstract strategic game which means each player has the knowledge of all the other players in the game and can sufficiently consider not only the situation of its own board but that of other players for each move to maximize its chance to win.  
Strategy: The strategy of this game is working to reduce the penalty scoring over the game. Also, setting up combos on the wall may offset the penalty loss. Special patterns on the board might help improve the scoring of the next round ("How to play Azul Solo | Game Rules | UltraBoardGames", 2020). 
Space Complexity: Players face a random rollout at the beginning of each round by randomly selecting 4 tiles from the bag to each factory display, which may yield 10^27different situations. Combining the situation on players’ board, the game state can reach the complexity of 𝑂(10^55). For the complexity of the game tree, we assume that the average branching factor is 50 and the depth is usually 50 plies (25 moves), which results in a lower bound game tree complexity of 1089. The complexity is increasing with the number of players. Under that high game tree complexity, the brute force approach is unfeasible.

3. Method
- Instant Reward Estimation Agent 
      (Instant Reward Estimation is used to evaluate the value of a placing action that brings to a certain situation on the wall.)
- Search Agent 
      (The shortcoming of the Instant Reward Estimation Agent is the lack of further observation, which means it behaves too greedy for the immediate score and ignore the potential danger of penalties coming in the next few turns, and the player have no strategy of choosing the moves that impact the opponents severely. Using tree search can provides strategies to future observation and evade from traps and find the optimal path of the game. )
- Search Agent with Predicting Opponents’ Moves 
      (The accuracy of the search tree depends on the precision of extending the opponents’ move. In this section, we will describe how we use Neural Networks to predict the action of Navïe Player and how we embedded it in our MCTS Agent. )
