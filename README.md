# The Application of Artificial Intelligence in Board Game - AZUL Case

This repository contains a framework to support policy learning for the boardgame AZUL, published by Plan B Games. The purpose of this framework is to allow students to implement algorithms for learning AI players for the game and evaluate the performance of these players against human/other AI players. 


1. Introduction
AZUL was first released in 2017 as a scoring-based abstract strategic board game for 2 – 4 players and soon became famous around the world. The rules of AZUL are simple, a reasonable human could have the ability to play the game with only the basic knowledge of the rules. 
The board of AZUL includes four main areas: the factory zone, the pattern line area, the floor line, and the wall area, as shown in Figure 1. Firstly, the factory zone which includes five-tonine displays (the number changes with the number of players) and there are four tiles on each display, players will be required to choose tiles from the factory zone; Second, the pattern line, players use this area to store the tiles they pick from factory display; Third, the floor line where players use to place the tiles that cannot fit into the available slots in pattern lines and the firsttoken-player deduction of marks (The first player taking tiles from central area); Fourth, the wall area where players put reasonable tiles at the end of each round and calculate marks based on the situation of wall area


2. Method
- Instant Reward Estimation Agent 
      (Instant Reward Estimation is used to evaluate the value of a placing action that brings to a certain situation on the wall.)
- Search Agent 
      (Using tree search can provides strategies to future observation and evade from traps and find the optimal path of the game. )
- Search Agent with Predicting Opponents’ Moves 
      (Use Neural Networks to predict the action of Navïe Player and embedded it in our MCTS Agent. )
