# RL_Maze
### Applying physical principles for reinforcement learning task of maze passage finding

Inspired by Frinston's [active inference model](https://www.researchgate.net/publication/323968061_Planning_and_navigation_as_active_inference/link/5ab5362045851515f59a48fc/download) which aims to minimize the *free energy* function by Markov decision process. In this repo the same function is minimized using mechanical approach instead of Bayesian inference.

## Task
Initilly the reinforcement agent (the blue ball) is placed at the upper left corner of the labirint. We want the agent to learn the shortest path to the destination, which is a square at bottom right corner.

At every step the agent decides on 4 acts: go *north, south, east* or *west*. The environment provides the agent's position and whether the task is completed after each step as a respond.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

## Environment
We use [gym](https://github.com/MattChanTK/gym-maze) to emulate the maze environment - this framework provides GUI for training visualization, maze samples generation and a q-learning baseline. It is completely independent on decision making algorithm.

## Theory

  * **What is the free energy F in physics?**
  
  In statistical physics and thermodynamics the complicated systems are often described by a variety of thermodynamical potentials, such as entropy, enthalpy or free energy function. The important feature about this potentials is that thay reach an extremum at the equilibrium state of the system. In out case the equilibrium state is defined as the bottom right corner of the maze, because at this point the agent does not have to do anything and can stay there forever, which is an equilibrium by definition.
  
  Statistically the free energy *F* can be expressed using the *statistical sum* Z:
  
  $Z = \sum_j e^{E_j}$
  
  $F = -kT ln Z$
  
  Hence the statistical description is acquired.
  
  Thermodynamical or statistical description of a physical system is a powerfull tool, suitable to describe complicated many-particle systems. However, in a one-particle system such as ours it't much more convenient to use mechanical approach.
  
  * **The mechanical equivallent of minimizing the free energy**
  It is showed in general course of physics, that the state with minimal free energy is fully equivallent to the state, which minimizes **action functional**. 
  
  
