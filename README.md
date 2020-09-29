# RL_Maze
### Applying physical principles for reinforcement learning task of maze passage finding

Inspired by Frinston's [active inference model](https://www.researchgate.net/publication/323968061_Planning_and_navigation_as_active_inference/link/5ab5362045851515f59a48fc/download), which aims to minimize the *free energy* function by Markov decision process. In this repo the same function is minimized using mechanical approach instead of Bayesian inference.

## Task
Initilly the reinforcement agent (the blue ball) is placed at the upper left corner of the labirint. We want the agent to learn the shortest path to the destination, which is a red square at bottom right corner.

At every step the agent decides on 4 acts: go *north, south, east* or *west*. The environment provides the agent's position and whether the task is completed after each step as a respond.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

## Environment
We use [gym](https://github.com/MattChanTK/gym-maze) to emulate the maze environment - this framework provides GUI for training visualization, maze samples generation and a q-learning baseline. The environment is completely independent on decision making algorithm.

## Theory

  * **What is the free energy F in physics?**
  
  In statistical physics and thermodynamics complicated systems are often described by a variety of thermodynamical potentials, such as entropy, enthalpy or free energy function. The important feature about this potentials is that all of them reach an extremum at the equilibrium state of the system. In out case the equilibrium state is defined as the bottom right corner of the maze, because at this point the agent does not have to do anything and can stay there forever, which is an equilibrium by definition.
  
  Statistically the free energy *F* can be expressed using the *statistical sum* **Z**:
  
<a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;\sum_j&space;e^{E_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\sum_j&space;e^{E_j}" title="Z = \sum_j e^{E_j}" /></a>, where E denoted the the energy of each state. Then, free energy is:
 
<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;-kT&space;ln&space;Z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;-kT&space;ln&space;Z" title="F = -kT ln Z" /></a>  ,

where *k* is the Boltzmann's constant and *T* is temperature.

  Hence the statistical description is acquired.
  
  Thermodynamical or statistical description of a physical system is a powerfull tool, suitable to describe complicated many-particle systems. However, in a one-particle system such as ours it't much more convenient to use mechanical approach.
  
  * **The mechanical equivallent of minimizing the free energy**
  It is showed in general course of physics, that the state with minimal free energy is fully equivallent to the state, which minimizes **action functional** **J**:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\int_{0}^{t}&space;L(p,&space;q,&space;t)&space;dt," target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\int_{0}^{t}&space;L(p,&space;q,&space;t)&space;dt," title="J = \int_{0}^{t} L(p, q, t) dt," /></a>
  
  where *L(p, q, t)* is the *Lagrange* function of physical system, defined as the differenece between kinetic and potential energy. It depends on impulses *p*, coordinates *q* and time *t*.
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;T&space;-&space;U&space;=&space;\frac{p^2}{2m}&space;-&space;U(q)," target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;T&space;-&space;U&space;=&space;\frac{p^2}{2m}&space;-&space;U(q)," title="L = T - U = \frac{p^2}{2m} - U(q)," /></a>
  
  where *U(q)* denotes the *potential* - an important function, which will be explained further. *m* denotes the mass of the agent, which will be set to 1.
  
  The conventional way to minimize any functional, including *J*, is to use the Euler–Lagrange equation, which leads to:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL}{dq}&space;=&space;\frac{d}{dt}&space;\frac{dL}{d&space;\dot{q}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL}{dq}&space;=&space;\frac{d}{dt}&space;\frac{dL}{d&space;\dot{q}}" title="\frac{dL}{dq} = \frac{d}{dt} \frac{dL}{d \dot{q}}" /></a>
  
  Substituting, we finally derive the simple Newton equation:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\ddot{q}&space;=&space;-&space;\frac{dU}{dq}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ddot{q}&space;=&space;-&space;\frac{dU}{dq}" title="\ddot{q} = - \frac{dU}{dq}" /></a>
  
  Let's provide some insight for the last equation. <a href="https://www.codecogs.com/eqnedit.php?latex=\ddot{q}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ddot{q}" title="\ddot{q}" /></a> is the second derivative of the coordinate, or, taking into account *m=1*, the force. The force represents the action that the agent takes on each step - whether to go north, south, east or west. Where the force should be directed? It should be directed into minimizing the potential.
  
  
  
