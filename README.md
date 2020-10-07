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
  
  In statistical physics and thermodynamics complicated systems are often described by a variety of thermodynamical potentials, such as entropy, enthalpy or free energy function. The important feature about this potentials is that all of them reach an extremum at the equilibrium state of the system. In out case the equilibrium state is defined as the agent reaching bottom right corner of the maze, because at this point the agent does not have to do anything and can stay there forever, which is an equilibrium by definition.
  
  Statistically the free energy *F* can be expressed using the *statistical sum* **Z**:
  
<a href="https://www.codecogs.com/eqnedit.php?latex=Z&space;=&space;\sum_j&space;e^{E_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;\sum_j&space;e^{E_j}" title="Z = \sum_j e^{E_j}" /></a>, 

where E denotes the energy of each state. Then, free energy is:
 
<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;-kT&space;ln&space;Z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;-kT&space;ln&space;Z" title="F = -kT ln Z" /></a>  ,

where *k* is the Boltzmann's constant and *T* is temperature.

  Hence the statistical description is acquired.
  
  Thermodynamical or statistical description of a physical system is a powerfull tool, suitable to describe complicated many-particle systems. However, in a one-particle system such as ours it's much more convenient to use mechanical approach.
  
  * **The mechanical equivallent of minimizing the free energy**
  
  It is showed in general course of physics, that the state with minimal free energy is fully equivallent to the state, which minimizes **action functional** **J**:
  
<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\int_0^t&space;L(\dot{q},&space;q,&space;t)&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\int_0^t&space;L(\dot{q},&space;q,&space;t)&space;dt" title="J = \int_0^t L(\dot{q}, q, t) dt" /></a>
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=J&space;\rightarrow&space;min&space;\Leftrightarrow&space;F&space;\rightarrow&space;min" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;\rightarrow&space;min&space;\Leftrightarrow&space;F&space;\rightarrow&space;min" title="J \rightarrow min \Leftrightarrow F \rightarrow min" /></a>
  
  where <a href="https://www.codecogs.com/eqnedit.php?latex=L(\dot{q},&space;q,&space;t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\dot{q},&space;q,&space;t)" title="L(\dot{q}, q, t)" /></a> is the *Lagrange* function of physical system, defined as the differenece between kinetic and potential energy. It depends on impulses <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{q}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{q}" title="\dot{q}" /></a>, coordinates *q* and time *t*.
  
<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;T&space;-&space;U&space;=&space;\frac{\dot{q}^2}{2m}&space;-&space;U(q)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;T&space;-&space;U&space;=&space;\frac{\dot{q}^2}{2m}&space;-&space;U(q)" title="L = T - U = \frac{\dot{q}^2}{2m} - U(q)" /></a>
  
  where *U(q)* denotes the *potential* - an important function, which will be explained further. *m* denotes the mass of the agent, which will be set to 1.
  
  The conventional way to minimize any functional, including *J*, is to use the Euler–Lagrange equation, which leads to:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL}{dq}&space;=&space;\frac{d}{dt}&space;\frac{dL}{d&space;\dot{q}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL}{dq}&space;=&space;\frac{d}{dt}&space;\frac{dL}{d&space;\dot{q}}" title="\frac{dL}{dq} = \frac{d}{dt} \frac{dL}{d \dot{q}}" /></a>
  
  Substituting, we finally derive the simple Newton equation:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\ddot{q}&space;=&space;-&space;\frac{dU}{dq}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ddot{q}&space;=&space;-&space;\frac{dU}{dq}" title="\ddot{q} = - \frac{dU}{dq}" /></a>
  
  Let's provide some insight for the last equation. <a href="https://www.codecogs.com/eqnedit.php?latex=\ddot{q}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ddot{q}" title="\ddot{q}" /></a> is the second derivative of the coordinate, or, taking into account *m=1*, the force. The force represents the action that the agent takes on each step - whether to go north, south, east or west. 
  
  Where the force should be directed? It should be directed into minimizing the potential *U(q)* at each point *q*. In contrast with the initial free energy *F*, the potential *U(q)* is a mechanical term - free energy at each state is a single value for the whole system, while potential is a function of coordinates, fixed for any system state.
  
  What is the physical equivallent for this task? Let's imagine a slide at the waterpark. The potential *U(q)* is exactly the form of that slide. At each moment a human inside the slope moves towards the gradient descent of *U(q)*, guided by the **principle of least action**, or principle of minimal free energy, which is equivallent. It might be shown that this trajectory is the fastest way to get to the final point.
  
  * **What the problem of shortest path in the maze is transformed into in this notation?**
  
  We figured out that the shortest path for the agent to complete it's goal is to follow the potential *U(q)* descent. We basically want it to roll down the slope in a waterpark. The only problem is that the agent does not know the environment, and, hence, does not know the potential. We are gonna make the agent to learn it interacting with the environment. 
  In order to learn *U(q)* efficiently we represent it as the sum of three different potentials:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=U(q)&space;=&space;U_{border}&space;&plus;&space;U_{optimal}&space;&plus;&space;U_{slope}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U(q)&space;=&space;U_{border}&space;&plus;&space;U_{optimal}&space;&plus;&space;U_{slope}" title="U(q) = U_{border} + U_{optimal} + U_{slope}" /></a>
  
  1) <a href="https://www.codecogs.com/eqnedit.php?latex=U_{border}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{border}" title="U_{border}" /></a> is needed to avoid walls and borders of the maze. Each time the agent tries to move from one cell to another and fails, we interpret it as an infinite potential wall between this cells. It automatically makes the agent not to go there the next time.
  
  2) <a href="https://www.codecogs.com/eqnedit.php?latex=U_{optimal}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{optimal}" title="U_{optimal}" /></a> represents the agent's approximation of potential.
  
  3) <a href="https://www.codecogs.com/eqnedit.php?latex=U_{slope}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{slope}" title="U_{slope}" /></a> is set to 1 at each point where agent has already been at each iteration. This part saves us from random walking back and forth at early stages and speeds up the training.
  
  The inference is simple: at each point the agent should go to the minimal potential in it's current poistion's neighbourhood. In order to increase exploration ability, sometimes the agent goes to random direction instead of known opimal. After the goal is reached, the potential is updated.
  
 ## Results
 
 Out approach called 'Mechanical' is implemented and compared with q-learning baseline provided by [ai-gym](https://github.com/MattChanTK/ai-gym).
 
 This graph is a learning curve averaged between 1000 different random initializations on the same environment (TODO: compare on different environments of different sizes). It is shown that in that setup our approach converges much faster at first, but it takes approximately as many steps to find a more optimal solution for us as for q-learning baseline.
 
 <img src="https://github.com/Andrey885/RL_Maze/blob/master/result_1000.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="400" height="300" />
