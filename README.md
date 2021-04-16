# ParallelComputingProject
Final project for Parallel Computing

exactSolution.c is a program that computes an exact solution for the Traveling Salesman Problem (TSP) for a 2D Euclidean distance case.
An example compilation would be:
  gcc exactSolution.c -o exactSolution-exe -Wall -lm

The first argument when running it is required which specifies the number of nodes in the TSP. By default the node positions are generated randomly.
Additional parameters can be specified in additional command line arguments.
An example execution of the executable would be:
  ./exactSolution-exe 12 -n NodesIn.txt -f NodesOut.txt -p BestPath.txt
  This will read in the 12 nodes from the file NodesIn.txt as well as write them to NodesOut.txt. It will also write out the order for the best path to BestPath.txt
  
  
ACOserial.c is a program that runs ant colony optimization for a TSP in serial.
An example compilation would be:
  gcc ACOserial.c -o ACOs-exe -Wall -lm
  
This program has the same format for command line arguments as exactSolution.c with some additional parameters for the ACO case.
For a list of parameters that can be specified with command line arguments, run the program with no arguments.
