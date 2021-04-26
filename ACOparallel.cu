#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// node for double linked list
struct NodeList
{
  int node;
  
  struct NodeList* prev;
  struct NodeList* next;
};

// prints the path and the path length
void printPath(int* path, int n, double length)
{
  printf("Path:\t%d",path[0]);
  for (int i = 1; i < n; i++)
    printf(", %d", path[i]);
  printf("\n\tLength: %f\n", length);
}

// function to parse a nodes file which has been read into arg
// writes node locations to nodes array
void parseNodes(char* arg, double** nodes, int n)
{
  char* str1; // stores overall list of nodes to parse
  char* str2; // stores node coordinates to parse
  char* ptr1; // keeps track of place in list of nodes
  char* ptr2; // keeps track of place in node coordinates
  
  char* nodeStr; // stores string for node coordinate
  char* coordStr; // stores string of given node coordinate

  int i, j;
  for (i = 0, str1 = arg; ; i++, str1 = NULL)
  {
    // get the next node coordinates to parse
    nodeStr = strtok_r(str1, "\n", &ptr1);
    if (nodeStr == NULL)
      break;
    // parse the coordinates
    for (j = 0, str2 = nodeStr; ; j++, str2 = NULL)
    {
      coordStr = strtok_r(str2, " ,", &ptr2);
      if (coordStr == NULL)
	break;
      // write coordinate to node array
      nodes[i][j] = atof(coordStr);
    }	
  }
}

// function to generate n nodes with 2D coordinates in the range [0, 1]
void generateRandomNodes(double** nodes, int n)
{
  // create n nodes
  for (int i = 0; i < n; i++)
  {
    // create 2 coordinates (x, y)
    for (int j = 0; j < 2; j++)
    {
      nodes[i][j] = ((double) rand()) / RAND_MAX;
    }
  }
}

// evaluates the euclidean distance between nodes n1 and n2
double dist(double* n1, double* n2)
{
  // get difference in x and y
  double diffX = n1[0] - n2[0];
  double diffY = n1[1] - n2[1];

  // euclidean distance function
  return sqrt(diffX*diffX + diffY*diffY);
}

// determines the length of the path (order of nodes to visit)
// given list of nodes
// Note: path is circular, so returns to first node in path at the end
double pathLength(int* path, double** nodes, int n)
{
  double totalDistance = 0.0;

  // add distance between adjacent nodes on the path
  for (int i = 1; i < n; i++)
    totalDistance += dist(nodes[path[i-1]], nodes[path[i]]);
  // add the distance of the return path
  totalDistance += dist(nodes[path[n-1]], nodes[path[0]]);

  return totalDistance;
}

// deep copy of path
void pathCopy(int* source, int* dest, int n)
{
  for (int i = 0; i < n; i++)
    dest[i] = source[i];
}

__device__ void pathCopyDev(int* source, int* dest, int n)
{
  for (int i = 0; i < n; i++)
    dest[i] = source[i];
}

// computes index in 1D array corresponding to nodes n1 and n2
// the array has n*(n-1)/2 elements
// the first n-1 elements correspond to edges between node 1 and all other nodes
// the next n-2 elements correspond to the edges between node 2 and all other nodes
// except for node 1 and so on ...
int getIndex(int n1, int n2, int n)
{
  int m = n1*(n1 < n2) + n2*(n2 < n1);
  int d = (n2-n1)*(n1 < n2) + (n1-n2)*(n2 < n1);
  return m*(2*n - m - 1)/2 + d - 1;
}

__device__ static inline int getIndexDev(int n1, int n2, int n)
{
  int m = n1*(n1 < n2) + n2*(n2 < n1);
  int d = (n2-n1)*(n1 < n2) + (n1-n2)*(n2 < n1);
  return m*(2*n - m - 1)/2 + d - 1;
}

// computes the distance between every pair of nodes and returns it
// as a n*(n-1)/2 length array to use minimum spacex
double* calcDistances(double** nodes, int n)
{
  //double* distances = calloc(sizeof(double), n*(n-1)/2);
  double* distances;
  cudaMallocManaged(&distances, ((n*(n-1)/2) * sizeof(double)));
  int index = 0;
  for (int i = 0; i < n-1; i++)
  {
    for (int j = i+1; j < n; j++)
    {
      distances[index] = dist(nodes[i], nodes[j]);
      index++;
    }
  }
  return distances;
}

// determines the desireability of an edge based on distance and pheremone trail
// the parameters alpha and beta affect to relative importance of each of these factors
__device__ static inline double getWeightDev(double dist, double pheremone,
			       double alpha, double beta)
{ 
  return pow(pheremone, alpha) * pow(1/dist, beta);
}

// determines the desireability of an edge based on distance and pheremone trail
// the parameters alpha and beta affect to relative importance of each of these factors
static inline double getWeight(double dist, double pheremone,
			       double alpha, double beta)
{ 
  return pow(pheremone, alpha) * pow(1/dist, beta);
}

// generates a "best path" based on current pheremone trails
// starting from node 0, it chooses the edge with the highest weight
// based on distance and pheremones on that edge
int* getBestPath(double* distances, double* pheremones, int n,
		 double alpha, double beta)
{
  int* path = (int*)calloc(sizeof(int), n);
  path[0] = 0; // start with node 0 for consistency
  // initialize a list of nodes to visit
  // nodes will be removed as they are visited
  // this ensures that no node is visited twice
  struct NodeList* head = (NodeList*)calloc(sizeof(struct NodeList),1);
  head->prev = NULL;
  struct NodeList* temp = head;
  struct NodeList* curr;
  // an array with pointers to all elements of the list for fast accessing
  struct NodeList** nodeAccessor = (NodeList**)calloc(sizeof(struct NodeList*), n);
  for (int i = 0; i < n; i++)
    nodeAccessor[i] = (NodeList*)calloc(sizeof(struct NodeList), 1);
  // initialize node elements in the list
  for (int i = 1; i < n; i++)
  {
    curr = nodeAccessor[i];
    curr->node = i;
    curr->prev = temp;
    temp->next = curr;
    temp = curr;
  }
  curr->next = NULL; // last node has no next

  int bestNode; // keeps track of best node ID
  double bestWeight; // keeps track of highest edge weight
  int currNode = 0; // the current node from which to find a next node
  int cNode; // a candidate node
  int index; // holds index for distance/pheremone array access
  double weight; // holds weight of given edge

  // need to visit n-1 nodes (first node is always 0)
  for (int i = 1; i < n; i++)
  {
    // initialize starting parameters with first node in the list
    curr = head->next;
    cNode = curr->node;
    bestNode = cNode;
    index = getIndex(currNode, cNode, n);
    bestWeight = getWeight(distances[index], pheremones[index], alpha, beta);

    // loop through all remaining nodes to visit
    while (curr->next)
    {
      curr = curr->next;
      cNode = curr->node;
      index = getIndex(currNode, cNode, n);
      // compute the weight between the currend node and the candidate node
      weight = getWeight(distances[index], pheremones[index], alpha, beta);
      // update best node/weight if this weight is higher
      if (weight > bestWeight)
      {
	bestNode = cNode;
	bestWeight = weight;
      }
    }

    // add the node with the highest edge weight as the next node to visit
    path[i] = bestNode;
    currNode = bestNode;

    // remove the node that was added from the nodes to visit list
    curr = nodeAccessor[bestNode];
    temp = curr->prev;
    temp->next = curr->next;
    if (curr->next)
      curr->next->prev = temp;
    // free the node, it is not needed anymore
    free(curr);
  }
  // memory clean up
  free(head);
  free(nodeAccessor[0]); // need to free dummy node 0 as it was not freed
  free(nodeAccessor);

  return path;
}

// performs a generational iteration with the specified number of ants
// the ant with the best path updates the pheremones based on its path and path length
__global__ void ACO_kernel(double* distances, double* pheremones,
			   int n, double alpha, double beta,
			   double* bestLengths, int* bestPaths,
			   int* tPaths, struct NodeList** tNacc,
			   struct NodeList* tNodes, struct NodeList* tHeads,
			   int* tPosN, double* tWeights)
{
  extern __shared__ double pathLengths[];
  int bindex = blockIdx.x;
  int tindex = threadIdx.x;
  curandState_t state;
  int currAnt = bindex*blockDim.x + tindex;

  curand_init(0, currAnt, 0, &state);

  int* path = &tPaths[n*currAnt]; // variable to hold paths

  struct NodeList* head = &tHeads[currAnt]; // head of node list
  head->prev = NULL;
  // variables for accessing nodes in the list
  struct NodeList* temp;
  struct NodeList* curr;
  // initialize accessor to nodes in the list for quick access of specific nodes
  struct NodeList** nodeAccessor = &tNacc[n*currAnt];
  for (int i = 0; i < n; i++)
  {
    nodeAccessor[i] = &tNodes[n*currAnt + i];
    nodeAccessor[i]->node = i;
  }
  
  int currNode; // current node
  int nextNode; // next nod visiting
  int cNode; // candidate node
  int index; // index in distance/pheremone array
  int windex; // index in list of candidate node wieghts
  double weight; // holds calculated weight
  double* weights = &tWeights[(n-1)*currAnt]; // holds weights of candidate nodes
  int* posNodes = &tPosN[(n-1)*currAnt]; // holds list of candidate nodes
  double totalWeight; // holds cumulative sum of weights to candidate nodes
  double prob; // holds individual probability of choosing a given node
  double totalProb; // holds cumulative probability of candidate nodes
  double targetProb; // desired probability for totalProb to reach/exceed

  // initialize list of nodes to search
  temp = head;
  for (int i = 0; i < n; i++)
  {
    curr = nodeAccessor[i];
    curr->prev = temp;
    temp->next = curr;
    temp = curr;
  }
  curr->next = NULL;

  // generate a path and keep track of the length
  double length = 0.0;
  
  // start from a random node
  currNode = curand(&state) % n;
  // remove the first node in the path from the list of nodes to visit
  curr = nodeAccessor[currNode];
  temp = curr->prev;
  temp->next = curr->next;
  if (curr->next)
    curr->next->prev = temp;
  path[0] = currNode;
  
  for (int i = 1; i < n; i++)
  {
    curr = head->next;
    windex = 0;
    totalWeight = 0.0;
    // loop through all of the possible nodes
    while (curr)
    {
      // get the candidate node information
      cNode = curr->node;
      index = getIndexDev(currNode, cNode, n);
      // compute weight between current and candidate node
      weight = getWeightDev(distances[index], pheremones[index], alpha, beta);
      // keep track of candidate node weight and node ID
      weights[windex] = weight;
      posNodes[windex] = cNode;
      totalWeight += weight;
      // iterate
      curr = curr->next;
      windex++;
    }

    // choose next node based on weighted probability
    targetProb = curand_uniform(&state); 
    totalProb = 0.0;
    char pickedNode = 0; // if no node was picked by end, then pick last candidate
    // loop through the candidate nodes
    for (int k = 0; k < windex; k++)
    {
      // compute probability based on candidate weight
      // and total weight of all the candidate node edges
      prob = weights[k] / totalWeight;
      totalProb += prob;
      // if the cumulative probability exceeds the target, then pick it
      if (totalProb >= targetProb)
      {
	nextNode = posNodes[k];
	pickedNode = 1;
	break;
      }
    }
    // pick the last node if no node was chosedn
    if (!pickedNode)
      nextNode = posNodes[windex-1];

    // add the picked node to the path and add the length to the node
    index = getIndexDev(currNode, nextNode, n);
    length += distances[index];
    path[i] = nextNode;
    currNode = nextNode;

    // remove the next node from the list of nodes to visit
    curr = nodeAccessor[nextNode];
    temp = curr->prev;
    temp->next = curr->next;
    if (curr->next)
      curr->next->prev = temp;
  }
  // add length from last node back to first node
  index = getIndexDev(path[0], path[n-1], n);
  length += distances[index];

  pathLengths[tindex] = length;
  
  // synchronize after data write, before reading data from shared memory
  __syncthreads();
  // first thread determines shortest path from ants in block
  // this could be parallelized further but is good enough
  if (tindex == 0)
  {
    int numAnts = blockDim.x; // number of ants running per block

    double bestLength = 0.0;
    int bestAnt = 0;
    for (int i = 0; i < numAnts; i++)
    {
      if (i == 0 || pathLengths[i] < bestLength)
      {
	bestLength = pathLengths[i];
	bestAnt = i;
      }
    }
    // mark best path by making path length negative
    pathLengths[bestAnt] = -pathLengths[bestAnt];
  }
  __syncthreads();

  // if this thread had the best path, then report it to output to be compared
  if (pathLengths[tindex] < 0)
  {
    bestLengths[bindex] = -pathLengths[tindex]; // revert to original length
    pathCopyDev(path, &bestPaths[bindex*n], n); // copy best path over
  }

}

void iterateParallel(double* distances, double* pheremones, int n, long ants,
		     double p, double alpha, double beta,
		     int generations, const char* genName, double** nodes)
{
  int combinations = n*(n-1)/2; // number of edge pairs

  if (ants % n != 0)
    ants = ants - (ants % n) + n;
  long numBlocks = ants / n;
  
  // array for each block to write its shortest path length to
  double* bestLengths;
  cudaMallocManaged(&bestLengths, (numBlocks * sizeof(double)));

  // array for each block to write its best path to
  int* bestPaths;
  cudaMallocManaged(&bestPaths, (n * numBlocks * sizeof(int)));
    
  int* bestPath = (int*)calloc(sizeof(int), n);
  double shortestLength;
  int bestBlock;

  int* tPaths;
  cudaMallocManaged(&tPaths, (n * ants * sizeof(int)));
  struct NodeList** tNacc;
  cudaMallocManaged(&tNacc, (n * ants * sizeof(NodeList*)));
  struct NodeList* tNodes;
  cudaMallocManaged(&tNodes, (n * ants * sizeof(NodeList)));
  struct NodeList* tHeads;
  cudaMallocManaged(&tHeads, (ants * sizeof(NodeList)));
  int* tPosN;
  cudaMallocManaged(&tPosN, ((n-1) * ants * sizeof(int)));
  double* tWeights;
  cudaMallocManaged(&tWeights, ((n-1) * ants * sizeof(double)));

  int index;
  // write out the generational progress
  FILE* gfile = fopen(genName, "w+");
  // run for the specified number of generations
  printf("Ants: %ld\tBlocks: %ld\tThreads: %d\n", ants, numBlocks, n);
  for (int g = 0; g < generations; g++)
  {
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, requiredMemory);
    ACO_kernel<<<numBlocks, n, n*sizeof(double)>>>(distances, pheremones, 
						   n, alpha, beta,
						   bestLengths, bestPaths,
						   tPaths, tNacc, tNodes,
						   tHeads, tPosN, tWeights);
    // sync the kernels so all results are in before continuing
    cudaDeviceSynchronize();

    // determine the shortest path from each of the blocks
    for (int i = 0; i < numBlocks; i++)
    {
      if (i == 0 || bestLengths[i] < shortestLength)
      {
	shortestLength = bestLengths[i];
	bestBlock = i;
      }
    }
    pathCopy(&bestPaths[n*bestBlock], bestPath, n);

    // update pheremones
    for (int i = 0; i < combinations; i++)
      pheremones[i] = (1-p)*pheremones[i];

    // add pheremones to edges of the best path found for this generation
    for (int i = 1; i < n; i++)
    {
      index = getIndex(bestPath[i-1], bestPath[i], n);
      pheremones[index] += p*n/(shortestLength);
    }
    // make sure to add return pheremones from last node to first node
    index = getIndex(bestPath[0], bestPath[n-1], n);
    pheremones[index] += p*n/(shortestLength);

    // get the current best path and output the resultant length
    int* path = getBestPath(distances, pheremones, n, alpha, beta);
    printf("Generation %d:\t %f\n", g, pathLength(path, nodes, n));
    fprintf(gfile, "%d,%f\n", g, pathLength(path, nodes, n));
    free(path);
  }
  fclose(gfile);

  // clean up memeory
  free(bestPath);


  cudaFree(bestLengths);
  cudaFree(bestPaths);

  cudaFree(tPaths);
  cudaFree(tNacc);
  cudaFree(tNodes);
  cudaFree(tHeads);
  cudaFree(tPosN);
  cudaFree(tWeights);
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    printf("Usage: Specify number of nodes in TSP, e.g. ./ACOs-exe 10\n\tadditional options:\n\n\t-a alpha\tparameter that controls weight of pheremone trails when ants choose their next node (default: 1.0)\n\t-b beta\t\tparameter that controls weight of distances when ants choose their next node (default: 5.0)\n\t-p rho\t\tparameter that controls how quickly pheremone trails evaporate (default: 0.3)\n\n\t-ants a\t\tnumber of ants to use in each generation (default: n^2)\n\t-g generations\t number of generations to run optimization for (default: 100)\n\n\t-n <filename>\tspecify file with node locations to use in the format:\n\t\tx1,y1\n\t\tx2,y2\n\t\t . .\n\t\t . .\n\t\t . .\n\t\txn,yn\n\t-s seed\t\tspecify seed for random node generations\n\t-gf <filename>\tspecify file to write shortest path length for each generation to (default: generations.txt)\n\t-f <filename>\tspecify file to write nodes to (default: nodes.txt)\n\t-p <filename>\tspecify file to write path to (default: path.txt)\n");

    exit(EXIT_FAILURE);
  }

  // variables to control ACO execution
  int n = atoi(argv[1]); // number of nodes
  long ants = n*n; // number of ants to use in each generation
  int generations = 100; // number of generations
  double p = 0.3;  // pheremone evaporation rate
  double alpha = 1; // relative weight of pheremones
  double beta = 5; // relative wieght of distances

  // holds coordinates of all the nodes
  double** nodes = (double**)calloc(sizeof(double*), n);
  for (int i = 0; i < n; i++)
    nodes[i] = (double*)calloc(sizeof(double*), 2);

  char randomNodes = 1; // flag for whether to generate random nodes or not

  // output files
  const char* filename = "nodes.txt";
  const char* pathname = "path.txt";
  const char* genName = "generations.txt";
  FILE* file;
  FILE* pfile;
  
  // flags for command line options to set parameters
  const char* nodeStr = "-n";
  const char* seedStr = "-s";
  const char* fileStr = "-f";
  const char* pathStr = "-p";
  const char* gfileStr = "-gf";
  const char* antsStr = "-ants";
  const char* genStr = "-g";
  const char* rhoStr = "-p";
  const char* alphaStr = "-a";
  const char* betaStr = "-b";
  
  // loop through any remaining arguments
  for (int i = 2; i < argc; i++)
  {
    // list of nodes to use has been specified
    if (strcmp(argv[i], nodeStr) == 0)
    {
      i++;
      // open the given file
      filename = argv[i];
      char * buffer = 0;
      long length;
      file = fopen (filename, "r");

      // if valid file, then open it
      if (file)
      {
	// determine total length
	fseek (file, 0, SEEK_END);
	length = ftell(file);
	fseek (file, 0, SEEK_SET);
	// read in entire file into a buffer
	buffer = (char*)calloc (sizeof(char), length+1);
	if (buffer)
	{
	  fread (buffer, sizeof(char), length, file);
	}
	fclose (file);

	// convert contents into desired node array format
	parseNodes(buffer, nodes, n);
	randomNodes = 0; // do not need to generate random nodes
	free(buffer);
      }
    }
    // seed the random number generator for repeatable results
    else if (strcmp(argv[i], seedStr) == 0)
    {
      i++;
      unsigned int seed = atoi(argv[i]);
      srand(seed);
    }
    // set output filename for list of nodes
    else if (strcmp(argv[i], fileStr) == 0)
    {
      i++;
      filename = argv[i];
    }
    // set output filename for best path
    else if (strcmp(argv[i], pathStr) == 0)
    {
      i++;
      pathname = argv[i];
    }
    // set output filename for generation data
    else if (strcmp(argv[i], gfileStr) == 0)
    {
      i++;
      genName = argv[i];
    }
    // set number of ants per generation to use
    else if (strcmp(argv[i], antsStr) == 0)
    {
      i++;
      ants = atol(argv[i]);
    }
    // set number of generations to run for
    else if (strcmp(argv[i], genStr) == 0)
    {
      i++;
      generations = atoi(argv[i]);
    }
    // set parameter rho
    else if (strcmp(argv[i], rhoStr) == 0)
    {
      i++;
      p = atof(argv[i]);
    }
    // set parameter alpha
    else if (strcmp(argv[i], alphaStr) == 0)
    {
      i++;
      alpha = atof(argv[i]);
    }
    // set parameter beta
    else if (strcmp(argv[i], betaStr) == 0)
    {
      i++;
      beta = atof(argv[i]);
    }
  }
  // if no node list was specified, generate n nodes randomly
  if (randomNodes)
    generateRandomNodes(nodes, n);

  // initialize distance and pheremones between each pair of nodes
  int combinations = n*(n-1)/2;
  double* distances = calcDistances(nodes, n);
  
  //double* pheremones = calloc(sizeof(double), combinations);
  double* pheremones;
  cudaMallocManaged(&pheremones, (combinations * sizeof(double)));
  for (int i = 0; i < combinations; i++)
    pheremones[i] = 1.0;

  iterateParallel(distances, pheremones, n, ants, p, alpha, beta,
		  generations, genName, nodes);

  // overall best path after all generations
  int* path = getBestPath(distances, pheremones, n, alpha, beta);
  double minLength = pathLength(path, nodes, n);

  // write out the nodes and the best path to files
  file = fopen(filename, "w+");
  pfile = fopen(pathname, "w+");
  printf("Nodes:\n");
  for (int i = 0; i < n; i++)
  {
    printf("%d:\t%f, %f\n", i, nodes[i][0], nodes[i][1]);
    fprintf(file, "%f,%f\n", nodes[i][0], nodes[i][1]);
    fprintf(pfile, "%d\n", path[i]);
  }
  fclose(file);
  fclose(pfile);
  
  // print out the best path
  printf("\nShortest Path\n");
  printPath(path, n, minLength);

  // memory clean up
  for (int i = 0; i < n; i++)
    free(nodes[i]);
  free(nodes);

  free(path);

  cudaFree(distances);
  cudaFree(pheremones);
  
  return EXIT_SUCCESS;
}
