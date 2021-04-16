#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

struct NodeList
{
  int node;
  
  struct NodeList* prev;
  struct NodeList* next;
};

void printPath(int* path, int n, double length)
{
  printf("Path:\t%d",path[0]);
  for (int i = 1; i < n; i++)
	printf(", %d", path[i]);
  printf("\n\tLength: %f\n", length);
}

void swap(int* path, int i, int j)
{
  int temp = path[i];
  path[i] = path[j];
  path[j] = temp;
}

void parseNodes(char* arg, double** nodes, int n)
{
  char* str1;
  char* str2;
  char* ptr1;
  char* ptr2;
  
  char* nodeStr;
  char* coordStr;

  int i, j;
  for (i = 0, str1 = arg; ; i++, str1 = NULL)
  {
	nodeStr = strtok_r(str1, "\n", &ptr1);
	if (nodeStr == NULL)
	  break;
	for (j = 0, str2 = nodeStr; ; j++, str2 = NULL)
	{
	  coordStr = strtok_r(str2, " ,", &ptr2);
	  if (coordStr == NULL)
		break;
	  nodes[i][j] = atof(coordStr);
	}	
  }
}

void generateRandomNodes(double** nodes, int n)
{
  for (int i = 0; i < n; i++)
  {
	for (int j = 0; j < 2; j++)
	{
	  nodes[i][j] = ((double) rand()) / RAND_MAX;
	}
  }
}

double dist(double* n1, double* n2)
{
  double diffX = n1[0] - n2[0];
  double diffY = n1[1] - n2[1];

  return sqrt(diffX*diffX + diffY*diffY);
}

double pathLength(int* path, double** nodes, int n)
{
  double totalDistance = 0.0;

  for (int i = 1; i < n; i++)
	totalDistance += dist(nodes[path[i-1]], nodes[path[i]]);
  totalDistance += dist(nodes[path[n-1]], nodes[path[0]]);

  return totalDistance;
}

void pathCopy(int* source, int* dest, int n)
{
  for (int i = 0; i < n; i++)
	dest[i] = source[i];
}

int getIndex(int n1, int n2, int n)
{
  int m = n1*(n1 < n2) + n2*(n2 < n1);
  int d = (n2-n1)*(n1 < n2) + (n1-n2)*(n2 < n1);
  return m*(2*n - m - 1)/2 + d - 1;
}

double* calcDistances(double** nodes, int n)
{
  double* distances = calloc(sizeof(double), n*(n-1)/2);
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

double getWeight(double dist, double pheremone, double alpha, double beta)
{
  return pow(1/dist, alpha) + pow(pheremone, beta);
}

int* getBestPath(double* distances, double* pheremones, int n,
				 double alpha, double beta)
{
  int* path = calloc(sizeof(int), n);
  path[0] = 0;
  struct NodeList* head = calloc(sizeof(struct NodeList),1);
  head->prev = NULL;
  struct NodeList* temp = head;
  struct NodeList* curr;
  struct NodeList** nodeAccessor = calloc(sizeof(struct NodeList*), n);
  for (int i = 0; i < n; i++)
	nodeAccessor[i] = calloc(sizeof(struct NodeList), 1);
  for (int i = 1; i < n; i++)
  {
	curr = nodeAccessor[i];
	curr->node = i;
	curr->prev = temp;
	temp->next = curr;
	temp = curr;
  }
  curr->next = NULL;

  int bestNode;
  double bestWeight;
  int currNode = 0;
  int cNode;
  int index;
  double weight;
  for (int i = 1; i < n; i++)
  {
	curr = head->next;
	cNode = curr->node;
	bestNode = cNode;
	index = getIndex(currNode, cNode, n);
	bestWeight = getWeight(distances[index], pheremones[index], alpha, beta);

	while (curr->next)
	{
	  curr = curr->next;
	  cNode = curr->node;
	  index = getIndex(currNode, cNode, n);
	  weight = getWeight(distances[index], pheremones[index], alpha, beta);
	  if (weight > bestWeight)
	  {
		bestNode = cNode;
		bestWeight = weight;
	  }
	}
	  
	path[i] = bestNode;
	currNode = bestNode;

	curr = nodeAccessor[bestNode];
	temp = curr->prev;
	temp->next = curr->next;
	if (curr->next)
	  curr->next->prev = temp;
	free(curr);
  }
  free(head);
  free(nodeAccessor[0]);
  free(nodeAccessor);

  return path;
}

void iterateGeneration(double* distances, double* pheremones, int n, int ants,
					   double p, double alpha, double beta)
{ 
  int* path = calloc(sizeof(int), n);
  struct NodeList* head = calloc(sizeof(struct NodeList),1);
  head->prev = NULL;
  struct NodeList* temp;
  struct NodeList* curr;
  struct NodeList** nodeAccessor = calloc(sizeof(struct NodeList*), n);
  for (int i = 0; i < n; i++)
  {
	nodeAccessor[i] = calloc(sizeof(struct NodeList), 1);
	nodeAccessor[i]->node = i;
  }
  
  int currNode;
  int nextNode;
  int cNode;
  int index;
  int windex;
  double weight;
  double* weights = calloc(sizeof(double), n-1);
  int* posNodes = calloc(sizeof(int), n-1);
  double totalWeight;
  double prob;
  double totalProb;
  double targetProb;

  double length;
  int combinations = n*(n-1)/2;
  double* newPheremones = calloc(sizeof(double), combinations);

  int* bestPath = calloc(sizeof(int), n);
  double shortestLength;

  for (int a = 0; a < ants; a++)
  {
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

	// start from a random node
	currNode = rand() % n;
	curr = nodeAccessor[currNode];
	temp = curr->prev;
	temp->next = curr->next;
	if (curr->next)
	  curr->next->prev = temp;
	path[0] = currNode;

	length = 0.0;
	for (int i = 1; i < n; i++)
	{
	  curr = head->next;
	  windex = 0;
	  totalWeight = 0.0;
	  while (curr)
	  {
		cNode = curr->node;
		index = getIndex(currNode, cNode, n);
		weight = getWeight(distances[index], pheremones[index], alpha, beta);
		weights[windex] = weight;
		posNodes[windex] = cNode;
		totalWeight += weight;
		curr = curr->next;
		windex++;
	  }

	  // choose next node based on weighted probability
	  targetProb = ((double) rand()) / RAND_MAX;
	  totalProb = 0.0;
	  char pickedNode = 0;
	  for (int k = 0; k < windex; k++)
	  {
		prob = weights[k] / totalWeight;
		totalProb += prob;
		if (totalProb >= targetProb)
		{
		  nextNode = posNodes[k];
		  pickedNode = 1;
		  break;
		}
	  }
	  if (!pickedNode)
		nextNode = posNodes[windex-1];
	  
	  index = getIndex(currNode, nextNode, n);
	  length += distances[index];
	  path[i] = nextNode;
	  currNode = nextNode;

	  curr = nodeAccessor[nextNode];
	  temp = curr->prev;
	  temp->next = curr->next;
	  if (curr->next)
		curr->next->prev = temp;
	}
	index = getIndex(path[0], path[n-1], n);
	length += distances[index];

	if (a == 0 || length < shortestLength)
	{
	  shortestLength = length;
	  pathCopy(path, bestPath, n);
	}
  }
  for (int i = 1; i < n; i++)
  {
	index = getIndex(bestPath[i-1], bestPath[i], n);
	newPheremones[index] = n/(shortestLength);
  }
  index = getIndex(bestPath[0], bestPath[n-1], n);
  newPheremones[index] = n/(shortestLength);
  
  for (int i = 0; i < combinations; i++)
	pheremones[i] = (1-p)*pheremones[i] + newPheremones[i];

  free(head);
  for (int i = 0; i < n; i++)
	free(nodeAccessor[i]);
  free(nodeAccessor);
  free(weights);
  free(posNodes);
  free(newPheremones);
  free(path);
  free(bestPath);
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
	printf("Usage: Specify number of nodes in TSP, e.g. ./ACOs-exe 10\n\tadditional options:\n\n\t-a alpha\tparameter that controls weight of distances when ants choose their next node (default: 1.0)\n\t-b beta\t\tparameter that controls weight of pheremone trails when ants choose their next node (default: 2.0)\n\t-p rho\t\tparameter that controls how quickly pheremone trails evaporate (default: 0.1)\n\n\t-ants a\t\tnumber of ants to use in each generation (default: n^2)\n\t-g generations\t number of generations to run optimization for (default: 100)\n\n\t-n <filename>\tspecify file with node locations to use in the format:\n\t\tx1,y1\n\t\tx2,y2\n\t\t . .\n\t\t . .\n\t\t . .\n\t\txn,yn\n\t-s seed\t\tspecify seed for random node generations\n\t-f <filename>\tspecify file to write nodes to (default: nodes.txt)\n\t-p <filename>\tspecify file to write path to (default: path.txt)\n");

	exit(EXIT_FAILURE);
  }
  
  int n = atoi(argv[1]);
  int ants = n*n;
  int generations = 100;
  double p = 0.1;
  double alpha = 1;
  double beta = 2;

  double** nodes = calloc(sizeof(double*), n);
  for (int i = 0; i < n; i++)
	nodes[i] = calloc(sizeof(double*), 2);

  char randomNodes = 1;
  char* filename = "nodes.txt";
  char* pathname = "path.txt";
  FILE* file;
  FILE* pfile;
  
  char* nodeStr = "-n";
  char* seedStr = "-s";
  char* fileStr = "-f";
  char* pathStr = "-p";
  char* antsStr = "-ants";
  char* genStr = "-g";
  char* rhoStr = "-p";
  char* alphaStr = "-a";
  char* betaStr = "-b";
  for (int i = 2; i < argc; i++)
  {
	if (strcmp(argv[i], nodeStr) == 0)
	{
	  i++;
	  filename = argv[i];
	  char * buffer = 0;
	  long length;
	  file = fopen (filename, "r");

	  if (file)
		{
		  fseek (file, 0, SEEK_END);
		  length = ftell(file);
		  fseek (file, 0, SEEK_SET);
		  buffer = calloc (sizeof(char), length+1);
		  if (buffer)
			{
			  fread (buffer, sizeof(char), length, file);
			}
		  fclose (file);
  
		  parseNodes(buffer, nodes, n);
		  randomNodes = 0;
		  free(buffer);
		}
	}
	else if (strcmp(argv[i], seedStr) == 0)
	{
	  i++;
	  unsigned int seed = atoi(argv[i]);
	  srand(seed);
	}
	else if (strcmp(argv[i], fileStr) == 0)
	{
	  i++;
	  filename = argv[i];
	}
	else if (strcmp(argv[i], pathStr) == 0)
	{
	  i++;
	  pathname = argv[i];
	}
	else if (strcmp(argv[i], antsStr) == 0)
	{
	  i++;
	  ants = atoi(argv[i]);
	}
	else if (strcmp(argv[i], genStr) == 0)
	{
	  i++;
	  generations = atoi(argv[i]);
	}
	else if (strcmp(argv[i], rhoStr) == 0)
	{
	  i++;
	  p = atof(argv[i]);
	}
	else if (strcmp(argv[i], alphaStr) == 0)
	{
	  i++;
	  alpha = atof(argv[i]);
	}
	else if (strcmp(argv[i], betaStr) == 0)
	{
	  i++;
	  beta = atof(argv[i]);
	}
  }
  if (randomNodes)
	generateRandomNodes(nodes, n);

  int combinations = n*(n-1)/2;
  double* distances = calcDistances(nodes, n);
  double* pheremones = calloc(sizeof(double), combinations);
  for (int i = 0; i < combinations; i++)
	pheremones[i] = 1.0;

  
  for (int i = 0; i < generations; i++)
  {
	iterateGeneration(distances, pheremones, n, ants, p, alpha, beta);
	int* path = getBestPath(distances, pheremones, n, alpha, beta);
	printf("Generation %d:\t %f\n", i, pathLength(path, nodes, n));
	free(path);
  }
  int* path = getBestPath(distances, pheremones, n, alpha, beta);
  double minLength = pathLength(path, nodes, n);
  
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
  
  
  printf("\nShortest Path\n");
  printPath(path, n, minLength);
  
  for (int i = 0; i < n; i++)
	free(nodes[i]);
  free(nodes);

  free(path);

  free(distances);
  free(pheremones);
  
  return EXIT_SUCCESS;
}
