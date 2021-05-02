#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

double generatePaths(int* path, double** nodes, int n)
{
  char* c = calloc(sizeof(char), n);
  double length = pathLength(path, nodes, n);
  double minLength = length;
  int* minPath = calloc(sizeof(int), n);
  pathCopy(path, minPath, n);
  
  //printPath(path, n, length);
  
  int i = 0;
  while (i < n-1)
  {
	if (c[i] < i)
	{
	  if (i % 2 == 0)
		swap(path, 1, i+1);
	  else
        swap(path, c[i]+1, i+1);
	  length = pathLength(path, nodes, n);
	  //printPath(path, n, length);
	  if (length < minLength)
	  {
		minLength = length;
		pathCopy(path, minPath, n);
	  }
	  c[i] += 1;
	  i = 0;
	}
	else
	{
	  c[i] = 0;
	  i++;
	}
  }
  pathCopy(minPath, path, n);
  free(c);
  free(minPath);
  return minLength;
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
	printf("Usage: Specify number of nodes in TSP, e.g. ./exactSolution-exe 10\n\tadditional options:\n\n\t-n <filename>\tspecify file with node locations to use in the format:\n\t\tx1,y1\n\t\tx2,y2\n\t\t . .\n\t\t . .\n\t\t . .\n\t\txn,yn\n\t-s seed\t\tspecify seed for random node generations\n\t-f <filename>\tspecify file to write nodes to (default: nodes.txt)\n\t-pf <filename>\tspecify file to write path to (default: path.txt)\n");

	exit(EXIT_FAILURE);
  }
  
  int n = atoi(argv[1]);

  int* path = calloc(sizeof(int), n);
  for (int i = 0; i < n; i++)
	path[i] = i;

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
  char* pathStr = "-pf";
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
  }
  if (randomNodes)
	generateRandomNodes(nodes, n);

  double minLength = generatePaths(path, nodes, n);
  
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
  
  return EXIT_SUCCESS;
}
