#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <omp.h>

#define N 60000
#define SOURCE 103
#define MAXINT 9999999

void dijkstra(int **graph, int source);

int **createConnectionMatrix2()
{
    int numStations = N;
    std::cout << numStations << std::endl;
    int **adjacencyMatrix = new int *[N]; // Using new instead of malloc
    for (int i = 0; i < N; i++)
    {
        adjacencyMatrix[i] = new int[N];
    }

    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections)
    {
        std::cerr << "Error opening file: NewYork/NewYork_Edgelist.csv" << std::endl;
        exit(1);
    }

    std::string t1;
    getline(fileConnections, t1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j)
            {
                adjacencyMatrix[i][j] = 0;
            }
            else
            {
                adjacencyMatrix[i][j] = MAXINT;
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        int source, target, c2;
        char c;
        float a, b, weight;
        fileConnections >> a >> c >> b >> c >> source >> c >> target >> c >> c2 >> c >> weight;
        getline(fileConnections, t1);
        if (source < numStations && target < numStations)
            adjacencyMatrix[source][target] = weight;
        else
            break;
    }
    fileConnections.close();
    std::cout << "kraj unose";
    return adjacencyMatrix;
}

int main()
{

    struct timeval tv;
    struct timezone tz;
    int **weight = createConnectionMatrix2();
    gettimeofday(&tv, &tz);
    double time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    dijkstra(weight, SOURCE);

    gettimeofday(&tv, &tz);
    double time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;

    std::cout << "\nNodes: " << N << " time cost is " << time_end - time_start << "\n\n";

    for (int i = 0; i < N; ++i)
    {
        delete[] weight[i];
    }
    delete[] weight;

    return 0;
}
#include <omp.h>

void dijkstra(int **graph, int source)
{
    int distance[N];
    int visited[N];
    int count, nextNode, minDistance;

    // Initialization
    for (int i = 0; i < N; i++)
    {
        distance[i] = graph[source][i];
        visited[i] = 0;
    }
    distance[source] = 0; // Distance from source to source
    visited[source] = 1;
    count = 1;

    while (count < N)
    {
        minDistance = MAXINT;
        nextNode = -1;

// Find the next node with the minimum distance
#pragma omp parallel for reduction(min : minDistance)
        for (int i = 0; i < N; i++)
        {
            if (distance[i] < minDistance && !visited[i])
            {
                #pragma omp critical
                {
                    if (distance[i] < minDistance && !visited[i])
                    {
                        minDistance = distance[i];
                        nextNode = i;
                    }
                }
            }
        }
        if (nextNode != -1)
        { // If a next node is found
            visited[nextNode] = 1;
            count++;

            // Parallelize the relaxation step
            #pragma omp parallel for
            for (int i = 0; i < N; i++)
            {
                if (!visited[i])
                {
                    int newDist = minDistance + graph[nextNode][i];
                    if (newDist < distance[i])
                    {
                        #pragma omp critical
                        {
                            if (newDist < distance[i])
                            {
                                distance[i] = newDist;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            break; // No more nodes are reachable
        }
    }

    std::cout << "Distance from source to node 161: " << distance[161] << "\n";
}
