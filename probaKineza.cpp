#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <sstream> // Include for std::istringstream
#include <sys/time.h>
#include <omp.h>

#define N 60000
#define SOURCE 103
#define MAXINT 9999999

int** createConnectionMatrix2() {
    std::cout <<"daj drugacije"<< N << std::endl;
    int** adjacencyMatrix = new int*[N];
    for (int i = 0; i < N; i++) {
        adjacencyMatrix[i] = new int[N];
    }

    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections) {
        std::cerr << "Error opening file: NewYork/NewYork_Edgelist.csv" << std::endl;
        exit(1);
    }

    std::string line;
    getline(fileConnections, line); // Skip header line if present
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            adjacencyMatrix[i][j] = (i == j) ? 0 : MAXINT;
        }
    }

    while (getline(fileConnections, line)) {
        std::istringstream iss(line);
        int source, target;
        float weight;
        char comma;

        iss >> source >> comma >> target >> comma >> weight;
        if (source < N && target < N) {
            adjacencyMatrix[source][target] = static_cast<int>(weight);
        }
    }
    
    fileConnections.close();
    std::cout << "Matrix loading complete.\n";
    return adjacencyMatrix;
}

void improvedParallelDijkstra(int** graph, int numCores, int numNodes, int source) {
    std::vector<int> dist(numNodes, MAXINT);
    std::vector<bool> visited(numNodes, false);
    std::vector<int> prev(numNodes, -1);
    dist[source] = 0;

    omp_set_num_threads(numCores);

    int w = 0;
    while (w != -1) {
        // Find the minimum distance node from the set of vertices not yet processed
        int minDistance = MAXINT;
        w = -1;
        for (int i = 0; i < numNodes; ++i) {
            if (!visited[i] && dist[i] < minDistance) {
                minDistance = dist[i];
                w = i;
            }
        }

        if (w == -1) break;

        visited[w] = true;

        // Parallel section to update distances
        #pragma omp parallel for
        for (int i = 0; i < numNodes; ++i) {
            if (!visited[i] && graph[w][i] != MAXINT) {
                int newDist = dist[w] + graph[w][i];
                if (newDist < dist[i]) {
                    dist[i] = newDist;
                    prev[i] = w;
                }
            }
        }
    }

    // Output the distances for demonstration purposes
    std::cout << "rezultat: " <<dist[161] << std::endl;
}

int main() {
  

    //int numCores = omp_get_max_threads(); // Use maximum number of available cores
    int numCores = 8;

    struct timeval tv;
    struct timezone tz;

    int** weight = createConnectionMatrix2();
    
    gettimeofday(&tv, &tz);
    double time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    
    improvedParallelDijkstra(weight, 8, N, SOURCE);
    
    gettimeofday(&tv, &tz);
    double time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    
    std::cout << "\nNodes: " << N << ", Time cost: " << time_end - time_start << " seconds.\n\n";

    for (int i = 0; i < N; ++i) {
        delete[] weight[i];
    }
    delete[] weight;

    return 0;
}
