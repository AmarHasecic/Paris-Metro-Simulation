#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <omp.h>    

#define N 60000
#define SOURCE 103
#define MAXINT 9999999

void dijkstra(int** graph, int source);
void parallelDijkstra(int** graph, int source, int M);

int** createConnectionMatrix2() {
    int numStations = N;
    std::cout << numStations << std::endl;
    int** adjacencyMatrix = new int*[N]; // Using new instead of malloc
    for (int i = 0; i < N; i++) {
        adjacencyMatrix[i] = new int[N];
    }

    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections) {
        std::cerr << "Error opening file: NewYork/NewYork_Edgelist.csv" << std::endl;
        exit(1);
    }

    std::string t1;
    getline(fileConnections, t1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(i==j){
                adjacencyMatrix[i][j]=0;
            }else{
                adjacencyMatrix[i][j]=MAXINT;
            }
            
        }
    }
    for (int i = 0; i < N; i++) {
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

int main() {

    struct timeval tv;
    struct timezone tz;
    int** weight = createConnectionMatrix2();
    gettimeofday(&tv, &tz);
    double time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    parallelDijkstra(weight, SOURCE, 8);

    gettimeofday(&tv, &tz);
    double time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;

    std::cout << "\nNodes: " << N << " time cost is " << time_end - time_start << "\n\n";

    for (int i = 0; i < N; ++i) {
        delete[] weight[i];
    }
    delete[] weight;

    return 0;
}



void parallelDijkstra(int** graph, int source, int M) {
    std::vector<int> dist(N, MAXINT);
    std::vector<bool> visited(N, false);
    std::vector<int> prev(N, -1);  // Array to store the path
    dist[source] = 0;

    int numThreads = std::min(M, N); // You can't have more threads than nodes
    omp_set_num_threads(numThreads);

    while (true) {
        // Find the minimum distance node from the set of vertices not yet processed
        // This step remains sequential since it's a reduction operation
        int u = -1, minDistance = MAXINT;
        for (int i = 0; i < N; i++) {
            if (!visited[i] && dist[i] < minDistance) {
                minDistance = dist[i];
                u = i;
            }
        }

        if (u == -1) break; // If there are no more nodes to process, exit the loop

        visited[u] = true;

        // Parallelize the relaxation step
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int start = (N / numThreads) * thread_num;
            int end = (thread_num == numThreads - 1) ? N : start + (N / numThreads);

            for (int i = start; i < end; i++) {
                if (!visited[i] && graph[u][i] != MAXINT) {
                    int newDist = dist[u] + graph[u][i];
                    if (newDist < dist[i]) {
                        dist[i] = newDist;
                        prev[i] = u; // Record the path
                    }
                }
            }
        } // Implicit barrier at the end of omp parallel
    }

    // Print the distances and the path
   std::cout << "rezultat: " <<dist[161] << std::endl;
}

// The rest of the code remains unchanged...




void dijkstra(int** graph, int source) {
    int distance[N];
    int visited[N];
    int count, nextNode, minDistance;

    for (int i = 0; i < N; i++) {
        distance[i] = graph[source][i];
        visited[i] = 0;
    }
    visited[source] = 1;
    count = 1;

    while (count < N) {
        minDistance = MAXINT;
        for (int i = 0; i < N; i++) {
            if (distance[i] < minDistance && !visited[i]) {
                minDistance = distance[i];
                nextNode = i;
            }
        }

        visited[nextNode] = 1;
        count++;

        for (int i = 0; i < N; i++) {
            if (!visited[i] && minDistance + graph[nextNode][i] < distance[i]) {
                distance[i] = minDistance + graph[nextNode][i];
            }
        }
    }
    std::cout<< " distance " << distance[161] << "\n";

}
