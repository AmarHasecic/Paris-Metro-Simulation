#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <omp.h> 
#include <limits>
#include <queue>
#include <functional>   

#define N 60000
#define SOURCE 103
#define MAXINT 9999999

void dijkstra(int** graph, int source);
void parallelDijkstra(int** graph, int source, int M);
void parallelDijkstraProba(int** graph, int source, int M);


struct Node {
    int vertex;
    int weight;
    bool operator>(const Node& other) const {
        return weight > other.weight;
    }
};

void parallelDijkstraProba(int** graph, int source, int numThreads) {
    std::vector<int> dist(N, MAXINT);
    std::vector<bool> visited(N, false);
    std::vector<int> prev(N, -1); // Array to store the path
    dist[source] = 0;

    omp_set_num_threads(numThreads);
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({source, 0});

    #pragma omp parallel
    {
        while (true) {
            Node u;
            bool found = false;

            #pragma omp single nowait
            {
                if (!pq.empty()) {
                    u = pq.top();
                    pq.pop();
                    found = true;
                }
            }

            if (!found) break;

            if (visited[u.vertex]) continue;

            #pragma omp single nowait
            visited[u.vertex] = true;

            #pragma omp for nowait
            for (int i = 0; i < N; i++) {
                if (graph[u.vertex][i] != 0 && graph[u.vertex][i] != MAXINT && !visited[i]) {
                    int newDist = u.weight + graph[u.vertex][i];
                    if (newDist < dist[i]) {
                        dist[i] = newDist;
                        prev[i] = u.vertex;
                        pq.push({i, newDist});
                    }
                }
            }
        }
    }

    // Print the distances and the path
    std::cout << "Distance from source to node 161: " << dist[161] << std::endl;
}

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
    parallelDijkstraProba(weight, SOURCE, 8);

    gettimeofday(&tv, &tz);
    double time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;

    std::cout << "\nNodes: " << N << " time cost is " << time_end - time_start << "\n\n";

    for (int i = 0; i < N; ++i) {
        delete[] weight[i];
    }
    delete[] weight;

    return 0;
}

