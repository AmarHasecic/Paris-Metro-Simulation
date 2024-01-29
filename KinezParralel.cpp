#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <omp.h>    

#define N 40000
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
    double sumaVremena = 0;
    for(int i=0; i<1000; i++){
        gettimeofday(&tv, &tz);
        double time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
        parallelDijkstra(weight, SOURCE, 8);

        gettimeofday(&tv, &tz);
        double time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
        sumaVremena +=  (time_end - time_start);       
    }
    

    std::cout << "\nNodes: " << N << " time cost is " << sumaVremena << "\n\n";

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
        int u = -1, minDistance = MAXINT;

        #pragma omp parallel
        {
            int localU = -1, localMinDistance = MAXINT;

            #pragma omp for nowait
            for (int i = 0; i < N; i++) {
                if (!visited[i] && dist[i] < localMinDistance) {
                    localMinDistance = dist[i];
                    localU = i;
                }
            }

            #pragma omp critical
            {
                if (localMinDistance < minDistance) {
                    minDistance = localMinDistance;
                    u = localU;
                }
            }
        } // Implicit barrier at the end of omp parallel

        if (u == -1) break; // If there are no more nodes to process, exit the loop

        visited[u] = true;

        // Parallelize the relaxation step
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (!visited[i] && graph[u][i] != MAXINT) {
                int newDist = dist[u] + graph[u][i];
                if (newDist < dist[i]) {
                    dist[i] = newDist;
                    prev[i] = u; // Record the path
                }
            }
        }
    }

}
