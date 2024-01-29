#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <queue>

#define N 60000
#define SOURCE 103
#define MAXINT 9999999


//rezultat 794 zadnja linija za ispisati

void dijkstra(int** graph, int source, int threads);

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
    // std::cout << "Please enter number of threads: ";
    // std::cin >> threads;
    

    double time_start, time_end;
    struct timeval tv;
    struct timezone tz;
    int** graph = createConnectionMatrix2();
    gettimeofday(&tv, &tz);
    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;  
    
    dijkstra(graph, SOURCE, 4);
    gettimeofday(&tv, &tz);

    time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    std::cout << "Nodes: " << N << "\n";
    std::cout << "Time cost is " << time_end - time_start << "\n";

    for (int i = 0; i < N; i++) {
        delete[] graph[i];
    }
    delete[] graph;


    return 0;
}



void dijkstra(int** graph, int source, int threads) {
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    std::vector<int> distance(N, MAXINT);
    std::vector<bool> visited(N, false);
    omp_set_num_threads(threads);

    distance[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;
        
        
        //#pragma omp parallel for directive se koristi za paralelizaciju petlje koja se ponavlja preko susjednih vrhova
        #pragma omp parallel for
        for (int v = 0; v < N; v++) {
            if (!visited[v] && graph[u][v] != MAXINT) {
                //osigurava da samo jedna nit istovremeno može izvršiti kritičnu sekciju, sprječavajući uvjete utrke prilikom ažuriranja niza udaljenosti.
                #pragma omp critical
                if (distance[u] + graph[u][v] < distance[v]) {
                    distance[v] = distance[u] + graph[u][v];
                    pq.push({distance[v], v});
                }
            }
        }
    }
    std::cout << "distance " << distance[161] << "\n";
}
