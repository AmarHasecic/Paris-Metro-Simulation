#include<cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <unordered_map>
#include <climits>
#include <chrono>
#include <omp.h>


std::vector<std::vector<int>> createConnectionMatrix() {

    std::unordered_map<std::string, int> stationIndices; 
    std::vector<std::string> stations; 
    std::vector<std::vector<int>> adjacencyMatrix; 

    std::ifstream fileStations("Data/stations.txt");
    if (!fileStations.is_open()) {
        std::cerr << "Error opening file: " << "Data/stations.txt" << std::endl;
    }

    for(int i = 0; i<376; i++){
       int nodeId;
        std::string nodeName;
        fileStations >> nodeId;
        std::getline(fileStations >> std::ws, nodeName);
        stations.push_back(nodeName);
    }
    fileStations.close();


    int numStations = stations.size();
    adjacencyMatrix.resize(numStations, std::vector<int>(numStations, 0));

    std::ifstream fileConnections("Data/connections.txt");
    if (!fileConnections.is_open()) {
        std::cerr << "Error opening file: " << "Data/connections.txt" << std::endl;
    }

    for(int i=0; i<=933; i++){
        int source, target, weight;
        fileConnections >> source >> target >> weight;
        adjacencyMatrix[source][target] = weight;
    }
    fileConnections.close();
    return adjacencyMatrix;
}

/*Chat GPT code - Dijkstra algorithm implementation*/
typedef std::vector<std::vector<int>> Graph;
using namespace std;

int minDistance(const vector<int>& dist, const vector<bool>& sptSet) {
    int min = INT_MAX, min_index;
    #pragma omp parallel for
    for (int v = 0; v < dist.size(); v++) {
        if (!sptSet[v] && dist[v] <= min) {
            #pragma omp critical
            {
                if (dist[v] <= min) {
                    min = dist[v];
                    min_index = v;
                }
            }
        }
    }
    return min_index;
}

void printPath(const vector<int>& parent, int dest) {
    if (parent[dest] == -1) {
        cout << dest;
        return;
    }
    printPath(parent, parent[dest]);
    cout << " -> " << dest;
}

void dijkstra(const Graph& graph, int src, int dest) {
    int V = graph.size();
    vector<int> dist(V, INT_MAX);
    vector<bool> sptSet(V, false);
    vector<int> parent(V, -1);

    dist[src] = 0;

    #pragma omp parallel for
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet);

        sptSet[u] = true;

        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                #pragma omp critical
                {
                    if (dist[u] + graph[u][v] < dist[v]) {
                        dist[v] = dist[u] + graph[u][v];
                        parent[v] = u;
                    }
                }
            }
        }
    }

    cout << "Shortest path from Station 1 " << src << " to Station 2 " << dest << ": ";
    printPath(parent, dest);
    cout << " (Total distance: " << dist[dest] << ")" << endl;
}
/*END of Chat GPT code*/



int main(){

    Graph graf = createConnectionMatrix();

    clock_t start = clock();

    dijkstra(graf, 0, 201);

    clock_t stop = clock();
    double duration = (double)(stop - start) / CLOCKS_PER_SEC;
    
    cout << "Time spent on execution: " << duration << " seconds" << endl;

    return 0;

}