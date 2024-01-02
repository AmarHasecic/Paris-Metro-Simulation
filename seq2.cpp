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

#define VERTICES 60000
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

int** createConnectionMatrix2() {
    int numStations = VERTICES;
    std::cout<<numStations<<std::endl;
    int** adjacencyMatrix; 
    adjacencyMatrix=(int**)malloc(sizeof(int*)*VERTICES);
    for(int i=0;i<VERTICES;i++)
    {
        adjacencyMatrix[i]=(int*)malloc(sizeof(int)*VERTICES);
    }
    std::cout<<"dsfs";
    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections.is_open()) {
        std::cerr << "Error opening file: " << "Data/NewYork_Edgelist.csv" << std::endl;
    }

    std::string t1;
    getline(fileConnections,t1);
    for(int i=0; VERTICES; i++){
        int source, target,c2;
        char c;
        float a,b,weight;
        fileConnections>>a>>c>>b>>c>>source>>c>>target>>c>>c2>>c>>weight;
        getline(fileConnections,t1);
        if(source<numStations&&target<numStations)
        adjacencyMatrix[source][target] = weight;
        else break;
        // std::cout<<source<<" "<<target<<" "<<weight<<std::endl;
    }
    fileConnections.close();
    std::cout<<"kraj unose";
    return adjacencyMatrix;
}
typedef int** Graph;
using namespace std;


int minDistance(int* dist, bool* sptSet) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < VERTICES; v++) {
        if (!sptSet[v] && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}


void printPath(int* parent, int dest) {
    if (parent[dest] == -1) {
        cout << dest;
        return;
    }
    printPath(parent, parent[dest]);
    cout << " -> " << dest;
}


void dijkstra(int** graph, int src, int dest) {
    int V =VERTICES;
    int* dist=(int*)malloc(sizeof(int)*VERTICES); 
    int* parent=(int*)malloc(sizeof(int)*VERTICES); 
    bool* sptSet=(bool*)malloc(sizeof(bool)*VERTICES); 
    for(int i=0;i<VERTICES;i++)
    {
        dist[i]=INT_MAX;
        sptSet[i]=false;
        parent[i]=-1; 
    }

    
    dist[src] = 0;


    for (int count = 0; count < V - 1; count++) {
       
        int u = minDistance(dist, sptSet);

  
        sptSet[u] = true;

        
        for (int v = 0; v < V; v++) {
        
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                parent[v] = u;
            }
        }
    }

   
    // cout << "Shortest path from Station 1 " << src << " to Staiton 2 " << dest << ": ";
    // printPath(parent, dest);
    // cout << " (Total distance: " << dist[dest] << ")" << endl;
}





int main(){

    // Graph graf = createConnectionMatrix();
    Graph graf = createConnectionMatrix2();

    int trys=1;
    clock_t start = clock();
    for(int i=0;i<trys;i++)
    dijkstra(graf, 0, 201);

    clock_t stop = clock();
    double duration = (double)(stop - start) / (CLOCKS_PER_SEC*trys);
    
    cout << "Time spent on execution: " << duration << " seconds" << endl;

    return 0;

}
