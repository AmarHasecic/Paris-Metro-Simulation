#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <vector>

#define N 2048
#define SOURCE 1
#define MAXINT 9999999

void dijkstra(int** graph, int source);

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
    int threads;
    std::cout << "Please enter number of threads: ";
    std::cin >> threads;
    omp_set_num_threads(threads);

    double time_start, time_end;
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;

    int** graph = createConnectionMatrix2();
   
    
    dijkstra(graph, SOURCE);
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




void dijkstra(int** graph, int source) {
    int visited[N];
    int md;
    int distance[N]; 
    int mv;
    int my_first; 
    int my_id; 
    int my_last; 
    int my_md; 
    int my_mv; 
    int my_step; 
    int nth;

    for (int i = 0; i < N; i++) {
        visited[i] = 0;
        distance[i] = graph[source][i];
    }
    visited[source] = 1;

    #pragma omp parallel private(my_first, my_id, my_last, my_md, my_mv, my_step) shared(visited, md, distance, mv, nth, graph)
    {
        my_id = omp_get_thread_num();
        nth = omp_get_num_threads();
        my_first = (my_id * N) / nth;
        my_last = ((my_id + 1) * N) / nth - 1;

        for (my_step = 1; my_step < N; my_step++) {
            #pragma omp single
            {
                md = MAXINT;
                mv = -1;
            }

            my_md = MAXINT;
            my_mv = -1;

            for (int k = my_first; k <= my_last; k++) {
                if (!visited[k] && distance[k] < my_md) {
                    my_md = distance[k];
                    my_mv = k;
                }
            }

            #pragma omp critical
            {
                if (my_md < md) {
                    md = my_md;
                    mv = my_mv;
                }
            }

            #pragma omp barrier

            #pragma omp single
            {
                if (mv != -1) {
                    visited[mv] = 1;
                }
            }

            #pragma omp barrier

            if (mv != -1) {
                for (int j = my_first; j <= my_last; j++) {
                    if (!visited[j] && graph[mv][j] < MAXINT && distance[mv] + graph[mv][j] < distance[j]) {
                        distance[j] = distance[mv] + graph[mv][j];
                    }
                }
            }

            #pragma omp barrier
        }
    }

    std::cout<<("\nThe distance vector is\n");
    for (int i = 0; i < N; i++) {
        std::cout<<distance[i]<< "\n";
    }
}
