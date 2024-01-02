/* openMPdijk.c
* The program that does OpenMP parallel Dijkstra’s Algorithm.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#define N 2048
#define SOURCE 1
#define MAXINT 9999999 //bilo 9999999



void dijkstra ( int graph[N][N], int source );
/* This program runs single source Dijkstra's algorithm. Given the distance
 matrix that defines a graph, we seek a minimum distance array between
 source vertex and all other vertices. */
int main(int argc, char **argv) {
    int i, j;
    char fn[255];
    FILE *fp;
    int graph[N][N];
    int threads ;

    printf("Please enter number of threads: ");
    scanf("%d", &threads);
    omp_set_num_threads(threads);

    double time_start, time_end;

    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    strcpy(fn, "input2048.txt");
    fp = fopen(fn, "r");
    if ((fp = fopen(fn, "r")) == NULL) {
    printf("Can't open the input file: %s\n\n", fn);
    exit(1);
    }
    //printf("\nThe adjacency matrix: \n");
    for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
    fscanf(fp, "%d", &graph[i][j]);
    //if (graph[i][j] == 9999999) printf("%4s", "INT");
    //else printf("%4d", graph[i][j]);
    }
    //printf("\n");
    }
    dijkstra(graph, SOURCE);
    gettimeofday(&tv, &tz);

        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
        printf("Nodes: %d\n", N);
        printf("time cost is %1f\n", time_end - time_start);
        return 0;
    }
    void dijkstra(int graph[N][N], int source){
    int visited[N];
    int i;
    int md;
    int distance[N]; /* This array holds the shortest distance from source to other vertices. */
    int mv;
    int my_first; /* The first vertex that stores in one thread locally. */
    int my_id; /* ID for threads */
    int my_last; /* The last vertex that stores in one thread locally. */
    int my_md; /* local minimum distance */
    int my_mv; /* local minimum vertex */
    int my_step; /* local vertex that is at the minimum distance from the source */
    int nth; /* number of threads */
    /* Initialize all vertices' distance and status. */
    for (i = 0; i < N; i++) {
        visited[i] = 0;
        distance[i] = graph[source][i];
    }
    visited[source] = 1;
    /* OpenMP parallelization starts here */
    # pragma omp parallel private ( my_first, my_id, my_last, my_md, my_mv, my_step ) \
    shared ( visited, md, distance, mv, nth, graph )
    {
    my_id = omp_get_thread_num ( );
    nth = omp_get_num_threads ( );
    my_first = (my_id * N ) / nth;
    my_last = ((my_id + 1) * N) / nth - 1;
    //fprintf(stdout, "P%d: First=%d Last=%d\n", my_id, my_first, my_last);
    for (my_step = 1; my_step < N; my_step++) {
    # pragma omp single
    {
        md = MAXINT;
        mv = -1;
    }
    int k;
    my_md = MAXINT;
    my_mv = -1;

    /* Each thread finds the minimum distance unconnected vertex inner of
    the graph */
    for (k = my_first; k <= my_last; k++) {
            if (!visited[k] && distance[k] < my_md) {
                my_md = distance[k];
                my_mv = k;
            }
        }
        /* 'critical' specifies that code is only be executed on one thread at a time,
        * because we need to determine the minimum of all the my_md here. */
        # pragma omp critical
        {
            if (my_md < md) {
                md = my_md;
                mv = my_mv;
            }
        }
        /* 'barrier' identifies a synchronization point at which threads in a parallel
        * region will wait until all other threads in this section reach the same point. So
        * that md and mv have the correct value. */
        # pragma omp barrier
        # pragma omp single
        {
        /* It means we find the vertex and set its status to true. */
            if (mv != - 1){
                visited[mv] = 1;
            }
        }
        # pragma omp barrier
        if ( mv != -1 ){
            int j;
            for (j = my_first; j <= my_last; j++) {
                if (!visited[j] && graph[mv][j] < MAXINT && distance[mv] + graph[mv][j] < distance[j]) {
                    distance[j] = distance[mv] + graph[mv][j];
                }
            }
        }
        #pragma omp barrier
    }
    }
    /*
    printf("\nThe distance vector is\n");
    for (i = 0; i < N; i++) {
    printf("%d ", distance[i]);
    }
    printf("
    \n");
    */
}
