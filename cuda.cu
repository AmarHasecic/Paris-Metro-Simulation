using namespace std;
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
#include <iomanip>
// 301 191 161  92
#define blockSize 256
#define TRY 10
#define TARGET 161
#define origin 103
#define TH_PER_BLOCK 1000
#define VERTICES 50000//2000  
/*

CUDA Avg Time (ms): 3883.098632813
103 -> 137 -> 165 -> 191 -> 161
 distance: 794
*/         //number of vertices
#define DENSITY 16              //minimum number of edges per vertex. DO NOT SET TO >= VERTICES
#define MAX_WEIGHT 1000000      //max edge length + 1
#define INF_DIST 1000000000     //"infinity" initial value of each node
#define CPU_IMP 1               //number of Dijkstra implementations (non-GPU)
#define GPU_IMP 1               //number of Dijkstra implementations (GPU)
#define THREADS 2               //number of OMP threads
#define RAND_SEED 1234          //random seed
#define THREADS_BLOCK 512
using namespace std;
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void printPath(const int * parent, int dest) {
    if (parent[dest] == -1) {
        cout << dest;
        return;
    }
    printPath(parent, parent[dest]);
    cout << " -> " << dest;
}
void setIntArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Initialize elements of a 1D data_t array with an initial value   */
void setDataArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, ": %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__ void reduce6NoVisitedV2(int *g_idata,int *distance,int *g_out,  const int n) {
    extern __shared__ int sdata;
    sdata=1;
    extern __shared__ int indexes1[VERTICES/TH_PER_BLOCK+2];
    int tip=blockIdx.x * blockDim.x + threadIdx.x;
    if(tip*sdata+sdata<n&&tip<n)
    {
        indexes1[threadIdx.x]= tip;
        //  __syncthreads();
           while (sdata<n) { 
                int i= 2*sdata*(threadIdx.x);
                if(i+sdata<n&&
                  distance[g_idata[indexes1[i]]]>distance[g_idata[indexes1[i+sdata]]])
                     indexes1[i]=indexes1[i+sdata];
                if(threadIdx.x==0)
                 sdata*=2;
                __syncthreads();
            }
    }
    if(threadIdx.x==0)
    {
        g_out[blockIdx.x]=g_idata[indexes1[0]];
    }
}
__global__ void closestNodeCUDA(int* node_dist, int* visited_node, int* global_closest, int num_vertices) {
    int dist = INF_DIST + 1;
    int node = -1;
    int i;
    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] != 1)) {
            dist = node_dist[i];
            node = i;
        }
    }
    global_closest[0] = node;
    visited_node[node] = 1;
}


__global__ void cudaRelax(int* graph, int* node_dist, int* parent_node, int* visited_node, int* global_closest) {
     int next = blockIdx.x*blockDim.x + threadIdx.x;    //global ID
     if(true||next<VERTICES)
     {
        int source = global_closest[0];

        int edge = graph[source*VERTICES + next];
        int new_dist = node_dist[source] + edge;

        if ((edge != 0) &&
            (visited_node[next] != 1) &&
            (new_dist < node_dist[next])) {
            node_dist[next] = new_dist;
            parent_node[next] = source;
        }
     }
}

__global__ void reduce6(int *g_idata, int *g_odata,int *visired,unsigned int n) {
    extern __shared__ int sdata;
    sdata=1;
    extern __shared__ int indexes[TH_PER_BLOCK];
    int tip=blockIdx.x * blockDim.x + threadIdx.x;
    if(tip*sdata+sdata<VERTICES&&tip<VERTICES)
    {
        //zbog ovog koristimo 2x vise threads, potential memory size improvements
        indexes[threadIdx.x]= tip;
        //  __syncthreads();
           while (sdata<TH_PER_BLOCK) { 
                int i= 2*sdata*(threadIdx.x);
                if(i+sdata<TH_PER_BLOCK&&
                  visired[indexes[i+sdata]]!=1&&
                  g_idata[indexes[i]]>g_idata[indexes[i+sdata]])
                     indexes[i]=indexes[i+sdata];
                else if(i+sdata<TH_PER_BLOCK&&visired[indexes[i+sdata]]!=1&&visired[indexes[i]]==1)
                     indexes[i]=indexes[i+sdata];
                   
                if(threadIdx.x==0)
                 sdata*=2;
                __syncthreads();
            }
    }
    if(threadIdx.x==0)
    {
        g_odata[blockIdx.x ]=indexes[0];
        //ovo ispod treba prebaciti izvan kernela jer min tek dobijem kad prodjem kroz sve blokove
        // visired[indexes[0]]=1;
    }
}
__global__ void reduce6NoVisited(int *g_idata,int *distance,int *visited,unsigned int n) {
    extern __shared__ int sdata;
    sdata=1;
    extern __shared__ int indexes2[TH_PER_BLOCK];
    int tip=blockIdx.x * blockDim.x + threadIdx.x;
    if(tip*sdata+sdata<n&&tip<n)
    {
        indexes2[threadIdx.x]= tip;
        //  __syncthreads();
           while (sdata<n) { 
                int i= 2*sdata*(threadIdx.x);
                if(i+sdata<n&&
                  distance[g_idata[indexes2[i]]]>distance[g_idata[indexes2[i+sdata]]])
                     indexes2[i]=indexes2[i+sdata];
                if(threadIdx.x==0)
                 sdata*=2;
                __syncthreads();
            }
    }
    if(threadIdx.x==0)
    {
        g_idata[0]=g_idata[indexes2[0]];
        visited[g_idata[0]]=1;
    }
}

int * createConnectionMatrix() {

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
    int *t=(int*)calloc(adjacencyMatrix.size()*adjacencyMatrix[0].size(),sizeof( int* ));
    for(int i=0;i<adjacencyMatrix.size();i++)
    {
        for(int j=0;j<adjacencyMatrix[i].size();j++)
        t[i*adjacencyMatrix.size()+j]=adjacencyMatrix[i][j];
    }
    printf("neki=%zd %zd\n",adjacencyMatrix.size(),adjacencyMatrix[0].size());
    return  t;
}

int* createConnectionMatrix2() {

    int numStations = VERTICES;
    int * adjacencyMatrix=(int*)malloc(sizeof(int)*numStations*numStations);
    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections.is_open()) {
        std::cerr << "Error opening file: " << "Data/connections.txt" << std::endl;
    }

    std::string t1;
    getline(fileConnections,t1);
    for(int i=0; true; i++){

        int source, target,c2;
        char c;
        float a,b,weight;

        fileConnections>>a>>c>>b>>c>>source>>c>>target>>c>>c2>>c>>weight;
        getline(fileConnections,t1);

        if(source<numStations&&target<numStations)
     {   adjacencyMatrix[source*numStations+target] = weight;
        adjacencyMatrix[target*numStations+source] = weight;
    }
        else break;
        // std::cout<<source<<" "<<target<<" "<<weight<<std::endl;
    }
    fileConnections.close();

    return  adjacencyMatrix; 
}

using namespace std;
int blockNum(int vertecies,int threads)
{
    int block=1;
    int tempBlock=VERTICES-TH_PER_BLOCK;
    while(tempBlock>0)
    {
        block++;
        tempBlock-=TH_PER_BLOCK;
    }
    return block;
}


int  main(){
    cout<<" dgao";
    int* graf = createConnectionMatrix2();
//     for(int i=0;i<VERTICES;i++)
// {
// cout<<graf[1*VERTICES+i]<<" ";
// }
    //performance measure, time
    float elapsed_exec;  
    cudaEvent_t exec_start, exec_stop; 
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_stop);
 //declare variables and allocate memory
 cout<<"dsfadfa";
 long long int tempSize=VERTICES;
    long long int graph_size = tempSize*tempSize*sizeof(int);             //memory in B required by adjacency matrix representation of graph
     long long int int_array       = VERTICES*sizeof(int);                         //memory in B required by array of vertex IDs. Vertices have int IDs.
     long long int data_array      = VERTICES*sizeof(int);                      //memory in B required by array of vertex distances (depends on type of data used)
    // int* graph       = (int*)malloc(graph_size);                  //graph itself
    int* node_dist   = (int*)malloc(data_array);        
    cout<<" alocatae1 11 ";          //distances from source indexed by node ID
    int* parent_node    = (int*)malloc(int_array);                       //number of edges per node indexed by node ID
    int* visited_node   = (int*)malloc(int_array);                      //pseudo-bool if node has been visited indexed by node ID
    // int *pn_matrix      = (int*)malloc((CPU_IMP+GPU_IMP)*int_array);    //matrix of parent_node arrays (one per each implementation)
    // int* dist_matrix = (int*)malloc((CPU_IMP + GPU_IMP)*data_array);

    printf("Variables created, allocated\n");

    //CUDA mallocs
    int* gpu_graph;
    int* gpu_node_dist;
    int* gpu_parent_node;
    int* gpu_visited_node;
    int* minOut;
    int *reduction,*reduction1;
    cudaMalloc((void**)&gpu_graph, graph_size);
    cudaMalloc((void**)&gpu_node_dist, data_array);
    cudaMalloc((void**)&gpu_parent_node, int_array);
    cudaMalloc((void**)&gpu_visited_node, int_array);
    cudaMalloc((void**)&reduction1, blockNum(VERTICES,TH_PER_BLOCK)*sizeof(int));
    cudaMalloc((void**)&reduction, blockNum(VERTICES,TH_PER_BLOCK)*sizeof(int));
    (cudaMalloc((void**)&minOut, int_array));
    
    int block=1;
    int tempBlock=VERTICES-TH_PER_BLOCK;
    while(tempBlock>0)
    {
        block++;
        tempBlock-=TH_PER_BLOCK;
    }
    int* closest_vertex = (int*)malloc(sizeof(int)*block);
    int* gpu_closest_vertex;
    closest_vertex[0] = origin;
    float totalTime=0.0;

    for(int k=0;k<TRY;k++)
    {
        setDataArrayValue(node_dist, VERTICES, INF_DIST);          //all node distances are infinity    
        node_dist[origin]=0;
        setIntArrayValue(parent_node, VERTICES, -1);            //parent nodes are -1 (no parents yet)
        setIntArrayValue(visited_node, VERTICES, 0); 
        cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int)*block));
        cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice);
        (cudaMemcpy(gpu_graph, graf, graph_size, cudaMemcpyHostToDevice));
        cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_parent_node, parent_node, int_array, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice);
        dim3 gridMin(1, 1, 1);
        dim3 blockMin(1, 1, 1);
        dim3 gridRelax(VERTICES / THREADS_BLOCK, 1, 1);
        dim3 blockRelax(THREADS_BLOCK, 1, 1);   
        
        printf("Krece exec\n");
        cudaEventRecord(exec_start);
        for (int i = 0; i < VERTICES; i++)
        {
            // closestNodeCUDA <<<gridMin, blockMin>>>(gpu_node_dist, gpu_visited_node, gpu_closest_vertex, VERTICES);                 //find min
            reduce6<<<block,TH_PER_BLOCK>>>(gpu_node_dist,reduction,gpu_visited_node,(unsigned int)VERTICES);
            
           
            // reduce6NoVisited<<<1,block>>>(gpu_closest_vertex,gpu_node_dist,gpu_visited_node,(unsigned int)block);
             int iBlock=block;
            int it=0;
             while(iBlock>TH_PER_BLOCK)
             {
                // cout<<"  petljaaaaa";
                if(it%2==0) 
                    reduce6NoVisitedV2<<<1,iBlock>>>(reduction,gpu_node_dist,reduction1,(unsigned int)iBlock);
                else
                    reduce6NoVisitedV2<<<1,iBlock>>>(reduction1,gpu_node_dist,reduction,(unsigned int)iBlock);
                
                it++;
                iBlock/=TH_PER_BLOCK;
             }

            
            //  cout<<" It="<<it<<endl;
            if(it==0)
            {
                 reduce6NoVisited<<<1,TH_PER_BLOCK>>>(reduction,gpu_node_dist,gpu_visited_node,(unsigned int)iBlock);
            
                cudaRelax <<<block, TH_PER_BLOCK>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, reduction); //relax
               
            }
            else if(it%2==0)
            {
                // cout<<"prva";
                reduce6NoVisited<<<1,TH_PER_BLOCK>>>(reduction,gpu_node_dist,gpu_visited_node,(unsigned int)iBlock);
            
                cudaRelax <<<block, TH_PER_BLOCK>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, reduction); //relax
               
               
            // // (cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost));
            // (cudaMemcpy(node_dist, gpu_visited_node, int_array, cudaMemcpyDeviceToHost));
            // // (cudaMemcpy(parent_node, reduction, block, cudaMemcpyDeviceToHost));

            // for(int j =0;j<VERTICES;j++)
            // {
            //     if(node_dist[j]==1000000000)
            //     cout<<setw(4)<<"x";
            //     else
            //     cout<<setw(4)<<node_dist[j];
            //     // cout<< setw(4)<<graf[parent_node[0]*VERTICES+j]<< setw(4)<<visited_node[j]<<" |";
            // }
                
            }else
            {
                reduce6NoVisited<<<1,TH_PER_BLOCK>>>(reduction1,gpu_node_dist,gpu_visited_node,(unsigned int)iBlock);
                cudaRelax <<<block, TH_PER_BLOCK>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, reduction1); //relax
             
            //    if(i==0)
            //     {
            //         cout<<"proslo1="<<it<<" "<<iBlock<<" "<<block<<endl;
            //         (cudaMemcpy(closest_vertex, reduction, sizeof(int)*block, cudaMemcpyDeviceToHost));
            //         for(int j =0;j<block;j++)
            //         {
            //             cout<< setw(4)<<closest_vertex[j]<<" ";
            //         }
            //         cout<<endl<<" kre";
            //         (cudaMemcpy(closest_vertex, reduction1, sizeof(int)*block, cudaMemcpyDeviceToHost));
            //         for(int j =0;j<block;j++)
            //         {
            //             cout<< setw(4)<<closest_vertex[j]<<" ";
            //         }
            //         cout<<endl<<" kre";
            //     }
            // for(int j =0;j<VERTICES;j++)
            // {
            //     if(node_dist[j]==1000000000)
            //     cout<<setw(4)<<"x";
            //     else
            //     cout<<setw(4)<<node_dist[j];
            //     // cout<< setw(4)<<graf[parent_node[0]*VERTICES+j]<< setw(4)<<visited_node[j]<<" |";
            // }
            }
        }
        cudaEventRecord(exec_stop);
        cudaEventSynchronize(exec_stop);
        cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop);        //elapsed execution time
        printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);
        totalTime+=elapsed_exec;
    }
    printf("\n\nCUDA Avg Time (ms): %7.9f\n", totalTime/TRY);
    (cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost));
    (cudaMemcpy(parent_node, gpu_parent_node, int_array, cudaMemcpyDeviceToHost));
    (cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < VERTICES; i++) {  
    //     // cout<<node_dist[i]<<" ";              //record resulting parent array and node distance
    //     pn_matrix[VERTICES + i] = parent_node[i];
    //     dist_matrix[VERTICES + i] = node_dist[i];
    // }
    printPath(parent_node,TARGET);
    printf("\n distance: %d",node_dist[TARGET]);
       //free memory
    (cudaFree(gpu_graph));
    (cudaFree(gpu_node_dist));
    (cudaFree(gpu_parent_node));
    (cudaFree(gpu_visited_node));
    (cudaFree(gpu_closest_vertex));
    (cudaFree(reduction1));
    (cudaFree(reduction));
    free(graf);
    free(closest_vertex);
    free(node_dist);
    free(parent_node);
    free(visited_node);
    // free(pn_matrix);
    // free(dist_matrix);
    return 0;
}
