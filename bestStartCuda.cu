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
#define TRY 1
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
void sortPart(int *array,int *array2,int start,int n)
{
    for(int i=0;i<n;i++)
    {
        int t=array[start+i];
        int t2=array2[start+i];

        for(int j=i;j<n;j++)
        {
           if(t>array[start+j]) 
           {
            array[start+i]=array[start+j];
            array2[start+i]=array2[start+j];
            array[start+j]=t;
            array2[start+j]=t2;
            t=array[start+i];
            t2=array2[start+i];
           }
        }
    }
}
int getLenFromFile()
{
    int numStations=VERTICES;
    int i=0;
     std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections.is_open()) {
        std::cerr << "Error opening file: " << "Data/connections.txt" << std::endl;
    }

    std::string t1;
    getline(fileConnections,t1);
    int k=0;
    int t=0;
    for( i=0; true; i++){

        int source, target,c2;
        char c;
        float a,b,weight;

        fileConnections>>a>>c>>b>>c>>source>>c>>target>>c>>c2>>c>>weight;
        getline(fileConnections,t1);

        if(source<numStations&&target<numStations)
        {  
            
        }
        else break;
        // std::cout<<source<<" "<<target<<" "<<weight<<std::endl;
    }
    fileConnections.close();
    return i;
}
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

void minItera(int *M, int*N1,int *MIN,int *W,int *E,int *NW)
{
    int N=VERTICES;
    int t=origin;
    cout<<" t="<<t<<" M="<<M[t]<<" n1="<<N1[t]<<endl;
    if (t < N -10&& M[t] == 1&&N1[t]!=-1) {
        cout<<" usao"<<endl;
        int tempT=t;
        for(int kh=t+1;kh<N-10;kh++)
        {
            if(N1[kh]!=-1)
            {
                tempT=kh;
                break;
            }
        }
        cout<<" n1="<<N1[t]<<" n12="<<N1[tempT]<<endl;
        for (int z = N1[t]; z < N1[tempT]; z++) { 
                int oldNW = NW[E[z]] ;
                cout<<" z="<<z<<" ez="<<E[z]<<" nwez="<<NW[E[z]]<<" w="<<W[z]<<" t"<<t<<" newt="<<NW[t]<<endl;
               if (oldNW > NW[t] + W[ z]) {
                    NW[E[z]]= NW[t] + W[ z];
                }

                if (*MIN> NW[E[z]]) {
                    *MIN=NW[E[z]];
                }
                cout<<" z="<<z<<endl;
                // N1[t] = z;
                break;
            
        }
    }
}

__global__ void minimumKernel(int *M, int *N1, int *NW, int *E, int *W, int *MIN, int N) {
    //NW node distance
    //M visited node
    //W graph
    //n1 is a copy of n
    //n array that has index from which index in E does start the edges
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    if (t < N -1&& M[t] == 1&&N1[t]!=-1) {
        int tempT=t;
        for(int kh=t+1;kh<N-1;kh++)
        {
            if(N1[kh]!=-1)
            {
                tempT=kh;
                break;
            }
        }
        for (int z = N1[t]; z < N1[tempT]; z++) { 
            if(M[E[z]]!=1)
            {
                int tempnw=NW[t]+W[z];
                if(NW[t]==INT_MAX)
                    tempnw=INT_MAX;
                int oldNW = atomicMin(&NW[E[z]], tempnw);
                if (oldNW > tempnw) {
                    atomicExch(&NW[E[z]],tempnw);
                }

                int minVal = atomicMin(MIN, NW[E[z]]);
                if (minVal > NW[E[z]]) {
                    atomicExch(MIN, NW[E[z]]);
                }

                N1[t] = z;
                break;
            }
        }
    }
}

/*
Algorithm 9: SET_FLAG (M, NW, MIN)
BEGIN
t=getThreadID
if(M[t]!=1 &&NW[t]==MIN) then
M[t]=1
if end
END
*/
__global__ void setFlag(int *m,int *nw,int *min)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if(t<VERTICES)
    {
         if(m[t]!=1&&nw[t]==*min)
            m[t]=1;
        __syncthreads();
        if(t==0)
        *min=INT_MAX;
    }
   
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

int* createConnectionMatrix3(int *e,int *w,int *n1, int*n) {

    int numStations = VERTICES;
    int len=getLenFromFile();
    for(int i=0;i<numStations;i++)
    {
        n[i]=-1;
        n1[i]=-1;
    }
    cout<<" len="<<len<<endl;
    
    std::ifstream fileConnections("NewYork/NewYork_Edgelist.csv");
    if (!fileConnections.is_open()) {
        std::cerr << "Error opening file: " << "Data/connections.txt" << std::endl;
    }

    std::string t1;
    getline(fileConnections,t1);
    int t=0,old;
    for(int i=0; true; i++){

        int source, target,c2;
        char c;
        float a,b,weight;

        fileConnections>>a>>c>>b>>c>>source>>c>>target>>c>>c2>>c>>weight;
        getline(fileConnections,t1);
        if(i==0)
        {
            old=source;
            t=0;
        }

        if(source<numStations&&i<VERTICES)
        {  
            if(target>numStations-1)
            {
                i--;
                continue;
            }
            if(old!=source)
            {
                sortPart(w,e,t,i-t);
                n[old]=t;
                n1[old]=t;
                t=i;
                old=source;
                //14 14 926 77 637 926 77 135 234 135 234 344 802 344 802 1252 61 61 77 564 proslo 
            }
            e[i]=target;
            w[i]=weight;
        }
        else{
            n[old]=t;
            n1[old]=t;
            t=i;
            n[old+1]=t;
            n1[old+1]=t;
            cout<<" kraje var"<<old<<" t="<<t<<" tart="<<target<<" sou="<<source<<endl;
            break;
        } 
        // std::cout<<source<<" "<<target<<" "<<weight<<std::endl;
    }
    fileConnections.close();
    return  n; 
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
    int* node_dist  = (int*)malloc(VERTICES*sizeof(int));        
    cout<<" alocatae1 11 ";          //distances from source indexed by node ID
    int* parent_node   = (int*)malloc(VERTICES*sizeof(int));                       //number of edges per node indexed by node ID
    int* visited_node= (int*)malloc(VERTICES*sizeof(int));                       //number of edges per node indexed by node ID
    int* visited_node2= (int*)malloc(VERTICES*sizeof(int));                       //number of edges per node indexed by node ID
    int* visited_node1= (int*)malloc(VERTICES*sizeof(int));                      //pseudo-bool if node has been visited indexed by node ID
    // int *pn_matrix      = (int*)malloc((CPU_IMP+GPU_IMP)*int_array);    //matrix of parent_node arrays (one per each implementation)
    // int* dist_matrix = (int*)malloc((CPU_IMP + GPU_IMP)*data_array);

    int * nn1=(int*)malloc(sizeof(int)*VERTICES);
     createConnectionMatrix3(visited_node,node_dist,parent_node,nn1);
    printf("Variables created, allocated\n");

    //CUDA mallocs
    int* e;
    int* w;
    int* n;
    int* n1;
    int* minOut;
    int *reduction,*reduction1;
    cudaMalloc((void**)&e, graph_size);
    cudaMalloc((void**)&w, data_array);
    cudaMalloc((void**)&n, int_array);
    cudaMalloc((void**)&n1, int_array);
    cudaMalloc((void**)&reduction1, blockNum(VERTICES,TH_PER_BLOCK)*sizeof(int));
    cudaMalloc((void**)&reduction, blockNum(VERTICES,TH_PER_BLOCK)*sizeof(int));
    cudaMalloc((void**)&minOut, int_array);
    int block=1;
    // for(int i=0;i<20;i++)
    // {
    //     cout<<nn1[i]<<" ";
    // }
    // cout<<endl<<";;;;;;;;;;;;;;;;;;;;;"<<endl;
    // for(int i=0;i<20;i++)
    // {
    //     cout<<node_dist[i]<<" ";
    // }
    int tempBlock=VERTICES-TH_PER_BLOCK;
    while(tempBlock>0)
    {
        block++;
        tempBlock-=TH_PER_BLOCK;
    }
    int* closest_vertex = (int*)malloc(sizeof(int)*block);
    int* gpu_closest_vertex,*gpu_m,*gpu_nw;
    closest_vertex[0] = INT_MAX;
    float totalTime=0.0;
    cout<<"proslo "<<endl;
    //    node_dist[origin]=0;
         cout<<" dga---"<<endl;
        cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int)*block));
        cudaMalloc((void**)&gpu_m, int_array);
        cudaMalloc((void**)&gpu_nw, int_array);
       CUDA_SAFE_CALL(cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice));
        (cudaMemcpy(e, visited_node, int_array, cudaMemcpyHostToDevice));
        setIntArrayValue(visited_node1, VERTICES, 0); 
        visited_node1[origin]=1;
        cout<<" dos =="<<endl;
        (cudaMemcpy(gpu_m, visited_node1, int_array, cudaMemcpyHostToDevice));
        cudaMemcpy(w, node_dist, data_array, cudaMemcpyHostToDevice);
        cudaMemcpy(n,nn1, int_array, cudaMemcpyHostToDevice);
        cudaMemcpy(n1, parent_node, int_array, cudaMemcpyHostToDevice);
        setDataArrayValue(visited_node2,VERTICES,INT_MAX);
        visited_node2[origin]=0;
        cudaMemcpy(gpu_nw,visited_node2, int_array, cudaMemcpyHostToDevice);
        // cout<<"koperia"<<endl;
        // for(int j=0;j<VERTICES;j++)
        // {
        //     if(j==103)
        //     cout<<"("<<setw(3)<<j<<","<<setw(3)<<parent_node[j]<<") ";
        //     else
        //     cout<<nn1[j]<<" ";
        // }
        // minItera(visited_node1,nn1,closest_vertex,node_dist,visited_node,visited_node2);
        cout<<" moe in="<<*closest_vertex<<endl;
        dim3 gridMin(1, 1, 1);
        dim3 blockMin(1, 1, 1);
        dim3 gridRelax(VERTICES / THREADS_BLOCK, 1, 1);
        dim3 blockRelax(THREADS_BLOCK, 1, 1);   
    for(int k=0;k<TRY;k++)
    {        
        printf("Krece exec\n");
        cudaEventRecord(exec_start);
        for (int i = 0; true; i++)
        {
            // cout<<" end"<<endl<<endl;
            //__global__ void minimumKernel(int *M, int *N1, int *NW, int *E, int *W, int *MIN, int N) {
                // (cudaMemcpy(closest_vertex, gpu_closest_vertex, sizeof(int), cudaMemcpyDeviceToHost));
                // cout<<" closest node="<<*closest_vertex<<endl;
                 (cudaMemcpy(visited_node, gpu_nw, sizeof(int)*VERTICES, cudaMemcpyDeviceToHost));
            //  if(i==2)
            //  for(int u=0;u<VERTICES;u++){
            //     if(visited_node[u]!=2147483647)
            //     // cout<<"x ";
            //     // else
            //     cout<<visited_node[u]<<" ";
            //  }
            minimumKernel<<<block,TH_PER_BLOCK>>>(gpu_m,n1,gpu_nw,e,w,gpu_closest_vertex,VERTICES);
             (cudaMemcpy(closest_vertex, gpu_closest_vertex, sizeof(int), cudaMemcpyDeviceToHost));
                // cout<<" closest node1="<<*closest_vertex<<endl;
                if(*closest_vertex==INT_MAX)
                break;
            setFlag<<<block,TH_PER_BLOCK>>>(gpu_m,gpu_nw,gpu_closest_vertex);
            
            
        }
        cudaEventRecord(exec_stop);
        cudaEventSynchronize(exec_stop);
        cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop);        //elapsed execution time
        printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);
        totalTime+=elapsed_exec;
    }
    printf("\n\nCUDA Avg Time (ms): %7.9f\n", totalTime/TRY);
    (cudaMemcpy(node_dist, gpu_nw, data_array, cudaMemcpyDeviceToHost));
    (cudaMemcpy(parent_node, n, int_array, cudaMemcpyDeviceToHost));
    (cudaMemcpy(visited_node, n1, int_array, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < VERTICES; i++) {  
    //     // cout<<node_dist[i]<<" ";              //record resulting parent array and node distance
    //     pn_matrix[VERTICES + i] = parent_node[i];
    //     dist_matrix[VERTICES + i] = node_dist[i];
    // }
    // printPath(parent_node,TARGET);
    printf("\n distance: %d %d",node_dist[TARGET],TARGET);//794
       //free memory
    (cudaFree(e));
    (cudaFree(w));
    (cudaFree(n));
    (cudaFree(n1));
    (cudaFree(gpu_closest_vertex));
    (cudaFree(reduction1));
    (cudaFree(reduction));
    // free(graf);
    free(closest_vertex);
    free(node_dist);
    free(parent_node);
    free(visited_node);
    // free(pn_matrix);
    // free(dist_matrix);
    return 0;
}
