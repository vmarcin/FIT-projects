#include <mpi.h>
#include <math.h>
#include <bits/stdc++.h> 

#define MEASURE_TIME    0

using namespace std; 

/**
 * @brief   Process 0 reads altitudes from a string given as first argument 
 *          of progam and scatters it to other processes. As n%comm_sz need not
 *          be zero, process 0 needs to take care of equal values distributions,
 *          by adding padding into input sequence of altitudes. Finally it 
 *          broadcasts an value of the observer and input_size value.
 * 
 * @param[in] rank
 * @param[out] local_alt    vector of local altitudes for each process 
 * @param[in] comm 
 * @param[in] argv 
 * @param[in] comm_sz       number of processors
 * @param[out] observer     an altitude of the observer
 * @param[out] n            input size
 */
void read_altitudes(int rank, vector<int> *local_alt, MPI_Comm comm, 
                char *argv[], int comm_sz, int *observer, int *n);

/**
 * @brief   Each process computes angles between observer and its altitudes.
 * 
 * @param[in] local_alt     vector of local altitudes for each process 
 * @param[in] observer      an altitude of the observer
 * @param[in] rank 
 * @param[in] n             input size
 * @param[in] comm_sz       number of processors
 * @param[out] local_angles vector of local angles for each process
 */
void compute_angle(vector<int> *local_alt, int observer, int rank,
                int n, int comm_sz, vector<double> *local_angles);

/**
 * @brief   Each process finds its max angle out of its local angles.
 * 
 * @param[out] max_angle    max_angle of each process 
 * @param[in] local_angles  vector of local angles for each process 
 */
void find_local_max(double *max_angle, vector<double> *local_angles);

/**
 * @brief   Up-sweep procedure. 
 * 
 * @param[in, out] value    max angle of each process. It is also an output 
 *                          argument as each process needs to remember its value.
 * @param[in] rank 
 * @param[in] comm_sz       number of processors
 * @param[in] comm 
 */
void up_sweep(double *value, int rank, int comm_sz, MPI_Comm comm);

/**
 * @brief Down-sweep procedure.
 * 
 * @param value             value of the angle of each process after down-sweep phase. 
 *                          It is also an output argument as each process needs to 
 *                          remember its value. 
 * @param rank 
 * @param comm_sz           number of processors
 * @param comm 
 */
void down_sweep(double *value, int rank, int comm_sz, MPI_Comm comm);


/**
 * @brief   As each process must process more than one value at the end of prescan
 *          each of them has only information if its biggest value is bigger than
 *          prev process biggest value. Because of that each of processes needs to
 *          compute visibility for every local altitude.          
 * 
 * @param[in] local_max 
 * @param[out] result 
 * @param[in] local_angles 
 */
void compute_local_visibility(double local_max,  vector<char> *result,
                                vector<double> *local_angles);

/**
 * @brief   Process 0 gathers all values and prints the result out.
 * 
 * @param[in] result    vector of resulting visibilities
 * @param[in] comm_sz   number of processes
 * @param[in] n         input size
 * @param[in] rank 
 * @param[in] comm 
 */
void print_result(vector<char> *result, int comm_sz, int n, int rank, MPI_Comm comm);

int main(int argc, char *argv[]) {
    // MPI initialization
    MPI_Init(&argc, &argv);

    // get number of processes
    int comm_sz = MPI::COMM_WORLD.Get_size();
    // get rank
    int rank = MPI::COMM_WORLD.Get_rank();
    int observer;
    int n;

    vector<int> local_alt;          // local altitudes for each process
    vector<double> local_angles;    // local angles for each process
    vector<char> local_visibility;  // resulting visibilities of input altitudes
    double max_angle;               // local maximum angle of each process

    double local_start, local_end;
    double local_elapsed, elapsed;

    // read input and distribute it to all processes
    read_altitudes(rank, &local_alt, MPI_COMM_WORLD, argv, comm_sz, &observer, &n);

    MPI_Barrier(MPI_COMM_WORLD);
    local_start = MPI_Wtime();

        // each process computes it's own angles
        compute_angle(&local_alt, observer, rank, n, comm_sz, &local_angles);

        // find max angle TODO if i have only one value dont make call
        find_local_max(&max_angle, &local_angles);
        
        // let's do some sweeping ᕦ(ò_óˇ)ᕤ (PRESCAN)
        up_sweep(&max_angle, rank, comm_sz, MPI_COMM_WORLD);
        down_sweep(&max_angle, rank, comm_sz, MPI_COMM_WORLD);
        // phew end of sweeping ( ဖ‿ဖ)人(စ‿စ ) we did it !

        // take the prescan result and determine visibility of local points
        compute_local_visibility(max_angle, &local_visibility, &local_angles);

    local_end = MPI_Wtime();
    local_elapsed = local_end - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

#   if (MEASURE_TIME)
    if ( rank == 0 ) {
        printf("Time = %g us\n", elapsed * 1000000);
    }
#   else
    print_result(&local_visibility, comm_sz, n, rank, MPI_COMM_WORLD);
#   endif

    MPI_Finalize();

    return 0;
}

void read_altitudes(int rank, vector<int> *local_alt, MPI_Comm comm, 
                char *argv[], int comm_sz, int *observer, int *input_size) {
    string input = argv[1];
    vector<int> altitudes;
    int local_data_size;

    if (rank == 0) {
        // WARNING ಠ_ಠ : only valid inputs are assumed (Natural Numbers).  
        // e.g. in case of the input in the form "a, 1, 2", the program will be aborted,
        // Also the input must be at least 2 values long 
        // (must conatain at least observer and one more point)
        stringstream ss(input);
        string num;
        while (ss.good()) {
            getline(ss, num, ',');
            altitudes.push_back(stol(num, nullptr, 0));
        }
        *observer = altitudes.at(0);
        altitudes.erase(altitudes.begin());

        unsigned n = altitudes.size();
        unsigned mod = n % comm_sz;
        *input_size = n;
        
        local_data_size = n / comm_sz;
        if ( mod ) {
            local_data_size += 1;
            // add padding into altitudes vector
            // padding is in form of -1s as -1 should never appears in the input (cause input \in N)
            for (int i = mod * local_data_size; i < local_data_size * comm_sz; i+= local_data_size) {
                altitudes.insert(altitudes.begin() + i, -1);
            }
        }
    }
    MPI_Bcast(&local_data_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(observer, 1, MPI_INT, 0, comm);
    MPI_Bcast(input_size, 1, MPI_INT, 0, comm);

    // only observer given
    if(*input_size == 0) {
        if(rank == 0)
            cout << "_\n";
        MPI_Finalize();
        exit(0);
    }
    
    // memory allocation for local values
    local_alt->resize(local_data_size);
    // scatter altitudes list to local vectors
    MPI_Scatter(altitudes.data(), local_data_size, MPI_INT, local_alt->data(), local_data_size, 
                MPI_INT, 0, comm);
}

void compute_angle(vector<int> *local_data, int observer, int rank, int n, 
                int comm_sz, vector<double> *local_angles) {
    
    double angle;
    int index = local_data->size() * rank;
    // ass there is a padding in the input sequence, index of local start needs to be adjusted
    if (rank >= n%comm_sz && n%comm_sz) { index -= (rank - n%comm_sz); }
    
    for (int i = 0,j = index; i < local_data->size(); i++, j++) {
        if (local_data->at(i) == -1) {
            j -= 1;
            continue;
        }
        angle = atan((double)(local_data->at(i) - observer)/(j+1));
        local_angles->push_back(angle);
    }
}

void find_local_max(double *max_angle, vector<double> *local_angles) {
    double max = local_angles->at(0);
    double tmp;

    // simple find max cycle
    for (int i = 0; i < local_angles->size(); i++) {
        tmp = local_angles->at(i);
        if ( tmp > max )
            max = tmp;
    }
    // return the result value
    *max_angle = max;
}

void up_sweep(double *value, int rank, int comm_sz, MPI_Comm comm) {
    int source, dest;       // source and  destination of communication in particular step
    double tmp;             // space for the received value
    int neighbour_offset;   // |[] 008 []| the house number of the adjacent process
    bool sender = false;    // sender flag tells the process if it's the sender in the current step
    MPI_Status status;

    // as number of processes is power of 2, the number of step will 
    // be always log2(number of processes)
    int numsteps = log2(comm_sz);
    
    // yaaaay (^‿^) we will be senders in first step 
    if((rank + 1) % 2) sender = true;

    for (int step = 1; step <= numsteps; step++) {
        neighbour_offset = 1 << (step - 1);

        // senders (°‿°) => |\/|
        if ( sender ) {
            dest = rank + neighbour_offset;
            MPI_Send(value, 1, MPI_DOUBLE, dest, 0, comm);

            // i'will not be the sender in the next step (ႎ_ႎ)
            sender = false;
        }
        // receivers (°‿°) <= |\/|  
        if ( ! ((rank + 1) % (1 << step)) ) {
            source = rank - neighbour_offset;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, source, 0, comm, &status);
            // max operation
            *value =  (tmp > * value) ? tmp : *value;

            // if true i'will be one of senders in the next step (^‿^)
            if ( ((rank + 1) % (1 << (step + 1))) ) sender = true;
        }
    }
}

void down_sweep(double *value, int rank, int comm_sz, MPI_Comm comm) {
    double tmp;
    int neighbour;
    int numsteps = log2(comm_sz);
    bool root_proc = false;
    MPI_Status status;

    // clear phase of down-sweep
    if ( (rank + 1) == comm_sz) { 
        *value = -DBL_MAX; 
        root_proc = true;
    }

    for (int step = (numsteps-1); step >= 0; step--) {
        neighbour = (root_proc) ? rank - (1 << step) : rank + (1 << step);

        // yaaaay (^‿^)  we are communicating in this step 
        if ( ! ((rank + 1) % (1 << step)) ) {
            MPI_Sendrecv(value, 1, MPI_DOUBLE, neighbour, 0, 
                        &tmp, 1, MPI_DOUBLE, neighbour, 0, comm, &status);
            // (⌐[]-[]) root
            // if ( !( (rank + 1) % (1 << (step + 1)) ) ) {
            if ( root_proc ) { 
                *value = (tmp > *value) ? tmp : *value; 
            } else { 
                *value = tmp; 
            }
            
            // yeah in the next step i will be also root (⌐[]-[])
            root_proc = true;
        }
    }
}

void compute_local_visibility(double local_max,  vector<char> *result,
                                vector<double> *local_angles) {
    double max = local_max;
    double tmp;
    for (int i = 0; i < local_angles->size(); i++) {
        tmp = local_angles->at(i);
        if (tmp > max) {
            result->push_back('v');
            max = tmp;
        } else result->push_back('u');
    }
}

void print_result(vector<char> *result, int comm_sz, int n, int rank, MPI_Comm comm) {
    char *final = NULL;
    int chunk_size;

    // all processes need to have the same size of data chunk (Gather's requirement)
    // so in case that some of them has less values, the padding is added
    if (rank >= (n % comm_sz) && n%comm_sz) { result->push_back('*'); }
    chunk_size = result->size();

    if (rank == 0) {
        final = (char *)malloc((chunk_size * comm_sz)*sizeof(char));
        MPI_Gather(result->data(), chunk_size, MPI_CHAR, final, 
                    chunk_size, MPI_CHAR, 0, comm);
        // printing the result
        cout << "_";
        for (int i = 0; i < (comm_sz * chunk_size); i++) {
            if (final[i] != '*') {
                cout << ",";
                cout << final[i];
            }
        } cout << "\n";
    } else {
        MPI_Gather(result->data(), chunk_size, MPI_CHAR, final,
                    chunk_size, MPI_CHAR, 0, comm);
    }
}