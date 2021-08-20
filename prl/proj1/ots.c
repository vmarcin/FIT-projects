/*
 * Paralelní a distribuované algoritmy (PRL 2020)
 * Projekt c. 2 (Odd-even transposition sort)
 * Login: xmarci10
 */

#include <stdio.h>
#include <stdlib.h>     
#include <mpi.h>    

#define DEBUG 0

/*function declarations*/
void read_numbers(int my_rank, unsigned char *local_value, MPI_Comm comm);
void sort(int my_rank, unsigned char *local_value, int proc_num, MPI_Comm comm);
void print_numbers(int my_rank, unsigned char *local_value, int proc_num, MPI_Comm comm);

int main(int argc, char *argv[]) {
    int comm_sz;                        // number of processes
    int my_rank;                        // rank
    unsigned char local_value;          // local number owned by one processor
    double local_start, local_finish;   // time
    double local_elapsed, elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    read_numbers(my_rank, &local_value, MPI_COMM_WORLD);

#   if (DEBUG > 1)
    printf("process %d has value: %u (%x)\n", my_rank, local_value, local_value);
#   endif

    MPI_Barrier(MPI_COMM_WORLD);
    local_start = MPI_Wtime();
    /* Code to be timed */
    sort(my_rank, &local_value, comm_sz, MPI_COMM_WORLD);

    local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    print_numbers(my_rank, &local_value, comm_sz, MPI_COMM_WORLD);

#   if (DEBUG > 0)
    if (my_rank == 0)
        printf("Sorting time = %g us\n", elapsed * 1000000); 
#   endif

    MPI_Finalize();

    return 0;
} /* main */

/*function definitions*/

/**
 * @brief   process 0 reads numbers from file and scatters it 
 *          to the other processes.
 * 
 * @param[in] my_rank 
 * @param[out] local_value 
 * @param[in] comm 
 */
void read_numbers(int my_rank, unsigned char *local_value, MPI_Comm comm) {
    FILE *fileptr;
    unsigned char *numbers;
    long int filelen;

    if (my_rank == 0) {
        // get the count of numbers in the file
        fileptr = fopen("numbers", "rb");
        fseek(fileptr, 0, SEEK_END);
        filelen = ftell(fileptr);
        rewind(fileptr);

        // allocate memory and read the file
        numbers = (unsigned char *)calloc(filelen, sizeof(unsigned char));
        fread(numbers, filelen, 1, fileptr);
        fclose(fileptr); 

        // print numbers in original order
        for (int i=0; i < filelen; i++) {
            printf("%u%c", numbers[i], ((i + 1) != filelen ? ' ' : '\n'));
        } 
    }
    // scatter numbers list containing all ints to local_value variables
    MPI_Scatter(numbers, 1, MPI_BYTE, local_value, 1, MPI_BYTE, 0, comm);

    // deallocate memory
    if (my_rank == 0) { free(numbers); }
} /* read_numbers */

/**
 * @brief sort input numbers by odd-even transposition sort
 * 
 * @param[in] my_rank 
 * @param[in] proc_num 
 * @param[in, out] local_value 
 */
void sort(int my_rank, unsigned char *local_value, int proc_num, MPI_Comm comm) {
    int even_partner;
    int odd_partner;
    int phase;
    MPI_Status status;
    unsigned char tmp;

    // compute partners for odd/even phase for every processor
    if (my_rank % 2) { /* Odd rank */
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if (odd_partner == proc_num) { odd_partner = MPI_PROC_NULL; }
    } else { /* Even rank */
        even_partner = my_rank + 1;
        if (even_partner == proc_num) { even_partner = MPI_PROC_NULL; }
        odd_partner = my_rank - 1;
        if (odd_partner == -1) { odd_partner = MPI_PROC_NULL; }
    }

#   if (DEBUG > 1)
    printf("proc %d: odd_partner= %d even_partner= %d\n", my_rank, odd_partner, even_partner); 
#   endif

    // Odd-even iter
    for (phase = 1; phase <= proc_num; phase++) {
        if (phase % 2) { /* Odd phase (step 1)*/
            MPI_Sendrecv(local_value, 1, MPI_BYTE, odd_partner, 0,
            &tmp, 1, MPI_BYTE, odd_partner, 0, comm, &status);
            //compare and switch
            if (odd_partner >= 0) {    
                // odd processor keeps lower value
                if(my_rank % 2) 
                    *local_value = (*local_value > tmp) ? tmp : *local_value; 
                // even processor keeps higher value
                else 
                    *local_value = (*local_value < tmp) ? tmp : *local_value; 
            }
        } else { /* Even phase (step 2) */        
            MPI_Sendrecv(local_value, 1, MPI_BYTE, even_partner, 0,
            &tmp, 1, MPI_BYTE, even_partner, 0, comm, &status);
            //compare and switch 
            if(even_partner >= 0) {
                // odd processor keeps higher value
                if(my_rank % 2) 
                    *local_value = (*local_value < tmp) ? tmp : *local_value; 
                // even processor keeps lower value
                else  
                    *local_value = (*local_value > tmp) ? tmp : *local_value; 
            }
        }
    }
} /* sort */

/**
 * @brief print all numbers
 * 
 * @param[in] my_rank 
 * @param[in] local_value 
 * @param[in] proc_num  number of sorted keys 
 *                      (number of values == number of processes; by assignment) 
 * @param[in] comm 
 */
void print_numbers(int my_rank, unsigned char *local_value, int proc_num, MPI_Comm comm) {
    unsigned char *sorted = NULL;
    
    if (my_rank == 0) {
        sorted = (unsigned char *)malloc(proc_num*sizeof(unsigned char));
        MPI_Gather(local_value, 1, MPI_BYTE, sorted, 1, MPI_BYTE, 0, comm);
        for (int i = 0; i < proc_num; i++) {
            printf("%u\n", sorted[i]);
        }
        free(sorted);
    } else {
        MPI_Gather(local_value, 1, MPI_BYTE, sorted, 1, MPI_BYTE, 0, comm);
    }
} /* print_sorted_numbers */
