/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xmarci10
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Time measurement
  struct timeval t1, t2;

  if (argc != 10)
  {
    printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    exit(1);
  }

  // Number of particles
  const int N           = std::stoi(argv[1]);
  // Length of time step
  const float dt        = std::stof(argv[2]);
  // Number of steps
  const int steps       = std::stoi(argv[3]);
  // Number of thread blocks
  const int thr_blc     = std::stoi(argv[4]);
  // Write frequency
  int writeFreq         = std::stoi(argv[5]);
  // number of reduction threads
  const int red_thr     = std::stoi(argv[6]);
  // Number of reduction threads/blocks
  const int red_thr_blc = std::stoi(argv[7]);

  // Size of the simulation CUDA gird - number of blocks
  const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
  // Size of the reduction CUDA grid - number of blocks
  const size_t reductionGrid  = (red_thr + red_thr_blc - 1) / red_thr_blc;
  // Size of the shared memory used in calculation_velocity kernel  
  const size_t shared_mem_size = thr_blc * 7 * sizeof(float);
  // Size of the shared memory used in centerOfMass kernel  
  const size_t reduction_shared_mem_size = (red_thr_blc/32) * 4 * sizeof(float);

  // Log benchmark setup
  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);
  printf("threads/block: %d\n", thr_blc);
  printf("blocks/grid: %lu\n", simulationGrid);
  printf("reduction threads/block: %d\n", red_thr_blc);
  printf("reduction blocks/grid: %lu\n", reductionGrid);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
  writeFreq = (writeFreq > 0) ?  writeFreq : 0;

  // CPU side memory allocation
  t_particles particles_cpu;
  float4 comOnGPU;

  cudaHostAlloc(&particles_cpu.pos_x, N*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&particles_cpu.pos_y, N*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&particles_cpu.pos_z, N*sizeof(float),cudaHostAllocDefault);
  
  cudaHostAlloc(&particles_cpu.vel_x, N*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&particles_cpu.vel_y, N*sizeof(float),cudaHostAllocDefault);
  cudaHostAlloc(&particles_cpu.vel_z, N*sizeof(float),cudaHostAllocDefault);
  
  cudaHostAlloc(&particles_cpu.weight, N*sizeof(float),cudaHostAllocDefault);

  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                      Stride of two               Offset of the first
   *  Data pointer        consecutive elements        element in floats,
   *                      in floats, not bytes        not bytes
  */
  MemDesc md(
        particles_cpu.pos_x,    1,                          0,              // Postition in X
        particles_cpu.pos_y,    1,                          0,              // Postition in Y
        particles_cpu.pos_z,    1,                          0,              // Postition in Z
        particles_cpu.vel_x,    1,                          0,              // Velocity in X
        particles_cpu.vel_y,    1,                          0,              // Velocity in Y
        particles_cpu.vel_z,    1,                          0,              // Velocity in Z
        particles_cpu.weight,   1,                          0,              // Weight
        N,                                                                  // Number of particles
        recordsNum);                                                        // Number of records in output file

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return -1;
  }

  // GPU side memory allocation
    // Step 3.*
  float4 *centerOfMassGPU;
  int *lock;

  cudaMalloc(&centerOfMassGPU, 4*sizeof(float));
  cudaMalloc(&lock, sizeof(int));
    // Step 0-2
  t_particles particles_gpuIn;
  t_particles particles_gpuOut;
  t_particles particles_tmp;

  cudaMalloc(&particles_gpuIn.pos_x, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.pos_y, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.pos_z, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.vel_x, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.vel_y, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.vel_z, N*sizeof(float));
  cudaMalloc(&particles_gpuIn.weight, N*sizeof(float));

  cudaMalloc(&particles_gpuOut.pos_x, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.pos_y, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.pos_z, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.vel_x, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.vel_y, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.vel_z, N*sizeof(float));
  cudaMalloc(&particles_gpuOut.weight, N*sizeof(float));

  // Transfer data to GPU 
    // Step 3.*
  cudaMemset(centerOfMassGPU, 0.0f, 4*sizeof(float));
  cudaMemset(lock, 0, sizeof(int));
    // Step 0-2
  cudaMemcpy(particles_gpuIn.pos_x, particles_cpu.pos_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.pos_y, particles_cpu.pos_y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.pos_z, particles_cpu.pos_z, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.vel_x, particles_cpu.vel_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.vel_y, particles_cpu.vel_y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.vel_z, particles_cpu.vel_z, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(particles_gpuIn.weight, particles_cpu.weight, N*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(particles_gpuOut.weight, particles_cpu.weight, N*sizeof(float), cudaMemcpyHostToDevice);

  // Streams and events allocation Step 4
  cudaStream_t stream_integrate, stream_com, stream_memcpy;
  cudaStreamCreate(&stream_integrate);
  cudaStreamCreate(&stream_com);
  cudaStreamCreate(&stream_memcpy);

  cudaEvent_t particles_finished, com_finished;
  cudaEventCreate(&particles_finished);
  cudaEventCreate(&com_finished);

  gettimeofday(&t1, 0);

  for(struct {int s = 0; int record_num = 0;} loop; loop.s < steps; loop.s++)
  {
    // Kernel invocation in stream_integrate
    calculate_velocity<<<simulationGrid, thr_blc, shared_mem_size, stream_integrate>>>(particles_gpuIn, particles_gpuOut, N, dt);
    // Inserting an event indicating the completion of the particle position calculation
    cudaEventRecord(particles_finished, stream_integrate);

    if (writeFreq > 0 && (loop.s % writeFreq == 0))
    {
      // Kernel invocation in stream_com
      cudaMemsetAsync(centerOfMassGPU, 0.0f, 4*sizeof(float), stream_com);
      centerOfMass<<<reductionGrid, red_thr_blc, reduction_shared_mem_size, stream_com>>>(particles_gpuIn, 
        &centerOfMassGPU->x, &centerOfMassGPU->y, &centerOfMassGPU->z, &centerOfMassGPU->w, lock, N);
      // Inserting an event indicating the completion of the center of mass calculation
      cudaEventRecord(com_finished, stream_com);

      // Transfer practicles to CPU in stream_memcpy 
      cudaMemcpyAsync(particles_cpu.pos_x, particles_gpuIn.pos_x, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      cudaMemcpyAsync(particles_cpu.pos_y, particles_gpuIn.pos_y, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      cudaMemcpyAsync(particles_cpu.pos_z, particles_gpuIn.pos_z, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      cudaMemcpyAsync(particles_cpu.vel_x, particles_gpuIn.vel_x, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      cudaMemcpyAsync(particles_cpu.vel_y, particles_gpuIn.vel_y, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      cudaMemcpyAsync(particles_cpu.vel_z, particles_gpuIn.vel_z, N*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);
      
      // CPU waits until particles data will be available
      cudaStreamSynchronize(stream_memcpy);

      // Putting a wait for the com_finished event in the stream_memcpy
      cudaStreamWaitEvent(stream_memcpy, com_finished, 0);
      // Transfer com to CPU in stream_memcpy
      cudaMemcpyAsync(&comOnGPU.x, centerOfMassGPU, 4*sizeof(float), cudaMemcpyDeviceToHost, stream_memcpy);

      // While com is copied from D2H, CPU is writing particles into output file
      h5Helper.writeParticleData(loop.record_num);

      // CPU waits until com data will be available
      // It also ensures that com ends before a new integrate step begins
      cudaStreamSynchronize(stream_memcpy);

      // CPU writes the com data into output file
      comOnGPU.x = comOnGPU.x / comOnGPU.w;
      comOnGPU.y = comOnGPU.y / comOnGPU.w;
      comOnGPU.z = comOnGPU.z / comOnGPU.w;
      h5Helper.writeCom(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w, loop.record_num++);
      
    }
    // stream_com needs to wait for an input data
    cudaStreamWaitEvent(stream_com, particles_finished, 0);
    // stream_memcpy also needs to wait until the particles computing is finished and after
    // that the data can be copied from D2H
    cudaStreamWaitEvent(stream_memcpy, particles_finished, 0);

    // swap pointers
    particles_tmp = particles_gpuOut;
    particles_gpuOut = particles_gpuIn;
    particles_gpuIn = particles_tmp;
  }

  cudaDeviceSynchronize();

  cudaMemset(centerOfMassGPU, 0.0f, 4*sizeof(float));
  // Kernel invoaction
  centerOfMass<<<reductionGrid, red_thr_blc, reduction_shared_mem_size>>>(particles_gpuIn, 
    &centerOfMassGPU->x, &centerOfMassGPU->y, &centerOfMassGPU->z, &centerOfMassGPU->w, lock, N);

  gettimeofday(&t2, 0);

  // Approximate simulation wall time
  double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
  printf("Time: %f s\n", t);

  // Transfer results back to the CPU 
    // Step 3.*
  cudaMemcpy(&comOnGPU.x, centerOfMassGPU, 4*sizeof(float), cudaMemcpyDeviceToHost);
    // Step 0-2
  cudaMemcpy(particles_cpu.pos_x, particles_gpuIn.pos_x, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_y, particles_gpuIn.pos_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_z, particles_gpuIn.pos_z, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_x, particles_gpuIn.vel_x, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_y, particles_gpuIn.vel_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_z, particles_gpuIn.vel_z, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.weight, particles_gpuIn.weight, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  // CPU completes the calculation of CenterOfMass
  comOnGPU.x = comOnGPU.x / comOnGPU.w;
  comOnGPU.y = comOnGPU.y / comOnGPU.w;
  comOnGPU.z = comOnGPU.z / comOnGPU.w;  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  float4 comOnCPU = centerOfMassCPU(md);

  std::cout << "Center of mass on CPU:" << std::endl
            << comOnCPU.x <<", "
            << comOnCPU.y <<", "
            << comOnCPU.z <<", "
            << comOnCPU.w
            << std::endl;

  std::cout << "Center of mass on GPU:" << std::endl
            << comOnGPU.x<<", "
            << comOnGPU.y<<", "
            << comOnGPU.z<<", "
            << comOnGPU.w
            << std::endl;

  // Writing final values to the file
  h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
  h5Helper.writeParticleDataFinal();

  // Free CPU memory
  cudaFreeHost(particles_cpu.pos_x );
  cudaFreeHost(particles_cpu.pos_y );
  cudaFreeHost(particles_cpu.pos_z );
  cudaFreeHost(particles_cpu.vel_x );
  cudaFreeHost(particles_cpu.vel_y );
  cudaFreeHost(particles_cpu.vel_z );
  cudaFreeHost(particles_cpu.weight);
  // Free GPU memory
  cudaFree(particles_gpuIn.pos_x);
  cudaFree(particles_gpuIn.pos_y);
  cudaFree(particles_gpuIn.pos_z);
  cudaFree(particles_gpuIn.vel_x);
  cudaFree(particles_gpuIn.vel_y);
  cudaFree(particles_gpuIn.vel_z);
  cudaFree(particles_gpuIn.weight);
  cudaFree(particles_gpuOut.pos_x);
  cudaFree(particles_gpuOut.pos_y);
  cudaFree(particles_gpuOut.pos_z);
  cudaFree(particles_gpuOut.vel_x);
  cudaFree(particles_gpuOut.vel_y);
  cudaFree(particles_gpuOut.vel_z);
  cudaFree(particles_gpuOut.weight);
  cudaFree(centerOfMassGPU);
  cudaFree(lock);

  cudaStreamDestroy(stream_integrate);
  cudaStreamDestroy(stream_memcpy);
  cudaStreamDestroy(stream_com);

  cudaEventDestroy(com_finished);
  cudaEventDestroy(particles_finished);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
