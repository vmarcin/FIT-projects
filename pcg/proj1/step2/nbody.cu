/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2020)
 * Projekt c. 1 (cuda)
 * Login: xmarci10
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate velocity and new position for each particle
 * @param p_in  - input particles
 * @param p_out - output particles
 * @param N     - Number of particles
 * @param dt    - Size of the time step
 */
__global__ void calculate_velocity(const t_particles p_in, t_particles p_out, int N, float dt)
{
  extern __shared__ float shared_particles[];

  float *shared_posx = &shared_particles[0];
  float *shared_posy = &shared_particles[blockDim.x];
  float *shared_posz = &shared_particles[2 * blockDim.x];

  float *shared_velx = &shared_particles[3 * blockDim.x];
  float *shared_vely = &shared_particles[4 * blockDim.x];
  float *shared_velz = &shared_particles[5 * blockDim.x];

  float *shared_weight = &shared_particles[6 * blockDim.x];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  float r, dx, dy, dz;
  float posx, posy, posz;
  float velx, vely, velz;
  float weight;
  float F;

  /**
   * Ressetting the registers for partial results.
   * note:  Using registers reduces the number of accesses to global memory.
   *        Partial results are saved at the end of the calculation.      
   */
  float tmpvelx = 0.0f;
  float tmpvely = 0.0f;
  float tmpvelz = 0.0f;

  /**
   * Loading positions, velocities and weights from the global memory into the registers.
   * note:  Pre-reading data from the global memory, reduces the number of 
   *        memory accesses and thus signigicantly speeds up the calculation.
   */
  posx = (thread_id < N) ? p_in.pos_x[thread_id] : 0.0f;
  posy = (thread_id < N) ? p_in.pos_y[thread_id] : 0.0f;    
  posz = (thread_id < N) ? p_in.pos_z[thread_id] : 0.0f;
  
  velx = (thread_id < N) ? p_in.vel_x[thread_id] : 0.0f;
  vely = (thread_id < N) ? p_in.vel_y[thread_id] : 0.0f;
  velz = (thread_id < N) ? p_in.vel_z[thread_id] : 0.0f;

  weight = (thread_id < N) ? p_in.weight[thread_id] : 0.0f;
  
  // Process the input in the form of "tiles" that are the same size as the blockDim.x.
  for ( struct {int i = 0; int tile = 0;} loop; 
        loop.i < N; 
        loop.i+=blockDim.x, loop.tile++) 
  {
    // Appropriate index into global memory.
    int idx = loop.tile * blockDim.x + threadIdx.x;
    
    /**
     * Loading a single "tile" into shared memory.
     * note:  Pre-reading data from the global memory 2 shared memory, reduces the number of 
     *        memory accesses and thus signigicantly speeds up the calculation.
     */
    shared_posx[threadIdx.x] = (idx < N) ? p_in.pos_x[idx] : 0.0f;
    shared_posy[threadIdx.x] = (idx < N) ? p_in.pos_y[idx] : 0.0f;
    shared_posz[threadIdx.x] = (idx < N) ? p_in.pos_z[idx] : 0.0f;

    shared_velx[threadIdx.x] = (idx < N) ? p_in.vel_x[idx] : 0.0f; 
    shared_vely[threadIdx.x] = (idx < N) ? p_in.vel_y[idx] : 0.0f;
    shared_velz[threadIdx.x] = (idx < N) ? p_in.vel_z[idx] : 0.0f;

    shared_weight[threadIdx.x] =(idx < N) ? p_in.weight[idx] : 0.0f;

    __syncthreads();

    // Process the tile.
    for (int j = 0; j < blockDim.x; j++) {
      /**
       * The calculation of the gravitational force is divided into several
       * several instructions in order to eliminate data dependencies, and thus
       * we have increased the ILP.
       */
      F = -G * dt * shared_weight[j];

      dx = posx - shared_posx[j];
      dy = posy - shared_posy[j];
      dz = posz - shared_posz[j];

      r = sqrt(dx*dx + dy*dy + dz*dz);

      // see previous comment
      F /= (r * r * r + FLT_MIN);

      tmpvelx += (r > COLLISION_DISTANCE) ? F * dx : 0.0f;
      tmpvely += (r > COLLISION_DISTANCE) ? F * dy : 0.0f;
      tmpvelz += (r > COLLISION_DISTANCE) ? F * dz : 0.0f;

      // Add the velocity obtained by the gravitational action of the body 'j'.
      if (r < COLLISION_DISTANCE) {
        /**
         * Reuseage of the registers of distances.
         * note:  The values are calculated only once and then used several times, see below.
         */
        dx = weight - shared_weight[j];
        dy = 2 * shared_weight[j];
        dz = weight + shared_weight[j];

        // Add the velocity obtained by the collision with the body 'j'.
        tmpvelx += (r > 0.0f) ? ((dx * velx + dy * shared_velx[j]) / dz) - velx : 0.0f;
        tmpvely += (r > 0.0f) ? ((dx * vely + dy * shared_vely[j]) / dz) - vely : 0.0f;
        tmpvelz += (r > 0.0f) ? ((dx * velz + dy * shared_velz[j]) / dz) - velz : 0.0f;
      }
    }
    
    __syncthreads();
  }
  /**
   * Update particle
   * note:  Write to global memory only once at the end of the cycle.
   */
  if (thread_id < N) {
    velx += tmpvelx;
    p_out.vel_x[thread_id] = velx;
    p_out.pos_x[thread_id] = velx * dt + posx;

    vely += tmpvely;
    p_out.vel_y[thread_id] = vely;
    p_out.pos_y[thread_id] = vely * dt + posy;

    velz += tmpvelz;
    p_out.vel_z[thread_id] = velz;
    p_out.pos_z[thread_id] = velz * dt + posz;
  }
}// end of calculate_velocity
//-----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
