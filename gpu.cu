#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
thrust::device_ptr<int> particle_ids;  // Stores sorted particle IDs per bin
thrust::device_ptr<int> bin_offsets;   // Offset array for bins
thrust::device_ptr<int> bin_counts;    // Counts number of particles per bin
int num_bins;
int grid_size;
int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// optimized version: compute forces between particles
__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int* particle_ids, int* bin_offsets, int grid_size) {
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= grid_size * grid_size)
        return;

    int row = bin_idx / grid_size;
    int col = bin_idx % grid_size;
    
    // Fetch bin start and end once to minimize global memory accesses
    int start_idx = bin_offsets[bin_idx];
    int end_idx = bin_offsets[bin_idx + 1];

    // Loop through neighboring bins
    for (int d_row = -1; d_row <= 1; ++d_row) {
        for (int d_col = -1; d_col <= 1; ++d_col) {
            int neighbor_row = row + d_row;
            int neighbor_col = col + d_col;

            // Ensure valid neighbor bin
            if (neighbor_row >= 0 && neighbor_row < grid_size && 
                neighbor_col >= 0 && neighbor_col < grid_size) {
                
                int neighbor_bin_idx = neighbor_row * grid_size + neighbor_col;
                int neighbor_start = bin_offsets[neighbor_bin_idx];
                int neighbor_end = bin_offsets[neighbor_bin_idx + 1];

                // Iterate over all particles in the bin
                for (int i = start_idx; i < end_idx; ++i) {
                    int p1 = particle_ids[i];

                    for (int j = neighbor_start; j < neighbor_end; ++j) {
                        int p2 = particle_ids[j];

                        // Apply force between the two particles
                        apply_force_gpu(parts[p1], parts[p2]);
                    }
                }
            }
        }
    }
}

// count particles in each bin
__global__ void count_particles_gpu(int* bin_counts, int num_parts, double size, particle_t* parts, int grid_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Precompute bin indices to reduce redundant calculations
    int bin_x = __double2int_rz(parts[tid].x / cutoff);
    int bin_y = __double2int_rz(parts[tid].y / cutoff);
    int bin_id = bin_x * grid_size + bin_y;

    // Ensure valid bin assignment
    if (bin_x >= 0 && bin_x < grid_size && bin_y >= 0 && bin_y < grid_size) {
        atomicAdd(&bin_counts[bin_id], 1);
    }

    // Reset particle acceleration (ensures fresh start)
    parts[tid].ax = 0.0;
    parts[tid].ay = 0.0;
}

// assign particles to bins
__global__ void assign_particles_gpu(particle_t* parts, int num_parts, int* particle_ids, int* bin_offsets, int grid_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    // Precompute bin indices
    int bin_x = __double2int_rz(parts[tid].x / cutoff);
    int bin_y = __double2int_rz(parts[tid].y / cutoff);
    int bin_id = bin_x * grid_size + bin_y;

    // Ensure valid bin assignment
    if (bin_x >= 0 && bin_x < grid_size && bin_y >= 0 && bin_y < grid_size) {
        int index = atomicAdd(&bin_offsets[bin_id], 1);
        particle_ids[index] = tid;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    grid_size = ceil(size / cutoff);
    num_bins = grid_size * grid_size;

    bin_counts = thrust::device_malloc<int>(num_bins);
    particle_ids = thrust::device_malloc<int>(num_parts);
    bin_offsets = thrust::device_malloc<int>(num_bins + 1);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset bin counts
    thrust::fill(bin_counts, bin_counts + num_bins, 0);

    // Count particles in each bin
    count_particles_gpu<<<blks, NUM_THREADS>>>(bin_counts.get(), num_parts, size, parts, grid_size);
    cudaDeviceSynchronize();

    // Compute cumulative bin offsets
    thrust::inclusive_scan(bin_counts, bin_counts + num_bins, bin_counts);

    // Reset bin offsets and copy from bin_counts
    cudaMemset(bin_offsets.get(), 0, (num_bins + 1) * sizeof(int));
    cudaMemcpy(bin_offsets.get() + 1, bin_counts.get(), num_bins * sizeof(int), cudaMemcpyDeviceToDevice);

    // Assign particles to bins
    assign_particles_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids.get(), bin_offsets.get(), grid_size);

    // Reset bin offsets again after assignment
    cudaMemset(bin_offsets.get(), 0, (num_bins + 1) * sizeof(int));
    cudaMemcpy(bin_offsets.get() + 1, bin_counts.get(), num_bins * sizeof(int), cudaMemcpyDeviceToDevice);

    // Compute forces
    compute_forces_gpu<<<(grid_size * grid_size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(parts, num_parts, particle_ids.get(), bin_offsets.get(), grid_size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
