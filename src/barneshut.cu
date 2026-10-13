#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct BoundingBox {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

struct Bodies {
    int n;
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
};

constexpr int THREADS = 256;
constexpr int NULL_NODE = -1;
constexpr int LOCK_NODE = -2;
constexpr int MAX_DEPTH = 20;


// Global Device Pointers

float *d_px = nullptr, *d_py = nullptr, *d_pz = nullptr, *d_mass = nullptr;
float *d_bmin_x = nullptr, *d_bmin_y = nullptr, *d_bmin_z = nullptr;
float *d_bmax_x = nullptr, *d_bmax_y = nullptr, *d_bmax_z = nullptr;
BoundingBox* d_bbox = nullptr;

int   *d_child = nullptr;
float *d_cell_cx = nullptr, *d_cell_cy = nullptr, *d_cell_cz = nullptr;
float *d_cell_s  = nullptr;
int   *d_bottom  = nullptr;

int *d_count = nullptr;
int *d_start = nullptr, *d_sorted = nullptr;

int *d_is_clump = nullptr;
int *d_body_clump = nullptr;
int *d_clump_fill = nullptr;

float *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
float *d_acc_x = nullptr, *d_acc_y = nullptr, *d_acc_z = nullptr;


// Kernels

__device__ unsigned int g_blocks_done = 0;

__global__ void bounding_box_kernel(
    const float* __restrict__ px, const float* __restrict__ py, const float* __restrict__ pz, int n,
    float* block_min_x, float* block_min_y, float* block_min_z,
    float* block_max_x, float* block_max_y, float* block_max_z,
    BoundingBox* result)
{
    extern __shared__ float smem[];
    float* s_min_x = smem + 0 * blockDim.x;
    float* s_min_y = smem + 1 * blockDim.x;
    float* s_min_z = smem + 2 * blockDim.x;
    float* s_max_x = smem + 3 * blockDim.x;
    float* s_max_y = smem + 4 * blockDim.x;
    float* s_max_z = smem + 5 * blockDim.x;

    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;

    float mn_x = INFINITY, mn_y = INFINITY, mn_z = INFINITY;
    float mx_x = -INFINITY, mx_y = -INFINITY, mx_z = -INFINITY;

    for (int i = gid; i < n; i += stride) {
        float x = px[i], y = py[i], z = pz[i];
        mn_x = fminf(mn_x, x); mx_x = fmaxf(mx_x, x);
        mn_y = fminf(mn_y, y); mx_y = fmaxf(mx_y, y);
        mn_z = fminf(mn_z, z); mx_z = fmaxf(mx_z, z);
    }

    s_min_x[tid] = mn_x; s_max_x[tid] = mx_x;
    s_min_y[tid] = mn_y; s_max_y[tid] = mx_y;
    s_min_z[tid] = mn_z; s_max_z[tid] = mx_z;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_min_z[tid] = fminf(s_min_z[tid], s_min_z[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
            s_max_z[tid] = fmaxf(s_max_z[tid], s_max_z[tid + s]);
        }
        __syncthreads();
    }

    __shared__ bool is_last_block;
    if (tid == 0) {
        block_min_x[blockIdx.x] = s_min_x[0];
        block_min_y[blockIdx.x] = s_min_y[0];
        block_min_z[blockIdx.x] = s_min_z[0];
        block_max_x[blockIdx.x] = s_max_x[0];
        block_max_y[blockIdx.x] = s_max_y[0];
        block_max_z[blockIdx.x] = s_max_z[0];
        __threadfence();
        unsigned int v = atomicInc(&g_blocks_done, gridDim.x);
        is_last_block = (v == gridDim.x - 1);
    }
    __syncthreads();

    if (!is_last_block) return;

    mn_x = INFINITY; mn_y = INFINITY; mn_z = INFINITY;
    mx_x = -INFINITY; mx_y = -INFINITY; mx_z = -INFINITY;
    for (int i = tid; i < gridDim.x; i += blockDim.x) {
        mn_x = fminf(mn_x, block_min_x[i]); mx_x = fmaxf(mx_x, block_max_x[i]);
        mn_y = fminf(mn_y, block_min_y[i]); mx_y = fmaxf(mx_y, block_max_y[i]);
        mn_z = fminf(mn_z, block_min_z[i]); mx_z = fmaxf(mx_z, block_max_z[i]);
    }
    s_min_x[tid] = mn_x; s_max_x[tid] = mx_x;
    s_min_y[tid] = mn_y; s_max_y[tid] = mx_y;
    s_min_z[tid] = mn_z; s_max_z[tid] = mx_z;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_min_z[tid] = fminf(s_min_z[tid], s_min_z[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
            s_max_z[tid] = fmaxf(s_max_z[tid], s_max_z[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result->min_x = s_min_x[0]; result->max_x = s_max_x[0];
        result->min_y = s_min_y[0]; result->max_y = s_max_y[0];
        result->min_z = s_min_z[0]; result->max_z = s_max_z[0];
        g_blocks_done = 0;
    }
}

__global__ void tree_init_kernel(
    int n_nodes, const BoundingBox* bbox, int* child,
    float* cell_cx, float* cell_cy, float* cell_cz, float* cell_s, int* bottom,
    float* mass, int* count, int* start)
{
    const int root = n_nodes - 1;
    const float ex = bbox->max_x - bbox->min_x;
    const float ey = bbox->max_y - bbox->min_y;
    const float ez = bbox->max_z - bbox->min_z;
    const float max_e = fmaxf(ex, fmaxf(ey, ez));
    cell_cx[root] = 0.5f * (bbox->min_x + max_e);
    cell_cy[root] = 0.5f * (bbox->min_y + max_e);
    cell_cz[root] = 0.5f * (bbox->min_z + max_e);
    cell_s[root]  = max_e;
    for (int k = 0; k < 8; ++k) child[8 * root + k] = NULL_NODE;
    *bottom = root;
    mass[root]  = -1.0f;
    count[root] = 0;
    start[root] = 0;
}

__global__ void build_tree_kernel(
    int n_bodies, int n_nodes,
    float* pos_x, float* pos_y, float* pos_z,
    volatile int* child,
    float* cell_cx, float* cell_cy, float* cell_cz, float* cell_s, int* bottom,
    float* mass, int* count, int* start,
    int* is_clump, int* body_clump)
{
    const int root = n_nodes - 1;
    const float root_cx = cell_cx[root], root_cy = cell_cy[root], root_cz = cell_cz[root];
    const float root_half = 0.5f * cell_s[root];

    const int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    for (int i = gid; i < n_bodies; i += gstride) {
        const float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
        const float mi = mass[i];
        bool success = false;
        while (!success) {
            int   cell = root;
            float cx = root_cx, cy = root_cy, cz = root_cz, half = root_half;
            int   oct = ((px > cx) ? 1 : 0) | ((py > cy) ? 2 : 0) | ((pz > cz) ? 4 : 0);
            int   slot = child[8 * cell + oct];
            int   depth = 0;

            while (slot >= n_bodies && !is_clump[slot]) {
                cell = slot; half *= 0.5f;
                cx += (oct & 1) ? half : -half;
                cy += (oct & 2) ? half : -half;
                cz += (oct & 4) ? half : -half;
                oct = ((px > cx) ? 1 : 0) | ((py > cy) ? 2 : 0) | ((pz > cz) ? 4 : 0);
                slot = child[8 * cell + oct];
                ++depth;
            }

            if (slot == LOCK_NODE) continue;

            const int slot_addr = 8 * cell + oct;
            if (atomicCAS((int*)&child[slot_addr], slot, LOCK_NODE) != slot) continue;

            if (slot == NULL_NODE) {
                __threadfence();
                child[slot_addr] = i;
                success = true;
            } else if (slot >= n_bodies) {
                const int C = slot;
                const float mC = mass[C];
                const float new_m = mC + mi;
                pos_x[C] = (mC * pos_x[C] + mi * px) / new_m;
                pos_y[C] = (mC * pos_y[C] + mi * py) / new_m;
                pos_z[C] = (mC * pos_z[C] + mi * pz) / new_m;
                mass[C]  = new_m;
                count[C] = count[C] + 1;
                body_clump[i] = C;
                success = true;
                __threadfence();
                child[slot_addr] = C;
            } else {
                const int   existing = slot;
                const float bx = pos_x[existing], by = pos_y[existing], bz = pos_z[existing];
                const float me = mass[existing];
                const int newcell = atomicSub(bottom, 1) - 1;

                for (int k = 0; k < 8; ++k) child[8 * newcell + k] = NULL_NODE;
                start[newcell] = -1;

                const float nhalf = 0.5f * half;
                const float ncx = cx + ((oct & 1) ? nhalf : -nhalf);
                const float ncy = cy + ((oct & 2) ? nhalf : -nhalf);
                const float ncz = cz + ((oct & 4) ? nhalf : -nhalf);
                cell_cx[newcell] = ncx; cell_cy[newcell] = ncy;
                cell_cz[newcell] = ncz; cell_s[newcell]  = 2.0f * nhalf;

                if (depth + 1 >= MAX_DEPTH) {
                    const float total = me + mi;
                    pos_x[newcell] = (me * bx + mi * px) / total;
                    pos_y[newcell] = (me * by + mi * py) / total;
                    pos_z[newcell] = (me * bz + mi * pz) / total;
                    mass[newcell]  = total;
                    count[newcell] = 2;
                    is_clump[newcell] = 1;
                    body_clump[existing] = newcell;
                    body_clump[i] = newcell;
                    success = true;
                } else {
                    mass[newcell]  = -1.0f;
                    count[newcell] = 0;
                    is_clump[newcell] = 0;
                    body_clump[newcell] = -1;

                    const int oct_old = ((bx > ncx) ? 1 : 0) | ((by > ncy) ? 2 : 0) | ((bz > ncz) ? 4 : 0);
                    int       oct_new = ((px > ncx) ? 1 : 0) | ((py > ncy) ? 2 : 0) | ((pz > ncz) ? 4 : 0);

                    if (oct_new == oct_old && nhalf <= 0.0f) {
                        oct_new = oct_old ^ 1;
                    }

                    child[8 * newcell + oct_old] = existing;
                    if (oct_new != oct_old) {
                        child[8 * newcell + oct_new] = i;
                        success = true;
                    }
                }
                __threadfence();
                child[slot_addr] = newcell;
            }
        }
    }
}

__global__ void summarize_kernel(
    int n_bodies, int n_nodes, int* child,
    volatile float* pos_x, volatile float* pos_y, volatile float* pos_z,
    volatile float* mass, int* count, const int* bottom_ptr,
    const int* __restrict__ is_clump)
{
    const int bottom = *bottom_ptr;
    const int root   = n_nodes - 1;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    int cache[8];
    int missing;
    float sum_m, sum_x, sum_y, sum_z;
    int cnt;

    for(int c = bottom + gid; c <= root; c += stride) {
        if (is_clump[c]) continue;
        sum_m = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        cnt = 0;
        missing = 0;

        bool success = false;

        while(!success) {
            if (missing == 0) {
                for (int i = 0; i < 8; ++i) {
                    int ch = child[8 * c + i];
                    if (ch < 0) continue;

                    if (mass[ch] < 0.0f) {
                        cache[missing++] = ch;
                        continue;
                    } else {
                        sum_m += mass[ch];
                        sum_x += mass[ch] * pos_x[ch];
                        sum_y += mass[ch] * pos_y[ch];
                        sum_z += mass[ch] * pos_z[ch];
                        cnt  += (ch < n_bodies) ? 1 : count[ch];
                    }
                }
            }
            if (missing != 0) {
                do {
                    int ch = cache[missing - 1];
                    if (*(volatile float *)&mass[ch] < 0.0f) {
                        break;
                    } else {
                        sum_m += mass[ch];
                        sum_x += mass[ch] * pos_x[ch];
                        sum_y += mass[ch] * pos_y[ch];
                        sum_z += mass[ch] * pos_z[ch];
                        cnt  += (ch < n_bodies) ? 1 : count[ch];
                        missing--;
                    }
                } while (missing != 0);
            }
            if (missing == 0){
                pos_x[c] = sum_x / sum_m;
                pos_y[c] = sum_y / sum_m;
                pos_z[c] = sum_z / sum_m;
                count[c] = cnt;
                __threadfence();
                mass[c] = sum_m;
                success = true;
            }
        }
    }
}

__global__ void sort_kernel(
    int n_bodies, int n_nodes, const int* __restrict__ child,
    const int* __restrict__ count, volatile int* start, int* sorted, const int* bottom_ptr)
{
    const int bottom = *bottom_ptr;
    const int root   = n_nodes - 1;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int c = root - gid; c >= bottom; c -= stride) {
        int s = -1;
        bool done = false;
        while (!done) {
            if (s < 0) {
                s = start[c];
            } else {
                int offset = 0;
                for (int k = 0; k < 8; ++k) {
                    const int ch = child[8 * c + k];
                    if (ch == NULL_NODE) continue;
                    if (ch < n_bodies) {
                        sorted[s + offset] = ch;
                        offset += 1;
                    } else {
                        start[ch] = s + offset;
                        offset += count[ch];
                    }
                }
                __threadfence();
                done = true;
            }
        }
    }
}

__global__ void fill_clump_kernel(
    int n_bodies, int* sorted, const int* __restrict__ start,
    int* fill_count, const int* __restrict__ body_clump)
{
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < n_bodies; i += stride) {
        const int C = body_clump[i];
        if (C < 0) continue;
        const int offset = atomicAdd(&fill_count[C], 1);
        sorted[start[C] + offset] = i;
    }
}

constexpr float    G         = 1.0f;
constexpr float    EPS2      = 0.05f;
constexpr float    THETA2    = 0.25f;
constexpr int      WARP      = 32;
constexpr int      WARPS     = THREADS / WARP;
constexpr int      K_MAX_DEPTH = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__global__ void force_kernel(
    int n_bodies, int n_nodes, const int* __restrict__ child,
    const float* __restrict__ pos_x, const float* __restrict__ pos_y, const float* __restrict__ pos_z,
    const float* __restrict__ mass, const float* __restrict__ cell_s,
    const int* __restrict__ sorted,
    const int* __restrict__ is_clump, const int* __restrict__ body_clump,
    float* acc_x, float* acc_y, float* acc_z)
{
    __shared__ float sh_dq  [K_MAX_DEPTH];
    __shared__ int   sh_pos [WARPS * K_MAX_DEPTH];
    __shared__ int   sh_node[WARPS * K_MAX_DEPTH * 8];

    const int tid           = threadIdx.x;
    const int lane          = tid & (WARP - 1);
    const int warp_in_block = tid >> 5;
    const int root          = n_nodes - 1;

    if (tid == 0) {
        const float root_s = cell_s[root];
        float s2_t2 = root_s * root_s * 0.25f / THETA2;
        sh_dq[0] = s2_t2;
        for (int d = 1; d < MAX_DEPTH; ++d) {
            s2_t2 *= 0.25f;
            sh_dq[d] = s2_t2;
        }
    }
    __syncthreads();

    int* const my_pos  = &sh_pos [warp_in_block * MAX_DEPTH];
    int* const my_node = &sh_node[warp_in_block * MAX_DEPTH * 8];

    const int n_padded = ((n_bodies + WARP - 1) / WARP) * WARP;
    const int gid      = blockIdx.x * blockDim.x + tid;
    const int gstride  = gridDim.x * blockDim.x;

    for (int idx = gid; idx < n_padded; idx += gstride) {
        const bool active = (idx < n_bodies);
        const int  i      = active ? sorted[idx] : 0;
        const float bx    = active ? pos_x[i] : 0.0f;
        const float by    = active ? pos_y[i] : 0.0f;
        const float bz    = active ? pos_z[i] : 0.0f;
        const float bm    = active ? mass[i]  : 0.0f;
        const int   bclump= active ? body_clump[i] : -1;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) my_node[k] = child[8 * root + k];
            my_pos[0] = 0;
        }
        int depth = 0;
        __syncwarp();

        while (depth >= 0) {
            int n = -1;
            if (lane == 0) {
                int p = my_pos[depth];
                while (p < 8) {
                    int v = my_node[depth * 8 + p++];
                    if (v != NULL_NODE) { n = v; break; }
                }
                my_pos[depth] = p;
            }
            n = __shfl_sync(FULL_MASK, n, 0);

            if (n < 0) { --depth; continue; }

            float nx = 0.0f, ny = 0.0f, nz = 0.0f, nm = 0.0f;
            int   nic = 0;
            if (lane == 0) {
                nx = pos_x[n]; ny = pos_y[n]; nz = pos_z[n]; nm = mass[n];
                nic = (n >= n_bodies) ? is_clump[n] : 0;
            }
            nx  = __shfl_sync(FULL_MASK, nx, 0);
            ny  = __shfl_sync(FULL_MASK, ny, 0);
            nz  = __shfl_sync(FULL_MASK, nz, 0);
            nm  = __shfl_sync(FULL_MASK, nm, 0);
            nic = __shfl_sync(FULL_MASK, nic, 0);

            const float dx = nx - bx, dy = ny - by, dz = nz - bz;
            const float r2 = dx * dx + dy * dy + dz * dz;

            const bool is_body    = (n < n_bodies);
            const bool is_clmp    = (nic != 0);
            const bool far_enough = !active || (r2 >= sh_dq[depth]);

            if (is_body || is_clmp || __all_sync(FULL_MASK, far_enough)) {
                if (active && !(is_body && n == i)) {
                    float eff_m = nm, eff_dx = dx, eff_dy = dy, eff_dz = dz;
                    if (is_clmp && bclump == n) {
                        eff_m = nm - bm;
                        if (eff_m > 0.0f) {
                            const float ex = (nm * nx - bm * bx) / eff_m;
                            const float ey = (nm * ny - bm * by) / eff_m;
                            const float ez = (nm * nz - bm * bz) / eff_m;
                            eff_dx = ex - bx; eff_dy = ey - by; eff_dz = ez - bz;
                        } else {
                            eff_m = 0.0f;
                        }
                    }
                    if (eff_m > 0.0f) {
                        const float r2e   = eff_dx * eff_dx + eff_dy * eff_dy + eff_dz * eff_dz + EPS2;
                        const float inv_r3 = rsqrtf(r2e * r2e * r2e);
                        const float f      = G * eff_m * inv_r3;
                        ax += f * eff_dx; ay += f * eff_dy; az += f * eff_dz;
                    }
                }
            } else {
                ++depth;
                if (lane == 0) {
                    const int base = 8 * n;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) my_node[depth * 8 + k] = child[base + k];
                    my_pos[depth] = 0;
                }
                __syncwarp();
            }
        }

        if (active) {
            acc_x[i] = ax; acc_y[i] = ay; acc_z[i] = az;
        }
    }
}

__global__ void integrate_kernel(
    int n_bodies, float dt, int do_drift,
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    const float* __restrict__ acc_x,
    const float* __restrict__ acc_y,
    const float* __restrict__ acc_z)
{
    const int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride  = gridDim.x * blockDim.x;
    const float h_dt  = 0.5f * dt;

    for (int i = gid; i < n_bodies; i += stride) {
        const float vxi = vel_x[i] + acc_x[i] * h_dt;
        const float vyi = vel_y[i] + acc_y[i] * h_dt;
        const float vzi = vel_z[i] + acc_z[i] * h_dt;
        vel_x[i] = vxi; vel_y[i] = vyi; vel_z[i] = vzi;

        if (do_drift) {
            pos_x[i] += vxi * dt;
            pos_y[i] += vyi * dt;
            pos_z[i] += vzi * dt;
        }
    }
}


// Host Allocation & Execution Wrappers

void allocateDeviceMemory(int n, int n_nodes, int blocks) {
    CUDA_CHECK(cudaMalloc(&d_px,   n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py,   n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz,   n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mass, n_nodes * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_bmin_x, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmin_y, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmin_z, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmax_x, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmax_y, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmax_z, blocks * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_bbox, sizeof(BoundingBox)));

    CUDA_CHECK(cudaMalloc(&d_child,   8 * n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_cx, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_cy, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_cz, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_s,  n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bottom,  sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_count, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted, n * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_is_clump,   n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_body_clump, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clump_fill, n_nodes * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_vx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_acc_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc_z, n * sizeof(float)));
}

void freeDeviceMemory() {
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_mass);
    cudaFree(d_bmin_x); cudaFree(d_bmin_y); cudaFree(d_bmin_z);
    cudaFree(d_bmax_x); cudaFree(d_bmax_y); cudaFree(d_bmax_z);
    cudaFree(d_bbox);
    cudaFree(d_child);
    cudaFree(d_cell_cx); cudaFree(d_cell_cy); cudaFree(d_cell_cz); cudaFree(d_cell_s);
    cudaFree(d_bottom);
    cudaFree(d_count);
    cudaFree(d_start); cudaFree(d_sorted);
    cudaFree(d_is_clump); cudaFree(d_body_clump); cudaFree(d_clump_fill);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_acc_x); cudaFree(d_acc_y); cudaFree(d_acc_z);
}

void initDeviceMemory(const Bodies& bodies, int n, int n_nodes) {
    CUDA_CHECK(cudaMemcpy(d_px, bodies.px.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, bodies.py.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, bodies.pz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, bodies.mass.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, bodies.vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, bodies.vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, bodies.vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

void copyStateToHost(Bodies& b, int n) {
    CUDA_CHECK(cudaMemcpy(b.px.data(), d_px, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.py.data(), d_py, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.pz.data(), d_pz, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vx.data(), d_vx, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vy.data(), d_vy, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.vz.data(), d_vz, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void runBoundingBox(int n, int blocks) {
    size_t smem_bytes = 6 * THREADS * sizeof(float);
    bounding_box_kernel<<<blocks, THREADS, smem_bytes>>>(
        d_px, d_py, d_pz, n,
        d_bmin_x, d_bmin_y, d_bmin_z, d_bmax_x, d_bmax_y, d_bmax_z, d_bbox);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runBuildTree(int n, int n_nodes, int blocks) {
    CUDA_CHECK(cudaMemset(d_is_clump,   0,    n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_body_clump, 0xFF, n * sizeof(int)));

    tree_init_kernel<<<1, 1>>>(
        n_nodes, d_bbox, d_child, d_cell_cx, d_cell_cy, d_cell_cz, d_cell_s, d_bottom, d_mass, d_count, d_start);
    CUDA_CHECK(cudaGetLastError());

    build_tree_kernel<<<blocks, THREADS>>>(
        n, n_nodes, d_px, d_py, d_pz,
        d_child, d_cell_cx, d_cell_cy, d_cell_cz, d_cell_s, d_bottom,
        d_mass, d_count, d_start, d_is_clump, d_body_clump);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runSummarizeCells(int n, int n_nodes, int cell_blocks) {
    summarize_kernel<<<cell_blocks, THREADS>>>(
        n, n_nodes, d_child, d_px, d_py, d_pz, d_mass, d_count, d_bottom, d_is_clump);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runSortBodies(int n, int n_nodes, int cell_blocks, int blocks) {
    sort_kernel<<<cell_blocks, THREADS>>>(
        n, n_nodes, d_child, d_count, d_start, d_sorted, d_bottom);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(d_clump_fill, 0, n_nodes * sizeof(int)));
    fill_clump_kernel<<<blocks, THREADS>>>(
        n, d_sorted, d_start, d_clump_fill, d_body_clump);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runComputeForces(int n, int n_nodes, int blocks) {
    force_kernel<<<blocks, THREADS>>>(
        n, n_nodes, d_child, d_px, d_py, d_pz, d_mass, d_cell_s,
        d_sorted, d_is_clump, d_body_clump, d_acc_x, d_acc_y, d_acc_z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runIntegrate(int n, int blocks, float dt, bool do_drift) {
    integrate_kernel<<<blocks, THREADS>>>(
        n, dt, do_drift ? 1 : 0,
        d_px, d_py, d_pz, d_vx, d_vy, d_vz,
        d_acc_x, d_acc_y, d_acc_z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Stage Timing

enum TimerIdx {
    T_BBOX = 0, T_BUILD, T_SUMM, T_SORT, T_FORCE, T_INTEG, T_EXPORT,
    N_TIMERS
};

struct Timer {
    const char* name;
    double total_ms;
    int    calls;
};

static Timer g_timers[N_TIMERS] = {
    {"bounding_box", 0.0, 0},
    {"build_tree",   0.0, 0},
    {"summarize",    0.0, 0},
    {"sort",         0.0, 0},
    {"forces",       0.0, 0},
    {"integrate",    0.0, 0},
    {"export",       0.0, 0},
};

template<typename F>
static inline void timed(int idx, F&& fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    g_timers[idx].total_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    g_timers[idx].calls++;
}

static void printTimings() {
    double total = 0.0;
    for (int i = 0; i < N_TIMERS; ++i) total += g_timers[i].total_ms;

    std::fprintf(stderr, "\n=== Stage timings ===\n");
    std::fprintf(stderr, "%-14s %12s %8s %12s %8s\n",
                 "stage", "total ms", "calls", "avg ms", "%total");
    for (int i = 0; i < N_TIMERS; ++i) {
        const Timer& t = g_timers[i];
        const double avg = t.calls > 0 ? t.total_ms / t.calls : 0.0;
        const double pct = total > 0.0 ? 100.0 * t.total_ms / total : 0.0;
        std::fprintf(stderr, "%-14s %12.3f %8d %12.4f %7.2f%%\n",
                     t.name, t.total_ms, t.calls, avg, pct);
    }
}


// I/O & Debug Helpers

static bool read_input(const char* path, Bodies& b) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;

    if (std::fscanf(f, "%d", &b.n) != 1 || b.n <= 0) {
        std::fclose(f); return false;
    }

    b.px.resize(b.n); b.py.resize(b.n); b.pz.resize(b.n);
    b.vx.resize(b.n); b.vy.resize(b.n); b.vz.resize(b.n);
    b.mass.resize(b.n);

    for (int i = 0; i < b.n; ++i) {
        if (std::fscanf(f, "%f %f %f %f %f %f %f",
            &b.px[i], &b.py[i], &b.pz[i], &b.vx[i], &b.vy[i], &b.vz[i], &b.mass[i]) != 7) {
            std::fclose(f); return false;
        }
    }
    std::fclose(f); return true;
}

static bool write_step(FILE* f, int step, const Bodies& b) {
    if (std::fprintf(f, "Step %d\n", step) < 0) return false;
    for (int i = 0; i < b.n; ++i) {
        if (std::fprintf(f, "%f %f %f %f %f %f\n",
            b.px[i], b.py[i], b.pz[i], b.vx[i], b.vy[i], b.vz[i]) < 0) return false;
    }
    return true;
}

void debugPrint(int n, int n_nodes) {
    BoundingBox h_bbox{};
    CUDA_CHECK(cudaMemcpy(&h_bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "Final bbox: x=[%g, %g] y=[%g, %g] z=[%g, %g]\n",
        h_bbox.min_x, h_bbox.max_x, h_bbox.min_y, h_bbox.max_y, h_bbox.min_z, h_bbox.max_z);

    std::vector<int> h_sorted(n);
    std::vector<float> h_acc_x(n), h_acc_y(n), h_acc_z(n);
    CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_sorted, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_acc_x.data(), d_acc_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_acc_y.data(), d_acc_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_acc_z.data(), d_acc_z, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::fprintf(stderr, "sorted bodies (in-order):");
    for (int i = 0; i < n; ++i) std::fprintf(stderr, " %d", h_sorted[i]);
    std::fputc('\n', stderr);

    std::fprintf(stderr, "accelerations:\n");
    for (int i = 0; i < n; ++i) {
        std::fprintf(stderr, "  body %d: a=(%g, %g, %g)\n",
            i, h_acc_x[i], h_acc_y[i], h_acc_z[i]);
    }
    const int root_idx = n_nodes - 1;
    std::vector<int>   h_child(8 * n_nodes);
    std::vector<float> h_cell_cx(n_nodes), h_cell_cy(n_nodes), h_cell_cz(n_nodes), h_cell_s(n_nodes);
    std::vector<float> h_pos_x(n_nodes), h_pos_y(n_nodes), h_pos_z(n_nodes), h_mass(n_nodes);
    CUDA_CHECK(cudaMemcpy(h_child.data(),   d_child,   8 * n_nodes * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_cx.data(), d_cell_cx, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_cy.data(), d_cell_cy, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_cz.data(), d_cell_cz, n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_s.data(),  d_cell_s,  n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_x.data(),   d_px,      n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y.data(),   d_py,      n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z.data(),   d_pz,      n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mass.data(),    d_mass,    n_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    auto walk = [&](auto& self, int node, int depth) -> void {
        for (int i = 0; i < depth; ++i) std::fputs("  ", stderr);
        if (node < n) {
            std::fprintf(stderr, "body %d: pos=(%g, %g, %g) mass=%g\n",
                node, h_pos_x[node], h_pos_y[node], h_pos_z[node], h_mass[node]);
        } else {
            std::fprintf(stderr, "cell %d: center=(%g, %g, %g) side=%g | com=(%g, %g, %g) mass=%g\n",
                node, h_cell_cx[node], h_cell_cy[node], h_cell_cz[node], h_cell_s[node],
                h_pos_x[node], h_pos_y[node], h_pos_z[node], h_mass[node]);
            for (int k = 0; k < 8; ++k) {
                int c = h_child[8 * node + k];
                if (c != NULL_NODE) self(self, c, depth + 1);
            }
        }
    };
    std::fprintf(stderr, "--- tree ---\n");
    walk(walk, root_idx, 0);
}


// Main Loop

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <input file> <timesteps> <export-rate> <time-step-millis> <output file>\n", argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const int   timesteps   = std::atoi(argv[2]);
    const int   export_rate = std::atoi(argv[3]);
    const float dt          = std::atof(argv[4]) * 1e-3f;
    const char* output_path = argv[5];

    Bodies bodies;
    if (!read_input(input_path, bodies)) return 1;

    std::fprintf(stderr, "loaded %d bodies | timesteps=%d export-rate=%d dt=%g s\n", bodies.n, timesteps, export_rate, dt);

    FILE* out = std::fopen(output_path, "w");
    if (!out) {
        std::fprintf(stderr, "error: cannot open output file '%s'\n", output_path);
        return 1;
    }

    const int n        = bodies.n;
    const int n_nodes  = 32 * n;
    
    int blocks = (n + THREADS - 1) / THREADS;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    const int cell_blocks = (n_nodes + THREADS - 1) / THREADS;

    allocateDeviceMemory(n, n_nodes, blocks);
    initDeviceMemory(bodies, n, n_nodes);

    timed(T_BBOX,  [&]{ runBoundingBox(n, blocks); });
    timed(T_BUILD, [&]{ runBuildTree(n, n_nodes, blocks); });
    timed(T_SUMM,  [&]{ runSummarizeCells(n, n_nodes, cell_blocks); });
    timed(T_SORT,  [&]{ runSortBodies(n, n_nodes, cell_blocks, blocks); });
    timed(T_FORCE, [&]{ runComputeForces(n, n_nodes, blocks); });

    for (int step = 1; step <= timesteps; ++step) {
        timed(T_INTEG, [&]{ runIntegrate(n, blocks, dt, true); });
        
        timed(T_BBOX,  [&]{ runBoundingBox(n, blocks); });
        timed(T_BUILD, [&]{ runBuildTree(n, n_nodes, blocks); });
        timed(T_SUMM,  [&]{ runSummarizeCells(n, n_nodes, cell_blocks); });
        timed(T_SORT,  [&]{ runSortBodies(n, n_nodes, cell_blocks, blocks); });
        timed(T_FORCE, [&]{ runComputeForces(n, n_nodes, blocks); });
        
        timed(T_INTEG, [&]{ runIntegrate(n, blocks, dt, false); });
        
        if (export_rate > 0 && step % export_rate == 0) {
            timed(T_EXPORT, [&]{
                copyStateToHost(bodies, n);
                if (!write_step(out, step, bodies)) {
                    std::fprintf(stderr, "error: failed writing step %d\n", step);
                }
            });
        }
    }
    
    // debugPrint(n, n_nodes);
    printTimings();

    freeDeviceMemory();
    std::fclose(out);

    return 0;
}