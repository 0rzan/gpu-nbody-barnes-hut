# GPU N-Body Simulation — Barnes–Hut in CUDA

A 3-D gravitational n-body simulator using the Barnes–Hut approximation, running
entirely on the GPU as a six-kernel pipeline (bounding box, tree build, mass
summarization, sort, force calculation, integration). The force kernel was
profiled and tuned with Nsight Compute/Systems (see [`profiling/`](profiling/)).

## Build & run

```sh
nvcc -O3 -o barneshut src/barneshut.cu
# barneshut <input> <timesteps> <export-rate> <time-step-millis> <output>
./barneshut examples/sample_small.txt 10 1 10 output.txt
```

Input: first line `n`, then `n` rows of 7 floats (mass, position, velocity).
Generate inputs with `tools/gen_input.py`.

## Attribution

Written for the HPC course at the University of Warsaw. All code here is mine;
the assignment spec, reference paper, and NVIDIA sample Makefile are not included.

[MIT](LICENSE) © 2026 Michał Orżanowski
