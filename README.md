# ParallelSTL
Benchmark of C++ Parallel Standard Library Algorithms (STL)

To setup Intel's C++ Standard Parallel Algorithms on Linux, the following are needed:
- (Windows) OneAPI
- (Linux)   Threading Building Blocks (TBB)

## Building on Ubuntu 20.04 Linux (or WSL on Windows)
To install g++ which supports C++17:
```
sudo apt install build-essential
```

To update gcc to support c++17 standard:
```
sudo apt update
sudo apt install libtbb-dev
```

To build, use g++ command and not gcc. The order of the following arguments matters!
```
g++ /mnt/c/repos/ParallelSTL/src/main.cpp -ltbb -std=c++17 -O3 -o benchmark_parallel_std
```

## Building NVidia's Parallel STL on Ubuntu 20.04 Linux (or WSL on Windows)
To setup NVidia's compiler on Linux follow these instructions:
https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/

To setup NVidia's compiler and C++ Standard parallel algorithms on Linux
```
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt-get update -y
sudo apt-get install -y nvhpc-23-5
```

To compile Benchmark using NVidia's compiler targetting Intel multicore CPU
```
nvc++ /mnt/c/repos/ParallelSTL/src/main.cpp -stdpar=multicore -O3 -o benchmark_nvc
```
## Building on Windows
In Windows/ParallelSTL sub-directory a Visual Studio 2022 solution/project can be used to build an executable using either Microsoft's compiler or Intel compiler.
Microsoft implements Parallel standard algorithms, but not all of them.
Intel implements more standard parallel algorithms with higher performance as show in [C++ Parallel STL Benchmark.](https://duvanenko.tech.blog/2023/05/21/c-parallel-stl-benchmark/)
[Intel OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) must be installed. If Intel implementations are not desired, then comment out "#define DPL_ALGORITHMS" at the top of "main.cpp".

Both compilers (Microsoft or Intel) can be used to build this project. To switch between compilers, select "Project/Intel-Compiler" from Visual Studio 2022 menu.

## Additional Resources
The following book provides examples, explanations, and a free code repository for developing your own parallel algorithms:
- [Practical Parallel Algorithms in C++ and C#: Part 1: Sorting on Multicore CPUs](https://www.amazon.com/Practical-Parallel-Algorithms-Sorting-Multicore-ebook/dp/B0C3TZPRKZ/ref=sr_1_2?crid=2WH4J28ICJ1DV&keywords=duvanenko&qid=1700855661&sprefix=duvanenko%2Caps%2C103&sr=8-2)

Related blogs to improve performance of C++ Standard Parallel Algorithms:
- [Sorting 19X Faster than C++ Parallel Sort](https://duvanenko.tech.blog/2023/10/29/sorting-19x-faster-than-c-parallel-sort/)
- [Can C++ Paralllel Standard Algorithms Accelerate, Even Small Arrays](https://duvanenko.tech.blog/2023/05/31/c-parallel-stl-performance-for-small-ish-arrays/)
- [Improving Parallel Performance for Small Arrays](https://duvanenko.tech.blog/2023/05/31/c-parallel-stl-performance-for-small-ish-arrays/)