# ParallelSTL
C++ Parallel STL Benchmarks and Other Developments

To setup Intel's C++ Standard Parallel Algorithms on Linux, the following are needed:
- OneAPI
- Threading Building Blocks (TBB)

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
g++ /mnt/c/repos/ParallelSTL/benchmarks/src/main.cpp -ltbb -std=c++17 -O3 -o benchmark_tbb
```


## Building NVidia's Parallel STL on Ubuntu 20.04 Linux (or WSL on Windows)
To setup NVidia's compiler on Linux
https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/

To compile Benchmark using NVidia's compiler targetting Intel multicore CPU
nvc++ main.cpp -stdpar=multicore -O3 -o benchmark_nvc

To setup NVidia's compiler and C++ Standard parallel algorithms on Linux

 curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
    4  ls
    5  ls -al
    6  curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
    7  rm tmp
    8  curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
    9  ls /usr/share
   10  ls /usr/share/keyrings/
   11  ls -lt /usr/share/keyrings/
   12  echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
   13  sudo apt-get update -y
   14  sudo apt-get install -y nvhpc-23-5
