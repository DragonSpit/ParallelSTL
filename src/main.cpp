// Additional reference besides oneapi/dpl/algorithm header itself: https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-parallel-stl.html (this reference provides links for each standard algorithm manual page)
// TODO: Figure out whose algorithm is std::sort (Microsoft's or Intel's) in the Intel environment versus Microsoft's environment - i.e. is it the same algorithm, but different allocator? Try running std::sort wihout dpl headers inclusion to see if performance is different.
//       possibly add print statements into standard headers to figure out which algorithm is running. I.E. Is is a better algorithm or a better allocator that is making the difference? TBB versus PPL, and/or allocator, or compiler?
// TODO: These benchmarks show that bandwidth limits parallel scaling. Maybe instead of large arrays, small enough arrays need to be used that fit in cache, using algorithms repeatedly within cache
//       to show parallel scaling when higher bandwidth is available. This would be a good demonstration of parallel scaling of each algorithm with higher bandwidth availability.
// TODO: Demonstrate a nice cache effect on performance, where a small array of 1,000,000 elements which fits into cache, first time run is much slower than the rest of runs, with different
//       implementation winning in performance. However, as an algorithm or an array is used again and again, performance goes up substatially. Need to measure not only the first run time
//       but also the rest of run times to show this clearly by showing run time for each time use. Show that for large arrays running once versus running again and again, the times stay the same.
//       This complicates parallel algorithm usage for arrays that fit into the cache. It becomes not clear which algorithm is best to use: parallel or serial and when to use each.
//       This belongs in a blog entry of its own!
// TODO: Make sure to page-in the buffer, even for fill, before benchmarking. Didn't seem to make much difference on my laptop
// Conclusion: Not all parallel algorithms are advantageous when arrays fit into cache, with serial algorithms outperforming the parallel on first few runs only for some.

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  #define DPL_ALGORITHMS          // Includes Intel's OneAPI parallel algorithm implementations
  #define MICROSOFT_ALGORITHMS    // Excludes single-core SIMD implementations, which Microsoft does not support
#endif

#ifdef DPL_ALGORITHMS
// oneDPL headers should be included before standard headers
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#else
#include <algorithm>
#include <execution>
#include <iterator>
#include <chrono>
#endif

#include <iomanip>
#include <iostream>
#include <random>

#include <immintrin.h>

//#include <sycl/sycl.hpp>

//#include "FillParallel.h"
//#include "BinarySearch.h"
//#include "ParallelMerge.h"

//using namespace sycl;
using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

void print_results(const char* const tag, const vector<int>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  Lowest: %d Highest: %d Time: %fms\n", tag, in_array.size(), in_array.front(), in_array.back(),
        duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void print_results(const char* const tag, std::vector<int>::iterator result, const vector<int>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  Result: %d  Lowest: %d  Highest: %d  Time: %fms\n", tag, in_array.size(), *result,
        in_array.front(), in_array.back(), duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void print_results(const char* const tag, size_t result, const vector<int>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  Result: %zu  Lowest: %d  Highest: %d  Time: %fms\n", tag, in_array.size(), result,
        in_array.front(), in_array.back(), duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void print_results(const char* const tag, const vector<long long>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  Lowest: %lld Highest: %lld Time: %fms\n", tag, in_array.size(), in_array.front(), in_array.back(),
        duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void print_results(const char* const tag, const vector<double>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  %p  Lowest: %g Highest: %g Time: %fms\n", tag, in_array.size(), in_array.data(), in_array.front(), in_array.back(),
        duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void print_results(const char* const tag, const vector<size_t>& in_array,
    high_resolution_clock::time_point startTime,
    high_resolution_clock::time_point endTime)
{
    printf("%s: size = %zu  %p  Lowest: %zu Highest: %zu Time: %fms\n", tag, in_array.size(), in_array.data(), in_array.front(), in_array.back(),
        duration_cast<duration<double, milli>>(endTime - startTime).count());
}

void fill_scalar_around_cache(vector<int>& data, int value)
{
    int* p_data = data.data();

    for (size_t i = 0; i < data.size(); i++, p_data++)
    {
        _mm_stream_si32(p_data, value);
    }
}

void fill_scalar_around_cache_64(vector<int>& data, int value)
{
    long long* p_data = (long long *)data.data();
    size_t end = data.size() / 2;

    for (size_t i = 0; i < end; i++, p_data++)
    {
        _mm_stream_si64(p_data, (long long)value);
    }
}

void fill_scalar_around_cache(int* p_data, size_t l, size_t r, int value)
{
    int* r_data = p_data + (r - l);
    for (; p_data != r_data; p_data++)
    {
        _mm_stream_si32(p_data, value);
    }
}

void fill_benchmark(size_t array_size, size_t num_times)
{
    high_resolution_clock::time_point startTime, endTime;
    std::vector<int>       data(array_size);

    printf("\n\n");

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::seq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial std::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        fill_scalar_around_cache(data, 42);
        endTime = high_resolution_clock::now();
        print_results("fill_scalar_around_cache", data, startTime, endTime);
    }

    //for (size_t i = 0; i < num_times; i++)
    //{
    //    startTime = high_resolution_clock::now();
    //    //ParallelAlgorithms::parallel_fill(data.data(), 42, 0, data.size() - 1, data.size() / 8);
    //    ParallelAlgorithms::parallel_fill(data.data(), 42, 0, data.size() - 1, data.size() / 8);
    //    endTime = high_resolution_clock::now();
    //    print_results("parallel_fill", data, startTime, endTime);
    //}
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::fill", data, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::par, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel std::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::par_unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::fill", data, startTime, endTime);
    }

#ifdef DPL_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::seq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::par, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::fill", data, startTime, endTime);
    }

//    for (size_t i = 0; i < num_times; i++)
//    {
//        startTime = high_resolution_clock::now();
//        std::fill(oneapi::dpl::execution::dpcpp_default, data.begin(), data.end(), 42);
//        endTime = high_resolution_clock::now();
//        print_results("Parallel DPCPP_DEFAULT dpl::fill", data, startTime, endTime);
//    }
#endif
}

void fill_long_long_benchmark(size_t array_size, int num_times)
{
    high_resolution_clock::time_point startTime, endTime;
    std::vector<long long>       data(array_size);

    printf("\n\n");

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::seq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial std::fill", data, startTime, endTime);
    }

    //startTime = high_resolution_clock::now();
    //std::fill(std::execution::unseq, data.begin(), data.end(), 42);
    //endTime = high_resolution_clock::now();
    //print_results("SIMD Fill", data, startTime, endTime);

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::par, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel std::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(std::execution::par_unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::fill", data, startTime, endTime);
    }

#ifdef DPL_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::seq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::par, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::fill", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::fill", data, startTime, endTime);
    }

//    for (size_t i = 0; i < num_times; i++)
//    {
//        startTime = high_resolution_clock::now();
//        std::fill(oneapi::dpl::execution::dpcpp_default, data.begin(), data.end(), 42);
//        endTime = high_resolution_clock::now();
//        print_results("Parallel DPCPP_DEFAULT dpl::fill", data, startTime, endTime);
//    }
#endif
}

void sort_benchmark(size_t array_size, size_t num_times)
{
    //std::cout << "Size of int: " << sizeof(int) << std::endl;
    printf("\n\n");

    std::vector<int>       data(     array_size);
    std::vector<int>       data_copy(array_size);

    high_resolution_clock::time_point startTime, endTime;
    random_device rd;
    std::mt19937_64 dist(1234);

    for (auto& d : data) {
        //d = static_cast<int>(rd());
        d = static_cast<int>(dist());   // way faster on Linux
    }

    // std::sort benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(std::execution::seq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial std::sort", data, startTime, endTime);
    }

#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(std::execution::unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::sort", data, startTime, endTime);
    }
#endif

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(std::execution::par, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(std::execution::par_unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::sort", data, startTime, endTime);
    }

    // dpl::sort benchmarks

#ifdef DPL_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(oneapi::dpl::execution::seq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(oneapi::dpl::execution::unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(oneapi::dpl::execution::par, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        sort(oneapi::dpl::execution::par_unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::sort", data, startTime, endTime);
    }
#endif
}

void sort_doubles_benchmark(size_t array_size, int num_times, bool reuse_array)
{
    //std::cout << "Size of int: " << sizeof(int) << std::endl;
    printf("\n\n");

    vector<double> data(array_size);
    high_resolution_clock::time_point startTime, endTime;
    random_device rd;

    for (auto& d : data) {
        d = static_cast<double>(rd());
    }

    // std::sort benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::seq, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Serial std::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::seq, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Serial std::sort<double>", data_loc, startTime, endTime);
        }
    }

    //for (auto& d : data) {
    //    d = static_cast<int>(rd());
    //}

    //startTime = high_resolution_clock::now();
    //sort(std::execution::unseq, data.begin(), data.end());
    //endTime = high_resolution_clock::now();
    //print_results("SIMD std::sort", data, startTime, endTime);

    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::par, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel std::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::par, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel std::sort<double>", data_loc, startTime, endTime);
        }
    }

    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::par_unseq, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel SIMD std::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(std::execution::par_unseq, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel SIMD std::sort<double>", data_loc, startTime, endTime);
        }
    }

    // dpl::sort benchmarks

#ifdef DPL_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::seq, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Serial dpl::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::seq, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Serial dpl::sort<double>", data_loc, startTime, endTime);
        }
    }

    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::unseq, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Serial SIMD dpl::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::unseq, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Serial SIMD dpl::sort<double>", data_loc, startTime, endTime);
        }
    }


    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::par, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel dpl::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::par, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel dpl::sort<double>", data_loc, startTime, endTime);
        }
    }

    for (size_t i = 0; i < num_times; i++)
    {
        if (reuse_array)
        {
            for (auto& d : data) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::par_unseq, data.begin(), data.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel SIMD dpl::sort<double>", data, startTime, endTime);
        }
        else
        {
            vector<double> data_loc(array_size);
            for (auto& d : data_loc) {
                d = static_cast<double>(rd());
            }
            startTime = high_resolution_clock::now();
            sort(oneapi::dpl::execution::par_unseq, data_loc.begin(), data_loc.end());
            endTime = high_resolution_clock::now();
            print_results("Parallel SIMD dpl::sort<double>", data_loc, startTime, endTime);
        }
    }
#endif
}

void stable_sort_benchmark(size_t array_size, size_t num_times)
{
    vector<int> data(array_size);
    vector<int> data_copy(array_size);
    high_resolution_clock::time_point startTime, endTime;
    random_device rd;
    std::mt19937_64 dist(1234);

    for (auto& d : data) {
        //d = static_cast<int>(rd());
        d = static_cast<int>(dist());   // way faster on Linux
    }

    // std::stable_sort benchmarks
    printf("\n\n");

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(std::execution::seq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial std::stable_sort", data, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(std::execution::unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::stable_sort", data, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(std::execution::par, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::stable_sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(std::execution::par_unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::stable_sort", data, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(oneapi::dpl::execution::seq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::stable_sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(oneapi::dpl::execution::unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::stable_sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(oneapi::dpl::execution::par, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::stable_sort", data, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data.begin(), data.end(), data_copy.begin());

        startTime = high_resolution_clock::now();
        stable_sort(oneapi::dpl::execution::par_unseq, data_copy.begin(), data_copy.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::stable_sort", data, startTime, endTime);
    }
#endif
}

void merge_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src_0(array_size);
    std::vector<int>       data_int_src_1(array_size);
    std::vector<int>       data_int_dst(  2 * array_size, 1);   // initializate destination to page in and cache it
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;
    mt19937_64 dist(1234);

    printf("\n\n");

    for (auto& d : data_int_src_0) {
        //d = static_cast<int>(rd());
        d = static_cast<int>(dist());   // way faster on Linux
    }
    for (auto& d : data_int_src_1) {
        //d = static_cast<int>(rd());
        d = static_cast<int>(dist());   // way faster on Linux
    }

    sort(std::execution::par, data_int_src_0.begin(), data_int_src_0.end());
    sort(std::execution::par, data_int_src_1.begin(), data_int_src_1.end());

    // std::merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::seq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial std::merge", data_int_dst, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::merge", data_int_dst, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par_unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::merge", data_int_dst, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::seq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par_unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::merge", data_int_dst, startTime, endTime);
    }

#endif
}

void inplace_merge_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int> data_int( array_size * 2);
    std::vector<int> data_copy(array_size * 2);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;
    mt19937_64 dist(1234);

    for (auto& d : data_int) {
        //d = static_cast<int>(rd());
        d = static_cast<int>(dist());   // way faster on Linux
    }

    printf("\n\n");

    // std::inplace_merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(std::execution::seq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Serial std::inplace_merge", data_int, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(std::execution::unseq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::inplace_merge", data_int, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(std::execution::par, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::inplace_merge", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(std::execution::par_unseq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::inplace_merge", data_int, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(oneapi::dpl::execution::seq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::inplace_merge", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(oneapi::dpl::execution::unseq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::inplace_merge", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(oneapi::dpl::execution::par, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::inplace_merge", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        copy(std::execution::par, data_int.begin(), data_int.end(), data_copy.begin());

        sort(std::execution::par, data_copy.begin(), data_copy.begin() + data_copy.size() / 2);  // left  half
        sort(std::execution::par, data_copy.begin() + data_copy.size() / 2, data_copy.end());    // right half

        startTime = high_resolution_clock::now();
        inplace_merge(oneapi::dpl::execution::par_unseq, data_int.begin(), data_int.begin() + data_int.size() / 2, data_int.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::inplace_merge", data_int, startTime, endTime);
    }

#endif
}

void merge_dual_buffer_benchmark(size_t array_size, int num_times)
{
    std::vector<int>       data_int_src(2 * array_size);
    std::vector<int>       data_int_dst(2 * array_size);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    for (auto& d : data_int_src) {
        d = static_cast<int>(rd());
    }
    for (auto& d : data_int_dst) {
        d = static_cast<int>(rd());
    }

    sort(std::execution::par, data_int_src.begin(), data_int_src.begin() + array_size);
    sort(std::execution::par, data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size);
    sort(std::execution::par, data_int_dst.begin(), data_int_dst.begin() + 2 * array_size);

    // std::merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::seq, data_int_src.begin(),              data_int_src.begin() + array_size,
                                   data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial single array std::merge", data_int_dst, startTime, endTime);
    }

    //startTime = high_resolution_clock::now();
    //merge(std::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
    //endTime = high_resolution_clock::now();
    //print_results("SIMD std::merge", data_int_dst, startTime, endTime);

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par, data_int_src.begin(), data_int_src.begin() + array_size,
                                   data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel single array std::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par_unseq, data_int_src.begin(), data_int_src.begin() + array_size,
                                         data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::merge", data_int_dst, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::seq, data_int_src.begin(), data_int_src.begin() + array_size,
                                           data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::unseq, data_int_src.begin(), data_int_src.begin() + array_size,
                                             data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par, data_int_src.begin(), data_int_src.begin() + array_size,
                                           data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::merge", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par_unseq, data_int_src.begin(), data_int_src.begin() + array_size,
                                                 data_int_src.begin() + array_size, data_int_src.begin() + 2 * array_size, data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::merge", data_int_dst, startTime, endTime);
    }

    printf("done\n");

#endif

    //for (size_t i = 0; i < num_times; i++)
    //{
    //    startTime = high_resolution_clock::now();
    //    merge_parallel_L5(data_int_src.data(), 0, array_size - 1, array_size, 2 * array_size - 1, data_int_dst.data(), 0);
    //    endTime = high_resolution_clock::now();
    //    print_results("Parallel Victor's merge", data_int_dst, startTime, endTime);
    //}

}
void merge_single_buffer_benchmark(size_t array_size, int num_times)
{
    std::vector<int>       data_int_src_dst(4 * array_size);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    for (auto& d : data_int_src_dst) {
        d = static_cast<int>(rd());
    }

    sort(std::execution::par, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size);
    sort(std::execution::par, data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size);
    sort(std::execution::par, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.end());

    // std::merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::seq, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                   data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Serial single array std::merge", data_int_src_dst, startTime, endTime);
    }

    //startTime = high_resolution_clock::now();
    //merge(std::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin(), data_int_src_1.end(), data_int_dst.begin());
    //endTime = high_resolution_clock::now();
    //print_results("SIMD std::merge", data_int_dst, startTime, endTime);

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                   data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Parallel single array std::merge", data_int_src_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(std::execution::par_unseq, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                         data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::merge", data_int_src_dst, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::seq, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                           data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::merge", data_int_src_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::unseq, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                             data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::merge", data_int_src_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                           data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::merge", data_int_src_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        merge(oneapi::dpl::execution::par_unseq, data_int_src_dst.begin(), data_int_src_dst.begin() + array_size,
                                                 data_int_src_dst.begin() + array_size, data_int_src_dst.begin() + 2 * array_size, data_int_src_dst.begin() + 2 * array_size);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::merge", data_int_src_dst, startTime, endTime);
    }
#endif

    //for (size_t i = 0; i < num_times; i++)
    //{
    //    startTime = high_resolution_clock::now();
    //    merge_parallel_L5(data_int_src_dst.data(), 0, array_size - 1, array_size, 2 * array_size - 1, data_int_src_dst.data(), 2 * array_size);
    //    endTime = high_resolution_clock::now();
    //    print_results("Parallel Victor's merge", data_int_src_dst, startTime, endTime);
    //}
}

void all_of_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int> data_int(array_size, 2);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    // std::inplace_merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(std::execution::seq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Serial std::all_of", data_int, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(std::execution::unseq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::all_of", data_int, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(std::execution::par, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel std::all_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(std::execution::par_unseq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::all_of", data_int, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(oneapi::dpl::execution::seq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::all_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(oneapi::dpl::execution::unseq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::all_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(oneapi::dpl::execution::par, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::all_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (all_of(oneapi::dpl::execution::par_unseq, data_int.begin(), data_int.end(), [](int i) { return i == 2; }))
            printf("All numbers in the array are equal to 2\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::all_of", data_int, startTime, endTime);
    }
#endif
}

void any_of_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int> data_int(array_size, 2);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    // std::inplace_merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(std::execution::seq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Serial std::any_of", data_int, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(std::execution::unseq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::any_of", data_int, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(std::execution::par, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel std::any_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(std::execution::par_unseq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::any_of", data_int, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(oneapi::dpl::execution::seq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::any_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(oneapi::dpl::execution::unseq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::any_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(oneapi::dpl::execution::par, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::any_of", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (!any_of(oneapi::dpl::execution::par_unseq, data_int.begin(), data_int.end(), [](int i) { return i == 3; }))
            printf("No numbers in the array are equal to 3\n");
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::any_of", data_int, startTime, endTime);
    }
#endif
}

void copy_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src(array_size);
    std::vector<int>       data_int_dst(array_size);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    for (size_t i = 0; i < array_size; i++)
    {
        data_int_src[i] = (int)i;
    }
    //for (auto& d : data_int_src) {
    //    d = static_cast<int>(rd());
    //}

    printf("\n\n");

    // std::merge benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(std::execution::seq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial std::copy", data_int_dst, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(std::execution::unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::copy", data_int_dst, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(std::execution::par, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::copy", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(std::execution::par_unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::copy", data_int_dst, startTime, endTime);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(oneapi::dpl::execution::seq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::copy", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(oneapi::dpl::execution::unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::copy", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(oneapi::dpl::execution::par, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::copy", data_int_dst, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        copy(oneapi::dpl::execution::par_unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::copy", data_int_dst, startTime, endTime);
    }

#endif
}

void equal_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src_0(100000000, 0);
    std::vector<int>       data_int_src_1(100000000, 0);
    high_resolution_clock::time_point startTime, endTime;

    printf("\n\n");

    // std::equal benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(std::execution::seq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Serial std::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        if (equal(std::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin()))
            printf("Arrays are equal\n");
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::equal", data_int_src_0, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(std::execution::par, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Parallel std::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(std::execution::par_unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Parallel SIMD std::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }

    // dpl::stable_sort benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(oneapi::dpl::execution::seq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Serial dpl::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(oneapi::dpl::execution::unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("SIMD dpl::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(oneapi::dpl::execution::par, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Parallel dpl::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        bool equals = equal(oneapi::dpl::execution::par_unseq, data_int_src_0.begin(), data_int_src_0.end(), data_int_src_1.begin());
        endTime = high_resolution_clock::now();
        if (equals)
            print_results("Parallel SIMD dpl::equal", data_int_src_0, startTime, endTime);
        else
            exit(1);
    }
#endif
}


void count_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src(array_size);
    size_t                 num_items;
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\n\n");

    for (size_t i = 0; i < array_size; i++)
    {
        data_int_src[i] = (int)i;
    }
    //for (auto& d : data_int_src) {
    //    d = static_cast<int>(rd());
    //}

    // std::count benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(std::execution::seq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial std::count", num_items, data_int_src, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(std::execution::unseq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::count", num_items, data_int_src, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(std::execution::par, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel std::count", num_items, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(std::execution::par_unseq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::count", num_items, data_int_src, startTime, endTime);
    }

    // dpl::count benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(oneapi::dpl::execution::seq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::count", num_items, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(oneapi::dpl::execution::unseq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::count", num_items, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(oneapi::dpl::execution::par, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::count", num_items, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        num_items = count(oneapi::dpl::execution::par_unseq, data_int_src.begin(), data_int_src.end(), 42);
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::count", num_items, data_int_src, startTime, endTime);
    }

#endif
}

void adjacent_find_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int(array_size, 2);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\nAdjacent Find\n");

    for (size_t i = 0; i < array_size; i++) // force all adjacent elements to be different - i.e. no matching pairs
    {
        if (i % 2 == 0)
            data_int[i] = 3;
    }

    // std::adjacent_find benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find (std::execution::seq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Serial std::adjacent_find", data_int, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(std::execution::unseq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Serial SIMD std::adjacent_find", data_int, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(std::execution::par, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Parallel std::adjacent_find", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(std::execution::par_unseq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Parallel SIMD std::adjacent_find", data_int, startTime, endTime);
    }
    // dpl::adjacent_find benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(oneapi::dpl::execution::seq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Serial dpl::adjacent_find", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(oneapi::dpl::execution::unseq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("SIMD dpl::adjacent_find", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(oneapi::dpl::execution::par, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Parallel dpl::adjacent_find", data_int, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        auto iequal = adjacent_find(oneapi::dpl::execution::par_unseq, data_int.begin(), data_int.end());
        endTime = high_resolution_clock::now();
        if (iequal == data_int.end()) printf("No equal pairs found\n");
        print_results("Parallel SIMD dpl::adjacent_find", data_int, startTime, endTime);
    }
#endif
}

void adjacent_difference_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src(array_size);
    std::vector<int>       data_int_dst(array_size, 10);
    high_resolution_clock::time_point startTime, endTime;

    random_device rd;

    printf("\nAdjacent Difference\n");

    for (size_t i = 0; i < array_size; i++)
    {
        data_int_src[i] = (int)i;
    }
    //for (auto& d : data_int_src) {
    //    d = static_cast<int>(rd());
    //}

    // std::adjacent_difference benchmarks
    printf("Benchmarks:\n");

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(std::execution::seq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial std::adjacent_difference", data_int_src, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(std::execution::unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::adjacent_difference", data_int_src, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(std::execution::par, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::adjacent_difference", data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(std::execution::par_unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::adjacent_difference", data_int_src, startTime, endTime);
    }
#if 0
    // dpl::adjacent_difference benchmarks (Intel doesn't implement!)
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(oneapi::dpl::execution::seq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::adjacent_difference", data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(oneapi::dpl::execution::unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::adjacent_difference", data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(oneapi::dpl::execution::par, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::adjacent_difference", data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        adjacent_difference(oneapi::dpl::execution::par_unseq, data_int_src.begin(), data_int_src.end(), data_int_dst.begin());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::adjacent_difference", data_int_src, startTime, endTime);
    }

#endif
#endif
}

void max_element_benchmark(size_t array_size, size_t num_times)
{
    std::vector<int>       data_int_src(array_size);
    high_resolution_clock::time_point startTime, endTime;
    std::vector<int>::iterator max_index;
    random_device rd;

    printf("\n\n");

    for (size_t i = 0; i < array_size; i++)
    {
        data_int_src[i] = (int)i;
    }
    //for (auto& d : data_int_src) {
    //    d = static_cast<int>(rd());
    //}
#if 1
    // std::max_element benchmarks

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(std::execution::seq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Serial std::max_element", max_index, data_int_src, startTime, endTime);
    }
#ifndef MICROSOFT_ALGORITHMS
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(std::execution::unseq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Serial SIMD std::max_element", max_index, data_int_src, startTime, endTime);
    }
#endif
    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(std::execution::par, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel std::max_element", max_index, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(std::execution::par_unseq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD std::max_element", max_index, data_int_src, startTime, endTime);
    }
#endif
    // dpl::max_element benchmarks
#ifdef DPL_ALGORITHMS

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(oneapi::dpl::execution::seq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Serial dpl::max_element", max_index, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(oneapi::dpl::execution::unseq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("SIMD dpl::max_element", max_index, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(oneapi::dpl::execution::par, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel dpl::max_element", max_index, data_int_src, startTime, endTime);
    }

    for (size_t i = 0; i < num_times; i++)
    {
        startTime = high_resolution_clock::now();
        max_index = max_element(oneapi::dpl::execution::par_unseq, data_int_src.begin(), data_int_src.end());
        endTime = high_resolution_clock::now();
        print_results("Parallel SIMD dpl::max_element", max_index, data_int_src, startTime, endTime);
    }

#endif
}


int main()
{
    size_t array_size = 100'000'000;
    size_t number_of_tests = 5;

    max_element_benchmark(        array_size, number_of_tests);   // for small arrays parallel implementation is much slower than serial
    adjacent_difference_benchmark(array_size, number_of_tests);   // for small arrays parallel implementation is much slower than serial
    adjacent_find_benchmark(      array_size, number_of_tests);   // for small arrays parallel implementation is much slower than serial
    all_of_benchmark(             array_size, number_of_tests);
    any_of_benchmark(             array_size, number_of_tests);
    count_benchmark(              array_size, number_of_tests);
    //count_benchmark(10000, 20);                 // for small arrays parallel implementations are much slower than serial
    equal_benchmark(              array_size, number_of_tests);
    copy_benchmark(               array_size, number_of_tests);
    //copy_benchmark(                   10000, 10);   // for small arrays parallel implementation is much slower than serial
    fill_benchmark(array_size, number_of_tests);
    merge_benchmark(              array_size, number_of_tests);
    inplace_merge_benchmark(      array_size, number_of_tests);
    //merge_dual_buffer_benchmark(  100000000, 10);
    //merge_single_buffer_benchmark(    10000, 10);
    //fill_long_long_benchmark(     100000000, 10);
    sort_benchmark(               array_size, number_of_tests);
    //sort_doubles_benchmark(         100000000, 10, true );
    //sort_doubles_benchmark(       100000000, 10, false);
    stable_sort_benchmark(        array_size, number_of_tests);

    return 0;
}
