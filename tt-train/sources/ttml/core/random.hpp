// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "random_sse.hpp"

namespace ttml::core {
namespace legacy {

template <typename T, typename DistGenFunc>
void sequential_generate(std::span<T> seq, const DistGenFunc& dist_factory, uint32_t seed) {
    auto dist = dist_factory();
    auto rng = std::mt19937{seed};
    std::generate(seq.begin(), seq.end(), [&dist, &rng]() { return dist(rng); });
}

template <typename T, typename DistGenFunc>
void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) {
    constexpr size_t min_size = 1 << 12;  // determined empirically that this is where we see an advantage over
                                          // sequential generation even with 2 threads.
    if (seq.size() < min_size) {
        sequential_generate(seq, dist_factory, seed);
        return;
    }

    // Fixed chunk size independent of thread count: seed is per-chunk so output
    // is identical regardless of how many threads process those chunks.
    static constexpr size_t CHUNK_SIZE = 512 * 512;
    const size_t num_threads =
        std::min(static_cast<size_t>(max_threads), static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t num_chunks = (seq.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const size_t actual_threads = std::min(num_threads, num_chunks);
    const size_t chunks_per_thread = num_chunks / actual_threads;
    const size_t leftover = num_chunks % actual_threads;

    std::vector<std::jthread> threads;
    threads.reserve(actual_threads);

    size_t start_chunk = 0;
    for (size_t t = 0; t < actual_threads; ++t) {
        const size_t end_chunk = start_chunk + chunks_per_thread + (t < leftover ? 1 : 0);
        threads.emplace_back([seq, start_chunk, end_chunk, seed, dist_factory]() {
            for (size_t chunk = start_chunk; chunk < end_chunk; ++chunk) {
                const size_t offset = chunk * CHUNK_SIZE;
                const size_t size = std::min(CHUNK_SIZE, seq.size() - offset);
                sequential_generate(seq.subspan(offset, size), dist_factory, seed + static_cast<uint32_t>(chunk));
            }
        });
        start_chunk = end_chunk;
    }
}

}  // namespace legacy

static inline bool use_simd_rng() {
    constexpr auto ENABLE_SIMD_RNG = "TT_TRAIN_ENABLE_SIMD_RNG";
    static bool simd_enabled = (std::getenv(ENABLE_SIMD_RNG) != nullptr);

    return simd_enabled;
}

template <typename T, typename DistGenFunc>
void sequential_generate(std::span<T> seq, const DistGenFunc& dist_factory, uint32_t seed) {
    if (use_simd_rng()) {
        return sse::sequential_generate(seq, dist_factory, seed);
    }
    return legacy::sequential_generate(seq, dist_factory, seed);
}

template <typename T, typename DistGenFunc>
void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) {
    if (use_simd_rng()) {
        return sse::parallel_generate(seq, dist_factory, seed, max_threads);
    }
    return legacy::parallel_generate(seq, dist_factory, seed, max_threads);
}

}  // namespace ttml::core
