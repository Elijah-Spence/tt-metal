// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests that ttml::core parallel RNG produces thread-count-independent output.
//
// Bug being diagnosed: seeds are assigned per-thread (thread_id << 32) rather
// than per-data-position.  Changing the thread count changes which thread owns
// which data positions, so element values differ across runs.  These tests will
// FAIL on the unfixed implementation and PASS after the fix.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "core/random.hpp"
#include "core/random_sse.hpp"

namespace {

constexpr size_t kSize = 1u << 20;  // 1M floats — large enough to span multiple chunks
constexpr uint32_t kSeed = 42;
constexpr size_t kMaxReportedMismatches = 10;

std::vector<float> run_sse_parallel(size_t size, uint32_t seed, uint32_t num_threads) {
    std::vector<float> out(size);
    ttml::core::sse::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed,
        num_threads);
    return out;
}

std::vector<float> run_legacy_parallel(size_t size, uint32_t seed, uint32_t num_threads) {
    std::vector<float> out(size);
    ttml::core::legacy::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); },
        seed,
        num_threads);
    return out;
}

// Returns total mismatch count; prints the first max_report of them with positions.
size_t compare_and_report_mismatches(
    const std::vector<float>& golden, const std::vector<float>& actual, size_t max_report = kMaxReportedMismatches) {
    size_t mismatches = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        if (golden[i] != actual[i]) {
            if (mismatches < max_report) {
                std::cout << "  position " << i << ": golden=" << golden[i] << "  actual=" << actual[i] << "\n";
            }
            ++mismatches;
        }
    }
    if (mismatches > max_report) {
        std::cout << "  ... (" << max_report << " shown of " << mismatches << " total mismatches)\n";
    }
    return mismatches;
}

}  // namespace

// ============================================================================
// Sanity: same seed + same thread count must always produce identical output
// ============================================================================

TEST(ParallelRngDeterminism, SseRepeatability) {
    auto v1 = run_sse_parallel(kSize, kSeed, 4);
    auto v2 = run_sse_parallel(kSize, kSeed, 4);
    EXPECT_EQ(v1, v2) << "SSE parallel generate with the same seed and thread count must be identical.";
}

TEST(ParallelRngDeterminism, LegacyRepeatability) {
    auto v1 = run_legacy_parallel(kSize, kSeed, 4);
    auto v2 = run_legacy_parallel(kSize, kSeed, 4);
    EXPECT_EQ(v1, v2) << "Legacy parallel generate with the same seed and thread count must be identical.";
}

// ============================================================================
// Parameterized: varying thread count must NOT change per-element values
// ============================================================================

class ParallelRngVsThreadCount : public ::testing::TestWithParam<uint32_t> {};

TEST_P(ParallelRngVsThreadCount, SseMatchesOneThread) {
    const uint32_t num_threads = GetParam();
    auto golden = run_sse_parallel(kSize, kSeed, 1);
    auto actual = run_sse_parallel(kSize, kSeed, num_threads);

    [[maybe_unused]] size_t mismatches = compare_and_report_mismatches(golden, actual);
    EXPECT_EQ(mismatches, 0u) << "SSE parallel_generate with " << num_threads << " threads produced " << mismatches
                              << " element(s) that differ from the 1-thread reference.";
}

TEST_P(ParallelRngVsThreadCount, LegacyMatchesOneThread) {
    const uint32_t num_threads = GetParam();
    auto golden = run_legacy_parallel(kSize, kSeed, 1);
    auto actual = run_legacy_parallel(kSize, kSeed, num_threads);

    size_t mismatches = compare_and_report_mismatches(golden, actual);
    EXPECT_EQ(mismatches, 0u) << "Legacy parallel_generate with " << num_threads << " threads produced " << mismatches
                              << " element(s) that differ from the 1-thread reference.";
}

INSTANTIATE_TEST_SUITE_P(
    ThreadCounts,
    ParallelRngVsThreadCount,
    ::testing::Values(2u, 4u, 8u, 16u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return std::to_string(info.param) + "threads"; });
