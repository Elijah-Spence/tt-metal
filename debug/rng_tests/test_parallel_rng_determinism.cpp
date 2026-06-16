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

// ============================================================================
// Statistical correctness: mean and variance, full-tensor and per-chunk
// ============================================================================

namespace {

// Uniform[-1, 1]: E[X] = 0, Var[X] = (b-a)^2/12 = 4/12 = 1/3
constexpr float kUniformMin = -1.0f;
constexpr float kUniformMax = 1.0f;
constexpr double kExpectedMean = 0.0;
constexpr double kExpectedVariance = (kUniformMax - kUniformMin) * (kUniformMax - kUniformMin) / 12.0;
constexpr double kAbsTol = 0.01;
constexpr double kRelTol = 0.01;
// Must match the RNG implementation's CHUNK_SIZE so per-chunk stats test exact chunk boundaries.
constexpr size_t kRngChunkSize = 512 * 512;

double stat_tolerance(double expected) { return std::max(kAbsTol, kRelTol * std::abs(expected)); }

std::pair<double, double> compute_mean_variance(std::span<const float> data) {
    double sum = 0.0, sum_sq = 0.0;
    for (float v : data) {
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    const double mean = sum / static_cast<double>(data.size());
    const double variance = sum_sq / static_cast<double>(data.size()) - mean * mean;
    return {mean, variance};
}

void check_stats(std::span<const float> data, const std::string& location) {
    const auto [mean, variance] = compute_mean_variance(data);
    EXPECT_NEAR(mean, kExpectedMean, stat_tolerance(kExpectedMean))
        << location << ": mean out of tolerance (n=" << data.size() << ")";
    EXPECT_NEAR(variance, kExpectedVariance, stat_tolerance(kExpectedVariance))
        << location << ": variance out of tolerance (n=" << data.size() << ")";
}

struct SizeParam {
    size_t rows;
    size_t cols;
};

}  // namespace

class UniformDistributionStats : public ::testing::TestWithParam<SizeParam> {};

TEST_P(UniformDistributionStats, SseStatisticsValid) {
    const auto [rows, cols] = GetParam();
    const size_t total = rows * cols;

    std::vector<float> out(total);
    ttml::core::sse::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::uniform_real_distribution<float>(kUniformMin, kUniformMax); },
        kSeed);

    const std::span<const float> all{out.data(), out.size()};
    const size_t num_chunks = (total + kRngChunkSize - 1) / kRngChunkSize;

    check_stats(all, "SSE full-tensor");

    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t offset = c * kRngChunkSize;
        const size_t size = std::min(kRngChunkSize, total - offset);
        check_stats(all.subspan(offset, size), "SSE chunk[" + std::to_string(c) + "]");
    }
}

TEST_P(UniformDistributionStats, LegacyStatisticsValid) {
    const auto [rows, cols] = GetParam();
    const size_t total = rows * cols;

    std::vector<float> out(total);
    ttml::core::legacy::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::uniform_real_distribution<float>(kUniformMin, kUniformMax); },
        kSeed);

    const std::span<const float> all{out.data(), out.size()};
    const size_t num_chunks = (total + kRngChunkSize - 1) / kRngChunkSize;

    check_stats(all, "Legacy full-tensor");

    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t offset = c * kRngChunkSize;
        const size_t size = std::min(kRngChunkSize, total - offset);
        check_stats(all.subspan(offset, size), "Legacy chunk[" + std::to_string(c) + "]");
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    UniformDistributionStats,
    ::testing::Values(
        SizeParam{512, 512},    // 262144 elements — exactly 1 chunk
        SizeParam{1024, 1024},  // 1048576 elements — exactly 4 chunks
        SizeParam{249, 1493},   // 371757 elements — 2 chunks, last one partial (109613 elems)
        SizeParam{2048, 2048},  // 4194304 elements — exactly 16 chunks
        SizeParam{4096, 4096},  // 16777216 elements — exactly 64 chunks
        SizeParam{4608, 4096}   // 18874368 elements — 72 chunks; on 16 cores: 8×5 + 8×4
        ),
    [](const ::testing::TestParamInfo<SizeParam>& info) {
        return std::to_string(info.param.rows) + "x" + std::to_string(info.param.cols);
    });

// ============================================================================
// Statistical correctness: mean and variance for Normal distribution
// ============================================================================

namespace {

// Normal(0, 1): E[X] = 0, Var[X] = 1
constexpr float kNormalMean = 0.0f;
constexpr float kNormalStddev = 1.0f;
constexpr double kExpectedNormalMean = 0.0;
constexpr double kExpectedNormalVariance = 1.0;

}  // namespace

class NormalDistributionStats : public ::testing::TestWithParam<SizeParam> {};

TEST_P(NormalDistributionStats, SseStatisticsValid) {
    const auto [rows, cols] = GetParam();
    const size_t total = rows * cols;

    std::vector<float> out(total);
    ttml::core::sse::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::normal_distribution<float>(kNormalMean, kNormalStddev); },
        kSeed);

    const std::span<const float> all{out.data(), out.size()};
    const size_t num_chunks = (total + kRngChunkSize - 1) / kRngChunkSize;

    {
        const auto [mean, variance] = compute_mean_variance(all);
        EXPECT_NEAR(mean, kExpectedNormalMean, stat_tolerance(kExpectedNormalMean))
            << "SSE full-tensor: mean out of tolerance (n=" << total << ")";
        EXPECT_NEAR(variance, kExpectedNormalVariance, stat_tolerance(kExpectedNormalVariance))
            << "SSE full-tensor: variance out of tolerance (n=" << total << ")";
    }

    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t offset = c * kRngChunkSize;
        const size_t size = std::min(kRngChunkSize, total - offset);
        const auto chunk = all.subspan(offset, size);
        const auto [mean, variance] = compute_mean_variance(chunk);
        EXPECT_NEAR(mean, kExpectedNormalMean, stat_tolerance(kExpectedNormalMean))
            << "SSE chunk[" << c << "]: mean out of tolerance (n=" << size << ")";
        EXPECT_NEAR(variance, kExpectedNormalVariance, stat_tolerance(kExpectedNormalVariance))
            << "SSE chunk[" << c << "]: variance out of tolerance (n=" << size << ")";
    }
}

TEST_P(NormalDistributionStats, LegacyStatisticsValid) {
    const auto [rows, cols] = GetParam();
    const size_t total = rows * cols;

    std::vector<float> out(total);
    ttml::core::legacy::parallel_generate(
        std::span{out.data(), out.size()},
        []() { return std::normal_distribution<float>(kNormalMean, kNormalStddev); },
        kSeed);

    const std::span<const float> all{out.data(), out.size()};
    const size_t num_chunks = (total + kRngChunkSize - 1) / kRngChunkSize;

    {
        const auto [mean, variance] = compute_mean_variance(all);
        EXPECT_NEAR(mean, kExpectedNormalMean, stat_tolerance(kExpectedNormalMean))
            << "Legacy full-tensor: mean out of tolerance (n=" << total << ")";
        EXPECT_NEAR(variance, kExpectedNormalVariance, stat_tolerance(kExpectedNormalVariance))
            << "Legacy full-tensor: variance out of tolerance (n=" << total << ")";
    }

    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t offset = c * kRngChunkSize;
        const size_t size = std::min(kRngChunkSize, total - offset);
        const auto chunk = all.subspan(offset, size);
        const auto [mean, variance] = compute_mean_variance(chunk);
        EXPECT_NEAR(mean, kExpectedNormalMean, stat_tolerance(kExpectedNormalMean))
            << "Legacy chunk[" << c << "]: mean out of tolerance (n=" << size << ")";
        EXPECT_NEAR(variance, kExpectedNormalVariance, stat_tolerance(kExpectedNormalVariance))
            << "Legacy chunk[" << c << "]: variance out of tolerance (n=" << size << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    NormalDistributionStats,
    ::testing::Values(
        SizeParam{512, 512},    // 262144 elements — exactly 1 chunk
        SizeParam{1024, 1024},  // 1048576 elements — exactly 4 chunks
        SizeParam{249, 1493},   // 371757 elements — 2 chunks, last one partial (109613 elems)
        SizeParam{2048, 2048},  // 4194304 elements — exactly 16 chunks
        SizeParam{4096, 4096},  // 16777216 elements — exactly 64 chunks
        SizeParam{4608, 4096}   // 18874368 elements — 72 chunks; on 16 cores: 8×5 + 8×4
        ),
    [](const ::testing::TestParamInfo<SizeParam>& info) {
        return std::to_string(info.param.rows) + "x" + std::to_string(info.param.cols);
    });
