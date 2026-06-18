// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/common/broadcast_ring.hpp"
#include "tt_metal/distributed/realtime_profiler_manager.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"

namespace tt::tt_metal {
namespace {

using tt::ProgramRealtimeRecord;
using tt::ProgramRealtimeRecordBatch;

ProgramRealtimeRecord make_record(uint32_t runtime_id, std::span<const std::string_view> sources = {}) {
    return ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = 7,
        .start_timestamp = 100 + runtime_id,
        .end_timestamp = 200 + runtime_id,
        .frequency = 1.5,
        .kernel_sources = sources,
    };
}

TEST(RealtimeProfilerCallbackRing, BroadcastRingCarriesProgramRealtimeRecordBatches) {
    std::string_view sources[] = {"kernel_a.cpp", "kernel_b.cpp"};
    BroadcastRing<ProgramRealtimeRecord> ring(8);
    auto reader_a = ring.make_reader();
    auto reader_b = ring.make_reader();

    ProgramRealtimeRecord records[] = {
        make_record(1, sources),
        make_record(2, sources),
        make_record(3, sources),
    };
    ring.writer().publish_batch(records);

    ProgramRealtimeRecord out_a[3];
    ProgramRealtimeRecord out_b[3];
    auto batch_a = reader_a.read_batch(out_a);
    auto batch_b = reader_b.read_batch(out_b);

    ASSERT_EQ(batch_a.size(), 3u);
    ASSERT_EQ(batch_b.size(), 3u);
    for (size_t i = 0; i < batch_a.size(); ++i) {
        const auto& record_a = batch_a[i];
        const auto& record_b = batch_b[i];
        EXPECT_EQ(record_a.runtime_id, i + 1);
        EXPECT_EQ(record_b.runtime_id, i + 1);
        ASSERT_EQ(record_a.kernel_sources.size(), 2u);
        EXPECT_EQ(record_a.kernel_sources[0], "kernel_a.cpp");
        EXPECT_EQ(record_a.kernel_sources[1], "kernel_b.cpp");
    }
}

TEST(RealtimeProfilerCallbackRing, SlowReaderDropsOldProgramRealtimeRecords) {
    BroadcastRing<ProgramRealtimeRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<ProgramRealtimeRecord> records;
    records.reserve(10);
    for (uint32_t i = 0; i < 10; ++i) {
        records.push_back(make_record(i));
    }
    ring.writer().publish_batch(records);

    ProgramRealtimeRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 6u);
    EXPECT_EQ(batch.front().runtime_id, 6u);
    EXPECT_EQ(batch.back().runtime_id, 9u);
}

TEST(RealtimeProfilerCallbackRing, BroadcastRingSupportsNonTrivialRecords) {
    struct NonTrivialRecord {
        uint32_t runtime_id = 0;
        std::string name;
    };
    static_assert(!BroadcastRing<NonTrivialRecord>::is_always_lock_free);

    BroadcastRing<NonTrivialRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<NonTrivialRecord> records;
    records.reserve(7);
    for (uint32_t i = 0; i < 7; ++i) {
        records.push_back({.runtime_id = i, .name = "kernel_" + std::to_string(i)});
    }
    ring.writer().publish_batch(records);

    NonTrivialRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 3u);
    EXPECT_EQ(batch.front().runtime_id, 3u);
    EXPECT_EQ(batch.front().name, "kernel_3");
    EXPECT_EQ(batch.back().runtime_id, 6u);
    EXPECT_EQ(batch.back().name, "kernel_6");
}

TEST(RealtimeProfilerCallbackRing, DataCollectorNotifiesCallbackListeners) {
    struct Listener : tt::RealtimeProfilerCallbackListener {
        void on_callback_registered(
            ProgramRealtimeProfilerCallbackHandle handle, const ProgramRealtimeProfilerCallback& callback) override {
            added_handle = handle;
            callback(ProgramRealtimeRecordBatch{std::span<const ProgramRealtimeRecord>(&record, 1), 0});
            add_count++;
        }
        void on_callback_unregistered(ProgramRealtimeProfilerCallbackHandle handle) override {
            removed_handle = handle;
            remove_count++;
        }

        ProgramRealtimeRecord record = make_record(7);
        ProgramRealtimeProfilerCallbackHandle added_handle = 0;
        ProgramRealtimeProfilerCallbackHandle removed_handle = 0;
        uint32_t add_count = 0;
        uint32_t remove_count = 0;
    };

    DataCollector collector;
    std::atomic<uint64_t> received{0};
    std::atomic<uint64_t> runtime_sum{0};
    Listener listener;
    collector.AttachRealtimeProfilerCallbackListener(&listener);

    auto handle = collector.RegisterProgramRealtimeProfilerCallback(
        [&received, &runtime_sum](const ProgramRealtimeRecordBatch& batch) {
            uint64_t local_sum = 0;
            for (const auto& record : batch.records) {
                local_sum += record.runtime_id;
            }
            runtime_sum.fetch_add(local_sum, std::memory_order_relaxed);
            received.fetch_add(batch.records.size(), std::memory_order_release);
        });

    EXPECT_EQ(listener.add_count, 1u);
    EXPECT_EQ(listener.added_handle, handle);
    EXPECT_EQ(received.load(std::memory_order_acquire), 1u);
    EXPECT_EQ(runtime_sum.load(std::memory_order_relaxed), 7u);
    collector.UnregisterProgramRealtimeProfilerCallback(handle);
    EXPECT_EQ(listener.remove_count, 1u);
    EXPECT_EQ(listener.removed_handle, handle);
    collector.DetachRealtimeProfilerCallbackListener(&listener);
}

}  // namespace
}  // namespace tt::tt_metal
