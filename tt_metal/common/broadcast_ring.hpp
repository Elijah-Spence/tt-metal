// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>

namespace tt::tt_metal {

// Single-producer, multi-consumer broadcast ring: every reader receives the whole stream.
// A reader that falls behind loses the oldest unread items (Reader::dropped()).
// Trivially copyable payloads use the atomic byte-copy fast path; other payloads use locked slots.
template <typename T>
class BroadcastRing {
    static constexpr bool kTriviallyCopyable = std::is_trivially_copyable_v<T>;
    static constexpr bool kAtomicWordAlwaysLockFree = std::atomic<uint64_t>::is_always_lock_free;

    static constexpr size_t kFalseSharingSize = 128;
    static constexpr size_t kSlotWordCount = (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    static constexpr size_t kPayloadWordCount = sizeof(T) / sizeof(uint64_t);
    static constexpr size_t kPayloadTailByteCount = sizeof(T) % sizeof(uint64_t);

    struct AtomicSlot final {
        std::array<std::atomic<uint64_t>, kSlotWordCount> words;
    };

    struct LockedSlot final {
        mutable std::mutex mutex;
        std::optional<T> value;
    };

    using Slot = std::conditional_t<kTriviallyCopyable, AtomicSlot, LockedSlot>;
    struct RingView;
    struct Counters;

public:
    static constexpr bool is_always_lock_free = kTriviallyCopyable && kAtomicWordAlwaysLockFree;

    // capacity is rounded up to the next power of two.
    explicit BroadcastRing(size_t capacity) :
        capacity_(std::bit_ceil(capacity)), slots_(std::make_unique<Slot[]>(capacity_)), writer_(&counters_, view()) {}

    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

    class alignas(kFalseSharingSize) Writer {
    public:
        void publish(const T& item) noexcept { publish_batch({&item, 1}); }

        // Publish items in order. A batch larger than capacity() keeps only its last capacity()
        // items.
        void publish_batch(std::span<const T> items) noexcept {
            const size_t n = items.size();
            const uint64_t head = head_cache_;
            counters_->claim.store(head + n, std::memory_order_relaxed);
            // publish `claim` before the payloads so a reader can detect a torn copy
            std::atomic_thread_fence(std::memory_order_release);
            const RingView view = view_;
            const size_t skip = n > view.capacity ? n - view.capacity : 0;
            for (size_t k = skip; k < n; k++) {
                store_payload(view.slot_at(head + k), items[k]);
            }
            counters_->head.store(head + n, std::memory_order_release);
            head_cache_ = head + n;
        }

        // Publish n items produced by make_item(k) for k in [0, n), in order, without staging a
        // temporary. make_item is called for every k even when n exceeds capacity().
        template <typename MakeItem>
        void publish_batch(size_t n, MakeItem make_item) noexcept {
            const uint64_t head = head_cache_;
            counters_->claim.store(head + n, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            const RingView view = view_;
            // Advance the generator over every index so a stateful make_item stays in step, but
            // store only the last capacity() items when the batch overflows the ring.
            const size_t skip = n > view.capacity ? n - view.capacity : 0;
            for (size_t k = 0; k < n; k++) {
                const T item = make_item(k);
                if (k >= skip) {
                    store_payload(view.slot_at(head + k), item);
                }
            }
            counters_->head.store(head + n, std::memory_order_release);
            head_cache_ = head + n;
        }

        Writer(const Writer&) = delete;
        Writer& operator=(const Writer&) = delete;
        Writer(Writer&&) = delete;
        Writer& operator=(Writer&&) = delete;

    private:
        friend class BroadcastRing;
        Writer(Counters* counters, RingView view) noexcept :
            counters_(counters), view_(view), head_cache_(counters->head.load(std::memory_order_relaxed)) {}

        Counters* counters_;
        RingView view_;
        uint64_t head_cache_;
    };

    // The single writer handle; use from one thread only.
    [[nodiscard]] Writer& writer() noexcept { return writer_; }

    class alignas(kFalseSharingSize) Reader {
    public:
        // Read the next available items into out (oldest first); returns the filled prefix, empty
        // when caught up. A reader that fell too far behind skips the oldest items (see dropped()).
        [[nodiscard]] std::span<T> read_batch(std::span<T> out) noexcept {
            if (out.empty()) {
                return out;
            }
            while (true) {
                const uint64_t head = counters_->head.load(std::memory_order_acquire);
                if (cursor_ >= head) {
                    return out.first(0);
                }
                const RingView view = view_;
                const uint64_t cap = view.capacity;
                const uint64_t want = std::min<uint64_t>(out.size(), cap);
                // jump past unread items we'd likely discard after a claim race
                const uint64_t keep = std::max(want, cap - std::max(want, cap >> 3));
                if (head - cursor_ > keep) {
                    dropped_ += (head - cursor_) - keep;
                    cursor_ = head - keep;
                }
                const uint64_t start = cursor_;
                const size_t n = std::min<uint64_t>(out.size(), head - start);
                for (size_t k = 0; k < n; k++) {
                    load_payload(view.slot_at(start + k), out[k]);
                }
                // re-read `claim` after copying (the acquire fence keeps the loads ahead of it);
                // if `claim` got more than `cap` past `start`, the writer overwrote what we read
                std::atomic_thread_fence(std::memory_order_acquire);
                const uint64_t claim = counters_->claim.load(std::memory_order_relaxed);
                if (claim - start > cap) {
                    const uint64_t oldest = claim - cap;
                    const uint64_t lost = oldest - start;
                    dropped_ += lost;
                    if (lost >= n) {
                        cursor_ = oldest;
                        continue;
                    }
                    const size_t valid = n - lost;
                    if constexpr (kTriviallyCopyable) {
                        std::memmove(out.data(), out.data() + lost, valid * sizeof(T));
                    } else {
                        for (size_t k = 0; k < valid; ++k) {
                            out[k] = out[k + lost];
                        }
                    }
                    cursor_ = start + n;
                    return out.first(valid);
                }
                cursor_ = start + n;
                return out.first(n);
            }
        }

        // Read a single item; returns false when caught up.
        [[nodiscard]] bool read(T& out) noexcept { return !read_batch({&out, 1}).empty(); }

        // Items this reader skipped after lagging too far behind; updated only during read_batch().
        [[nodiscard]] uint64_t dropped() const noexcept { return dropped_; }

        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;
        Reader(Reader&&) = default;
        Reader& operator=(Reader&&) = default;

    private:
        friend class BroadcastRing;
        Reader(const Counters* counters, RingView view, uint64_t start) noexcept :
            counters_(counters), view_(view), cursor_(start) {}

        const Counters* counters_;
        RingView view_;
        uint64_t cursor_;
        uint64_t dropped_ = 0;
    };

    // Create a reader positioned at the current end of the stream: it sees only items published
    // after this call. Each reader is single-threaded and must not outlive the ring.
    [[nodiscard]] Reader make_reader() const noexcept {
        return Reader(&counters_, view(), counters_.head.load(std::memory_order_acquire));
    }

private:
    struct RingView {
        Slot* slots;
        size_t capacity;
        Slot& slot_at(uint64_t position) const noexcept { return slots[position & (capacity - 1)]; }
    };

    struct alignas(kFalseSharingSize) Counters {
        std::atomic<uint64_t> head{0};   // count of fully written items a reader may consume
        std::atomic<uint64_t> claim{0};  // count the writer has started writing; always >= `head`
    };

    static void store_payload(Slot& s, const T& v) noexcept {
        if constexpr (!kTriviallyCopyable) {
            std::lock_guard lock(s.mutex);
            s.value = v;
        } else {
            const std::byte* src = reinterpret_cast<const std::byte*>(&v);
#pragma GCC unroll 8
            for (size_t k = 0; k < kPayloadWordCount; k++) {
                uint64_t w;
                std::memcpy(&w, src + k * sizeof(uint64_t), sizeof(uint64_t));
                s.words[k].store(w, std::memory_order_relaxed);
            }
            if constexpr (kPayloadTailByteCount != 0) {
                uint64_t w = 0;
                std::memcpy(&w, src + kPayloadWordCount * sizeof(uint64_t), kPayloadTailByteCount);
                s.words[kPayloadWordCount].store(w, std::memory_order_relaxed);
            }
        }
    }

    static void load_payload(const Slot& s, T& out) noexcept {
        if constexpr (!kTriviallyCopyable) {
            std::lock_guard lock(s.mutex);
            out = *s.value;
        } else {
            std::byte* dst = reinterpret_cast<std::byte*>(&out);
#pragma GCC unroll 8
            for (size_t k = 0; k < kPayloadWordCount; k++) {
                const uint64_t w = s.words[k].load(std::memory_order_relaxed);
                std::memcpy(dst + k * sizeof(uint64_t), &w, sizeof(uint64_t));
            }
            if constexpr (kPayloadTailByteCount != 0) {
                const uint64_t w = s.words[kPayloadWordCount].load(std::memory_order_relaxed);
                std::memcpy(dst + kPayloadWordCount * sizeof(uint64_t), &w, kPayloadTailByteCount);
            }
        }
    }

    RingView view() const noexcept { return {slots_.get(), capacity_}; }

    const size_t capacity_;
    const std::unique_ptr<Slot[]> slots_;
    Counters counters_;
    Writer writer_;
};

}  // namespace tt::tt_metal
