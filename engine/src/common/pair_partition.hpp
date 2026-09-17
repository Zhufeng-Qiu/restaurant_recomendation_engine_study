// How work is divided between devices, on either of the two axes.
//
// The engine has always split the RATING DIMENSION: every device computes
// every candidate pair over its own slice of the dimensions, and an AllReduce
// sums the six partial statistics. That is what makes the compressed payload
// study meaningful, because it is the reduction that carries the bytes.
//
// It is not the only division available. Pairs are independent of each other,
// and both devices already hold the entire CSR input, so a device could
// instead take half the PAIRS and compute them to completion over the full
// dimension range -- no collective at all. Which of the two is faster is an
// open question the repository never asked, and this header is what lets it
// be asked: one splitting rule, two axes, chosen by a flag.
//
// The dangerous part is not the split. It is the index arithmetic on the way
// back out, which is why it is spelled out and tested here on the host rather
// than discovered on a rented GPU.
#pragma once

#include <cstdint>
#include <vector>

namespace engine {

enum class Partition { Dim, Pair };

inline const char* partition_name(Partition p) {
  return p == Partition::Pair ? "pair" : "dim";
}

// A half-open [begin, end).
struct Range {
  int64_t begin = 0;
  int64_t end = 0;
  int64_t count() const { return end - begin; }
  bool empty() const { return end <= begin; }
};

// Split [0, total) into n_parts balanced pieces: begin = floor(total*g/n).
// This is the rule nccl_main.cu already uses for the dimension axis; reusing
// it rather than writing a second one is the point. It leaves the piece sizes
// differing by at most one, and produces empty pieces -- correctly -- when
// total < n_parts.
inline Range split_range(int64_t total, int g, int n_parts) {
  if (n_parts <= 0) return {0, 0};
  return {total * g / n_parts, total * (g + 1) / n_parts};
}

// What device g owns.
struct Shard {
  Range pairs;  // candidate pairs this device computes
  Range dims;   // rating dimensions it scans
};

inline Shard shard_for(Partition p, int64_t n_pairs, int64_t n_dims, int g,
                       int n_gpus) {
  Shard s;
  if (p == Partition::Pair) {
    s.pairs = split_range(n_pairs, g, n_gpus);
    s.dims = {0, n_dims};  // each device scans every dimension, to completion
  } else {
    s.pairs = {0, n_pairs};  // each device computes every pair, partially
    s.dims = split_range(n_dims, g, n_gpus);
  }
  return s;
}

// Do these ranges cover [0, total) exactly once -- no gap, no overlap?
//
// "Every pair is computed exactly once" is the entire correctness claim of
// the pair partition, and confirming it costs O(n_gpus). Cheap enough to run
// at startup rather than trust.
inline bool ranges_tile(const std::vector<Range>& r, int64_t total) {
  int64_t cursor = 0;
  for (const Range& x : r) {
    if (x.begin != cursor || x.end < x.begin) return false;
    cursor = x.end;
  }
  return cursor == total;
}

// THE INDEX RULE, and why the pair partition refuses a sorted order.
//
// finalize writes sims[order[slot]] -- a GLOBAL output position, not a local
// one. Device g is handed `order + pair_begin` and its own pair_count, so it
// writes to sims[order[pair_begin + local]] for local in [0, count).
//
// The minimal safe implementation keeps sims at full length N on every
// device and offsets the ORDER pointer, never the sims pointer. Then the
// device-to-host copy moves sims + pair_begin into host_sims + pair_begin,
// count elements -- which is correct only if those writes were contiguous and
// began at pair_begin. That holds exactly when order is the identity.
//
// Under --pair-order bylen, order is a permutation: device 1's writes scatter
// across the whole output, the contiguous window copies back whatever
// happened to be sitting there, and every similarity still looks plausible.
// So the combination is refused, and this predicate confirms the array really
// is what the flag claims rather than trusting the flag.
//
// Three ways to get this wrong, all of which the partition test reproduces
// and detects: offsetting the sims pointer by begin as well as the order
// pointer; copying the second device's results from sims[0]; and copying all
// N results from each device and keeping half.
inline bool order_is_identity(const int32_t* order, int64_t n) {
  for (int64_t k = 0; k < n; ++k)
    if (order[k] != static_cast<int32_t>(k)) return false;
  return true;
}

}  // namespace engine
