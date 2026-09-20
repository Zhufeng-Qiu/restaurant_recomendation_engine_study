// Work division on both axes, and the index arithmetic on the way back out.
//
// The split itself is three lines and hard to get wrong. What is easy to get
// wrong -- and expensive to discover on a rented GPU -- is that the finalize
// kernel writes sims[order[slot]], a GLOBAL output position, while the stats
// buffer is indexed by LOCAL slot. Get the two confused and every similarity
// still looks like a similarity.
//
// So the second half of this file does not test a predicate. It simulates the
// writes: sentinel-fills each device's output, replays the exact index
// arithmetic the kernels would execute, copies back through the same window
// the host code would use, and checks that every global position was written
// exactly once by exactly one device. Then it does the same for the three
// wrong implementations the design note warns about, and requires that the
// check CATCHES each of them -- a sentinel scan that cannot fail is not a
// check.
//
// No GPU, no fixtures, no arguments.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "common/pair_partition.hpp"

namespace {

int checks = 0;
int failures = 0;

void expect(bool ok, const std::string& what) {
  ++checks;
  if (ok) {
    std::printf("ok   %s\n", what.c_str());
  } else {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}

using engine::Partition;
using engine::Range;
using engine::Shard;

// Pair counts that bracket every boundary the kernels have: single pairs,
// warp edges at 64, the G=4 sub-warp block tail, finalize block tails at
// 512, and the two real workloads.
const std::vector<int64_t> kPairCounts = {0,   1,   2,   3,   63,  64,
                                          65,  127, 128, 129, 511, 512,
                                          513, 1171857, 1411864};
const std::vector<int> kGpuCounts = {1, 2, 3, 4, 8};

const double kSentinel = std::numeric_limits<double>::quiet_NaN();

// What the output SHOULD hold: a distinct, recognisable value per pair.
double expected_at(int64_t k) { return 1000.0 + static_cast<double>(k); }

enum class Bug { None, OffsetSimsToo, CopyFromZero, ForgetSecondDevice };

// Replay the pair-partitioned write path on the host.
//
//   stats  -> device g computes its own [begin, end) into LOCAL slots
//   finalize -> writes sims[order[begin + local]]  (global position)
//   D2H    -> copies sims + begin into host + begin, count elements
//
// Returns the host array the CPU would end up holding, plus a per-position
// write count so a double-write is distinguishable from a wrong value.
struct Replay {
  std::vector<double> host;
  std::vector<int> writes;
  bool out_of_bounds = false;
};

Replay replay(int64_t n_pairs, int n_gpus, const std::vector<int32_t>& order,
              Bug bug) {
  Replay r;
  r.host.assign(static_cast<size_t>(n_pairs), kSentinel);
  r.writes.assign(static_cast<size_t>(n_pairs), 0);

  for (int g = 0; g < n_gpus; ++g) {
    const Shard s = engine::shard_for(Partition::Pair, n_pairs, 64, g, n_gpus);
    if (s.pairs.empty()) continue;  // zero-length launch, skipped

    // Every device keeps a full-length sims array; only the ORDER pointer is
    // offset. Allocating `count` and indexing globally is the out-of-bounds
    // write this arrangement exists to avoid.
    std::vector<double> sims(static_cast<size_t>(n_pairs), kSentinel);

    for (int64_t local = 0; local < s.pairs.count(); ++local) {
      const int64_t slot = s.pairs.begin + local;
      int64_t dst = order[static_cast<size_t>(slot)];
      if (bug == Bug::OffsetSimsToo) dst += s.pairs.begin;
      if (dst < 0 || dst >= n_pairs) {
        r.out_of_bounds = true;
        continue;
      }
      sims[static_cast<size_t>(dst)] = expected_at(order[static_cast<size_t>(slot)]);
    }

    if (bug == Bug::ForgetSecondDevice && g > 0) continue;

    for (int64_t i = 0; i < s.pairs.count(); ++i) {
      const int64_t src = (bug == Bug::CopyFromZero) ? i : s.pairs.begin + i;
      r.host[static_cast<size_t>(s.pairs.begin + i)] =
          sims[static_cast<size_t>(src)];
      r.writes[static_cast<size_t>(s.pairs.begin + i)] += 1;
    }
  }
  return r;
}

// Did the replay produce a complete, correctly ordered output?
bool output_is_complete(const Replay& r, int64_t n_pairs) {
  if (r.out_of_bounds) return false;
  for (int64_t k = 0; k < n_pairs; ++k) {
    if (r.writes[static_cast<size_t>(k)] != 1) return false;
    if (!std::isfinite(r.host[static_cast<size_t>(k)])) return false;
    if (r.host[static_cast<size_t>(k)] != expected_at(k)) return false;
  }
  return true;
}

}  // namespace

int main() {
  // ---- the split tiles, on both axes, at every boundary ----
  for (int64_t n : kPairCounts) {
    for (int gpus : kGpuCounts) {
      std::vector<Range> pair_ranges, dim_ranges;
      int64_t sum = 0, min_c = -1, max_c = 0;
      bool dims_full_pairs = true;
      const int64_t n_dims = 97;  // deliberately not a multiple of anything

      for (int g = 0; g < gpus; ++g) {
        const Shard p = engine::shard_for(Partition::Pair, n, n_dims, g, gpus);
        const Shard d = engine::shard_for(Partition::Dim, n, n_dims, g, gpus);
        pair_ranges.push_back(p.pairs);
        dim_ranges.push_back(d.dims);
        sum += p.pairs.count();
        if (min_c < 0 || p.pairs.count() < min_c) min_c = p.pairs.count();
        if (p.pairs.count() > max_c) max_c = p.pairs.count();
        // Under the pair split a device scans every dimension; under the
        // dimension split it computes every pair. The two are duals.
        if (p.dims.begin != 0 || p.dims.end != n_dims) dims_full_pairs = false;
        if (d.pairs.begin != 0 || d.pairs.end != n) dims_full_pairs = false;
      }

      const std::string tag =
          "N=" + std::to_string(n) + " G=" + std::to_string(gpus);
      expect(engine::ranges_tile(pair_ranges, n), tag + ": pairs tile [0,N)");
      expect(engine::ranges_tile(dim_ranges, n_dims), tag + ": dims tile [0,D)");
      expect(sum == n, tag + ": shard counts sum to N");
      expect(max_c - min_c <= 1, tag + ": shards differ by at most one pair");
      expect(dims_full_pairs, tag + ": each axis leaves the other whole");
      // Empty shards appear exactly when there is not enough work to go round,
      // and must be skipped rather than launched at length zero.
      expect((min_c == 0) == (n < gpus), tag + ": empty shard iff N < G");
    }
  }

  // ---- the identity requirement ----
  {
    std::vector<int32_t> ident(16);
    std::iota(ident.begin(), ident.end(), 0);
    expect(engine::order_is_identity(ident.data(), 16),
           "order_is_identity accepts the source order");

    std::vector<int32_t> swapped = ident;
    std::swap(swapped[4], swapped[9]);
    expect(!engine::order_is_identity(swapped.data(), 16),
           "order_is_identity rejects a single transposition");

    std::vector<int32_t> shifted = ident;
    for (int32_t& x : shifted) x = (x + 1) % 16;
    expect(!engine::order_is_identity(shifted.data(), 16),
           "order_is_identity rejects a rotation");
  }

  // ---- the write path: correct implementation ----
  for (int64_t n : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{63},
                    int64_t{64}, int64_t{65}, int64_t{129}, int64_t{513}}) {
    for (int gpus : {1, 2, 4}) {
      std::vector<int32_t> order(static_cast<size_t>(n));
      std::iota(order.begin(), order.end(), 0);
      const Replay r = replay(n, gpus, order, Bug::None);
      const std::string tag =
          "N=" + std::to_string(n) + " G=" + std::to_string(gpus);
      expect(output_is_complete(r, n),
             tag + ": every pair written once, in its own place");
    }
  }

  // ---- and the three wrong ones, which the check must CATCH ----
  //
  // A sentinel scan that passes everything proves nothing, so each mistake
  // named in the design note is reproduced and required to fail.
  {
    const int64_t n = 129;  // odd, so the two shards are 65 and 64
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    const Replay a = replay(n, 2, order, Bug::OffsetSimsToo);
    expect(!output_is_complete(a, n),
           "caught: offsetting the sims pointer as well as the order pointer");
    expect(a.out_of_bounds,
           "caught: and it is detected as an out-of-bounds write, not a wrong value");

    const Replay b = replay(n, 2, order, Bug::CopyFromZero);
    expect(!output_is_complete(b, n),
           "caught: copying the second device's results from sims[0]");

    const Replay c = replay(n, 2, order, Bug::ForgetSecondDevice);
    expect(!output_is_complete(c, n),
           "caught: copying only the first device's half");

    // Positive control on the same shape, so the three above are not passing
    // because the checker rejects everything.
    expect(output_is_complete(replay(n, 2, order, Bug::None), n),
           "control: the correct implementation on the same odd split");
  }

  // ---- a sorted order breaks the contiguous copy, which is why it is refused --
  {
    const int64_t n = 64;
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[0], order[63]);  // the mildest possible bylen-style sort
    expect(!engine::order_is_identity(order.data(), n),
           "a sorted order is not the identity");
    const Replay r = replay(n, 2, order, Bug::None);
    expect(!output_is_complete(r, n),
           "and the contiguous copy-back silently loses results under it");
  }

  std::printf("\n%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
