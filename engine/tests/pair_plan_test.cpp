// Boundary tests for warp packing's pair planning.
//
// cuda_emulation_test runs the device algorithm against a shipped fixture,
// which is a good end-to-end check and a poor boundary check: item_tiny
// happens to contain no pair whose restricted slice is empty, no trailing
// partial warp at every group size, and no interesting tie structure. This
// file builds small synthetic fixtures that hit those cases on purpose.
//
// The fixtures are constructed in memory rather than on disk: the point is to
// control the shorter-slice length of each pair exactly, which is easier to
// state in code than to encode in a binary CSR and easier to trust when the
// expected lane-slot count is derived from the same statement.
//
// Ratings stay small non-negative integers so the packed payload's domain
// holds and its round trip can be checked here too.
//
// Usage: pair_plan_test           (no arguments; everything is synthetic)

#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "common/fixture.hpp"
#include "common/pair_order.hpp"
#include "common/pearson.hpp"
#include "cuda/pair_kernel.cuh"

namespace {

int failures = 0;
int checks = 0;

void expect(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

template <typename T>
void expect_eq(T got, T want, const std::string& what) {
  ++checks;
  if (!(got == want)) {
    ++failures;
    std::printf("  FAIL  %s: got %lld want %lld\n", what.c_str(),
                static_cast<long long>(got), static_cast<long long>(want));
  }
}

// --- fixture construction ---------------------------------------------------

using Row = std::vector<std::pair<int32_t, double>>;  // (dim, value), ascending

// Builds a Fixture from explicit rows and pairs, with golden similarities from
// an independent naive intersection -- not from anything the code under test
// shares.
engine::Fixture make_fixture(const std::vector<Row>& rows,
                             const std::vector<std::pair<int32_t, int32_t>>& prs) {
  engine::Fixture fx;
  fx.offsets.push_back(0);
  for (const Row& r : rows) {
    for (auto& [d, v] : r) {
      fx.dims.push_back(d);
      fx.vals.push_back(v);
    }
    fx.offsets.push_back(static_cast<int64_t>(fx.dims.size()));
  }
  for (auto& [i, j] : prs) {
    fx.pairs.push_back(i);
    fx.pairs.push_back(j);
    std::map<int32_t, double> a;
    for (auto& [d, v] : rows[i]) a[d] = v;
    engine::PairStats s;
    for (auto& [d, v] : rows[j]) {
      auto it = a.find(d);
      if (it != a.end()) s.add(it->second, v);
    }
    fx.golden.push_back(engine::pearson_finalize(s));
  }
  return fx;
}

// A row of `n` consecutive dims starting at `base`, ratings cycling 1..5.
Row dense_row(int32_t base, int32_t n) {
  Row r;
  for (int32_t k = 0; k < n; ++k)
    r.push_back({base + k, static_cast<double>(1 + (k % 5))});
  return r;
}

// --- an independent model of the lane-slot cost -----------------------------
//
// Deliberately written the other way round from evaluate_order: that one walks
// warps and strides inside them, this one labels every slot with its warp and
// folds by label. If both are wrong they have to be wrong in the same way.
int64_t reference_lane_slots(const std::vector<int32_t>& order,
                             const std::vector<int32_t>& len, int group) {
  std::map<int64_t, int64_t> rounds_by_warp;
  const int per_warp = 32 / group;
  for (size_t slot = 0; slot < order.size(); ++slot) {
    const int64_t warp = static_cast<int64_t>(slot) / per_warp;
    const int64_t L = len[order[slot]];
    const int64_t rounds = (L + group - 1) / group;
    auto it = rounds_by_warp.find(warp);
    if (it == rounds_by_warp.end()) rounds_by_warp[warp] = rounds;
    else it->second = std::max(it->second, rounds);
  }
  int64_t total = 0;
  for (auto& [w, r] : rounds_by_warp) total += 32 * r;
  return total;
}

const int kGroups[] = {1, 2, 4, 8, 16, 32};

// --- tests ------------------------------------------------------------------

void test_group_validation() {
  std::printf("group validation\n");
  for (int g : kGroups) expect(engine::valid_group(g), "valid_group accepts " + std::to_string(g));
  for (int g : {-8, -1, 0, 3, 5, 6, 7, 9, 12, 17, 31, 33, 64, 96})
    expect(!engine::valid_group(g), "valid_group rejects " + std::to_string(g));

  const engine::Fixture fx = make_fixture({dense_row(0, 8), dense_row(0, 8)}, {{0, 1}});
  for (int g : {0, 3, 31, 33, 64, -2}) {
    bool threw = false;
    try { engine::plan_pairs(fx, 0, 8, engine::PairOrder::kByShortLen, g); }
    catch (const std::exception&) { threw = true; }
    expect(threw, "plan_pairs throws on group=" + std::to_string(g));
    threw = false;
    try { engine::evaluate_order({0}, {8}, g); }
    catch (const std::exception&) { threw = true; }
    expect(threw, "evaluate_order throws on group=" + std::to_string(g));
  }
}

void test_empty_and_single() {
  std::printf("empty and single-pair fixtures\n");
  const engine::Fixture empty = make_fixture({dense_row(0, 4), dense_row(0, 4)}, {});
  for (int g : kGroups) {
    for (auto how : {engine::PairOrder::kSource, engine::PairOrder::kByShortLen}) {
      const engine::PairPlan p = engine::plan_pairs(empty, 0, 4, how, g);
      const engine::PlanMetrics m = engine::measure_plan(empty, p, 0, 4);
      expect(p.order.empty(), "zero pairs -> empty order");
      expect_eq<int64_t>(m.lane_slots, 0, "zero pairs -> zero slots");
      expect_eq<int64_t>(m.effective_elements, 0, "zero pairs -> zero elements");
      expect(m.utilisation() == 0.0, "zero pairs -> utilisation 0, not NaN");
    }
  }

  // One pair, shorter slice = 5: one warp, ceil(5/G) rounds, 32 slots each.
  const engine::Fixture one =
      make_fixture({dense_row(0, 5), dense_row(0, 40)}, {{0, 1}});
  for (int g : kGroups) {
    const engine::PairPlan p =
        engine::plan_pairs(one, 0, 64, engine::PairOrder::kByShortLen, g);
    const engine::PlanMetrics m = engine::measure_plan(one, p, 0, 64);
    expect_eq<int64_t>(m.effective_elements, 5, "single pair elements");
    expect_eq<int64_t>(m.lane_slots, 32 * ((5 + g - 1) / g),
                       "single pair slots at g=" + std::to_string(g));
  }
}

void test_short_len_edges() {
  std::printf("shorter-slice lengths at the boundaries\n");
  // A long partner row so the SHORT side is always the one we set.
  std::vector<Row> rows{dense_row(0, 4096)};
  std::vector<std::pair<int32_t, int32_t>> prs;
  std::vector<int32_t> want_len;
  for (int g : kGroups)
    for (int L : {0, 1, g - 1, g, g + 1, 31, 32, 33}) {
      if (L < 0) continue;
      rows.push_back(dense_row(0, L));
      prs.push_back({static_cast<int32_t>(rows.size() - 1), 0});
      want_len.push_back(L);
    }
  const engine::Fixture fx = make_fixture(rows, prs);

  for (size_t k = 0; k < prs.size(); ++k) {
    expect_eq<int64_t>(engine::pair_short_len(fx, static_cast<int64_t>(k), 0, 4096),
                       want_len[k], "short_len of pair " + std::to_string(k));
    // Evaluated alone, a pair occupies exactly one warp of ceil(L/G) rounds.
    for (int g : kGroups) {
      const int64_t got =
          engine::evaluate_order({static_cast<int32_t>(k)},
                                 engine::short_lens(fx, 0, 4096), g).lane_slots;
      expect_eq<int64_t>(got, 32 * ((want_len[k] + g - 1) / g),
                         "isolated slots L=" + std::to_string(want_len[k]) +
                             " g=" + std::to_string(g));
    }
  }

  // An empty restricted slice: row 1 lives entirely outside [100, 200).
  const engine::Fixture out =
      make_fixture({dense_row(0, 10), dense_row(0, 4096)}, {{0, 1}});
  expect_eq<int64_t>(engine::pair_short_len(out, 0, 100, 200), 0,
                     "restricted slice can be empty");
  expect_eq<int64_t>(
      [&] {
        const engine::PairPlan q =
            engine::plan_pairs(out, 100, 200, engine::PairOrder::kByShortLen, 8);
        return engine::measure_plan(out, q, 100, 200).lane_slots;
      }(),
      0, "a warp with nothing to scan costs no rounds");
}

void test_bijection_and_reference_model() {
  std::printf("bijection, trailing warps, and the model against a reference\n");
  // 77 pairs: not a multiple of 32/G for any G > 1, so every group size ends
  // with a partial warp. Lengths deliberately repeat.
  std::vector<Row> rows{dense_row(0, 4096)};
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 77; ++k) {
    rows.push_back(dense_row(0, (k * 7) % 40));
    prs.push_back({static_cast<int32_t>(rows.size() - 1), 0});
  }
  const engine::Fixture fx = make_fixture(rows, prs);
  const int64_t n = fx.n_pairs();
  expect_eq<int64_t>(n, 77, "fixture pair count");

  for (int32_t hi : {4096, 20}) {  // full range, and a slice that truncates rows
    const std::vector<int32_t> len = engine::short_lens(fx, 0, hi);
    for (int g : kGroups) {
      for (auto how : {engine::PairOrder::kSource, engine::PairOrder::kByShortLen}) {
        const engine::PairPlan p = engine::plan_pairs(fx, 0, hi, how, g);
        std::set<int32_t> seen(p.order.begin(), p.order.end());
        expect_eq<size_t>(seen.size(), static_cast<size_t>(n), "order is a bijection");
        expect(*seen.begin() == 0 && *seen.rbegin() == n - 1,
               "order covers exactly [0, n)");
        expect_eq<int64_t>(engine::evaluate_order(p.order, len, g).lane_slots,
                           reference_lane_slots(p.order, len, g),
                           "model == reference, g=" + std::to_string(g) +
                               " hi=" + std::to_string(hi));
        // The trailing partial warp still costs a whole warp -- but only
        // warps that have something to scan cost anything at all. The `hi=20`
        // slice truncates some rows to nothing, and a warp of empty slices is
        // correctly free.
        const int per_warp = 32 / g;
        int64_t busy_warps = 0;
        for (int64_t s0 = 0; s0 < n; s0 += per_warp) {
          for (int64_t t = s0; t < std::min(s0 + per_warp, n); ++t)
            if (len[p.order[t]] > 0) { ++busy_warps; break; }
        }
        expect(engine::measure_plan(fx, p, 0, hi).lane_slots >= 32 * busy_warps,
               "each warp with work costs at least one full round");
      }
    }
  }
}

void test_tie_determinism() {
  std::printf("deterministic tie order\n");
  std::vector<Row> rows{dense_row(0, 4096)};
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 50; ++k) {           // ten distinct lengths, five each
    rows.push_back(dense_row(0, 3 + (k % 10)));
    prs.push_back({static_cast<int32_t>(rows.size() - 1), 0});
  }
  const engine::Fixture fx = make_fixture(rows, prs);
  const std::vector<int32_t> len = engine::short_lens(fx, 0, 4096);

  const auto a = engine::plan_pairs(fx, 0, 4096, engine::PairOrder::kByShortLen, 8);
  const auto b = engine::plan_pairs(fx, 0, 4096, engine::PairOrder::kByShortLen, 8);
  expect(a.order == b.order, "same inputs -> same order");

  bool sorted = true, ties_in_index_order = true;
  for (size_t i = 1; i < a.order.size(); ++i) {
    if (len[a.order[i - 1]] > len[a.order[i]]) sorted = false;
    if (len[a.order[i - 1]] == len[a.order[i]] && a.order[i - 1] > a.order[i])
      ties_in_index_order = false;
  }
  expect(sorted, "bylen is ascending");
  expect(ties_in_index_order, "equal lengths keep source index order");

  const auto src = engine::plan_pairs(fx, 0, 4096, engine::PairOrder::kSource, 8);
  std::vector<int32_t> identity(src.order.size());
  std::iota(identity.begin(), identity.end(), 0);
  expect(src.order == identity, "source order is the identity");
}

void test_per_device_metrics() {
  std::printf("one shared order, evaluated per device slice\n");
  std::vector<Row> rows;
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 64; ++k) rows.push_back(dense_row(0, 20 + (k % 60)));
  for (int k = 0; k + 1 < 64; k += 2) prs.push_back({k, k + 1});
  const engine::Fixture fx = make_fixture(rows, prs);
  const int32_t n_dims = 80;

  for (int g : kGroups) {
    // The order every device must share: sorted on the FULL range.
    const engine::PairPlan plan =
        engine::plan_pairs(fx, 0, n_dims, engine::PairOrder::kByShortLen, g);
    engine::PlanMetrics agg;
    int64_t agg_slots = 0, agg_eff = 0, critical = 0;
    for (int d = 0; d < 2; ++d) {
      const int32_t lo = n_dims * d / 2, hi = n_dims * (d + 1) / 2;
      const std::vector<int32_t> len = engine::short_lens(fx, lo, hi);
      const engine::PlanMetrics m = engine::evaluate_order(plan.order, len, g);
      expect_eq<int64_t>(m.lane_slots, reference_lane_slots(plan.order, len, g),
                         "per-device model == reference");
      // The device's own effective elements come from its own slice, not the
      // full range: this is exactly what the full-dimension plan misreports.
      int64_t eff = 0;
      for (int32_t l : len) eff += l;
      expect_eq<int64_t>(m.effective_elements, eff, "per-device elements");

      const engine::PlanMetrics ideal =
          engine::counterfactual_per_device_ideal(fx, g, lo, hi);
      expect(ideal.lane_slots <= m.lane_slots,
             "sorting a slice for itself cannot cost more than a shared order");
      expect_eq<int64_t>(ideal.effective_elements, m.effective_elements,
                         "counterfactual does the same work, packed better");
      agg_slots += m.lane_slots;
      agg_eff += m.effective_elements;
      agg.lane_slots += m.lane_slots;
      agg.effective_elements += m.effective_elements;
      critical = std::max(critical, m.lane_slots);
    }
    expect(critical * 2 >= agg_slots, "critical device >= the average one");
    // Elements survive a contiguous split -- each pair's work is divided, not
    // duplicated -- so the aggregate matching the full range is expected here
    // and is NOT what makes the full-dimension figure misleading.
    const engine::PlanMetrics full = engine::measure_plan(fx, plan, 0, n_dims);
    expect_eq<int64_t>(agg_eff, full.effective_elements,
                       "a contiguous split conserves elements");
    // Lane slots do not survive it: every device pays its own tail, so the
    // executed cost is strictly worse than the full-range plan suggests. This
    // is the reason the two must be reported under different names.
    expect(agg_slots >= full.lane_slots,
           "splitting the dimensions can only add tail waste");
    expect(agg.utilisation() <= full.utilisation() + 1e-12,
           "executed utilisation is never better than the full-range figure");
  }
}

// --- end-to-end: slot-indexed stats, scatter, and the payload round trip ----

void test_scatter_and_payloads() {
  std::printf("slot indexing, scatter, and payload round trips\n");
  std::vector<Row> rows;
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 40; ++k) rows.push_back(dense_row((k % 3), 8 + (k % 37)));
  for (int k = 0; k + 1 < 40; ++k) prs.push_back({k, k + 1});
  const engine::Fixture fx = make_fixture(rows, prs);
  const int64_t n = fx.n_pairs();
  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  for (int g : kGroups) {
    for (auto how : {engine::PairOrder::kSource, engine::PairOrder::kByShortLen}) {
      const engine::PairPlan plan = engine::plan_pairs(fx, 0, n_dims, how, g);
      for (int ranges : {1, 2, 3}) {
        for (int payload = 0; payload < 3; ++payload) {  // f64, i32, packed
          std::vector<double> sims(static_cast<size_t>(n),
                                   std::numeric_limits<double>::quiet_NaN());
          for (int64_t slot = 0; slot < n; ++slot) {
            const int64_t k = plan.order[slot];
            double total[6] = {};
            int32_t as_i32[6] = {};
            uint64_t as_packed[2] = {0, 0};
            for (int r = 0; r < ranges; ++r) {
              const int32_t lo = static_cast<int32_t>(int64_t(n_dims) * r / ranges);
              const int32_t hi =
                  static_cast<int32_t>(int64_t(n_dims) * (r + 1) / ranges);
              double lanes[32][6] = {};
              for (int lane = 0; lane < g; ++lane)
                engine_cuda::pair_lane_stats(fx.offsets.data(), fx.dims.data(),
                                             fx.vals.data(), fx.pairs[2 * k],
                                             fx.pairs[2 * k + 1], lo, hi, lane,
                                             g, lanes[lane]);
              for (int off = g / 2; off > 0; off >>= 1)
                for (int l = 0; l < off; ++l)
                  for (int i = 0; i < 6; ++i) lanes[l][i] += lanes[l + off][i];
              if (payload == 0) {
                for (int i = 0; i < 6; ++i) total[i] += lanes[0][i];
              } else if (payload == 1) {
                for (int i = 0; i < 6; ++i)
                  as_i32[i] += static_cast<int32_t>(lanes[0][i]);
              } else {
                uint64_t w[2];
                engine_cuda::pack_six(lanes[0], w);
                engine_cuda::pack_add(as_packed, w, as_packed);
              }
            }
            if (payload == 1)
              for (int i = 0; i < 6; ++i) total[i] = static_cast<double>(as_i32[i]);
            else if (payload == 2)
              engine_cuda::unpack_six(as_packed, total);
            sims[plan.order[slot]] = engine_cuda::finalize_six(total);
          }
          int64_t bad = 0;
          for (int64_t k = 0; k < n; ++k)
            if (!(std::fabs(sims[k] - fx.golden[k]) <= engine::kTol)) ++bad;
          expect_eq<int64_t>(bad, 0,
                             "g=" + std::to_string(g) + " ranges=" +
                                 std::to_string(ranges) + " payload=" +
                                 std::to_string(payload) + " order=" +
                                 engine::pair_order_name(how));
        }
      }
    }
  }
}

// The split that the cold-path fix depends on: building the order is required
// to run, describing it is not. A regression here would put a ~50 ms length
// pass back inside cold_data_path_s for callers that do no packing at all.
void test_required_vs_diagnostic() {
  std::printf("required planning vs diagnostic metrics\n");
  std::vector<Row> rows{dense_row(0, 4096)};
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 60; ++k) {
    rows.push_back(dense_row(0, (k * 11) % 50));
    prs.push_back({static_cast<int32_t>(rows.size() - 1), 0});
  }
  const engine::Fixture fx = make_fixture(rows, prs);
  const std::vector<int32_t> len = engine::short_lens(fx, 0, 4096);

  for (int g : kGroups) {
    for (auto how : {engine::PairOrder::kSource, engine::PairOrder::kByShortLen}) {
      const engine::PairPlan p = engine::plan_pairs(fx, 0, 4096, how, g);
      // measure_plan must be a pure description of the plan, agreeing with a
      // direct evaluation and repeatable.
      const engine::PlanMetrics a = engine::measure_plan(fx, p, 0, 4096);
      const engine::PlanMetrics b = engine::measure_plan(fx, p, 0, 4096);
      const engine::PlanMetrics direct = engine::evaluate_order(p.order, len, g);
      expect_eq<int64_t>(a.lane_slots, direct.lane_slots, "measure == evaluate");
      expect_eq<int64_t>(a.effective_elements, direct.effective_elements,
                         "measure == evaluate (elements)");
      expect_eq<int64_t>(a.lane_slots, b.lane_slots, "measure is repeatable");
      // ...and must not change the plan it describes.
      const engine::PairPlan q = engine::plan_pairs(fx, 0, 4096, how, g);
      expect(p.order == q.order, "describing a plan does not alter it");
    }
  }
  // kSource is the identity regardless of group, so it needs no lengths at all.
  for (int g : kGroups) {
    const engine::PairPlan p =
        engine::plan_pairs(fx, 0, 4096, engine::PairOrder::kSource, g);
    std::vector<int32_t> identity(p.order.size());
    std::iota(identity.begin(), identity.end(), 0);
    expect(p.order == identity, "source order needs no length pass");
  }
}

// Hoisting must be a pure relocation of work. The device version computes the
// slice bounds in one lane and broadcasts them; __shfl cannot run here, but the
// semantics can: compute once, hand the same bounds to every lane. If that ever
// diverges from each lane computing its own, the hoisted kernel is not the same
// kernel and its A/B against the unhoisted one measures two things at once.
void test_hoist_is_behaviour_preserving() {
  std::printf("hoisted bounds == per-lane bounds\n");
  std::vector<Row> rows;
  std::vector<std::pair<int32_t, int32_t>> prs;
  for (int k = 0; k < 48; ++k) rows.push_back(dense_row(k % 4, 5 + (k * 13) % 70));
  for (int k = 0; k + 1 < 48; ++k) prs.push_back({k, k + 1});
  const engine::Fixture fx = make_fixture(rows, prs);
  int32_t n_dims = 0;
  for (int32_t d : fx.dims) n_dims = std::max(n_dims, d + 1);

  for (int g : kGroups) {
    for (int ranges : {1, 2, 3}) {
      for (int r = 0; r < ranges; ++r) {
        const int32_t lo = static_cast<int32_t>(int64_t(n_dims) * r / ranges);
        const int32_t hi = static_cast<int32_t>(int64_t(n_dims) * (r + 1) / ranges);
        for (int64_t k = 0; k < fx.n_pairs(); ++k) {
          const int32_t ei = fx.pairs[2 * k], ej = fx.pairs[2 * k + 1];
          double per_lane[6] = {}, hoisted[6] = {};
          const engine_cuda::PairSlices shared =
              engine_cuda::pair_slice_bounds(fx.offsets.data(), fx.dims.data(),
                                             ei, ej, lo, hi);
          for (int lane = 0; lane < g; ++lane) {
            double a[6] = {}, b[6] = {};
            engine_cuda::pair_lane_stats(fx.offsets.data(), fx.dims.data(),
                                         fx.vals.data(), ei, ej, lo, hi, lane,
                                         g, a);
            engine_cuda::pair_lane_scan(fx.dims.data(), fx.vals.data(), shared,
                                        lane, g, b);
            for (int i = 0; i < 6; ++i) { per_lane[i] += a[i]; hoisted[i] += b[i]; }
          }
          for (int i = 0; i < 6; ++i)
            // Bit-identical, not close: same additions in the same order.
            expect(per_lane[i] == hoisted[i],
                   "hoist matches at g=" + std::to_string(g) + " stat "
                       + std::to_string(i));
        }
      }
    }
  }
}

}  // namespace

int main() {
  test_group_validation();
  test_empty_and_single();
  test_short_len_edges();
  test_bijection_and_reference_model();
  test_tie_determinism();
  test_per_device_metrics();
  test_required_vs_diagnostic();
  test_hoist_is_behaviour_preserving();
  test_scatter_and_payloads();
  std::printf("pair planning: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
