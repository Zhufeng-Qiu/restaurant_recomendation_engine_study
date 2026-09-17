// Domain gate test for the compressed collective payloads (contract 9.4).
//
// The README's limitations section records that check_payload_domain()'s
// rejection branches are never exercised by the shipped fixtures: every one is
// Yelp stars {1,2,3,4,5}, worst statistic 34,075 against a 2^21-1 field, so the
// gate always says yes. A gate that has only ever returned "accept" is not
// evidence of anything. This test supplies the inputs that make it say no.
//
// Fixtures come from tools/make_domain_fixtures.py (committed under
// data/fixtures/domain/), each a handful of ratings over 3 entities:
//
//   ok               integer 1-5                      accept f64, i32, packed
//   fractional       one rating 3.5                   accept f64 only
//   negative         one rating -2                    accept f64 only
//   packed_overflow  vmax=1000, N=5   -> 5.0e6        accept f64, i32
//   i32_overflow     vmax=100000, N=3 -> 3.0e10       accept f64 only
//
// The in-domain control matters as much as the rejections: it proves the gate
// is discriminating between inputs rather than refusing everything.
//
// Usage: payload_domain_test <domain_fixture_root>

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "common/fixture.hpp"
#include "common/payload_domain.hpp"

namespace {

using engine::check_payload_domain;
using engine::Fixture;
using engine::PayloadKind;
using engine::payload_kind_name;

struct Expect {
  const char* fixture;
  bool f64, i32, packed;
  const char* why;
};

const std::vector<Expect> kExpect = {
    {"ok",              true, true,  true,  "in-domain control"},
    {"fractional",      true, false, false, "3.5 is not an integer"},
    {"negative",        true, false, false, "-2 is negative"},
    {"packed_overflow", true, true,  false, "5.0e6 > 2^21-1, still < 2^31-1"},
    {"i32_overflow",    true, false, false, "3.0e10 > 2^31-1"},
};

bool expected_for(const Expect& e, PayloadKind p) {
  switch (p) {
    case PayloadKind::I32: return e.i32;
    case PayloadKind::Packed: return e.packed;
    default: return e.f64;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <domain_fixture_root>\n", argv[0]);
    return 2;
  }
  const std::string root = argv[1];
  int failures = 0;
  int checks = 0;

  for (const Expect& e : kExpect) {
    Fixture fx;
    try {
      fx = Fixture::load(root + "/" + e.fixture);
    } catch (const std::exception& ex) {
      std::printf("FAIL %-16s cannot load: %s\n", e.fixture, ex.what());
      ++failures;
      continue;
    }

    for (PayloadKind p : {PayloadKind::F64, PayloadKind::I32,
                          PayloadKind::Packed}) {
      const engine::DomainVerdict v = check_payload_domain(fx, p);
      const bool want = expected_for(e, p);
      ++checks;
      if (v.ok != want) {
        std::printf("FAIL %-16s payload=%-6s got %s, expected %s  (%s)\n",
                    e.fixture, payload_kind_name(p),
                    v.ok ? "accept" : "reject", want ? "accept" : "reject",
                    e.why);
        if (!v.ok) std::printf("       reason: %s\n", v.reason.c_str());
        ++failures;
        continue;
      }
      // A rejection must explain itself; a silent refusal is as unhelpful as
      // a silent overflow.
      if (!v.ok && v.reason.empty()) {
        std::printf("FAIL %-16s payload=%-6s rejected with empty reason\n",
                    e.fixture, payload_kind_name(p));
        ++failures;
        continue;
      }
      std::printf("ok   %-16s payload=%-6s %s%s\n", e.fixture,
                  payload_kind_name(p), v.ok ? "accept" : "reject",
                  v.ok ? "" : "  <- gate fired");
    }
  }

  // ---- capacity boundary, on the pure decision function ----
  //
  // These cannot be fixtures: proving that a row of 2^31 elements is refused
  // would mean building one. check_payload_bounds() takes the extents
  // directly, so the boundary is testable at a cost of nothing.
  //
  // For integer ratings only two of the three bounds can ever fire first --
  // vmax >= 1 makes vmax^2*max_row >= vmax*max_row >= max_row, so the
  // second-moment bound dominates, and at vmax = 0 the other two collapse to
  // zero and only `n <= max_row` is left holding anything. The sum(x) bound
  // is stated anyway, because a reader should not have to re-derive that it
  // is implied. The vmax = 0 row below is the case the single old bound got
  // wrong.
  {
    const int64_t kPackedL = 2097151;      // 2^21 - 1
    const int64_t kI32L = 2147483647;      // 2^31 - 1
    struct Case {
      int64_t max_row;
      double vmax;
      PayloadKind p;
      bool want;
      const char* why;
    };
    const std::vector<Case> kBounds = {
        {kPackedL - 1, 1.0, PayloadKind::Packed, true,  "vmax=1, one inside"},
        {kPackedL,     1.0, PayloadKind::Packed, true,  "vmax=1, exactly at capacity"},
        {kPackedL + 1, 1.0, PayloadKind::Packed, false, "vmax=1, one past"},
        {kPackedL,     0.0, PayloadKind::Packed, true,  "all-zero ratings, n at capacity"},
        {kPackedL + 1, 0.0, PayloadKind::Packed, false,
         "all-zero ratings, n one past -- the hole the old single bound left"},
        {83886,        5.0, PayloadKind::Packed, true,  "Yelp stars, longest row that fits"},
        {83887,        5.0, PayloadKind::Packed, false, "Yelp stars, one row too long"},
        {1363,         5.0, PayloadKind::Packed, true,  "item_full's real extents"},
        {kI32L,        1.0, PayloadKind::I32,    true,  "i32 exactly at capacity"},
        {kI32L + 1,    1.0, PayloadKind::I32,    false, "i32 one past"},
        {kI32L + 1,    0.0, PayloadKind::I32,    false, "i32, all-zero ratings, n one past"},
        {kI32L + 1,    9e9, PayloadKind::F64,    true,  "f64 has no capacity bound"},
    };
    for (const Case& c : kBounds) {
      const engine::DomainVerdict v =
          engine::check_payload_bounds(c.max_row, c.vmax, c.p);
      ++checks;
      if (v.ok != c.want) {
        std::printf("FAIL bounds max_row=%lld vmax=%g payload=%-6s got %s, "
                    "expected %s  (%s)\n",
                    static_cast<long long>(c.max_row), c.vmax,
                    payload_kind_name(c.p), v.ok ? "accept" : "reject",
                    c.want ? "accept" : "reject", c.why);
        ++failures;
        continue;
      }
      if (!v.ok && v.reason.empty()) {
        std::printf("FAIL bounds max_row=%lld rejected with empty reason\n",
                    static_cast<long long>(c.max_row));
        ++failures;
        continue;
      }
      std::printf("ok   bounds max_row=%-11lld vmax=%-4g %-6s %-6s  %s\n",
                  static_cast<long long>(c.max_row), c.vmax,
                  payload_kind_name(c.p), v.ok ? "accept" : "reject", c.why);
    }
  }

  // ---- non-finite ratings are refused on every path, f64 included ----
  //
  // A NaN rating cannot be caught downstream: pearson_finalize's
  // zero-variance branch turns an abnormal intermediate into a clean 0.0, so
  // a finite output is not evidence of a finite input.
  {
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
      Fixture fx;
      fx.offsets = {0, 2, 4};
      fx.dims = {0, 1, 0, 1};
      fx.vals = {3.0, bad, 4.0, 2.0};
      fx.pairs = {0, 1};
      fx.golden = {0.0};
      for (PayloadKind p : {PayloadKind::F64, PayloadKind::I32,
                            PayloadKind::Packed}) {
        const engine::DomainVerdict v = check_payload_domain(fx, p);
        ++checks;
        if (v.ok) {
          std::printf("FAIL non-finite rating %g accepted for payload=%s\n",
                      bad, payload_kind_name(p));
          ++failures;
        } else {
          std::printf("ok   non-finite rating %-4g payload=%-6s reject  %s\n",
                      bad, payload_kind_name(p), v.reason.substr(0, 40).c_str());
        }
      }
    }
    // Positive control on the same shape: valid ratings still accepted.
    Fixture fx;
    fx.offsets = {0, 2, 4};
    fx.dims = {0, 1, 0, 1};
    fx.vals = {3.0, 5.0, 4.0, 2.0};
    fx.pairs = {0, 1};
    fx.golden = {0.0};
    for (PayloadKind p : {PayloadKind::F64, PayloadKind::I32,
                          PayloadKind::Packed}) {
      const engine::DomainVerdict v = check_payload_domain(fx, p);
      ++checks;
      if (!v.ok) {
        std::printf("FAIL valid in-memory fixture rejected for payload=%s: %s\n",
                    payload_kind_name(p), v.reason.c_str());
        ++failures;
      } else {
        std::printf("ok   finite ratings   payload=%-6s accept\n",
                    payload_kind_name(p));
      }
    }
  }

  if (failures) {
    std::printf("\nFAIL: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  std::printf("\nOK: all %d domain checks pass\n", checks);
  return 0;
}
