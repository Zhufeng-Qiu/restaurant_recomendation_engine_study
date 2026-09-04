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

  if (failures) {
    std::printf("\nFAIL: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  std::printf("\nOK: all %d domain checks pass (%zu fixtures)\n", checks,
              kExpect.size());
  return 0;
}
