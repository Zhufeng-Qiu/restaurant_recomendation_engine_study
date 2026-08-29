// Loader for pearson-fixture-v1 directories (see spark_pipeline/export_fixture.py).
// Array shapes are derived from file sizes, so no JSON parsing is required;
// meta.json remains the human-readable description and hash manifest.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {

template <typename T>
std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = f.tellg();
  if (bytes % static_cast<std::streamsize>(sizeof(T)) != 0)
    throw std::runtime_error(path + ": size not a multiple of element size");
  std::vector<T> v(static_cast<size_t>(bytes) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  if (!f) throw std::runtime_error("short read on " + path);
  return v;
}

struct Fixture {
  // CSR ratings: entity-major, dim indices strictly ascending per row.
  std::vector<int64_t> offsets;  // n_entities + 1
  std::vector<int32_t> dims;     // nnz
  std::vector<double> vals;      // nnz
  // Candidate pairs (i, j), i < j, lexicographically sorted; golden sim each.
  std::vector<int32_t> pairs;    // 2 * n_pairs
  std::vector<double> golden;    // n_pairs

  int64_t n_entities() const { return static_cast<int64_t>(offsets.size()) - 1; }
  int64_t nnz() const { return static_cast<int64_t>(vals.size()); }
  int64_t n_pairs() const { return static_cast<int64_t>(golden.size()); }

  static Fixture load(const std::string& dir) {
    Fixture fx;
    fx.offsets = read_bin<int64_t>(dir + "/offsets.bin");
    fx.dims = read_bin<int32_t>(dir + "/dims.bin");
    fx.vals = read_bin<double>(dir + "/vals.bin");
    fx.pairs = read_bin<int32_t>(dir + "/pairs.bin");
    fx.golden = read_bin<double>(dir + "/golden.bin");

    if (fx.offsets.empty() || fx.offsets.front() != 0 ||
        fx.offsets.back() != static_cast<int64_t>(fx.vals.size()))
      throw std::runtime_error(dir + ": inconsistent offsets.bin");
    if (fx.dims.size() != fx.vals.size())
      throw std::runtime_error(dir + ": dims.bin/vals.bin length mismatch");
    if (fx.pairs.size() != 2 * fx.golden.size())
      throw std::runtime_error(dir + ": pairs.bin/golden.bin length mismatch");
    return fx;
  }
};

}  // namespace engine
