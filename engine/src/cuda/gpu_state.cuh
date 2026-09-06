// GPU clock / power / temperature at a chosen instant, via NVML.
//
// The plan-diagnostics A/B showed that doing ~490 ms of host work before the
// timed region makes the two-GPU stats kernel ~17% faster. It did NOT show why:
// that work is not a sleep, it is 1.17M binary searches, a sort and a full pass
// over the lane model, so it burns CPU, memory bandwidth and cache alongside
// whatever idle time it gives the GPUs. Distinguishing "the GPUs got to cool
// and re-boost" from "the CPU was busy elsewhere" needs the device state read
// at the moment the timed region opens, which is what this does.
//
// Optional: if NVML is not available the fields are simply absent, and the
// delay sweep still works without them.
#pragma once

#include <cstdio>

#if defined(ENGINE_HAVE_NVML)
#include <nvml.h>
#endif

namespace engine_cuda {

struct GpuState {
  bool valid = false;
  unsigned sm_clock_mhz = 0, mem_clock_mhz = 0;
  unsigned power_mw = 0, temperature_c = 0;
  unsigned throttle_reasons_lo = 0;   // nvmlClocksThrottleReasons, low 32 bits
};

inline GpuState read_gpu_state(int device) {
  GpuState s;
#if defined(ENGINE_HAVE_NVML)
  static bool inited = (nvmlInit_v2() == NVML_SUCCESS);
  if (!inited) return s;
  nvmlDevice_t d;
  if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(device), &d) != NVML_SUCCESS)
    return s;
  unsigned v = 0;
  unsigned long long r = 0;
  if (nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &v) == NVML_SUCCESS) s.sm_clock_mhz = v;
  if (nvmlDeviceGetClockInfo(d, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) s.mem_clock_mhz = v;
  if (nvmlDeviceGetPowerUsage(d, &v) == NVML_SUCCESS) s.power_mw = v;
  if (nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS)
    s.temperature_c = v;
  if (nvmlDeviceGetCurrentClocksThrottleReasons(d, &r) == NVML_SUCCESS)
    s.throttle_reasons_lo = static_cast<unsigned>(r & 0xffffffffu);
  s.valid = true;
#else
  (void)device;
#endif
  return s;
}

// JSON object for one device, or null when NVML is unavailable.
inline int gpu_state_json(char* buf, size_t n, const GpuState& s) {
  if (!s.valid) return snprintf(buf, n, "null");
  return snprintf(buf, n,
                  "{\"sm_clock_mhz\":%u,\"mem_clock_mhz\":%u,\"power_w\":%.1f,"
                  "\"temperature_c\":%u,\"throttle_reasons\":%u}",
                  s.sm_clock_mhz, s.mem_clock_mhz, s.power_mw / 1000.0,
                  s.temperature_c, s.throttle_reasons_lo);
}

}  // namespace engine_cuda
