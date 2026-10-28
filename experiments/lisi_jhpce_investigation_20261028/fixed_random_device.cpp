// LD_PRELOAD shim: makes std::random_device (libstdc++) return one fixed value, so the seed that scib 1.1.7's
// compiled knn_graph helper draws (mt19937 gen(random_device()())) is the same on every machine.
// The value is read from the environment variable LISI_RANDOM_DEVICE_VALUE (decimal or 0x hex, 32 bit); a missing or
// malformed value aborts, so a job can never run on an unintended default. 0xFFFFFFFF is what RDRAND/RDSEED return on the
// workstation (AMD Ryzen 5 3600; see rdrand_probe.py). It replaces only the exported members the helper calls
// (_M_init, _M_fini, _M_getval); it does not touch scib, the helper binary, or any metric code.
// Build (links libc only, so it cannot pull a second libstdc++ into the scorer's process):
//   g++ -O1 -fPIC -shared -nostdlib -fno-exceptions -fno-rtti -o fixed_random_device.so fixed_random_device.cpp -lc
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
namespace std {
void random_device::_M_init(const std::string&) {}
void random_device::_M_init_pretr1(const std::string&) {}
void random_device::_M_fini() {}
random_device::result_type random_device::_M_getval() {
   const char* s = std::getenv("LISI_RANDOM_DEVICE_VALUE");
   char* end = nullptr;
   unsigned long long v = s ? std::strtoull(s, &end, 0) : 0;
   if (!s || end == s || *end != '\0' || v > 0xFFFFFFFFull) {
      std::fprintf(stderr, "fixed_random_device: LISI_RANDOM_DEVICE_VALUE missing or not a 32-bit integer\n");
      std::abort();
   }
   return static_cast<result_type>(v);
}
}
