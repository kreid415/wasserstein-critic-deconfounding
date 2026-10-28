"""Reads the CPU's RDRAND and RDSEED instructions directly (machine code run through ctypes) and the CPUID feature bits.
libstdc++'s std::random_device, which seeds the random row subsample inside scib's knn_graph helper, uses RDRAND when
CPUID reports it. Prints one JSON line. Usage: python rdrand_probe.py"""
import ctypes, json, mmap, os, platform
KEEP = []   # the mappings must outlive the calls
def call(code):
    m = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC); m.write(code); KEEP.append(m)
    f = ctypes.CFUNCTYPE(ctypes.c_uint32)(ctypes.addressof(ctypes.c_char.from_buffer(m))); return f
# rdrand eax ; setc cl ; (carry flag => valid)  -> return value in eax, valid flag via second function
rd = call(bytes([0x0F,0xC7,0xF0, 0xC3]))                              # rdrand eax; ret
rds = call(bytes([0x0F,0xC7,0xF8, 0xC3]))                             # rdseed eax; ret
cpuid1 = call(bytes([0x53, 0xB8,1,0,0,0, 0x31,0xC9, 0x0F,0xA2, 0x89,0xC8, 0x5B, 0xC3]))  # ecx of leaf 1 -> eax
fl = open("/proc/cpuinfo").read()
ecx = cpuid1()
out = dict(host=os.uname().nodename, cpu=[l.split(":")[1].strip() for l in fl.splitlines() if l.startswith("model name")][0],
           cpuid_rdrand_bit=bool(ecx >> 30 & 1), proc_cpuinfo_rdrand=" rdrand" in fl, proc_cpuinfo_rdseed=" rdseed" in fl,
           rdrand=[hex(rd()) for _ in range(6)], rdseed=[hex(rds()) for _ in range(6)], kernel=platform.release())
print(json.dumps(out))
