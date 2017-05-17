#ifndef __PROC_READER_H__
#define __PROC_READER_H__

#include <map>
#include <set>
#include <string>

#define CPU_INFO_NEON_STR "neon"
#define CPU_INFO_NEON2_STR "neon2"
#define CPU_INFO_VFPV3_STR "vfpv3"
#define CPU_INFO_VFPV3D16_STR "vfpv3d16"

#define CPU_INFO_SSE_STR "sse"
#define CPU_INFO_SSE2_STR "sse2"
#define CPU_INFO_SSSE3_STR "ssse3"

#define CPU_INFO_ARCH_ARMV7_STR "ARMv7"
#define CPU_INFO_ARCH_ARMV6_STR "ARMv6"
#define CPU_INFO_ARCH_ARMV5_STR "ARMv5"

#define CPU_INFO_ARCH_X86_STR "x86"


// public part
std::map<std::string, std::string> GetCpuInfo();

#endif