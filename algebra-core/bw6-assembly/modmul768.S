#if defined(__arm64) || defined(__arm64__) || defined(__ARM64_ARCH_8__) || defined(__ARM_ARCH_8A) || (__ARM_ARCH==8) || defined(__ARM_ARCH_ISA_A64) || defined(__ARM_PCS_AAPCS64)
#include "modmul768-armv8.SCR.S"

#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(__amd64__)
#include "modmul768-x86_64.SCR.S"

#endif
