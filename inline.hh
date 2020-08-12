#ifndef __always_inline

#ifndef __GNUC__
#warning "__always_inline: prng functions might not be properly inlined."
#define __always_inline inline
#else
#define __always_inline __attribute__((__always_inline__)) inline
#endif

#endif
