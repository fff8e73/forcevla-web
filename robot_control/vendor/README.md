# Isolated DAQP source

Upstream: `https://github.com/darnstrom/daqp.git`

Pinned tag: `v0.7.2`  
Pinned commit: `653e22316c1d54ee7c71500f45f3ff7afaadad07`

License: MIT; see `daqp/LICENSE`.

Only `src/*.c` and `include/*.h` participate in the local static build. `build_native.sh` checks the pinned clean checkout. No upstream install step is run; no system or robot-controller library is replaced.

This revision supplies a 112-byte DAQPSettings layout compatible with the observed installed QP caller. The exact DAQP source revision originally embedded in the online executable is unknown. Reusing the installed QP core and passing numerical tests does not by itself prove complete online wrapper/solver equivalence.
