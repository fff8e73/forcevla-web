# Offline worker dependencies

- Installed `marvin_qp_controller` core/header and installed robot models are referenced in place, not copied. The worker contains no ROS wrapper or robot-driver code.
- DAQP v0.7.2 is vendored under `vendor/daqp-0.7.2`; retain its LICENSE. Source: `https://github.com/darnstrom/daqp/tree/v0.7.2`.
- nlohmann/json v3.11.3 single header is vendored under `vendor/nlohmann/json.hpp`; its license and copyright notice are in the header. Source: `https://github.com/nlohmann/json/tree/v3.11.3`.
- Pinocchio 4.0.0, Coal 3.0.3, Eigen and yaml-cpp are linked from the existing system installation.

The installed controller's DAQP settings ABI is 112 bytes. The separately installed Python DAQP 0.8.7 writes an additional field and is deliberately not linked. The worker checks the compatible settings size at compile time. Matching the core ABI and functional tests does not assert bitwise equality with every version of the original controller's dependencies.
