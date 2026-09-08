#pragma once

namespace planning::transport {

enum class ZmqSocketOptionCriticality {
    REQUIRED_FOR_CORRECTNESS,
    REQUIRED_FOR_EXACTLY_ONCE,
    REQUIRED_FOR_LIFECYCLE_CLEANUP,
    OPTIONAL_PERFORMANCE,
    OPTIONAL_RECOVERY_TUNING,
    UNKNOWN,
};

ZmqSocketOptionCriticality ZmqSocketOptionCriticalityFor(int option);
const char* ZmqSocketOptionName(int option);
const char* ZmqSocketOptionCriticalityName(
    ZmqSocketOptionCriticality criticality);

}
