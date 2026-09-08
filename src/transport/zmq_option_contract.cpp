#include <planning/transport/zmq_option_contract.hpp>

#include <zmq.h>

namespace planning::transport {
namespace {

struct OptionDescriptor {
    int option;
    const char* name;
    ZmqSocketOptionCriticality criticality;
};

constexpr OptionDescriptor kProductionOptions[] = {
    {ZMQ_LINGER, "ZMQ_LINGER",
     ZmqSocketOptionCriticality::REQUIRED_FOR_LIFECYCLE_CLEANUP},
    {ZMQ_SNDHWM, "ZMQ_SNDHWM",
     ZmqSocketOptionCriticality::OPTIONAL_PERFORMANCE},
    {ZMQ_RCVHWM, "ZMQ_RCVHWM",
     ZmqSocketOptionCriticality::OPTIONAL_PERFORMANCE},
    {ZMQ_ROUTER_HANDOVER, "ZMQ_ROUTER_HANDOVER",
     ZmqSocketOptionCriticality::REQUIRED_FOR_EXACTLY_ONCE},
    {ZMQ_SUBSCRIBE, "ZMQ_SUBSCRIBE",
     ZmqSocketOptionCriticality::REQUIRED_FOR_CORRECTNESS},
};

const OptionDescriptor* FindOption(int option) {
    for (const OptionDescriptor& descriptor : kProductionOptions) {
        if (descriptor.option == option) return &descriptor;
    }
    return nullptr;
}

}

ZmqSocketOptionCriticality ZmqSocketOptionCriticalityFor(int option) {
    const OptionDescriptor* descriptor = FindOption(option);
    return descriptor == nullptr
        ? ZmqSocketOptionCriticality::UNKNOWN
        : descriptor->criticality;
}

const char* ZmqSocketOptionName(int option) {
    const OptionDescriptor* descriptor = FindOption(option);
    return descriptor == nullptr ? "UNKNOWN_ZMQ_OPTION" : descriptor->name;
}

const char* ZmqSocketOptionCriticalityName(
    ZmqSocketOptionCriticality criticality) {
    switch (criticality) {
        case ZmqSocketOptionCriticality::REQUIRED_FOR_CORRECTNESS:
            return "REQUIRED_FOR_CORRECTNESS";
        case ZmqSocketOptionCriticality::REQUIRED_FOR_EXACTLY_ONCE:
            return "REQUIRED_FOR_EXACTLY_ONCE";
        case ZmqSocketOptionCriticality::REQUIRED_FOR_LIFECYCLE_CLEANUP:
            return "REQUIRED_FOR_LIFECYCLE_CLEANUP";
        case ZmqSocketOptionCriticality::OPTIONAL_PERFORMANCE:
            return "OPTIONAL_PERFORMANCE";
        case ZmqSocketOptionCriticality::OPTIONAL_RECOVERY_TUNING:
            return "OPTIONAL_RECOVERY_TUNING";
        case ZmqSocketOptionCriticality::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

}
