#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <planning/PrimitiveExecution.h>

namespace planning::bridge_util {

int64_t WallTimeNs();
std::string CurrentThreadId();
std::string SocketThreadDiagnosticFields(const std::string& owner_thread_id);
std::string HexBytes(const uint8_t* data, size_t size);
std::string HexBytes(const std::vector<uint8_t>& data);
std::string ExecutablePath();
std::string FileSha256(const std::string& path);
std::string PrimitiveCommandSequenceHashHex(const planning::PrimitiveExecution& message);
int ZmqSocketEvents(void* socket);
std::string RuntimeInstanceIdFromDealerIdentity(const std::vector<uint8_t>& identity);
void WritePythonDiagnostic(std::ofstream* stream, const std::string& body);
const char* ZmqMonitorEventName(uint16_t event);

}  // namespace planning::bridge_util
