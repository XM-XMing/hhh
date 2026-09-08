#include <planning/bridge/bridge_util.hpp>

#include <array>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <exception>
#include <iterator>
#include <thread>
#include <unistd.h>

#include <openssl/sha.h>
#include <zmq.h>

#include <planning/protocol/xm_protocol.hpp>

namespace xp = planning::xm_protocol;

namespace planning::bridge_util {

int64_t WallTimeNs() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string CurrentThreadId() {
  std::ostringstream stream;
  stream << std::this_thread::get_id();
  return stream.str();
}

std::string SocketThreadDiagnosticFields(const std::string& owner_thread_id) {
  const std::string access_thread_id = CurrentThreadId();
  return "\"socket_owner_thread_id\":\"" + owner_thread_id +
      "\",\"socket_access_thread_id\":\"" + access_thread_id +
      "\",\"socket_owner_match\":" +
      std::string(owner_thread_id == access_thread_id ? "true" : "false");
}

std::string HexBytes(const uint8_t* data, size_t size) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (size_t index = 0; index < size; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(data[index]);
  }
  return stream.str();
}

std::string HexBytes(const std::vector<uint8_t>& data) {
  return HexBytes(data.data(), data.size());
}

std::string ExecutablePath() {
  std::array<char, 4096> buffer{{}};
  const ssize_t length =
      readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) return "";
  return std::string(buffer.data(), static_cast<size_t>(length));
}

std::string FileSha256(const std::string& path) {
  if (path.empty()) return "";
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream) return "";
  const std::string contents(
      (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (!stream.eof() && stream.fail()) return "";
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest{{}};
  SHA256(
      reinterpret_cast<const unsigned char*>(contents.data()), contents.size(),
      digest.data());
  return HexBytes(digest.data(), digest.size());
}

std::string PrimitiveCommandSequenceHashHex(
    const planning::PrimitiveExecution& message) {
  try {
    std::vector<xp::v4::PrimitiveExecutionFrame> frames;
    frames.reserve(message.commands.size());
    for (size_t index = 0; index < message.commands.size(); ++index) {
      const auto& command = message.commands[index];
      xp::v4::PrimitiveExecutionFrame frame;
      frame.frame_index = static_cast<uint32_t>(index);
      frame.command_id = message.command_ids[index];
      frame.action = {
          static_cast<float>(command.linear.x),
          static_cast<float>(command.linear.y),
          static_cast<float>(command.linear.z),
          static_cast<float>(command.angular.z)};
      frames.push_back(frame);
    }
    const std::vector<uint8_t> canonical =
        xp::v4::canonical_command_sequence_bytes(frames);
    std::array<uint8_t, SHA256_DIGEST_LENGTH> digest{{}};
    SHA256(canonical.data(), canonical.size(), digest.data());
    return HexBytes(digest.data(), digest.size());
  } catch (const std::exception&) {
    return "";
  }
}

int ZmqSocketEvents(void* socket) {
  if (socket == nullptr) return -1;
  int events = 0;
  size_t size = sizeof(events);
  if (zmq_getsockopt(socket, ZMQ_EVENTS, &events, &size) != 0) return -1;
  return events;
}

std::string RuntimeInstanceIdFromDealerIdentity(
    const std::vector<uint8_t>& identity) {
  for (const uint8_t value : identity) {
    const unsigned char character = static_cast<unsigned char>(value);
    if (!(std::isalnum(character) || character == '-' || character == '_' ||
          character == '.')) {
      return "";
    }
  }
  return std::string(identity.begin(), identity.end());
}

void WritePythonDiagnostic(
    std::ofstream* stream, const std::string& body) {
  if (stream == nullptr || !stream->is_open()) return;
  (*stream) << "{\"timestamp_monotonic_ns\":" << WallTimeNs()
            << "," << body << "}\n";
  stream->flush();
}

const char* ZmqMonitorEventName(uint16_t event) {
  switch (event) {
    case ZMQ_EVENT_ACCEPTED: return "ACCEPTED";
    case ZMQ_EVENT_ACCEPT_FAILED: return "ACCEPT_FAILED";
    case ZMQ_EVENT_CLOSED: return "CLOSED";
    case ZMQ_EVENT_CLOSE_FAILED: return "CLOSE_FAILED";
    case ZMQ_EVENT_CONNECTED: return "CONNECTED";
    case ZMQ_EVENT_CONNECT_DELAYED: return "CONNECT_DELAYED";
    case ZMQ_EVENT_CONNECT_RETRIED: return "CONNECT_RETRIED";
    case ZMQ_EVENT_DISCONNECTED: return "DISCONNECTED";
    case ZMQ_EVENT_HANDSHAKE_FAILED_NO_DETAIL:
      return "HANDSHAKE_FAILED_NO_DETAIL";
    case ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL:
      return "HANDSHAKE_FAILED_PROTOCOL";
    case ZMQ_EVENT_HANDSHAKE_FAILED_AUTH:
      return "HANDSHAKE_FAILED_AUTH";
    case ZMQ_EVENT_HANDSHAKE_SUCCEEDED: return "HANDSHAKE_SUCCEEDED";
    default: return "UNKNOWN";
  }
}

}  // namespace planning::bridge_util
