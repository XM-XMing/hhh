#include "planning/xm_protocol.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using planning::xm_protocol::v4::EndpointObservationSnapshot;
using planning::xm_protocol::v4::ObservationRef;
using planning::xm_protocol::v4::ObservationSnapshotIdentityOutcome;
using planning::xm_protocol::v4::ObservationSnapshotIdentityRegistry;
using planning::xm_protocol::v4::SnapshotAck;
using planning::xm_protocol::v4::SnapshotRequest;

std::string read_text(const std::string& path) {
  std::ifstream stream(path.c_str());
  if (!stream) throw std::runtime_error("cannot read " + path);
  std::ostringstream content;
  content << stream.rdbuf();
  std::string value = content.str();
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' '))
    value.pop_back();
  return value;
}

std::vector<uint8_t> decode_hex(const std::string& text) {
  if (text.size() % 2 != 0) throw std::runtime_error("odd hex length");
  std::vector<uint8_t> bytes;
  for (size_t index = 0; index < text.size(); index += 2) {
    unsigned int value = 0;
    std::istringstream stream(text.substr(index, 2));
    stream >> std::hex >> value;
    if (stream.fail()) throw std::runtime_error("invalid hex");
    bytes.push_back(static_cast<uint8_t>(value));
  }
  return bytes;
}

std::array<uint8_t, 32> decode_hash(const std::string& text) {
  const std::vector<uint8_t> bytes = decode_hex(text);
  if (bytes.size() != 32) throw std::runtime_error("hash must contain 32 bytes");
  std::array<uint8_t, 32> value{{}};
  std::copy(bytes.begin(), bytes.end(), value.begin());
  return value;
}

std::string sha256_hex(const std::vector<uint8_t>& bytes) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(bytes.data(), bytes.size(), digest);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned char value : digest) output << std::setw(2) << static_cast<int>(value);
  return output.str();
}

void assert_equal(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected,
                  const std::string& label) {
  if (actual != expected) throw std::runtime_error(label + " bytes mismatch");
}

ObservationRef observation_ref() {
  ObservationRef value;
  value.schema_version = 4;
  value.runtime_instance_id = "worker-00-runtime-test";
  value.episode_id = "episode-54";
  value.reset_id = "reset-54";
  value.state_id = 1514;
  value.depth_id = "depth-1514";
  value.sim_time_ns = 559999987ULL;
  return value;
}

EndpointObservationSnapshot snapshot(const std::array<uint8_t, 32>& snapshot_hash) {
  EndpointObservationSnapshot value;
  value.observation_ref = observation_ref();
  value.state_bytes = decode_hex("00ff1073746174652d3135313400");
  value.depth_bytes = decode_hex("0102030405060708090a0b0c0d0e0f10");
  value.snapshot_hash = snapshot_hash;
  return value;
}

SnapshotRequest request() {
  SnapshotRequest value;
  value.observation_ref = observation_ref();
  value.execution_id = 44000000000054ULL;
  value.result_payload_hash = decode_hash(std::string(64, 'a'));
  value.command_sequence_hash = decode_hash(std::string(64, 'b'));
  return value;
}

SnapshotAck ack(const std::array<uint8_t, 32>& snapshot_hash) {
  SnapshotAck value;
  value.observation_ref = observation_ref();
  value.snapshot_hash = snapshot_hash;
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string fixture_dir = argv[1];
  const std::array<uint8_t, 32> snapshot_hash = decode_hash(
      "a91d4155d93c6dbaddec2b7bf617065c92f04507fd60505cae4845db13684e5a");
  const EndpointObservationSnapshot value = snapshot(snapshot_hash);

  const std::vector<uint8_t> ref_bytes =
      planning::xm_protocol::v4::canonical_observation_ref_bytes(value.observation_ref);
  const std::vector<uint8_t> snapshot_bytes =
      planning::xm_protocol::v4::canonical_endpoint_observation_snapshot_bytes(value);
  const std::vector<uint8_t> request_bytes =
      planning::xm_protocol::v4::canonical_snapshot_request_bytes(request());
  const std::vector<uint8_t> ack_bytes =
      planning::xm_protocol::v4::canonical_snapshot_ack_bytes(ack(snapshot_hash));

  assert_equal(ref_bytes, decode_hex(read_text(fixture_dir + "/observation_ref.msgpack.hex")),
               "observation ref");
  assert_equal(snapshot_bytes, decode_hex(read_text(fixture_dir + "/snapshot.msgpack.hex")),
               "snapshot");
  assert_equal(request_bytes, decode_hex(read_text(fixture_dir + "/snapshot_request.msgpack.hex")),
               "snapshot request");
  assert_equal(ack_bytes, decode_hex(read_text(fixture_dir + "/snapshot_ack.msgpack.hex")),
               "snapshot ack");
  if (sha256_hex(ref_bytes) !=
      "98a5377a5bafba35c2b4ce54e6b6dd4419fa973bc1abbdc3be212d605dc364dc")
    throw std::runtime_error("observation ref hash mismatch");
  if (sha256_hex(snapshot_bytes) !=
      "a91d4155d93c6dbaddec2b7bf617065c92f04507fd60505cae4845db13684e5a")
    throw std::runtime_error("snapshot hash mismatch");
  if (sha256_hex(request_bytes) !=
      "f5141c37a5c09d1f10be31898b151d5871ed68dce6b8d64261eece85087e5474")
    throw std::runtime_error("snapshot request hash mismatch");
  if (sha256_hex(ack_bytes) !=
      "a240ca1087508618158106ffdef1bfc0212c0d1a1eec7749d1d6ec422d71d80c")
    throw std::runtime_error("snapshot ack hash mismatch");
  for (int index = 0; index < 100; ++index) {
    if (planning::xm_protocol::v4::canonical_endpoint_observation_snapshot_bytes(value) != snapshot_bytes)
      throw std::runtime_error("snapshot serialization is not deterministic");
  }

  ObservationSnapshotIdentityRegistry registry;
  if (registry.Observe(value) != ObservationSnapshotIdentityOutcome::kFirst)
    throw std::runtime_error("first snapshot identity outcome mismatch");
  if (registry.Observe(value) != ObservationSnapshotIdentityOutcome::kDuplicate)
    throw std::runtime_error("duplicate snapshot identity outcome mismatch");
  EndpointObservationSnapshot conflicting = value;
  conflicting.snapshot_hash[0] ^= 0xff;
  try {
    registry.Observe(conflicting);
    throw std::runtime_error("conflicting snapshot was accepted");
  } catch (const std::invalid_argument&) {
  }
  return 0;
}
