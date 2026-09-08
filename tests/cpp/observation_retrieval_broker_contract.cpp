#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/observation_retrieval_broker.hpp"

namespace {

namespace xp = planning::xm_protocol;
namespace broker = planning::observation_retrieval_broker;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::array<uint8_t, 32> Hash(uint8_t value) {
  std::array<uint8_t, 32> hash{{}};
  hash.fill(value);
  return hash;
}

xp::v4::PrimitiveExecutionResult CompleteResult() {
  xp::v4::PrimitiveExecutionResult result;
  result.runtime_instance_id = "worker-00-runtime-test";
  result.execution_id = 0x0102030405060708ULL;
  result.status = "COMPLETE";
  result.applied_frame_count = 25;
  result.has_first_applied_state_id = true;
  result.first_applied_state_id = 4000;
  result.has_endpoint_state_id = true;
  result.endpoint_state_id = 4024;
  result.last_applied_frame_index = 24;
  result.reason_code = "NONE";
  result.command_sequence_hash = Hash(0x11);
  result.has_endpoint_sim_time_ns = true;
  result.endpoint_sim_time_ns = 5000000000ULL;
  result.has_endpoint_observation_ref = true;
  result.endpoint_observation_ref.schema_version = 4;
  result.endpoint_observation_ref.runtime_instance_id = result.runtime_instance_id;
  result.endpoint_observation_ref.episode_id = "episode-v4-0001";
  result.endpoint_observation_ref.reset_id = "reset-v4-0001";
  result.endpoint_observation_ref.state_id = 4024;
  result.endpoint_observation_ref.depth_id = "depth-v4-0001";
  result.endpoint_observation_ref.sim_time_ns = 5000000000ULL;
  result.result_payload_hash = broker::ResultPayloadHash(result);
  return result;
}

xp::v4::EndpointObservationSnapshot SnapshotFor(
    const xp::v4::SnapshotRequest& request) {
  xp::v4::EndpointObservationSnapshot snapshot;
  snapshot.observation_ref = request.observation_ref;
  snapshot.state_bytes = {0x01, 0x02, 0x03};
  snapshot.depth_bytes = {0x04, 0x05};
  snapshot.snapshot_hash = broker::SnapshotHash(snapshot);
  return snapshot;
}

class FakeUnityRequester : public broker::IUnitySnapshotRequester {
 public:
  bool available = true;
  std::vector<xp::v4::SnapshotRequest> requests;

  bool Request(const xp::v4::SnapshotRequest& request) override {
    if (!available) return false;
    requests.push_back(request);
    return true;
  }
};

class FakeSnapshotAckSink : public broker::ISnapshotAckSink {
 public:
  std::vector<xp::v4::SnapshotAck> acknowledgements;

  void Send(const xp::v4::SnapshotAck& ack) override {
    acknowledgements.push_back(ack);
  }
};

void NormalRetrievalStoresExactImmutableSnapshot() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  const xp::v4::PrimitiveExecutionResult result = CompleteResult();

  Require(retrieval.ReceiveComplete(result, 0ULL) == broker::RequestOutcome::REQUESTED,
          "normal COMPLETE must request snapshot");
  Require(requester.requests.size() == 1, "one exact Unity request");
  const xp::v4::SnapshotRequest request = requester.requests.front();
  Require(request.execution_id == result.execution_id, "request execution id");
  Require(request.result_payload_hash == result.result_payload_hash, "request result hash");
  Require(request.command_sequence_hash == result.command_sequence_hash, "request command hash");
  Require(request.observation_ref.runtime_instance_id == result.runtime_instance_id,
          "request runtime identity");
  Require(request.observation_ref.depth_id == "depth-v4-0001", "request depth identity");

  xp::v4::EndpointObservationSnapshot snapshot = SnapshotFor(request);
  Require(retrieval.ReceiveSnapshot(request, snapshot) == broker::SnapshotOutcome::STORED,
          "matching snapshot must be stored");
  Require(retrieval.pending_count() == 0, "stored snapshot clears pending request");
  Require(ack_sink.acknowledgements.size() == 1, "snapshot receipt acknowledgement");
  Require(ack_sink.acknowledgements[0].snapshot_hash == snapshot.snapshot_hash,
          "ack snapshot hash");

  xp::v4::EndpointObservationSnapshot stored;
  Require(retrieval.GetSnapshot(request, &stored), "snapshot exposed by exact request");
  Require(stored.observation_ref.depth_id == request.observation_ref.depth_id,
          "stored depth identity");
  Require(stored.state_bytes == snapshot.state_bytes, "stored immutable state bytes");
}

void ExactObservationKeyMismatchIsRejected() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "exact-key setup request");
  const xp::v4::SnapshotRequest request = requester.requests.front();
  xp::v4::EndpointObservationSnapshot wrong = SnapshotFor(request);
  wrong.observation_ref.depth_id = "depth-v4-wrong";
  wrong.snapshot_hash = broker::SnapshotHash(wrong);

  Require(retrieval.ReceiveSnapshot(request, wrong) ==
              broker::SnapshotOutcome::PROTOCOL_ERROR,
          "wrong observation ref must be rejected");
  Require(retrieval.pending_count() == 1, "wrong ref preserves pending request");
  Require(cache.size() == 0, "wrong ref cannot populate cache");
}

void MissingSnapshotIsExplicitAndRetryable() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "missing setup request");
  const xp::v4::SnapshotRequest request = requester.requests.front();

  Require(retrieval.ReportMissingSnapshot(request) == broker::SnapshotOutcome::MISSING,
          "missing snapshot must be explicit");
  Require(retrieval.pending_count() == 1, "missing snapshot remains pending");
  Require(cache.size() == 0, "missing snapshot cannot populate cache");
}

void DuplicateSnapshotIsSafeAndCacheIsImmutable() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "duplicate setup request");
  const xp::v4::SnapshotRequest request = requester.requests.front();
  xp::v4::EndpointObservationSnapshot snapshot = SnapshotFor(request);

  Require(retrieval.ReceiveSnapshot(request, snapshot) == broker::SnapshotOutcome::STORED,
          "first snapshot stored");
  snapshot.state_bytes[0] = 0xff;
  Require(retrieval.ReceiveSnapshot(request, SnapshotFor(request)) ==
              broker::SnapshotOutcome::DUPLICATE,
          "same immutable snapshot is duplicate");
  xp::v4::EndpointObservationSnapshot stored;
  Require(retrieval.GetSnapshot(request, &stored), "stored snapshot lookup");
  Require(stored.state_bytes[0] == 0x01, "stored bytes cannot be overwritten");
  Require(ack_sink.acknowledgements.size() == 2, "duplicate snapshot is acknowledged");
}

void ConflictingSnapshotIsProtocolError() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "conflict setup request");
  const xp::v4::SnapshotRequest request = requester.requests.front();
  const xp::v4::EndpointObservationSnapshot first = SnapshotFor(request);
  Require(retrieval.ReceiveSnapshot(request, first) == broker::SnapshotOutcome::STORED,
          "conflict first snapshot stored");
  xp::v4::EndpointObservationSnapshot conflict = first;
  conflict.depth_bytes[0] ^= 0xff;
  conflict.snapshot_hash = broker::SnapshotHash(conflict);

  Require(retrieval.ReceiveSnapshot(request, conflict) ==
              broker::SnapshotOutcome::PROTOCOL_ERROR,
          "same ref different hash must be protocol error");
  Require(cache.size() == 1, "conflicting snapshot cannot overwrite cache");
  Require(retrieval.conflict_count() == 1, "conflict counter");
}

void RetryResendsTheSameExactRequest() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "retry setup request");
  const xp::v4::SnapshotRequest first = requester.requests.front();

  Require(retrieval.Poll(9ULL) == 0, "retry before deadline");
  Require(retrieval.Poll(10ULL) == 1, "one retry at deadline");
  Require(requester.requests.size() == 2, "one retry request");
  Require(xp::v4::canonical_snapshot_request_bytes(requester.requests[1]) ==
              xp::v4::canonical_snapshot_request_bytes(first),
          "retry request bytes are immutable");
  Require(retrieval.pending_count() == 1, "retry keeps pending request");
}

void DisconnectReconnectResendsPendingRequest() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "disconnect setup request");
  const xp::v4::SnapshotRequest first = requester.requests.front();
  retrieval.OnUnityDisconnected();
  Require(retrieval.Poll(10ULL) == 0, "disconnected poll cannot send");
  Require(retrieval.OnUnityReconnected(20ULL) == 1,
          "reconnect resends pending request");
  Require(requester.requests.size() == 2, "reconnect request count");
  Require(xp::v4::canonical_snapshot_request_bytes(requester.requests[1]) ==
              xp::v4::canonical_snapshot_request_bytes(first),
          "reconnect request bytes are immutable");
}

void WrongExecutionOrResultIdentityIsRejected() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);
  Require(retrieval.ReceiveComplete(CompleteResult(), 0ULL) ==
              broker::RequestOutcome::REQUESTED,
          "identity setup request");
  const xp::v4::SnapshotRequest request = requester.requests.front();
  xp::v4::SnapshotRequest wrong_execution = request;
  ++wrong_execution.execution_id;
  Require(retrieval.ReceiveSnapshot(wrong_execution, SnapshotFor(wrong_execution)) ==
              broker::SnapshotOutcome::NOT_FOUND,
          "unknown execution identity must not be accepted");
  Require(cache.size() == 0, "wrong execution cannot populate cache");

  xp::v4::SnapshotRequest wrong = request;
  wrong.result_payload_hash[0] ^= 0xff;
  const xp::v4::EndpointObservationSnapshot snapshot = SnapshotFor(request);

  Require(retrieval.ReceiveSnapshot(wrong, snapshot) ==
              broker::SnapshotOutcome::PROTOCOL_ERROR,
          "wrong result identity must be rejected");
  Require(retrieval.pending_count() == 1, "wrong identity preserves pending request");
  Require(cache.size() == 0, "wrong identity cannot populate cache");
}

void ResetRequestsWithReservedExecutionIdentityRemainIsolated() {
  broker::InMemorySnapshotCache cache;
  FakeUnityRequester requester;
  FakeSnapshotAckSink ack_sink;
  broker::ObservationRetrievalBroker retrieval(cache, requester, ack_sink, 10ULL);

  const xp::v4::PrimitiveExecutionResult result = CompleteResult();
  xp::v4::SnapshotRequest first;
  first.observation_ref = broker::ObservationRefFor(result);
  first.execution_id = 0;
  first.result_payload_hash.fill(0);
  first.command_sequence_hash.fill(0);

  xp::v4::SnapshotRequest second = first;
  second.observation_ref.episode_id = "episode-v4-0002";
  second.observation_ref.reset_id = "reset-v4-0002";
  second.observation_ref.state_id += 1;
  second.observation_ref.depth_id = "depth-v4-0002";
  second.observation_ref.sim_time_ns += 1000000ULL;

  Require(retrieval.ReceiveRequest(first, 0ULL) == broker::RequestOutcome::REQUESTED,
          "first reset snapshot request must be accepted");
  Require(retrieval.ReceiveRequest(second, 1ULL) == broker::RequestOutcome::REQUESTED,
          "different reset snapshot identity must be accepted");
  Require(requester.requests.size() == 2,
          "different reset identities must create two Unity requests");
  Require(retrieval.pending_count() == 2,
          "different reset identities must remain independently pending");

  Require(retrieval.ReceiveRequest(first, 2ULL) == broker::RequestOutcome::DUPLICATE,
          "same reset request must remain idempotent");
  Require(retrieval.request_count() == 2,
          "duplicate reset request must not increase request count");
}

}  // namespace

int main() {
  try {
    NormalRetrievalStoresExactImmutableSnapshot();
    ExactObservationKeyMismatchIsRejected();
    MissingSnapshotIsExplicitAndRetryable();
    DuplicateSnapshotIsSafeAndCacheIsImmutable();
    ConflictingSnapshotIsProtocolError();
    RetryResendsTheSameExactRequest();
    DisconnectReconnectResendsPendingRequest();
    WrongExecutionOrResultIdentityIsRejected();
    ResetRequestsWithReservedExecutionIdentityRemainIsolated();
    std::cout << "C++ observation retrieval broker tests: 9 passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
