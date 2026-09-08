"""Canonical runtime port profiles and fail-closed port validation.

Every process-facing endpoint used by a managed worker or a direct component
smoke is resolved by this module. Launchers consume projections from the
profile; they do not derive endpoint offsets independently.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from planning.protocol.constants import SCHEMA_VERSION

from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT


WORKER_RUNTIME_SPEC_SCHEMA = "p3_worker_runtime_spec_v2"
STARTUP_WORKER_ENDPOINT_MISMATCH = "STARTUP_WORKER_ENDPOINT_MISMATCH"

# These are configuration inputs to the canonical resolver, not defaults owned
# by a launcher or a transport implementation.  Keep the historical resolved
# values unchanged while making their ownership explicit.
MANAGED_TRAINING_PORT_DEFAULTS = {
    "master_port_base": 11621,
    "command_port_base": 10553,
    "depth_port_base": 12554,
    "port_stride": 20,
}
# The formal collection topology is intentionally still configured at twelve
# workers by ``config/pre_bc.yaml``.  This is the explicit upper bound for a
# caller that has passed the separate runtime/resource gates; it is not a
# protocol, Unity, or Bridge schema version.
MANAGED_COLLECTION_WORKER_COUNT_MAX = 24
LEGACY_COLLECTION_PORT_DEFAULTS = {
    "master_port_base": 11321,
    "command_port_base": 10253,
    "depth_port_base": 12254,
    "port_stride": 20,
}

_PORT_FIELDS = (
    "master_port",
    "command_port",
    "state_port",
    "depth_port",
    "python_command_port",
    "unity_command_port",
    "unity_result_port",
    "python_result_port",
    "unity_snapshot_port",
    "python_snapshot_port",
)


class PortContractError(ValueError):
    """A runtime port profile is missing, mixed, invalid, or colliding."""


class WorkerEndpointMismatchError(PortContractError):
    """Compatibility error name used by the reliable worker lifecycle."""


def _fail(message: str) -> None:
    raise WorkerEndpointMismatchError(
        "{}: {}".format(STARTUP_WORKER_ENDPOINT_MISMATCH, message)
    )


def _check_port(name: str, value: int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        _fail("{} is not an integer port".format(name))
    if not 1 <= port <= 65535:
        _fail("{}={} is outside TCP/UDP range".format(name, port))
    return port


def _endpoint(port: int) -> str:
    return "tcp://127.0.0.1:{}".format(int(port))


def _validate_profile_ports(profile: object) -> None:
    owners: Dict[int, str] = {}
    for name in _PORT_FIELDS:
        port = _check_port(name, getattr(profile, name))
        previous = owners.get(port)
        if previous is not None:
            raise PortContractError(
                "port collision: {} and {} both resolve to {}".format(
                    previous, name, port
                )
            )
        owners[port] = name


class _PortProjectionMixin:
    """Shared projections from one immutable port owner."""

    @property
    def ros_master_uri(self) -> str:
        return _endpoint(self.master_port).replace("tcp://", "http://", 1)

    @property
    def python_command_endpoint(self) -> str:
        return _endpoint(self.python_command_port)

    @property
    def unity_command_endpoint(self) -> str:
        return _endpoint(self.unity_command_port)

    @property
    def python_result_endpoint(self) -> str:
        return _endpoint(self.python_result_port)

    @property
    def unity_result_endpoint(self) -> str:
        return _endpoint(self.unity_result_port)

    @property
    def python_snapshot_endpoint(self) -> str:
        return _endpoint(self.python_snapshot_port)

    @property
    def unity_snapshot_endpoint(self) -> str:
        return _endpoint(self.unity_snapshot_port)

    @property
    def python_reset_endpoint(self) -> str:
        return self.python_command_endpoint

    @property
    def unity_reset_endpoint(self) -> str:
        return self.unity_command_endpoint

    @property
    def all_ports(self) -> Tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in _PORT_FIELDS)

    def port_mapping(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in _PORT_FIELDS}

    def bridge_launch_args(self) -> Dict[str, object]:
        """Exact ROS/Bridge projection; no endpoint is inferred downstream."""

        return {
            "cmd_port": self.command_port,
            "state_port": self.state_port,
            "depth_port": self.depth_port,
            "python_command_port": self.python_command_port,
            "unity_command_port": self.unity_command_port,
            "command_runtime_instance_id": self.runtime_instance_id,
            "execution_result_port": self.unity_result_port,
            "observation_snapshot_port": self.unity_snapshot_port,
            "python_result_port": self.python_result_port,
            "python_snapshot_port": self.python_snapshot_port,
        }

    def unity_launch_args(self) -> Dict[str, object]:
        return {
            "cmdSubPort": self.command_port,
            "statePubPort": self.state_port,
            "depthPubPort": self.depth_port,
            "executionResultPort": self.unity_result_port,
            "observationSnapshotPort": self.unity_snapshot_port,
            "reliableCommandPort": self.unity_command_port,
            "primitiveResultSchema": SCHEMA_VERSION,
            "runtimeInstanceId": self.runtime_instance_id,
        }

    def unity_launch_argv(self) -> Tuple[str, ...]:
        args = self.unity_launch_args()
        return (
            "-cmdSubPort",
            str(args["cmdSubPort"]),
            "-statePubPort",
            str(args["statePubPort"]),
            "-depthPubPort",
            str(args["depthPubPort"]),
            "-executionResultPort",
            str(args["executionResultPort"]),
            "-observationSnapshotPort",
            str(args["observationSnapshotPort"]),
            "-reliableCommandPort",
            str(args["reliableCommandPort"]),
            "-runtimeInstanceId",
            str(args["runtimeInstanceId"]),
        )

    def python_backend_config(self) -> Dict[str, object]:
        endpoints = {
            "command": self.python_command_endpoint,
            "result": self.python_result_endpoint,
            "snapshot": self.python_snapshot_endpoint,
            "reset": self.python_reset_endpoint,
        }
        return {
            "runtime_instance_id": self.runtime_instance_id,
            "command_endpoint": endpoints["command"],
            "result_endpoint": endpoints["result"],
            "snapshot_endpoint": endpoints["snapshot"],
            "reset_endpoint": endpoints["reset"],
            "endpoints": endpoints,
            "ports": self.port_mapping(),
            "observation_semantics": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        }


@dataclass(frozen=True)
class WorkerRuntimeSpec(_PortProjectionMixin):
    """All process-facing endpoints for one deterministic managed worker."""

    worker_id: int
    runtime_instance_id: str
    master_port: int
    ros_home: str
    command_port: int
    state_port: int
    depth_port: int
    python_command_port: int
    unity_command_port: int
    unity_result_port: int
    python_result_port: int
    unity_snapshot_port: int
    python_snapshot_port: int
    training_run_id: str = "legacy-unspecified"
    runtime_launch_nonce: str = "legacy-unspecified"

    def __post_init__(self) -> None:
        if int(self.worker_id) < 0:
            _fail("worker_id must be non-negative")
        if not str(self.runtime_instance_id):
            _fail("runtime_instance_id is empty")
        if not str(self.training_run_id):
            _fail("training_run_id is empty")
        if not str(self.runtime_launch_nonce):
            _fail("runtime_launch_nonce is empty")
        if not str(self.ros_home):
            _fail("ros_home is empty")
        try:
            _validate_profile_ports(self)
        except PortContractError as error:
            _fail(str(error))
        expected = {
            "state_port": self.command_port + 1,
            "unity_result_port": self.command_port + 2,
            "python_result_port": self.command_port + 3,
            "unity_snapshot_port": self.command_port + 4,
            "python_snapshot_port": self.command_port + 5,
            "python_command_port": self.command_port + 6,
            "unity_command_port": self.command_port + 7,
        }
        mismatches = {
            name: {"expected": value, "actual": getattr(self, name)}
            for name, value in expected.items()
            if int(getattr(self, name)) != int(value)
        }
        if mismatches:
            _fail("worker {} derived port mismatch: {}".format(self.worker_id, mismatches))

    def to_mapping(self) -> Dict[str, object]:
        return {
            "profile": "managed",
            "worker_id": self.worker_id,
            "training_run_id": self.training_run_id,
            "runtime_launch_nonce": self.runtime_launch_nonce,
            "runtime_instance_id": self.runtime_instance_id,
            "master_port": self.master_port,
            "ros_master_uri": self.ros_master_uri,
            "ros_home": self.ros_home,
            "cmd_port": self.command_port,
            "command_port": self.command_port,
            "state_port": self.state_port,
            "depth_port": self.depth_port,
            "python_command_port": self.python_command_port,
            "unity_command_port": self.unity_command_port,
            "unity_result_port": self.unity_result_port,
            "python_result_port": self.python_result_port,
            "unity_snapshot_port": self.unity_snapshot_port,
            "python_snapshot_port": self.python_snapshot_port,
            "reliable_command_port": self.python_command_port,
            "reliable_result_port": self.python_result_port,
            "reliable_snapshot_port": self.python_snapshot_port,
            "python_command_endpoint": self.python_command_endpoint,
            "unity_command_endpoint": self.unity_command_endpoint,
            "python_result_endpoint": self.python_result_endpoint,
            "unity_result_endpoint": self.unity_result_endpoint,
            "python_snapshot_endpoint": self.python_snapshot_endpoint,
            "unity_snapshot_endpoint": self.unity_snapshot_endpoint,
            "python_reset_endpoint": self.python_reset_endpoint,
            "unity_reset_endpoint": self.unity_reset_endpoint,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "WorkerRuntimeSpec":
        if not isinstance(mapping, Mapping):
            _fail("managed worker spec must be a mapping")
        if "profile" in mapping and str(mapping["profile"]) != "managed":
            _fail("managed worker spec profile is not managed")
        required = (
            "worker_id",
            "runtime_instance_id",
            "master_port",
            "ros_home",
            "command_port",
            "state_port",
            "depth_port",
            "python_command_port",
            "unity_command_port",
            "unity_result_port",
            "python_result_port",
            "unity_snapshot_port",
            "python_snapshot_port",
        )
        missing = [name for name in required if name not in mapping]
        if missing:
            _fail("worker spec missing required port/field {}".format(",".join(missing)))
        spec = cls(
            worker_id=int(mapping["worker_id"]),
            runtime_instance_id=str(mapping["runtime_instance_id"]),
            master_port=int(mapping["master_port"]),
            ros_home=str(mapping["ros_home"]),
            command_port=int(mapping["command_port"]),
            state_port=int(mapping["state_port"]),
            depth_port=int(mapping["depth_port"]),
            python_command_port=int(mapping["python_command_port"]),
            unity_command_port=int(mapping["unity_command_port"]),
            unity_result_port=int(mapping["unity_result_port"]),
            python_result_port=int(mapping["python_result_port"]),
            unity_snapshot_port=int(mapping["unity_snapshot_port"]),
            python_snapshot_port=int(mapping["python_snapshot_port"]),
            training_run_id=str(mapping.get("training_run_id", "legacy-unspecified")),
            runtime_launch_nonce=str(
                mapping.get("runtime_launch_nonce", "legacy-unspecified")
            ),
        )
        for name in (
            "ros_master_uri",
            "python_command_endpoint",
            "unity_command_endpoint",
            "python_result_endpoint",
            "unity_result_endpoint",
            "python_snapshot_endpoint",
            "unity_snapshot_endpoint",
            "python_reset_endpoint",
            "unity_reset_endpoint",
        ):
            if name in mapping and str(mapping[name]) != str(getattr(spec, name)):
                _fail("worker {} {} disagrees with port fields".format(spec.worker_id, name))
        reliable_aliases = {
            "reliable_command_port": spec.python_command_port,
            "reliable_result_port": spec.python_result_port,
            "reliable_snapshot_port": spec.python_snapshot_port,
        }
        for name, expected in reliable_aliases.items():
            if name in mapping and int(mapping[name]) != int(expected):
                _fail(
                    "worker {} {} disagrees with Python-facing port fields".format(
                        spec.worker_id, name
                    )
                )
        return spec


@dataclass(frozen=True)
class DirectRuntimePortProfile(_PortProjectionMixin):
    """Explicit endpoint profile for a direct/manual component smoke."""

    profile: str
    runtime_instance_id: str
    master_port: int
    command_port: int
    state_port: int
    depth_port: int
    python_command_port: int
    unity_command_port: int
    unity_result_port: int
    python_result_port: int
    unity_snapshot_port: int
    python_snapshot_port: int

    def __post_init__(self) -> None:
        if self.profile != "direct":
            raise PortContractError(
                "direct profile required, got {!r}".format(self.profile)
            )
        if not str(self.runtime_instance_id):
            raise PortContractError("direct profile runtime_instance_id is empty")
        _validate_profile_ports(self)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "runtime_instance_id": self.runtime_instance_id,
            **self.port_mapping(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "DirectRuntimePortProfile":
        if not isinstance(mapping, Mapping):
            raise PortContractError("direct profile must be a mapping")
        if str(mapping.get("profile", "")) != "direct":
            raise PortContractError("direct profile required")
        required = ("runtime_instance_id",) + _PORT_FIELDS
        missing = [name for name in required if name not in mapping]
        if missing:
            raise PortContractError(
                "direct profile missing required port/field {}".format(",".join(missing))
            )
        return cls(
            profile="direct",
            runtime_instance_id=str(mapping["runtime_instance_id"]),
            **{name: int(mapping[name]) for name in _PORT_FIELDS},
        )

    @classmethod
    def frozen_defaults(
        cls, *, runtime_instance_id: str = "direct-frozen-default"
    ) -> "DirectRuntimePortProfile":
        """Explicit direct profile matching the frozen Unity defaults."""

        return cls(
            profile="direct",
            runtime_instance_id=runtime_instance_id,
            master_port=11321,
            command_port=10253,
            state_port=10254,
            depth_port=11254,
            unity_result_port=11255,
            unity_snapshot_port=11256,
            python_result_port=11257,
            python_snapshot_port=11258,
            python_command_port=11259,
            unity_command_port=11260,
        )

    @classmethod
    def allocate(
        cls, *, runtime_instance_id: str = "direct-allocated"
    ) -> "DirectRuntimePortProfile":
        """Allocate one explicit profile for a local, non-managed smoke."""

        ports = []
        while len(ports) < len(_PORT_FIELDS):
            port = _free_port()
            if port not in ports:
                ports.append(port)
        return cls(
            profile="direct",
            runtime_instance_id=runtime_instance_id,
            **dict(zip(_PORT_FIELDS, ports)),
        )


def build_managed_evaluation_profile(
    *,
    runtime_instance_id: str,
    ros_home: Path,
    overrides: Optional[Mapping[str, object]] = None,
) -> WorkerRuntimeSpec:
    """Build the single-worker managed-evaluation profile in one place."""

    values = {
        "master_port": 11521,
        "command_port": 10453,
        "state_port": 10454,
        "depth_port": 12454,
        "python_command_port": 10459,
        "unity_command_port": 10460,
        "unity_result_port": 10455,
        "python_result_port": 10456,
        "unity_snapshot_port": 10457,
        "python_snapshot_port": 10458,
    }
    if overrides:
        unknown = set(overrides) - set(values)
        if unknown:
            raise PortContractError(
                "managed evaluation contains unknown port fields: {}".format(
                    ",".join(sorted(unknown))
                )
            )
        values.update({name: int(value) for name, value in overrides.items()})
    return WorkerRuntimeSpec(
        worker_id=0,
        runtime_instance_id=str(runtime_instance_id),
        ros_home=str(Path(ros_home)),
        training_run_id="managed-evaluation",
        runtime_launch_nonce="managed-evaluation",
        **values
    )


def resolve_runtime_port_profile(
    mapping: Mapping[str, object], *, mode: str
) -> object:
    """Resolve exactly one declared profile; reject absent or mixed modes."""

    if mode not in ("managed", "direct"):
        raise PortContractError("unknown runtime port profile mode: {}".format(mode))
    if not isinstance(mapping, Mapping):
        raise PortContractError("runtime port profile must be a mapping")
    declared = str(mapping.get("profile", ""))
    if declared != mode:
        raise PortContractError(
            "{} runtime requires profile={!r}, got {!r}".format(mode, mode, declared)
        )
    if mode == "managed":
        return WorkerRuntimeSpec.from_mapping(mapping)
    return DirectRuntimePortProfile.from_mapping(mapping)


def validate_port_profiles(profiles: Sequence[object]) -> Tuple[object, ...]:
    """Validate profiles before any child process starts."""

    normalized = tuple(profiles)
    ports: Dict[int, str] = {}
    runtime_ids = set()
    for index, profile in enumerate(normalized):
        if not isinstance(profile, (WorkerRuntimeSpec, DirectRuntimePortProfile)):
            raise PortContractError("unknown runtime port profile type")
        _validate_profile_ports(profile)
        runtime_id = str(profile.runtime_instance_id)
        if runtime_id in runtime_ids:
            raise PortContractError("runtime_instance_id collision: {}".format(runtime_id))
        runtime_ids.add(runtime_id)
        for name, port in profile.port_mapping().items():
            owner = ports.get(port)
            if owner is not None:
                raise PortContractError(
                    "port collision between {} and profile {} ({})".format(
                        owner, index, name
                    )
                )
            ports[port] = "profile {} {}".format(index, name)
    return normalized


def validate_worker_runtime_specs(
    specs: Sequence[WorkerRuntimeSpec],
) -> Tuple[WorkerRuntimeSpec, ...]:
    normalized = tuple(specs)
    if [spec.worker_id for spec in normalized] != list(range(len(normalized))):
        _fail("worker ids must be contiguous and ordered from zero")
    validate_port_profiles(normalized)
    return normalized


def build_worker_runtime_specs(
    *,
    worker_count: int,
    master_port_base: Optional[int] = None,
    command_port_base: Optional[int] = None,
    depth_port_base: Optional[int] = None,
    port_stride: Optional[int] = None,
    runtime_instance_template: str,
    ros_root: Path,
    training_run_id: str = "legacy-unspecified",
    runtime_launch_nonce: str = "legacy-unspecified",
) -> Tuple[WorkerRuntimeSpec, ...]:
    defaults = MANAGED_TRAINING_PORT_DEFAULTS
    master_base = (
        defaults["master_port_base"]
        if master_port_base is None
        else int(master_port_base)
    )
    command_base = (
        defaults["command_port_base"]
        if command_port_base is None
        else int(command_port_base)
    )
    depth_base = (
        defaults["depth_port_base"]
        if depth_port_base is None
        else int(depth_port_base)
    )
    stride_value = defaults["port_stride"] if port_stride is None else int(port_stride)
    count = int(worker_count)
    if not 1 <= count <= MANAGED_COLLECTION_WORKER_COUNT_MAX:
        _fail(
            "worker_count must be in [1,{}]".format(
                MANAGED_COLLECTION_WORKER_COUNT_MAX
            )
        )
    stride = int(stride_value)
    if stride <= 0:
        _fail("port_stride must be positive")
    specs = []
    for worker_id in range(count):
        command_port = command_base + worker_id * stride
        master_port = master_base + worker_id
        depth_port = depth_base + worker_id * stride
        specs.append(
            WorkerRuntimeSpec(
                worker_id=worker_id,
                runtime_instance_id=str(runtime_instance_template).format(
                    worker_id=worker_id,
                    training_run_id=str(training_run_id),
                    runtime_launch_nonce=str(runtime_launch_nonce),
                ),
                master_port=master_port,
                ros_home=str(Path(ros_root) / "worker_{:02d}".format(worker_id)),
                command_port=command_port,
                state_port=command_port + 1,
                depth_port=depth_port,
                python_command_port=command_port + 6,
                unity_command_port=command_port + 7,
                unity_result_port=command_port + 2,
                python_result_port=command_port + 3,
                unity_snapshot_port=command_port + 4,
                python_snapshot_port=command_port + 5,
                training_run_id=str(training_run_id),
                runtime_launch_nonce=str(runtime_launch_nonce),
            )
        )
    return validate_worker_runtime_specs(specs)


# Legacy collection topology remains available as a compatibility projection.
@dataclass(frozen=True)
class CollectionWorkerRuntimeSpec:
    worker_id: int
    master_port: int
    command_port: int
    state_port: int
    depth_port: int
    ros_home: Path
    worker_dir: Path

    @property
    def ros_master_uri(self) -> str:
        return "http://127.0.0.1:{}".format(self.master_port)

    @property
    def ports(self) -> Tuple[int, int, int, int]:
        return self.master_port, self.command_port, self.state_port, self.depth_port

    @property
    def port_profile(self) -> DirectRuntimePortProfile:
        """Project the legacy collection worker onto the canonical owner."""

        return DirectRuntimePortProfile(
            profile="direct",
            runtime_instance_id="legacy-worker-{:02d}".format(self.worker_id),
            master_port=self.master_port,
            command_port=self.command_port,
            state_port=self.state_port,
            depth_port=self.depth_port,
            unity_result_port=self.command_port + 2,
            python_result_port=self.command_port + 3,
            unity_snapshot_port=self.command_port + 4,
            python_snapshot_port=self.command_port + 5,
            python_command_port=self.command_port + 6,
            unity_command_port=self.command_port + 7,
        )


def build_collection_worker_specs(
    *,
    workers: int,
    out_dir: Path,
    master_base: int,
    command_base: int,
    depth_base: int,
    port_stride: int,
) -> Tuple[CollectionWorkerRuntimeSpec, ...]:
    root = Path(out_dir).expanduser().resolve()
    specs = []
    for worker_id in range(int(workers)):
        command_port = int(command_base) + int(port_stride) * worker_id
        specs.append(
            CollectionWorkerRuntimeSpec(
                worker_id=worker_id,
                master_port=int(master_base) + worker_id,
                command_port=command_port,
                state_port=command_port + 1,
                depth_port=int(depth_base) + int(port_stride) * worker_id,
                ros_home=root / "ros" / "worker_{:02d}".format(worker_id),
                worker_dir=root / "workers" / "worker_{:02d}".format(worker_id),
            )
        )
    validate_unique_ports(specs)
    return tuple(specs)


def validate_unique_ports(specs: Iterable[CollectionWorkerRuntimeSpec]) -> None:
    owners: Dict[int, int] = {}
    for spec in specs:
        for port in spec.ports:
            if not 1 <= int(port) <= 65535:
                raise ValueError(
                    "worker {} port outside TCP range: {}".format(spec.worker_id, port)
                )
            previous = owners.get(int(port))
            if previous is not None:
                raise ValueError(
                    "port {} is shared by workers {} and {}".format(
                        port, previous, spec.worker_id
                    )
                )
            owners[int(port)] = spec.worker_id


def is_tcp_port_available(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


__all__ = [
    "CollectionWorkerRuntimeSpec",
    "DirectRuntimePortProfile",
    "LEGACY_COLLECTION_PORT_DEFAULTS",
    "MANAGED_TRAINING_PORT_DEFAULTS",
    "MANAGED_COLLECTION_WORKER_COUNT_MAX",
    "PortContractError",
    "STARTUP_WORKER_ENDPOINT_MISMATCH",
    "WORKER_RUNTIME_SPEC_SCHEMA",
    "WorkerEndpointMismatchError",
    "WorkerRuntimeSpec",
    "build_collection_worker_specs",
    "build_managed_evaluation_profile",
    "build_worker_runtime_specs",
    "is_tcp_port_available",
    "resolve_runtime_port_profile",
    "validate_port_profiles",
    "validate_unique_ports",
    "validate_worker_runtime_specs",
]
