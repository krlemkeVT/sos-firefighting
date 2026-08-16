import json
from datetime import datetime
from pathlib import Path

import numpy as np

from examples.wildfire.firefighter_model.awareness import (
    ClusterInfo,
    FireAwarenessManager,
)
from examples.wildfire.paths import DATA_DIR
from examples.wildfire.simulation import AgentInput
from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SelectPOIType,
)


class _DummyParams:
    enable_awareness_model = True
    awareness_update_period_s = 5.0
    base_comms_radius_m = 2000.0
    base_camera_radius_m = 1000.0
    altitude_visibility_gain_m_per_m = 8.0
    smoke_cell_factor = 0.001
    smoke_distance_scale_m = 2000.0
    smoke_min_factor = 0.35
    cluster_reservation_ttl_s = 30.0
    reacquire_time_s = 10.0
    front_cluster_resolution_cells = 8


class _DummyEnv:
    def get_elevation(self, _):
        return 0.0


class _DummySimulation:
    def __init__(self):
        self.environment = _DummyEnv()
        self.timer = type("Timer", (), {"mission_time": datetime.now()})()


class _DummyAgent:
    def __init__(
        self,
        unique_id: int,
        pos: tuple[float, float],
        *,
        camera_strength: float = 1.0,
        radio_strength: float = 1.0,
    ) -> None:
        self.unique_id = unique_id
        self.pos = np.array(pos, dtype=np.float64)
        self.altitude = 50.0
        self.camera_strength = camera_strength
        self.radio_strength = radio_strength
        self.current_base = None


class _DummyModel:
    def __init__(self):
        self.parameters = _DummyParams()
        self.simulation = _DummySimulation()
        self.firefighters = []
        self.recon_uavs = []
        self.protection_locations = []


def test_agent_input_defaults_for_recon_fields():
    input_file = DATA_DIR / "aircraft" / "example_aircraft_1.json"
    data = json.loads(Path(input_file).read_text())
    data["agents_per_base"] = (1,)
    model = AgentInput.model_validate(data)
    assert model.recon is False
    assert model.radio_strength == 1.0
    assert model.camera_strength == 1.0
    assert model.recon_tactic is SelectPOIType.MAX_LOS


def test_camera_range_increases_with_camera_strength():
    model = _DummyModel()
    manager = FireAwarenessManager(model)
    cluster = ClusterInfo(
        cluster_id=0,
        centroid_pos=np.array([100.0, 0.0], dtype=np.float64),
        burning_count=20,
    )
    weak = _DummyAgent(0, (0.0, 0.0), camera_strength=1.0)
    strong = _DummyAgent(1, (0.0, 0.0), camera_strength=2.0)
    assert manager._camera_range_for_agent_cluster(
        strong, cluster
    ) > manager._camera_range_for_agent_cluster(weak, cluster)


def test_smoke_factor_reduces_visibility_for_large_cluster():
    model = _DummyModel()
    manager = FireAwarenessManager(model)
    agent = _DummyAgent(0, (0.0, 0.0))
    small_cluster = ClusterInfo(
        cluster_id=0,
        centroid_pos=np.array([200.0, 0.0], dtype=np.float64),
        burning_count=5,
    )
    large_cluster = ClusterInfo(
        cluster_id=1,
        centroid_pos=np.array([200.0, 0.0], dtype=np.float64),
        burning_count=200,
    )
    assert manager._camera_range_for_agent_cluster(
        agent, small_cluster
    ) > manager._camera_range_for_agent_cluster(agent, large_cluster)


def test_relay_visibility_propagates_over_comm_graph():
    model = _DummyModel()
    source = _DummyAgent(0, (0.0, 0.0), camera_strength=2.0)
    sink = _DummyAgent(1, (1500.0, 0.0), camera_strength=0.2)
    model.recon_uavs = [source]
    model.firefighters = [sink]
    manager = FireAwarenessManager(model)
    manager.clusters = {
        0: ClusterInfo(
            cluster_id=0,
            centroid_pos=np.array([100.0, 0.0], dtype=np.float64),
            burning_count=1,
        )
    }
    manager._compute_visibility_maps()
    assert manager._direct_visibility[source.unique_id][0] is True
    assert manager._relay_visibility[sink.unique_id][0] is True


def test_lock_transitions_for_loss_and_reacquire():
    model = _DummyModel()
    suppressor = _DummyAgent(5, (0.0, 0.0))
    model.firefighters = [suppressor]
    manager = FireAwarenessManager(model)
    manager.clusters = {
        0: ClusterInfo(
            cluster_id=0,
            centroid_pos=np.array([0.0, 0.0], dtype=np.float64),
            burning_count=1,
        )
    }
    manager.target_cluster_id = lambda _agent: 0

    manager._direct_visibility = {5: {0: False}}
    manager._relay_visibility = {5: {0: True}}
    manager._update_lock_states(dt_s=5.0)
    assert manager.lock_state[5].has_lock is True

    manager._direct_visibility = {5: {0: False}}
    manager._relay_visibility = {5: {0: False}}
    manager._update_lock_states(dt_s=5.0)
    assert manager.lock_state[5].has_lock is False
    assert manager.lock_state[5].needs_reacquire is True

    manager._direct_visibility = {5: {0: True}}
    manager._relay_visibility = {5: {0: False}}
    manager._update_lock_states(dt_s=5.0)
    assert manager.lock_state[5].has_lock is False

    manager._update_lock_states(dt_s=5.0)
    assert manager.lock_state[5].has_lock is True
