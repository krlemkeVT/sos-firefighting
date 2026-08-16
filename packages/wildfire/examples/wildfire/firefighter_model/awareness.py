from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from sosid.model.abm.agent import Agent
from sosid.model.transform import index_to_pos, pos_to_index
from sosid.typedef import Position


@dataclass
class ClusterInfo:
    cluster_id: int
    centroid_pos: np.ndarray
    burning_count: int


@dataclass
class LockState:
    cluster_id: int | None = None
    has_lock: bool = False
    needs_reacquire: bool = True
    reacquire_remaining_s: float = 0.0
    direct_visible: bool = False
    relay_visible: bool = False


@dataclass
class ReservationInfo:
    agent_id: int
    expires_at: datetime


class FireAwarenessManager:
    """Maintains shared front-cluster awareness and lock state."""

    def __init__(self, model) -> None:
        self.model = model
        self.parameters = model.parameters
        self.clusters: dict[int, ClusterInfo] = {}
        self.cluster_by_cell: dict[tuple[int, int], int] = {}
        self.lock_state: dict[int, LockState] = {}
        self.reservations: dict[int, ReservationInfo] = {}
        self._seconds_since_update = 0.0
        self._direct_visibility: dict[int, dict[int, bool]] = {}
        self._relay_visibility: dict[int, dict[int, bool]] = {}

    @property
    def enabled(self) -> bool:
        return self.parameters.enable_awareness_model

    @property
    def now(self) -> datetime:
        return self.model.simulation.timer.mission_time

    @property
    def update_period_s(self) -> float:
        return self.parameters.awareness_update_period_s

    @property
    def reservation_ttl(self) -> timedelta:
        return timedelta(seconds=self.parameters.cluster_reservation_ttl_s)

    def update(self, dt_s: float) -> None:
        """Update awareness state periodically."""
        if not self.enabled:
            return
        self._seconds_since_update += dt_s
        if self._seconds_since_update < self.update_period_s:
            return
        dt_s = self._seconds_since_update
        self._seconds_since_update = 0.0
        self._cleanup_reservations()
        self._build_clusters()
        self._compute_visibility_maps()
        self._update_lock_states(dt_s)

    def _build_clusters(self) -> None:
        """Create front clusters using a simple coarse grid binning."""
        self.clusters.clear()
        self.cluster_by_cell.clear()
        burning_indices = self.model.wildfire.burning_indices
        if not burning_indices.size:
            return

        bins: dict[tuple[int, int], list[np.ndarray]] = {}
        resolution = self.parameters.front_cluster_resolution_cells
        for idx in burning_indices:
            i, j = map(int, idx)
            key = (i // resolution, j // resolution)
            bins.setdefault(key, []).append(idx)

        for cluster_id, (_, cells) in enumerate(bins.items()):
            cells_arr = np.array(cells, dtype=np.int64)
            centroid_idx = np.round(np.mean(cells_arr, axis=0)).astype(
                np.int64
            )
            centroid_pos = np.array(
                index_to_pos(
                    centroid_idx,
                    grid_description=self.model.simulation.environment.terrain.grid_description,
                    origin=self.model.simulation.environment.terrain.origin,
                ),
                dtype=np.float64,
            )
            self.clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                centroid_pos=centroid_pos,
                burning_count=len(cells),
            )
            for i, j in cells_arr:
                self.cluster_by_cell[(int(i), int(j))] = cluster_id

    def _airborne_agents(self) -> list:
        agents = []
        for agent in self.model.firefighters:
            if agent.current_base is None:
                agents.append(agent)
        for agent in self.model.recon_uavs:
            if agent.current_base is None:
                agents.append(agent)
        return agents

    def _camera_range_for_agent_cluster(self, agent, cluster: ClusterInfo) -> float:
        base_range = self.parameters.base_camera_radius_m * getattr(
            agent, "camera_strength", 1.0
        )
        agl = max(
            0.0,
            float(agent.altitude - self.model.simulation.environment.get_elevation(agent.pos)),
        )
        altitude_factor = 1.0 + (
            self.parameters.altitude_visibility_gain_m_per_m * agl / 1000.0
        )
        distance = float(Agent.distance(agent.pos, cluster.centroid_pos))
        smoke_factor = np.exp(
            -self.parameters.smoke_cell_factor * cluster.burning_count
        ) * np.exp(-distance / self.parameters.smoke_distance_scale_m)
        smoke_factor = max(self.parameters.smoke_min_factor, float(smoke_factor))
        return base_range * altitude_factor * smoke_factor

    def _compute_visibility_maps(self) -> None:
        self._direct_visibility = {}
        self._relay_visibility = {}
        agents = self._airborne_agents()
        if not agents or not self.clusters:
            return
        for agent in agents:
            self._direct_visibility[agent.unique_id] = {}
            self._relay_visibility[agent.unique_id] = {}
            for cluster_id, cluster in self.clusters.items():
                distance = float(Agent.distance(agent.pos, cluster.centroid_pos))
                max_range = self._camera_range_for_agent_cluster(agent, cluster)
                self._direct_visibility[agent.unique_id][cluster_id] = (
                    distance <= max_range
                )
                self._relay_visibility[agent.unique_id][cluster_id] = False

        # Build undirected comm graph.
        graph: dict[int, set[int]] = {agent.unique_id: set() for agent in agents}
        for i, agent_i in enumerate(agents):
            for agent_j in agents[i + 1 :]:
                max_range = self.parameters.base_comms_radius_m * min(
                    getattr(agent_i, "radio_strength", 1.0),
                    getattr(agent_j, "radio_strength", 1.0),
                )
                if Agent.distance(agent_i.pos, agent_j.pos) <= max_range:
                    graph[agent_i.unique_id].add(agent_j.unique_id)
                    graph[agent_j.unique_id].add(agent_i.unique_id)

        # Relay visibility through connected components for each cluster.
        for cluster_id in self.clusters:
            sources = [
                agent_id
                for agent_id, by_cluster in self._direct_visibility.items()
                if by_cluster.get(cluster_id, False)
            ]
            if not sources:
                continue
            visited: set[int] = set(sources)
            queue = deque(sources)
            while queue:
                current = queue.popleft()
                self._relay_visibility[current][cluster_id] = True
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    def _update_lock_states(self, dt_s: float) -> None:
        for agent in self.model.firefighters:
            cluster_id = self.target_cluster_id(agent)
            state = self.lock_state.setdefault(agent.unique_id, LockState())

            if cluster_id is None:
                state.cluster_id = None
                state.has_lock = False
                state.direct_visible = False
                state.relay_visible = False
                continue

            if state.cluster_id != cluster_id:
                state.cluster_id = cluster_id
                state.has_lock = False
                state.needs_reacquire = True
                state.reacquire_remaining_s = self._reacquire_time_s()

            direct = self._direct_visibility.get(agent.unique_id, {}).get(
                cluster_id, False
            )
            relay = self._relay_visibility.get(agent.unique_id, {}).get(
                cluster_id, False
            )
            state.direct_visible = direct
            state.relay_visible = relay

            if relay:
                state.has_lock = True
                state.needs_reacquire = False
                state.reacquire_remaining_s = 0.0
                continue

            if direct:
                if state.needs_reacquire:
                    state.reacquire_remaining_s = max(
                        0.0, state.reacquire_remaining_s - dt_s
                    )
                    state.has_lock = state.reacquire_remaining_s <= 0.0
                    if state.has_lock:
                        state.needs_reacquire = False
                else:
                    state.has_lock = True
                continue

            if state.has_lock or not state.needs_reacquire:
                state.needs_reacquire = True
                state.reacquire_remaining_s = self._reacquire_time_s()
            state.has_lock = False

    def _reacquire_time_s(self) -> float:
        value = self.parameters.reacquire_time_s
        return float(value)

    def _cleanup_reservations(self) -> None:
        now = self.now
        expired = [
            cluster_id
            for cluster_id, reservation in self.reservations.items()
            if reservation.expires_at <= now
        ]
        for cluster_id in expired:
            self.reservations.pop(cluster_id, None)

    def cluster_id_from_index(self, index: tuple[int, int]) -> int | None:
        return self.cluster_by_cell.get(index)

    def target_cluster_id(self, agent) -> int | None:
        if agent.destination is None or not self.clusters:
            return None
        destination_idx = pos_to_index(
            agent.destination,
            grid_description=self.model.simulation.environment.terrain.grid_description,
            origin=self.model.simulation.environment.terrain.origin,
        )
        cluster_id = self.cluster_id_from_index(
            (int(destination_idx[0]), int(destination_idx[1]))
        )
        if cluster_id is not None:
            return cluster_id
        # Fallback to nearest cluster centroid.
        positions = np.array(
            [cluster.centroid_pos for cluster in self.clusters.values()],
            dtype=np.float64,
        )
        nearest_idx = int(np.argmin(Agent.distance(agent.destination, positions)))
        return nearest_idx

    def can_suppress(self, agent, cluster_id: int) -> bool:
        state = self.lock_state.get(agent.unique_id)
        return bool(
            state is not None
            and state.cluster_id == cluster_id
            and state.has_lock
        )

    def is_reserved_for_other(self, agent, cluster_id: int | None) -> bool:
        if cluster_id is None:
            return False
        self._cleanup_reservations()
        reservation = self.reservations.get(cluster_id)
        if reservation is None:
            return False
        return reservation.agent_id != agent.unique_id

    def reserve_cluster(self, agent, cluster_id: int | None) -> None:
        if cluster_id is None:
            return
        self.reservations[cluster_id] = ReservationInfo(
            agent_id=agent.unique_id,
            expires_at=self.now + self.reservation_ttl,
        )

    def release_cluster_reservations(self, agent) -> None:
        to_remove = [
            cluster_id
            for cluster_id, reservation in self.reservations.items()
            if reservation.agent_id == agent.unique_id
        ]
        for cluster_id in to_remove:
            self.reservations.pop(cluster_id, None)

    def get_recon_target(
        self, agent, prioritize_protection: bool = False
    ) -> Position | None:
        if not self.clusters:
            return None
        coverage_radius = self.parameters.base_camera_radius_m * getattr(
            agent, "camera_strength", 1.0
        )
        centroids = np.array(
            [cluster.centroid_pos for cluster in self.clusters.values()],
            dtype=np.float64,
        )
        scores = []
        for cluster_id, cluster in self.clusters.items():
            distances = Agent.distance(cluster.centroid_pos, centroids)
            los_score = float(np.count_nonzero(distances <= coverage_radius))
            if prioritize_protection and self.model.protection_locations:
                protection_pos = np.array(
                    [loc.pos for loc in self.model.protection_locations],
                    dtype=np.float64,
                )
                min_dist = np.min(
                    Agent.distance(cluster.centroid_pos, protection_pos)
                )
                protection_score = 1.0 / max(min_dist, 1.0)
                los_score += protection_score * 1000.0
            scores.append((los_score, cluster_id))
        best_cluster_id = max(scores)[1]
        return self.clusters[best_cluster_id].centroid_pos
