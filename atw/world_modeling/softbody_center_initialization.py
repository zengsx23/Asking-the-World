"""Observation-only shared-center initialization."""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import warp as wp
from atw.polyworld.soft_body_swr import SoftBodySceneData, ParameterLayout, parameter_bounds, initial_distribution, minimum_distribution_std, project_parameters, _target_penetration_allowances, _obb_contact, _contact_strain, _deformed_support_radius, _project_strain_support, _deformation_matrices, _shape_parameter_indices, _project_output_surface_positions, softbody_metrics, GRAVITY_M_PER_S2, DEFAULT_SEEDS, DEFAULT_CUDA_BLOCK_DIM, MIN_DEFORMED_SUPPORT_SCALE

@dataclass(frozen=True)
class SoftBodyCEMConfig:
    center_weight: float = 1.0
    population: int = 768
    elite_fraction: float = 0.08
    generations_per_prefix: int = 48
    prefix_frames: tuple[int, ...] = (24, 48, 72, 96, 0)
    distribution_smoothing: float = 0.2
    exploration_fraction: float = 0.02
    huber_delta_m: float = 0.03
    shape_weight: float = 1.0
    deformation_focus: float = 4.0
    joint_generations_per_prefix: int = 48
    joint_initial_std_scale: float = 0.5
    shape_refinement_generations: int = 48
    shape_event_weight: float = 1.0
    seed: int = 23


def make_parameter_layout(object_count: int) -> ParameterLayout:
    if object_count <= 0:
        raise ValueError("object_count must be positive")
    offset = 0

    def take(width: int) -> slice:
        nonlocal offset
        result = slice(offset, offset + width)
        offset += width
        return result

    return ParameterLayout(
        object_count=object_count,
        velocity_slice=take(2 * object_count),
        log_gravity_scale_slice=take(1),
        log_radius_scale_slice=take(object_count),
        log_frequency_slice=take(object_count),
        damping_ratio_slice=take(object_count),
        friction_slice=take(object_count),
        deformation_gain_slice=take(object_count),
        log_shape_frequency_slice=take(object_count),
        shape_damping_ratio_slice=take(object_count),
        drag_slice=take(object_count),
        size=offset,
    )


def decode_parameters(values: np.ndarray, layout: ParameterLayout) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "initial_velocity": values[layout.velocity_slice].reshape(-1, 2),
        "gravity_scale": np.repeat(np.exp(values[layout.log_gravity_scale_slice]), layout.object_count),
        "radius_scale": np.exp(values[layout.log_radius_scale_slice]),
        "natural_frequency_rad_per_s": np.exp(values[layout.log_frequency_slice]),
        "damping_ratio": values[layout.damping_ratio_slice],
        "contact_friction": values[layout.friction_slice],
        "deformation_gain": values[layout.deformation_gain_slice],
        "shape_frequency_rad_per_s": np.exp(
            values[layout.log_shape_frequency_slice]
        ),
        "shape_damping_ratio": values[layout.shape_damping_ratio_slice],
        "linear_drag_per_s": values[layout.drag_slice],
    }


@wp.kernel
def _simulate_soft_worlds(
    theta: wp.array2d(dtype=float),
    initial_centers: wp.array(dtype=wp.vec2),
    radii: wp.array(dtype=float),
    first_offsets: wp.array(dtype=int),
    obstacle_centers: wp.array(dtype=wp.vec2),
    obstacle_axes: wp.array(dtype=wp.vec2),
    obstacle_half_extents: wp.array(dtype=wp.vec2),
    static_penetration_allowance: wp.array3d(dtype=float),
    pair_penetration_allowance: wp.array3d(dtype=float),
    object_count: int,
    obstacle_count: int,
    frame_count: int,
    fps: float,
    substeps: int,
    log_gravity_scale_start: int,
    log_radius_scale_start: int,
    log_frequency_start: int,
    damping_start: int,
    friction_start: int,
    deformation_gain_start: int,
    log_shape_frequency_start: int,
    shape_damping_start: int,
    drag_start: int,
    enable_soft_body_contacts: int,
    enable_shape_projection: int,
    positions: wp.array2d(dtype=wp.vec2),
    velocities: wp.array2d(dtype=wp.vec2),
    strains: wp.array2d(dtype=wp.vec3),
    strain_velocities: wp.array2d(dtype=wp.vec3),
    accelerations: wp.array2d(dtype=wp.vec2),
    target_strains: wp.array2d(dtype=wp.vec3),
    contact_strengths: wp.array2d(dtype=float),
    predicted_positions: wp.array3d(dtype=wp.vec2),
    predicted_strains: wp.array3d(dtype=wp.vec3),
    contact_obstacle: wp.array3d(dtype=int),
):
    world = wp.tid()
    dt = 1.0 / (fps * float(substeps))
    for object_index in range(object_count):
        velocity_offset = 2 * object_index
        positions[world, object_index] = initial_centers[object_index]
        velocities[world, object_index] = wp.vec2(
            theta[world, velocity_offset], theta[world, velocity_offset + 1]
        )
        strains[world, object_index] = wp.vec3(0.0, 0.0, 0.0)
        strain_velocities[world, object_index] = wp.vec3(0.0, 0.0, 0.0)

    for frame_index in range(frame_count):
        for object_index in range(object_count):
            contact_obstacle[world, frame_index, object_index] = -1
        if frame_index > 0:
            for _substep in range(substeps):
                for object_index in range(object_count):
                    accelerations[world, object_index] = wp.vec2(0.0, 0.0)
                    target_strains[world, object_index] = wp.vec3(0.0, 0.0, 0.0)
                    contact_strengths[world, object_index] = 0.0
                    if frame_index <= first_offsets[object_index]:
                        continue
                    position = positions[world, object_index]
                    velocity = velocities[world, object_index]
                    radius = radii[object_index] * wp.exp(
                        theta[world, log_radius_scale_start + object_index]
                    )
                    frequency = wp.exp(
                        theta[world, log_frequency_start + object_index]
                    )
                    damping = theta[world, damping_start + object_index]
                    friction = theta[world, friction_start + object_index]
                    drag = theta[world, drag_start + object_index]
                    gravity_scale = wp.exp(
                        theta[world, log_gravity_scale_start]
                    )
                    acceleration = (
                        wp.vec2(0.0, float(GRAVITY_M_PER_S2) * gravity_scale)
                        - drag * velocity
                    )
                    target_strain = wp.vec3(0.0, 0.0, 0.0)
                    for obstacle_index in range(obstacle_count):
                        geometry = _obb_contact(
                            position,
                            obstacle_centers[obstacle_index],
                            obstacle_axes[obstacle_index],
                            obstacle_half_extents[obstacle_index],
                        )
                        normal = wp.vec2(geometry[0], geometry[1])
                        penetration = radius - geometry[2]
                        if penetration <= 0.0:
                            continue
                        contact_strength = penetration / radius
                        if contact_strength > contact_strengths[world, object_index]:
                            contact_strengths[world, object_index] = contact_strength
                            contact_obstacle[world, frame_index, object_index] = (
                                obstacle_index
                            )
                        normal_speed = wp.dot(velocity, normal)
                        normal_acceleration = frequency * frequency * penetration
                        if normal_speed < 0.0:
                            normal_acceleration -= 2.0 * damping * frequency * normal_speed
                        normal_acceleration = wp.clamp(normal_acceleration, 0.0, 2000.0)
                        acceleration += normal_acceleration * normal
                        tangent = wp.vec2(-normal[1], normal[0])
                        tangent_speed = wp.dot(velocity, tangent)
                        friction_acceleration = wp.min(
                            wp.abs(tangent_speed) / dt,
                            friction * normal_acceleration,
                        )
                        if tangent_speed > 0.0:
                            acceleration -= friction_acceleration * tangent
                        elif tangent_speed < 0.0:
                            acceleration += friction_acceleration * tangent
                        compression = wp.clamp(
                            normal_acceleration
                            / (frequency * frequency * radius),
                            0.0,
                            0.7,
                        )
                        deformation_gain = theta[
                            world, deformation_gain_start + object_index
                        ]
                        amplitude = deformation_gain * compression
                        target_strain += _contact_strain(normal, amplitude)
                    accelerations[world, object_index] = acceleration
                    target_strains[world, object_index] = target_strain

                for first_index in range(object_count):
                    if enable_soft_body_contacts == 0:
                        continue
                    if frame_index <= first_offsets[first_index]:
                        continue
                    for second_index in range(first_index + 1, object_count):
                        if frame_index <= first_offsets[second_index]:
                            continue
                        first_position = positions[world, first_index]
                        second_position = positions[world, second_index]
                        delta = first_position - second_position
                        distance = wp.length(delta)
                        normal = wp.vec2(1.0, 0.0)
                        if distance > 1.0e-8:
                            normal = delta / distance
                        first_radius = radii[first_index] * wp.exp(
                            theta[world, log_radius_scale_start + first_index]
                        )
                        second_radius = radii[second_index] * wp.exp(
                            theta[world, log_radius_scale_start + second_index]
                        )
                        penetration = first_radius + second_radius - distance
                        if penetration <= 0.0:
                            continue
                        first_frequency = wp.exp(
                            theta[world, log_frequency_start + first_index]
                        )
                        second_frequency = wp.exp(
                            theta[world, log_frequency_start + second_index]
                        )
                        frequency = wp.sqrt(first_frequency * second_frequency)
                        damping = 0.5 * (
                            theta[world, damping_start + first_index]
                            + theta[world, damping_start + second_index]
                        )
                        relative_velocity = (
                            velocities[world, first_index]
                            - velocities[world, second_index]
                        )
                        normal_speed = wp.dot(relative_velocity, normal)
                        pair_acceleration = 0.5 * frequency * frequency * penetration
                        if normal_speed < 0.0:
                            pair_acceleration -= damping * frequency * normal_speed
                        pair_acceleration = wp.clamp(
                            pair_acceleration, 0.0, 2000.0
                        )
                        accelerations[world, first_index] = (
                            accelerations[world, first_index]
                            + pair_acceleration * normal
                        )
                        accelerations[world, second_index] = (
                            accelerations[world, second_index]
                            - pair_acceleration * normal
                        )
                        tangent = wp.vec2(-normal[1], normal[0])
                        tangent_speed = wp.dot(relative_velocity, tangent)
                        friction = 0.5 * (
                            theta[world, friction_start + first_index]
                            + theta[world, friction_start + second_index]
                        )
                        friction_acceleration = wp.min(
                            0.5 * wp.abs(tangent_speed) / dt,
                            friction * pair_acceleration,
                        )
                        if tangent_speed > 0.0:
                            accelerations[world, first_index] -= (
                                friction_acceleration * tangent
                            )
                            accelerations[world, second_index] += (
                                friction_acceleration * tangent
                            )
                        elif tangent_speed < 0.0:
                            accelerations[world, first_index] += (
                                friction_acceleration * tangent
                            )
                            accelerations[world, second_index] -= (
                                friction_acceleration * tangent
                            )
                        first_compression = wp.clamp(
                            pair_acceleration
                            / (
                                0.5 * frequency * frequency * first_radius
                            ),
                            0.0,
                            0.7,
                        )
                        second_compression = wp.clamp(
                            pair_acceleration
                            / (
                                0.5 * frequency * frequency * second_radius
                            ),
                            0.0,
                            0.7,
                        )
                        first_amplitude = (
                            theta[world, deformation_gain_start + first_index]
                            * first_compression
                        )
                        second_amplitude = (
                            theta[world, deformation_gain_start + second_index]
                            * second_compression
                        )
                        target_strains[world, first_index] = (
                            target_strains[world, first_index]
                            + _contact_strain(normal, first_amplitude)
                        )
                        target_strains[world, second_index] = (
                            target_strains[world, second_index]
                            + _contact_strain(-normal, second_amplitude)
                        )
                        first_strength = 0.5 * penetration / first_radius
                        second_strength = 0.5 * penetration / second_radius
                        if first_strength > contact_strengths[world, first_index]:
                            contact_strengths[world, first_index] = first_strength
                            contact_obstacle[world, frame_index, first_index] = (
                                obstacle_count + second_index
                            )
                        if second_strength > contact_strengths[world, second_index]:
                            contact_strengths[world, second_index] = second_strength
                            contact_obstacle[world, frame_index, second_index] = (
                                obstacle_count + first_index
                            )

                for object_index in range(object_count):
                    if frame_index <= first_offsets[object_index]:
                        continue
                    position = positions[world, object_index]
                    velocity = velocities[world, object_index]
                    strain = strains[world, object_index]
                    strain_velocity = strain_velocities[world, object_index]
                    shape_frequency = wp.exp(
                        theta[world, log_shape_frequency_start + object_index]
                    )
                    shape_damping = theta[
                        world, shape_damping_start + object_index
                    ]
                    velocity += accelerations[world, object_index] * dt
                    position += velocity * dt
                    strain_acceleration = (
                        shape_frequency * shape_frequency
                        * (target_strains[world, object_index] - strain)
                        - 2.0 * shape_damping * shape_frequency * strain_velocity
                    )
                    strain_velocity += strain_acceleration * dt
                    strain += strain_velocity * dt
                    strain_norm = wp.length(strain)
                    if strain_norm > 0.7:
                        strain *= 0.7 / strain_norm
                        strain_velocity *= 0.5
                    positions[world, object_index] = position
                    velocities[world, object_index] = velocity
                    strains[world, object_index] = strain
                    strain_velocities[world, object_index] = strain_velocity

                # Soft bodies accommodate geometric overlap through strain rather
                # than by moving observed centers away from their fitted paths.
                for _projection_iteration in range(2):
                    if enable_shape_projection == 0:
                        continue
                    for object_index in range(object_count):
                        if frame_index <= first_offsets[object_index]:
                            continue
                        position = positions[world, object_index]
                        strain = strains[world, object_index]
                        radius = radii[object_index]
                        for obstacle_index in range(obstacle_count):
                            geometry = _obb_contact(
                                position,
                                obstacle_centers[obstacle_index],
                                obstacle_axes[obstacle_index],
                                obstacle_half_extents[obstacle_index],
                            )
                            normal = wp.vec2(geometry[0], geometry[1])
                            minimum_support = float(MIN_DEFORMED_SUPPORT_SCALE) * radius
                            support = _deformed_support_radius(radius, strain, normal)
                            maximum_support = wp.max(
                                geometry[2]
                                + static_penetration_allowance[
                                    frame_index, object_index, obstacle_index
                                ],
                                minimum_support,
                            )
                            penetration = support - maximum_support
                            if penetration <= 0.0:
                                continue
                            strain = _project_strain_support(
                                radius,
                                strain,
                                normal,
                                maximum_support,
                            )
                            strength = penetration / radius
                            if strength > contact_strengths[world, object_index]:
                                contact_strengths[world, object_index] = strength
                                contact_obstacle[
                                    world, frame_index, object_index
                                ] = obstacle_index
                        strain_norm = wp.length(strain)
                        if strain_norm > 0.7:
                            strain *= 0.7 / strain_norm
                        strains[world, object_index] = strain

                    if enable_soft_body_contacts != 0:
                        for first_index in range(object_count):
                            if frame_index <= first_offsets[first_index]:
                                continue
                            for second_index in range(first_index + 1, object_count):
                                if frame_index <= first_offsets[second_index]:
                                    continue
                                delta = (
                                    positions[world, first_index]
                                    - positions[world, second_index]
                                )
                                distance = wp.length(delta)
                                normal = wp.vec2(1.0, 0.0)
                                if distance > 1.0e-8:
                                    normal = delta / distance
                                first_radius = radii[first_index]
                                second_radius = radii[second_index]
                                first_strain = strains[world, first_index]
                                second_strain = strains[world, second_index]
                                first_support = _deformed_support_radius(
                                    first_radius, first_strain, normal
                                )
                                second_support = _deformed_support_radius(
                                    second_radius, second_strain, normal
                                )
                                maximum_support_sum = (
                                    distance
                                    + pair_penetration_allowance[
                                        frame_index, first_index, second_index
                                    ]
                                )
                                penetration = (
                                    first_support + second_support
                                    - maximum_support_sum
                                )
                                if penetration <= 0.0:
                                    continue
                                radius_sum = first_radius + second_radius
                                first_limit = wp.max(
                                    maximum_support_sum
                                    * first_radius / radius_sum,
                                    float(MIN_DEFORMED_SUPPORT_SCALE) * first_radius,
                                )
                                second_limit = wp.max(
                                    maximum_support_sum
                                    * second_radius / radius_sum,
                                    float(MIN_DEFORMED_SUPPORT_SCALE) * second_radius,
                                )
                                first_strain = _project_strain_support(
                                    first_radius, first_strain, normal, first_limit
                                )
                                second_strain = _project_strain_support(
                                    second_radius, second_strain, -normal, second_limit
                                )
                                strains[world, first_index] = first_strain
                                strains[world, second_index] = second_strain
                                first_strength = 0.5 * penetration / first_radius
                                second_strength = 0.5 * penetration / second_radius
                                if first_strength > contact_strengths[world, first_index]:
                                    contact_strengths[world, first_index] = first_strength
                                    contact_obstacle[
                                        world, frame_index, first_index
                                    ] = obstacle_count + second_index
                                if second_strength > contact_strengths[world, second_index]:
                                    contact_strengths[world, second_index] = second_strength
                                    contact_obstacle[
                                        world, frame_index, second_index
                                    ] = obstacle_count + first_index
        for object_index in range(object_count):
            position = positions[world, object_index]
            strain = strains[world, object_index]
            predicted_positions[world, frame_index, object_index] = position
            predicted_strains[world, frame_index, object_index] = strain


class BatchWarpSoftBodyEngine:
    def __init__(
        self, scene: SoftBodySceneData, *, capacity: int, device: str = "cuda:0",
        substeps: int = 8,
        projection_tolerance: float = 1.0e-9,
        strict_projection: bool = True,
        block_dim: int = DEFAULT_CUDA_BLOCK_DIM,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        if not 1 <= block_dim <= 1024:
            raise ValueError("block_dim must be between 1 and 1024")
        self.scene = scene
        self.projection_tolerance = float(projection_tolerance)
        self.strict_projection = strict_projection
        self.projection_diagnostics = []
        self.capacity = int(capacity)
        self.device = device
        self.torch_device = torch.device(device)
        self.substeps = int(substeps)
        self.block_dim = int(block_dim)
        self.layout = make_parameter_layout(scene.object_count)
        self.enable_soft_body_contacts = True
        self.enable_shape_projection = True
        static_allowance, pair_allowance = _target_penetration_allowances(scene)
        obstacle_centers = np.asarray([item.center for item in scene.obstacles], np.float32)
        obstacle_axes = np.asarray([item.axis_x for item in scene.obstacles], np.float32)
        obstacle_half = np.asarray([item.half_extents for item in scene.obstacles], np.float32)
        wp.init()
        with wp.ScopedDevice(device):
            self.theta = wp.empty((capacity, self.layout.size), dtype=float, device=device)
            self.initial_centers = wp.array(scene.initial_centers.astype(np.float32), dtype=wp.vec2, device=device)
            self.radii = wp.array(scene.radii.astype(np.float32), dtype=float, device=device)
            self.first_offsets = wp.array(np.asarray(scene.first_offsets, np.int32), dtype=int, device=device)
            self.obstacle_centers = wp.array(obstacle_centers, dtype=wp.vec2, device=device)
            self.obstacle_axes = wp.array(obstacle_axes, dtype=wp.vec2, device=device)
            self.obstacle_half = wp.array(obstacle_half, dtype=wp.vec2, device=device)
            self.static_penetration_allowance = wp.array(
                static_allowance, dtype=float, device=device
            )
            self.pair_penetration_allowance = wp.array(
                pair_allowance, dtype=float, device=device
            )
            state_shape = (capacity, scene.object_count)
            self.positions = wp.empty(state_shape, dtype=wp.vec2, device=device)
            self.velocities = wp.empty(state_shape, dtype=wp.vec2, device=device)
            self.strains = wp.empty(state_shape, dtype=wp.vec3, device=device)
            self.strain_velocities = wp.empty(state_shape, dtype=wp.vec3, device=device)
            self.accelerations = wp.empty(
                state_shape, dtype=wp.vec2, device=device
            )
            self.target_strains = wp.empty(
                state_shape, dtype=wp.vec3, device=device
            )
            self.contact_strengths = wp.empty(
                state_shape, dtype=float, device=device
            )
            self.predicted_positions = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=wp.vec2, device=device
            )
            self.predicted_strains = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=wp.vec3, device=device
            )
            self.contacts = wp.empty(
                (capacity, scene.frame_count, scene.object_count), dtype=int, device=device
            )
        self.theta_torch = wp.to_torch(self.theta)
        self.predicted_positions_torch = wp.to_torch(self.predicted_positions)
        self.predicted_strains_torch = wp.to_torch(self.predicted_strains)
        self.contacts_torch = wp.to_torch(self.contacts)
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                value, dtype=torch.float32, device=self.torch_device
            )
        self.target_subset_centers = tensor(scene.target_subset_centers)
        self.target_centers = tensor(scene.target_centers)
        self.center_weight = 1.0
        self.reference_subset_offsets = tensor(scene.reference_subset_offsets)
        self.target_deformations = tensor(scene.target_deformations)
        self.observed_mask = tensor(scene.observed_mask)
        self.shape_observed_mask = tensor(scene.shape_observed_mask)
        self.radii_torch = tensor(scene.radii)
        identity = torch.eye(2, device=self.torch_device).reshape(1, 1, 2, 2)
        self.deformation_signal = torch.sqrt(
            (self.target_deformations - identity).square().mean(dim=(-1, -2))
        )

    def rollout(self, samples: torch.Tensor | np.ndarray, end_offset: int | None = None) -> torch.Tensor:
        values = torch.as_tensor(samples, dtype=torch.float32, device=self.torch_device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[1] != self.layout.size:
            raise ValueError(f"expected {self.layout.size} parameters, got {values.shape[1]}")
        count = int(values.shape[0])
        if count > self.capacity:
            raise ValueError(f"batch size {count} exceeds engine capacity {self.capacity}")
        final_offset = self.scene.frame_count - 1 if end_offset is None else min(
            max(int(end_offset), 0), self.scene.frame_count - 1
        )
        self.theta_torch[:count].copy_(values)
        layout = self.layout
        wp.launch(
            _simulate_soft_worlds,
            dim=count,
            block_dim=self.block_dim,
            inputs=[
                self.theta, self.initial_centers, self.radii, self.first_offsets,
                self.obstacle_centers, self.obstacle_axes, self.obstacle_half,
                self.static_penetration_allowance,
                self.pair_penetration_allowance,
                self.scene.object_count, len(self.scene.obstacles), final_offset + 1,
                float(self.scene.fps), self.substeps,
                layout.log_gravity_scale_slice.start,
                layout.log_radius_scale_slice.start, layout.log_frequency_slice.start,
                layout.damping_ratio_slice.start, layout.friction_slice.start,
                layout.deformation_gain_slice.start,
                layout.log_shape_frequency_slice.start,
                layout.shape_damping_ratio_slice.start,
                layout.drag_slice.start,
                int(self.enable_soft_body_contacts),
                int(self.enable_shape_projection),
                self.positions, self.velocities, self.strains, self.strain_velocities,
                self.accelerations, self.target_strains, self.contact_strengths,
                self.predicted_positions, self.predicted_strains, self.contacts,
            ],
            device=self.device,
        )
        return torch.cat(
            (
                self.predicted_positions_torch[:count, : final_offset + 1],
                self.predicted_strains_torch[:count, : final_offset + 1],
            ),
            dim=-1,
        )

    def rollout_numpy(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted = self.rollout(values)
        wp.synchronize_device(self.device)
        result = predicted[0].detach().cpu().numpy().astype(np.float64)
        if self.enable_shape_projection:
            self.projection_diagnostics = []
            result = _project_output_surface_positions(self.scene, result, tolerance=self.projection_tolerance, strict=self.strict_projection, diagnostics=self.projection_diagnostics)
        return result, self.contacts_torch[0].detach().cpu().numpy().astype(np.int32)


def candidate_loss(
    engine: BatchWarpSoftBodyEngine, predicted: torch.Tensor, *, end_offset: int,
    huber_delta_m: float, shape_weight: float, deformation_focus: float,
    per_object: bool = False,
) -> torch.Tensor:
    frame_count = predicted.shape[1]
    deformation = _deformation_matrices(predicted)
    reference_offsets = engine.reference_subset_offsets[:frame_count]
    predicted_subset_centers = predicted[..., :2] + torch.einsum(
        "pfnij,fnj->pfni", deformation, reference_offsets
    )
    center_residual = predicted_subset_centers - engine.target_subset_centers[:frame_count]
    absolute = torch.abs(center_residual)
    center_loss = torch.where(
        absolute <= huber_delta_m,
        0.5 * center_residual.square(),
        huber_delta_m * (absolute - 0.5 * huber_delta_m),
    ).mean(dim=-1)
    centroid_residual = predicted[..., :2] - engine.target_centers[:frame_count]
    centroid_absolute = torch.abs(centroid_residual)
    centroid_loss = torch.where(
        centroid_absolute <= huber_delta_m,
        0.5 * centroid_residual.square(),
        huber_delta_m * (centroid_absolute - 0.5 * huber_delta_m),
    ).mean(dim=-1)
    radius = engine.radii_torch.reshape(1, 1, -1, 1, 1)
    shape_loss = (
        radius * (deformation - engine.target_deformations[:frame_count])
    ).square().mean(dim=(-1, -2))
    shape_available = engine.shape_observed_mask[:frame_count].unsqueeze(0)
    deformation_weight = 1.0 + deformation_focus * torch.clamp(
        engine.deformation_signal[:frame_count] / 0.15, 0.0, 1.0
    )
    mask = engine.observed_mask[:frame_count].unsqueeze(0)
    frame_indices = torch.arange(frame_count, device=predicted.device)
    prefix_mask = (frame_indices <= end_offset).float().reshape(1, -1, 1)
    weights = mask * prefix_mask
    object_losses = (
        (
            (
                engine.center_weight * centroid_loss + center_loss
                + shape_weight * deformation_weight * shape_loss * shape_available
            )
            * weights
        ).sum(dim=1)
        / weights.sum(dim=1).clamp_min(1.0)
    )
    return object_losses if per_object else object_losses.mean(dim=1)


def _object_parameter_indices(
    layout: ParameterLayout, object_index: int,
) -> tuple[int, ...]:
    return (
        2 * object_index,
        2 * object_index + 1,
        layout.log_gravity_scale_slice.start,
        layout.log_radius_scale_slice.start + object_index,
        layout.log_frequency_slice.start + object_index,
        layout.damping_ratio_slice.start + object_index,
        layout.friction_slice.start + object_index,
        layout.deformation_gain_slice.start + object_index,
        layout.log_shape_frequency_slice.start + object_index,
        layout.shape_damping_ratio_slice.start + object_index,
        layout.drag_slice.start + object_index,
    )


def refine_shape_parameters(
    engine: BatchWarpSoftBodyEngine, config: SoftBodyCEMConfig,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    generations = config.shape_refinement_generations
    if config.shape_event_weight < 0.0:
        raise ValueError("shape_event_weight cannot be negative")
    if generations <= 0:
        return np.asarray(initial).copy(), {"generations": 0}
    layout = engine.layout
    device = engine.torch_device
    lower_np, upper_np = parameter_bounds(layout)
    _, std_np = initial_distribution(layout)
    minimum_std_np = minimum_distribution_std(layout)
    lower = torch.as_tensor(lower_np, device=device)
    upper = torch.as_tensor(upper_np, device=device)
    mean = torch.as_tensor(initial, dtype=torch.float32, device=device).clone()
    std = torch.as_tensor(std_np, device=device)
    minimum_std = torch.as_tensor(minimum_std_np, device=device)
    best = mean.clone()
    shape_indices = tuple(
        _shape_parameter_indices(layout, object_index)
        for object_index in range(engine.scene.object_count)
    )
    flat_indices = torch.as_tensor(
        tuple(index for indices in shape_indices for index in indices), device=device
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 100_003)
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    def score(predicted):
        return candidate_loss(engine, predicted, end_offset=engine.scene.frame_count - 1,
                              huber_delta_m=config.huber_delta_m,
                              shape_weight=config.shape_weight,
                              deformation_focus=config.deformation_focus)

    best_loss = score(engine.rollout(best))[0]
    initial_loss = float(best_loss.item())
    for _ in range(generations):
        samples = best.unsqueeze(0).repeat(config.population, 1)
        sampled_shape = (
            mean[flat_indices]
            + std[flat_indices]
            * torch.randn(
                (config.population, len(flat_indices)),
                generator=generator,
                device=device,
            )
        )
        sampled_shape.clamp_(lower[flat_indices], upper[flat_indices])
        samples[:, flat_indices] = sampled_shape
        samples[0] = best
        losses = score(engine.rollout(samples))
        elite_losses, elite_rows = torch.topk(losses, elite_count, largest=False)
        elites = samples[elite_rows][:, flat_indices]
        smoothing = config.distribution_smoothing
        mean[flat_indices] = smoothing * mean[flat_indices] + (1.0-smoothing) * elites.mean(dim=0)
        std[flat_indices] = smoothing * std[flat_indices] + (1.0-smoothing) * elites.std(dim=0,unbiased=False)
        if elite_losses[0] < best_loss:
            best_loss = elite_losses[0].clone()
            best = samples[elite_rows[0]].clone()
        std[flat_indices] = torch.maximum(
            std[flat_indices], minimum_std[flat_indices]
        )
        if config.exploration_fraction > 0.0:
            std[flat_indices] = torch.maximum(
                std[flat_indices],
                config.exploration_fraction
                * (upper[flat_indices] - lower[flat_indices]),
            )
    return best.detach().cpu().numpy(), {
        "generations": generations,
        "objective": "unified_centroid_subset_shape_loss",
        "initial_loss": initial_loss,
        "final_loss": float(best_loss.item()),
    }


def run_cem(
    scene: SoftBodySceneData, engine: BatchWarpSoftBodyEngine,
    config: SoftBodyCEMConfig, *, initial: np.ndarray | None = None,
    initial_std_scale: float = 1.0, joint: bool = True,
    generations_per_prefix: int | None = None, verbose: bool = True,
) -> dict[str, Any]:
    # A shared gravity couples every object's loss even without ball contacts.
    # Use scene-scored block updates in both phases instead of splicing winners
    # that were evaluated under different gravity values.
    joint = True
    if config.population > engine.capacity:
        raise ValueError("CEM population exceeds engine capacity")
    if initial_std_scale <= 0.0:
        raise ValueError("initial_std_scale must be positive")
    layout = engine.layout
    lower_np, upper_np = parameter_bounds(layout)
    mean_np, std_np = initial_distribution(layout)
    if initial is not None:
        mean_np = project_parameters(initial, layout).astype(np.float32)
        std_np *= float(initial_std_scale)
    minimum_std_np = minimum_distribution_std(layout)
    device = engine.torch_device
    lower = torch.as_tensor(lower_np, device=device)
    upper = torch.as_tensor(upper_np, device=device)
    mean = torch.as_tensor(mean_np, device=device)
    std = torch.as_tensor(std_np, device=device)
    minimum_std = torch.as_tensor(minimum_std_np, device=device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    generation_count = (
        config.generations_per_prefix
        if generations_per_prefix is None
        else int(generations_per_prefix)
    )
    if config.population < 2 or generation_count <= 0:
        raise ValueError("CEM requires at least two samples and one generation")
    elite_count = max(2, int(round(config.population * config.elite_fraction)))
    trace = []
    best = mean.clone()
    best_loss = math.inf
    prefixes = tuple(
        scene.frame_count - 1 if value <= 0 else min(value - 1, scene.frame_count - 1)
        for value in config.prefix_frames
    )
    for stage_index, end_offset in enumerate(prefixes):
        active_objects = np.any(
            scene.observed_mask[: end_offset + 1] > 0.0, axis=0
        )
        active_object_indices = torch.as_tensor(
            np.flatnonzero(active_objects), device=device
        )
        stage_best_loss = math.inf
        stage_object_best = np.where(active_objects, math.inf, 0.0)
        stage_candidate = best.clone()
        for generation in range(generation_count):
            joint_indices = None
            if joint:
                active_sequence = np.flatnonzero(active_objects)
                object_index = int(active_sequence[generation % len(active_sequence)])
                joint_indices = torch.as_tensor(
                    _object_parameter_indices(layout, object_index), device=device
                )
                samples = best.unsqueeze(0).repeat(config.population, 1)
                samples[:, joint_indices] = (
                    mean[joint_indices]
                    + std[joint_indices]
                    * torch.randn(
                        (config.population, len(joint_indices)),
                        generator=generator,
                        device=device,
                    )
                )
                samples[:, joint_indices] = torch.clamp(
                    samples[:, joint_indices],
                    lower[joint_indices],
                    upper[joint_indices],
                )
            else:
                samples = mean + std * torch.randn(
                    (config.population, layout.size),
                    generator=generator,
                    device=device,
                )
                samples.clamp_(lower, upper)
            if generation == 0:
                samples[0] = best
            predicted = engine.rollout(samples, end_offset)
            object_losses = candidate_loss(
                engine, predicted, end_offset=end_offset,
                huber_delta_m=config.huber_delta_m, shape_weight=config.shape_weight,
                deformation_focus=config.deformation_focus,
                per_object=True,
            )
            smoothing = config.distribution_smoothing
            if joint:
                assert joint_indices is not None
                losses = object_losses[:, active_object_indices].mean(dim=1)
                elite_losses, elite_indices = torch.topk(
                    losses, elite_count, largest=False
                )
                elites = samples[elite_indices][:, joint_indices]
                mean[joint_indices] = (
                    smoothing * mean[joint_indices]
                    + (1.0 - smoothing) * elites.mean(dim=0)
                )
                std[joint_indices] = (
                    smoothing * std[joint_indices]
                    + (1.0 - smoothing) * elites.std(dim=0, unbiased=False)
                )
                generation_loss = float(elite_losses[0].item())
                if generation_loss < stage_best_loss:
                    stage_best_loss = generation_loss
                    best[joint_indices] = samples[elite_indices[0], joint_indices]
                    stage_candidate = best.clone()
            else:
                for object_index in np.flatnonzero(active_objects):
                    elite_losses, elite_indices = torch.topk(
                        object_losses[:, object_index], elite_count, largest=False
                    )
                    indices = torch.as_tensor(
                        _object_parameter_indices(layout, int(object_index)),
                        device=device,
                    )
                    elites = samples[elite_indices][:, indices]
                    mean[indices] = (
                        smoothing * mean[indices]
                        + (1.0 - smoothing) * elites.mean(dim=0)
                    )
                    std[indices] = (
                        smoothing * std[indices]
                        + (1.0 - smoothing)
                        * elites.std(dim=0, unbiased=False)
                    )
                    object_loss = float(elite_losses[0].item())
                    if object_loss < stage_object_best[object_index]:
                        stage_object_best[object_index] = object_loss
                        stage_candidate[indices] = samples[elite_indices[0], indices]
            std = torch.maximum(std, minimum_std)
            if config.exploration_fraction > 0.0:
                std = torch.maximum(std, config.exploration_fraction * (upper - lower))
        best = stage_candidate
        best_loss = (
            stage_best_loss if joint else float(np.mean(stage_object_best))
        )
        entry = {
            "stage": stage_index + 1,
            "end_frame": int(scene.frames[end_offset]),
            "best_loss": best_loss,
            "optimization": "joint" if joint else "independent",
        }
        trace.append(entry)
        mean = best.clone()
        if verbose:
            print(json.dumps({"phase": "softbody_cem", **entry}, sort_keys=True), flush=True)
    return {"best": best.detach().cpu().numpy(), "best_loss": best_loss, "trace": trace}


def fit_softbody_scene(
    scene: SoftBodySceneData, *, configs: tuple[SoftBodyCEMConfig, ...] | None = None,
    device: str = "cuda:0", substeps: int = 8, verbose: bool = True,
    projection_tolerance: float = 1.0e-9,
    strict_projection: bool = True,
) -> tuple[dict[str, Any], SoftBodyCEMConfig, float]:
    if configs is None:
        configs = tuple(SoftBodyCEMConfig(seed=seed) for seed in DEFAULT_SEEDS)
    if not configs:
        raise ValueError("at least one CEM config is required")
    start = time.perf_counter()
    engine = BatchWarpSoftBodyEngine(
        scene, capacity=max(config.population for config in configs), device=device,
        substeps=substeps, projection_tolerance=projection_tolerance, strict_projection=strict_projection,
    )
    selected = None
    selected_config = None
    portfolio = []
    for config in configs:
        candidate_start = time.perf_counter()
        engine.center_weight = config.center_weight
        engine.enable_shape_projection = False
        engine.enable_soft_body_contacts = False
        independent = run_cem(
            scene, engine, config, joint=False, verbose=verbose
        )
        engine.enable_soft_body_contacts = True
        cem = run_cem(
            scene,
            engine,
            config,
            initial=independent["best"],
            initial_std_scale=config.joint_initial_std_scale,
            joint=True,
            generations_per_prefix=config.joint_generations_per_prefix,
            verbose=verbose,
        )
        engine.enable_shape_projection = True
        refined, shape_refinement = refine_shape_parameters(
            engine, config, cem["best"]
        )
        cem["best"] = refined
        raw_predicted = engine.rollout(cem["best"])
        selection_score = float(candidate_loss(
            engine, raw_predicted, end_offset=scene.frame_count-1,
            huber_delta_m=config.huber_delta_m, shape_weight=config.shape_weight,
            deformation_focus=config.deformation_focus,
        )[0].item())
        raw_metrics = softbody_metrics(scene, raw_predicted[0].detach().cpu().numpy())
        predicted, contacts = engine.rollout_numpy(cem["best"])
        metrics = softbody_metrics(scene, predicted)
        row = {
            "seed": config.seed,
            **metrics,
            "portfolio_selection_score_m": selection_score,
            "elapsed_seconds": time.perf_counter() - candidate_start,
        }
        portfolio.append(row)
        if (
            selected is None
            or selection_score < selected["portfolio_selection_score_m"]
        ):
            selected = {
                "best": cem["best"],
                "predicted": predicted,
                "contacts": contacts,
                "cem_trace": independent["trace"] + cem["trace"],
                "shape_refinement": shape_refinement,
                **metrics,
                "portfolio_selection_score_m": selection_score,
                "raw_metrics": raw_metrics,
            }
            selected_config = config
    assert selected is not None and selected_config is not None
    selected["portfolio"] = portfolio
    return selected, selected_config, time.perf_counter() - start
