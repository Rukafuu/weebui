use glam::{Mat4, Quat, Vec3};

use crate::loader::VrmModelData;
use crate::skeleton::SkeletonRuntime;

/// Um elo individual de uma cadeia física SpringBone (cabelo, fita, saia, manga).
#[derive(Debug, Clone)]
struct SpringJoint {
    node_index: usize,
    parent_index: Option<usize>,
    /// Direção local da ponta do osso no espaço de repouso do nó.
    local_dir: Vec3,
    /// Comprimento do segmento no espaço global escalado.
    bone_length: f32,
    current_tail: Vec3,
    prev_tail: Vec3,
    stiffness: f32,
    drag_force: f32,
    gravity_dir: Vec3,
    gravity_power: f32,
    hit_radius: f32,
    collider_indices: Vec<usize>,
}

/// Simulador físico Verlet de SpringBones compatível com VRM 0.x/1.0.
pub struct SpringBoneSimulator {
    joints: Vec<SpringJoint>,
    prev_hips_pos: Option<Vec3>,
    pub enabled: bool,
}

impl SpringBoneSimulator {
    pub fn new(model: &VrmModelData, skeleton: &SkeletonRuntime) -> Self {
        let mut joints = Vec::new();
        let scale = model.auto_scale;

        for group in &model.spring_bone_groups {
            for &root_idx in &group.root_bones {
                Self::collect_chain(
                    root_idx,
                    model,
                    skeleton,
                    scale,
                    group.stiffness,
                    group.drag_force,
                    group.gravity_dir,
                    group.gravity_power,
                    group.hit_radius * scale,
                    &group.collider_indices,
                    &mut joints,
                );
            }
        }

        let prev_hips_pos = model
            .humanoid_bones
            .get("hips")
            .map(|&idx| skeleton.world_positions[idx]);

        Self {
            joints,
            prev_hips_pos,
            enabled: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_chain(
        node_idx: usize,
        model: &VrmModelData,
        skeleton: &SkeletonRuntime,
        scale: f32,
        stiffness: f32,
        drag_force: f32,
        gravity_dir: Vec3,
        gravity_power: f32,
        hit_radius: f32,
        collider_indices: &[usize],
        out: &mut Vec<SpringJoint>,
    ) {
        let Some(node) = model.nodes.get(node_idx) else {
            return;
        };

        let head_pos = skeleton.world_positions[node_idx];
        let (local_dir, tail_pos) = if let Some(&first_child) = node.children.first() {
            let child_t = model.nodes[first_child].rest_translation;
            let dir = child_t.normalize_or_zero();
            let dir = if dir.length_squared() > 1e-6 {
                dir
            } else {
                Vec3::new(0.0, -1.0, 0.0)
            };
            let tp = skeleton.world_positions[first_child];
            (dir, tp)
        } else {
            let dir = node.rest_translation.normalize_or_zero();
            let dir = if dir.length_squared() > 1e-6 {
                dir
            } else {
                Vec3::new(0.0, -1.0, 0.0)
            };
            let tp = head_pos + (skeleton.world_rotations[node_idx] * (dir * 0.07 * scale));
            (dir, tp)
        };

        let bone_length = (tail_pos - head_pos).length().max(0.01 * scale);

        out.push(SpringJoint {
            node_index: node_idx,
            parent_index: node.parent,
            local_dir,
            bone_length,
            current_tail: tail_pos,
            prev_tail: tail_pos,
            stiffness,
            drag_force,
            gravity_dir,
            gravity_power: gravity_power * scale,
            hit_radius,
            collider_indices: collider_indices.to_vec(),
        });

        for &child_idx in &node.children {
            Self::collect_chain(
                child_idx,
                model,
                skeleton,
                scale,
                stiffness,
                drag_force,
                gravity_dir,
                gravity_power,
                hit_radius,
                collider_indices,
                out,
            );
        }
    }

    /// Executa um passo de integração Verlet e resolve colisões esféricas.
    pub fn step(&mut self, model: &VrmModelData, skeleton: &mut SkeletonRuntime, dt: f32) {
        if !self.enabled || self.joints.is_empty() {
            return;
        }
        let dt = dt.clamp(0.001, 1.0 / 30.0);

        // Compensação parcial de movimento de corpo inteiro para evitar chicoteamento excessivo
        let cur_hips = model
            .humanoid_bones
            .get("hips")
            .map(|&idx| skeleton.world_positions[idx]);
        let body_delta = match (self.prev_hips_pos, cur_hips) {
            (Some(prev), Some(cur)) => (cur - prev) * 0.55,
            _ => Vec3::ZERO,
        };
        self.prev_hips_pos = cur_hips;

        // Pré-calcula posição global e raio escalado de todas as esferas de colisão
        let scale = model.auto_scale;
        let world_colliders: Vec<(Vec3, f32)> = model
            .spring_colliders
            .iter()
            .map(|c| {
                let pos = skeleton
                    .world_matrices
                    .get(c.node_index)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY)
                    .transform_point3(c.offset);
                (pos, c.radius * scale)
            })
            .collect();

        for i in 0..self.joints.len() {
            let node_idx = self.joints[i].node_index;
            let parent_idx = self.joints[i].parent_index;

            // Atualiza a posição da base deste elo a partir da matriz global do pai atualizado
            let (parent_world_mat, parent_world_rot) = if let Some(p) = parent_idx {
                (skeleton.world_matrices[p], skeleton.world_rotations[p])
            } else {
                (skeleton.root_transform, skeleton.root_rotation)
            };

            let rest_local_rot = model.nodes[node_idx].rest_rotation;
            let rest_local_pos = skeleton.local_translations[node_idx];
            let head_pos = parent_world_mat.transform_point3(rest_local_pos);

            let current_tail = self.joints[i].current_tail + body_delta;
            let prev_tail = self.joints[i].prev_tail + body_delta;

            // 1. Termo de inércia amortecida (Verlet)
            let inertia = (current_tail - prev_tail) * (1.0 - self.joints[i].drag_force);

            // 2. Força restauradora de rigidez (stiffness) em direção à pose de repouso
            let rest_world_rot = (parent_world_rot * rest_local_rot).normalize();
            let stiffness_dir = (rest_world_rot * self.joints[i].local_dir).normalize_or_zero();
            let stiffness_force = stiffness_dir * (self.joints[i].stiffness * dt * 2.2);

            // 3. Força de gravidade
            let gravity_force = self.joints[i].gravity_dir * (self.joints[i].gravity_power * dt);

            let mut next_tail = current_tail + inertia + stiffness_force + gravity_force;

            // Restringe ao comprimento do osso
            let to_tail = (next_tail - head_pos).normalize_or_zero();
            if to_tail.length_squared() > 1e-6 {
                next_tail = head_pos + to_tail * self.joints[i].bone_length;
            } else {
                next_tail = head_pos + stiffness_dir * self.joints[i].bone_length;
            }

            // 4. Resolução de colisão com esferas do corpo/cabeça
            let hit_r = self.joints[i].hit_radius;
            for &col_idx in &self.joints[i].collider_indices {
                if let Some(&(col_pos, col_r)) = world_colliders.get(col_idx) {
                    let min_dist = hit_r + col_r;
                    let diff = next_tail - col_pos;
                    let dist_sq = diff.length_squared();
                    if dist_sq < min_dist * min_dist && dist_sq > 1e-10 {
                        let normal = diff / dist_sq.sqrt();
                        next_tail = col_pos + normal * min_dist;
                        // Re-projeta para manter o comprimento do osso a partir da base
                        let dir = (next_tail - head_pos).normalize_or_zero();
                        if dir.length_squared() > 1e-6 {
                            next_tail = head_pos + dir * self.joints[i].bone_length;
                        }
                    }
                }
            }

            self.joints[i].prev_tail = current_tail;
            self.joints[i].current_tail = next_tail;

            // 5. Calcula a nova rotação local do osso a partir da direção da ponta
            let new_world_dir = (next_tail - head_pos).normalize_or_zero();
            if new_world_dir.length_squared() > 1e-6 && stiffness_dir.length_squared() > 1e-6 {
                let swing = Quat::from_rotation_arc(stiffness_dir, new_world_dir);
                let new_world_rot = (swing * rest_world_rot).normalize();
                let new_local_rot = (parent_world_rot.inverse() * new_world_rot).normalize();

                skeleton.local_rotations[node_idx] = new_local_rot;
                let local_mat = Mat4::from_scale_rotation_translation(
                    skeleton.local_scales[node_idx],
                    new_local_rot,
                    rest_local_pos,
                );
                skeleton.world_matrices[node_idx] = parent_world_mat * local_mat;
                skeleton.world_rotations[node_idx] = new_world_rot;
                skeleton.world_positions[node_idx] = head_pos;
            }
        }
    }
}
