use std::collections::HashMap;
use std::sync::Arc;

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

use crate::loader::{VrmModelData, VrmaClip, VrmaRotationTrack, is_finger_bone};

/// Estado de reprodução de um clipe VRMA no mixer esquelético.
#[derive(Debug, Clone)]
struct ActiveClipState {
    clip: Arc<VrmaClip>,
    elapsed: f32,
    looping: bool,
    time_scale: f32,
}

/// Gerenciador de esqueleto, retargeting VRMA, respiração procedural e Look-At IK.
pub struct SkeletonRuntime {
    pub local_translations: Vec<Vec3>,
    pub local_rotations: Vec<Quat>,
    pub local_scales: Vec<Vec3>,
    pub world_matrices: Vec<Mat4>,
    pub world_rotations: Vec<Quat>,
    pub world_positions: Vec<Vec3>,
    /// Matriz raiz do modelo (aplica rotação 180° Y para VRM 0.x + auto_scale + auto_offset).
    pub root_transform: Mat4,
    pub root_rotation: Quat,

    // Mixer VRMA com crossfade suave
    active_clip: Option<ActiveClipState>,
    prev_clip: Option<ActiveClipState>,
    crossfade_elapsed: f32,
    crossfade_duration: f32,
    pub strip_fingers: bool,
    pub arm_clearance_deg: f32,

    // Tempo acumulado para respiração procedural, sacadas oculares e sway
    pub time_sec: f32,

    // Look-At suave (coordenadas normalizadas -1..1 na tela)
    pub look_target: Vec2,
    pub look_current: Vec2,
    saccade_offset: Vec2,
    saccade_timer: f32,
}

impl SkeletonRuntime {
    pub fn new(model: &VrmModelData) -> Self {
        let local_translations = model.nodes.iter().map(|nd| nd.rest_translation).collect();
        let local_rotations = model.nodes.iter().map(|nd| nd.rest_rotation).collect();
        let local_scales = model.nodes.iter().map(|nd| nd.rest_scale).collect();
        let world_matrices = model.nodes.iter().map(|nd| nd.rest_world_matrix).collect();
        let world_rotations = model.nodes.iter().map(|nd| nd.rest_world_rotation).collect();
        let world_positions = model.nodes.iter().map(|nd| nd.rest_world_position).collect();

        // Em VRM 0.x o modelo aponta para -Z. Rotacionamos 180° em Y (idêntico a VRMUtils.rotateVRM0)
        // para que o personagem olhe para +Z em direção à câmera em (0, 1.25, 3.8).
        let root_rotation = if model.is_vrm0 {
            Quat::from_rotation_y(std::f32::consts::PI)
        } else {
            Quat::IDENTITY
        };
        let rotated_offset = root_rotation * model.auto_offset;
        let root_transform = Mat4::from_scale_rotation_translation(
            Vec3::splat(model.auto_scale),
            root_rotation,
            rotated_offset,
        );

        let mut rt = Self {
            local_translations,
            local_rotations,
            local_scales,
            world_matrices,
            world_rotations,
            world_positions,
            root_transform,
            root_rotation,
            active_clip: None,
            prev_clip: None,
            crossfade_elapsed: 0.0,
            crossfade_duration: 0.45,
            strip_fingers: true,
            arm_clearance_deg: 26.0,
            time_sec: 0.0,
            look_target: Vec2::ZERO,
            look_current: Vec2::ZERO,
            saccade_offset: Vec2::ZERO,
            saccade_timer: 1.2,
        };
        rt.update_pose(model, 0.0);
        rt
    }

    /// Inicia a reprodução de um clipe `.vrma` com transição suave (crossfade).
    pub fn play_vrma(&mut self, clip: Arc<VrmaClip>, looping: bool, crossfade: f32) {
        if let Some(cur) = self.active_clip.take() {
            self.prev_clip = Some(cur);
            self.crossfade_elapsed = 0.0;
            self.crossfade_duration = crossfade.max(0.05);
        }
        self.active_clip = Some(ActiveClipState {
            clip,
            elapsed: 0.0,
            looping,
            time_scale: 1.0,
        });
    }

    pub fn stop_vrma(&mut self, crossfade: f32) {
        if let Some(cur) = self.active_clip.take() {
            self.prev_clip = Some(cur);
            self.crossfade_elapsed = 0.0;
            self.crossfade_duration = crossfade.max(0.05);
        }
    }

    pub fn current_clip_name(&self) -> Option<&str> {
        self.active_clip.as_ref().map(|c| c.clip.name.as_str())
    }

    /// Atualiza todas as transformações locais e globais do esqueleto.
    pub fn update_pose(&mut self, model: &VrmModelData, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.time_sec += dt;

        // 1. Restaura a pose base
        for (i, node) in model.nodes.iter().enumerate() {
            self.local_translations[i] = node.rest_translation;
            self.local_rotations[i] = node.rest_rotation;
            self.local_scales[i] = node.rest_scale;
        }

        // 2. Avança timers de animação VRMA
        let mut clip_finished = false;
        if let Some(active) = &mut self.active_clip {
            active.elapsed += dt * active.time_scale;
            if active.clip.duration > 0.0 && active.elapsed >= active.clip.duration {
                if active.looping {
                    active.elapsed %= active.clip.duration;
                } else {
                    active.elapsed = active.clip.duration;
                    clip_finished = true;
                }
            }
        }
        if let Some(prev) = &mut self.prev_clip {
            prev.elapsed += dt * prev.time_scale;
            self.crossfade_elapsed += dt;
            if self.crossfade_elapsed >= self.crossfade_duration {
                self.prev_clip = None;
            }
        }
        if clip_finished {
            // Quando um gesto único termina, faz crossfade suave de volta
            let finished = self.active_clip.take();
            self.prev_clip = finished;
            self.crossfade_elapsed = 0.0;
            self.crossfade_duration = 0.45;
        }

        // 3. Aplica VRMA ou Pose Procedural VTuber
        if self.active_clip.is_some() || self.prev_clip.is_some() {
            self.apply_vrma_pose(model);
        } else {
            self.apply_procedural_idle_pose(model);
        }

        // 4. Aplica respiração sutil sempre ativa + Look-At IK + micro-sacadas oculares
        self.apply_breathing_and_look_at(model, dt);

        // 5. Propaga matrizes globais na ordem topológica
        self.recompute_world_transforms(model);
    }

    fn apply_vrma_pose(&mut self, model: &VrmModelData) {
        let blend_alpha = if self.prev_clip.is_some() && self.crossfade_duration > 1e-4 {
            (self.crossfade_elapsed / self.crossfade_duration).clamp(0.0, 1.0)
        } else {
            1.0
        };

        for (bone_name, &node_idx) in &model.humanoid_bones {
            if self.strip_fingers && is_finger_bone(bone_name) {
                continue;
            }

            let cur_q = self
                .active_clip
                .as_ref()
                .and_then(|st| sample_vrma_rotation(&st.clip.rotation_tracks, bone_name, st.elapsed));
            let prev_q = self
                .prev_clip
                .as_ref()
                .and_then(|st| sample_vrma_rotation(&st.clip.rotation_tracks, bone_name, st.elapsed));

            let norm_q = match (prev_q, cur_q) {
                (Some(pq), Some(cq)) => pq.slerp(cq, blend_alpha),
                (None, Some(cq)) => Quat::IDENTITY.slerp(cq, blend_alpha),
                (Some(pq), None) => pq.slerp(Quat::IDENTITY, blend_alpha),
                (None, None) => continue,
            };

            let vrm_q = if model.is_vrm0 {
                Quat::from_xyzw(-norm_q.x, norm_q.y, -norm_q.z, norm_q.w)
            } else {
                norm_q
            };

            let node = &model.nodes[node_idx];
            let retargeted = node.rest_rotation
                * node.rest_world_rotation.inverse()
                * vrm_q
                * node.rest_world_rotation;
            self.local_rotations[node_idx] = retargeted.normalize();
        }

        // Translação do quadril (hips) escalada proporcionalmente à altura das pernas do modelo
        if let Some(&hips_idx) = model.humanoid_bones.get("hips") {
            let vrm_hips_y = model.nodes[hips_idx].rest_world_position.y.max(0.1);
            let cur_t = self.active_clip.as_ref().and_then(|st| {
                let tr = st.clip.hips_translation.as_ref()?;
                let d = sample_vec3_track(&tr.times, &tr.deltas, st.elapsed);
                Some(d * (vrm_hips_y / tr.rest_hips_y))
            });
            let prev_t = self.prev_clip.as_ref().and_then(|st| {
                let tr = st.clip.hips_translation.as_ref()?;
                let d = sample_vec3_track(&tr.times, &tr.deltas, st.elapsed);
                Some(d * (vrm_hips_y / tr.rest_hips_y))
            });
            let delta = match (prev_t, cur_t) {
                (Some(pt), Some(ct)) => pt.lerp(ct, blend_alpha),
                (None, Some(ct)) => ct * blend_alpha,
                (Some(pt), None) => pt * (1.0 - blend_alpha),
                (None, None) => Vec3::ZERO,
            };
            let vrm_delta = if model.is_vrm0 {
                Vec3::new(-delta.x, delta.y, -delta.z)
            } else {
                delta
            };
            self.local_translations[hips_idx] = model.nodes[hips_idx].rest_translation + vrm_delta;
        }

        // Correção pós-mixer (arm clearance) para não prender as mangas de sino no tronco
        let rad = self.arm_clearance_deg.to_radians();
        if rad.abs() > 1e-4 {
            if let Some(&l_arm) = model.humanoid_bones.get("leftUpperArm") {
                let corr = Quat::from_euler(EulerRot::XYZ, rad * 0.15, 0.0, -rad);
                self.local_rotations[l_arm] = (self.local_rotations[l_arm] * corr).normalize();
            }
            if let Some(&r_arm) = model.humanoid_bones.get("rightUpperArm") {
                let corr = Quat::from_euler(EulerRot::XYZ, rad * 0.15, 0.0, rad);
                self.local_rotations[r_arm] = (self.local_rotations[r_arm] * corr).normalize();
            }
        }
    }

    fn apply_procedural_idle_pose(&mut self, model: &VrmModelData) {
        let t = self.time_sec;
        // Braços relaxados ao lado do corpo (idêntico a Usami.jsx linhas 231-245)
        let arm_down = 66.0_f32.to_radians();
        let arm_fwd = 4.0_f32.to_radians();
        let arm_sway = (t * 1.4).sin() * 0.025;

        if let Some(&l_upper) = model.humanoid_bones.get("leftUpperArm") {
            let q = Quat::from_euler(EulerRot::XYZ, arm_fwd, 0.0, arm_down + arm_sway);
            self.local_rotations[l_upper] = (model.nodes[l_upper].rest_rotation * q).normalize();
        }
        if let Some(&r_upper) = model.humanoid_bones.get("rightUpperArm") {
            let q = Quat::from_euler(EulerRot::XYZ, arm_fwd, 0.0, -arm_down - arm_sway);
            self.local_rotations[r_upper] = (model.nodes[r_upper].rest_rotation * q).normalize();
        }
        let forearm_bend = 7.0_f32.to_radians();
        if let Some(&l_lower) = model.humanoid_bones.get("leftLowerArm") {
            let q = Quat::from_rotation_x(forearm_bend);
            self.local_rotations[l_lower] = (model.nodes[l_lower].rest_rotation * q).normalize();
        }
        if let Some(&r_lower) = model.humanoid_bones.get("rightLowerArm") {
            let q = Quat::from_rotation_x(forearm_bend);
            self.local_rotations[r_lower] = (model.nodes[r_lower].rest_rotation * q).normalize();
        }
    }

    fn apply_breathing_and_look_at(&mut self, model: &VrmModelData, dt: f32) {
        let t = self.time_sec;

        // Suavização exponencial do cursor Look-At
        let lerp_factor = (dt * 7.5).clamp(0.0, 1.0);
        self.look_current = self.look_current.lerp(self.look_target, lerp_factor);

        // Micro-sacadas oculares naturais a cada ~1.5-3.2s
        self.saccade_timer -= dt;
        if self.saccade_timer <= 0.0 {
            let phase = (t * 12.9898).sin() * 43758.5453;
            let rx = (phase.fract() - 0.5) * 0.04;
            let ry = ((phase * 1.7).fract() - 0.5) * 0.025;
            self.saccade_offset = Vec2::new(rx, ry);
            self.saccade_timer = 1.4 + (phase.abs().fract()) * 1.8;
        }

        // Respiração suave na coluna e tórax
        let breath_x = (t * 1.85).sin() * 0.016;
        let sway_z = (t * 0.85).sin() * 0.014;
        if let Some(&spine) = model.humanoid_bones.get("spine") {
            let q = Quat::from_euler(EulerRot::XYZ, breath_x * 0.5, 0.0, sway_z * 0.5);
            self.local_rotations[spine] = (self.local_rotations[spine] * q).normalize();
        }
        if let Some(&chest) = model.humanoid_bones.get("chest") {
            let q = Quat::from_euler(EulerRot::XYZ, breath_x, 0.0, sway_z * 0.4);
            self.local_rotations[chest] = (self.local_rotations[chest] * q).normalize();
        }

        // Orientação da cabeça/pescoço acompanhando o mouse (com compensação de sinal VRM 0.x)
        let sign = if model.is_vrm0 { -1.0 } else { 1.0 };
        let yaw = self.look_current.x * 0.28 * sign;
        let pitch = -self.look_current.y * 0.18 * sign;
        let head_tilt = (t * 0.65).sin() * 0.022;

        if let Some(&neck) = model.humanoid_bones.get("neck") {
            let q = Quat::from_euler(EulerRot::YXZ, yaw * 0.35, pitch * 0.35, head_tilt * 0.3);
            self.local_rotations[neck] = (self.local_rotations[neck] * q).normalize();
        }
        if let Some(&head) = model.humanoid_bones.get("head") {
            let q = Quat::from_euler(EulerRot::YXZ, yaw * 0.65, pitch * 0.65, head_tilt * 0.7);
            self.local_rotations[head] = (self.local_rotations[head] * q).normalize();
        }

        // Olhos (leftEye / rightEye) com Look-At + micro-sacadas
        let eye_yaw = (self.look_current.x * 0.12 + self.saccade_offset.x) * sign;
        let eye_pitch = (-self.look_current.y * 0.08 + self.saccade_offset.y) * sign;
        let eye_q = Quat::from_euler(EulerRot::YXZ, eye_yaw, eye_pitch, 0.0);
        for eye_name in ["leftEye", "rightEye"] {
            if let Some(&eye_idx) = model.humanoid_bones.get(eye_name) {
                self.local_rotations[eye_idx] = (self.local_rotations[eye_idx] * eye_q).normalize();
            }
        }
    }

    /// Propaga transformações locais para globais na ordem topológica.
    pub fn recompute_world_transforms(&mut self, model: &VrmModelData) {
        for &idx in &model.topological_order {
            let local_mat = Mat4::from_scale_rotation_translation(
                self.local_scales[idx],
                self.local_rotations[idx],
                self.local_translations[idx],
            );
            if let Some(p) = model.nodes[idx].parent {
                let p_mat = self.world_matrices[p];
                let p_rot = self.world_rotations[p];
                let w_mat = p_mat * local_mat;
                self.world_matrices[idx] = w_mat;
                self.world_rotations[idx] = (p_rot * self.local_rotations[idx]).normalize();
                self.world_positions[idx] = w_mat.transform_point3(Vec3::ZERO);
            } else {
                let w_mat = self.root_transform * local_mat;
                self.world_matrices[idx] = w_mat;
                self.world_rotations[idx] =
                    (self.root_rotation * self.local_rotations[idx]).normalize();
                self.world_positions[idx] = w_mat.transform_point3(Vec3::ZERO);
            }
        }
    }

    /// Calcula as matrizes de juntas (`joint_matrix = world_matrix[joint] * inverse_bind_matrix`)
    /// para uma skin específica, até `max_joints`.
    pub fn compute_skin_matrices(
        &self,
        model: &VrmModelData,
        skin_index: Option<usize>,
        out: &mut [[f32; 16]],
    ) {
        if let Some(s_idx) = skin_index {
            if let Some(skin) = model.skins.get(s_idx) {
                let count = skin.joints.len().min(out.len());
                for i in 0..count {
                    let joint_node = skin.joints[i];
                    let ibm = skin.inverse_bind_matrices[i];
                    let mat = self.world_matrices[joint_node] * ibm;
                    out[i] = mat.to_cols_array();
                }
                return;
            }
        }
        if !out.is_empty() {
            out[0] = self.root_transform.to_cols_array();
        }
    }
}

fn sample_vrma_rotation(
    tracks: &HashMap<String, VrmaRotationTrack>,
    bone_name: &str,
    time: f32,
) -> Option<Quat> {
    let track = tracks.get(bone_name)?;
    if track.rotations.is_empty() {
        return None;
    }
    if track.rotations.len() == 1 || time <= track.times[0] {
        return Some(track.rotations[0]);
    }
    let last = track.rotations.len() - 1;
    if time >= track.times[last] {
        return Some(track.rotations[last]);
    }
    let idx = match track
        .times
        .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(i) => return Some(track.rotations[i]),
        Err(i) => i.saturating_sub(1),
    };
    let t0 = track.times[idx];
    let t1 = track.times[idx + 1];
    let alpha = if t1 > t0 {
        ((time - t0) / (t1 - t0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some(track.rotations[idx].slerp(track.rotations[idx + 1], alpha))
}

fn sample_vec3_track(times: &[f32], values: &[Vec3], time: f32) -> Vec3 {
    if values.is_empty() {
        return Vec3::ZERO;
    }
    if values.len() == 1 || time <= times[0] {
        return values[0];
    }
    let last = values.len() - 1;
    if time >= times[last] {
        return values[last];
    }
    let idx = match times
        .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(i) => return values[i],
        Err(i) => i.saturating_sub(1),
    };
    let t0 = times[idx];
    let t1 = times[idx + 1];
    let alpha = if t1 > t0 {
        ((time - t0) / (t1 - t0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    values[idx].lerp(values[idx + 1], alpha)
}
