use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use serde_json::Value;

/// Modos de transparência do material (glTF / MToon).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Modo de culling de faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// Propriedades unificadas PBR + MToon (Anime Cel-Shading).
#[derive(Debug, Clone)]
pub struct VrmMaterial {
    pub name: String,
    pub base_color: Vec4,
    pub shade_color: Vec4,
    pub emissive_color: Vec3,
    pub base_texture: Option<usize>,
    pub shade_texture: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub cull_mode: CullMode,
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub rim_color: Vec3,
    pub rim_fresnel_power: f32,
    pub rim_lift: f32,
    pub outline_width: f32,
    pub outline_color: Vec4,
    pub has_outline: bool,
    pub render_queue: i32,
    pub is_mtoon: bool,
}

impl Default for VrmMaterial {
    fn default() -> Self {
        Self {
            name: "default".into(),
            base_color: Vec4::ONE,
            shade_color: Vec4::new(0.88, 0.82, 0.86, 1.0),
            emissive_color: Vec3::ZERO,
            base_texture: None,
            shade_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            cull_mode: CullMode::Back,
            shade_shift: -0.1,
            shade_toony: 0.5,
            rim_color: Vec3::new(0.18, 0.14, 0.22),
            rim_fresnel_power: 3.0,
            rim_lift: 0.0,
            outline_width: 0.0,
            outline_color: Vec4::new(0.15, 0.10, 0.14, 1.0),
            has_outline: false,
            render_queue: 2000,
            is_mtoon: false,
        }
    }
}

/// Textura RGBA8 decodificada pronta para upload na GPU.
#[derive(Debug, Clone)]
pub struct DecodedTexture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Deltas de um Morph Target (BlendShape) para uma malha.
#[derive(Debug, Clone)]
pub struct MorphTargetDelta {
    pub name: String,
    /// Lista esparsa de `(vertex_index, position_delta)` para acumulação ultra-rápida.
    pub position_deltas: Vec<(u32, Vec3)>,
    /// Lista esparsa de `(vertex_index, normal_delta)`.
    pub normal_deltas: Vec<(u32, Vec3)>,
}

/// Primitiva de renderização (compartilha os vértices de sua `VrmMesh`).
#[derive(Debug, Clone)]
pub struct VrmPrimitive {
    pub indices: Vec<u32>,
    pub material_index: usize,
}

/// Malha 3D deduplicada (múltiplas primitivas compartilham os mesmos buffers de vértices e morphs).
#[derive(Debug, Clone)]
pub struct VrmMesh {
    pub name: String,
    pub mesh_index: usize,
    pub node_index: usize,
    pub skin_index: Option<usize>,
    pub visible: bool,
    pub base_positions: Vec<Vec3>,
    pub base_normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub morph_targets: Vec<MorphTargetDelta>,
    pub primitives: Vec<VrmPrimitive>,
}

/// Nó da hierarquia de cena glTF/VRM.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub index: usize,
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
    pub rest_world_matrix: Mat4,
    pub rest_world_rotation: Quat,
    pub rest_world_position: Vec3,
}

/// Skin (esqueleto de deformação de malha).
#[derive(Debug, Clone)]
pub struct VrmSkin {
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// Vínculo de um BlendShapeGroup do VRM a um alvo de morph específico.
#[derive(Debug, Clone)]
pub struct BlendShapeBind {
    pub mesh_index: usize,
    pub target_index: usize,
    /// Peso normalizado em `0.0..=1.0` (no VRM 0.x vem como `0..=100`).
    pub weight: f32,
}

/// Grupo de expressão VRM (ex.: `a`, `i`, `u`, `e`, `o`, `blink`, `01-prefab_happy01`).
#[derive(Debug, Clone)]
pub struct BlendShapeGroup {
    pub name: String,
    pub preset_name: String,
    pub is_binary: bool,
    pub binds: Vec<BlendShapeBind>,
}

/// Esfera de colisão para física de SpringBone.
#[derive(Debug, Clone)]
pub struct SpringCollider {
    pub node_index: usize,
    pub offset: Vec3,
    pub radius: f32,
}

/// Grupo de cadeia física SpringBone (cabelos, fitas, saia, mangas).
#[derive(Debug, Clone)]
pub struct SpringBoneGroup {
    pub comment: String,
    pub stiffness: f32,
    pub gravity_power: f32,
    pub gravity_dir: Vec3,
    pub drag_force: f32,
    pub hit_radius: f32,
    pub root_bones: Vec<usize>,
    pub collider_indices: Vec<usize>,
}

/// Modelo VRM/glTF completo carregado na memória.
#[derive(Debug, Clone)]
pub struct VrmModelData {
    pub title: String,
    pub is_vrm0: bool,
    pub nodes: Vec<SceneNode>,
    pub root_nodes: Vec<usize>,
    /// Ordem topológica (pais sempre antes dos filhos) para atualização de transformações sem recursão.
    pub topological_order: Vec<usize>,
    pub skins: Vec<VrmSkin>,
    pub meshes: Vec<VrmMesh>,
    pub materials: Vec<VrmMaterial>,
    pub textures: Vec<DecodedTexture>,
    pub humanoid_bones: HashMap<String, usize>,
    pub blend_shape_groups: Vec<BlendShapeGroup>,
    pub spring_colliders: Vec<SpringCollider>,
    pub spring_bone_groups: Vec<SpringBoneGroup>,
    /// Escala automática para normalizar a altura do avatar para ~2.25m (padrão Usami).
    pub auto_scale: f32,
    /// Offset de centralização do modelo após auto-scale.
    pub auto_offset: Vec3,
}

/// Faixa de keyframe de rotação para um osso humanoide em um arquivo `.vrma`.
#[derive(Debug, Clone)]
pub struct VrmaRotationTrack {
    pub bone_name: String,
    pub times: Vec<f32>,
    /// Quaternions já normalizados no espaço T-pose humanoide (`Q_parent * q_raw * Q_bone^-1`).
    pub rotations: Vec<Quat>,
}

/// Faixa de keyframe de translação do quadril (`hips`) em um arquivo `.vrma`.
#[derive(Debug, Clone)]
pub struct VrmaTranslationTrack {
    pub times: Vec<f32>,
    /// Deslocamento em relação à posição de repouso do quadril no `.vrma`.
    pub deltas: Vec<Vec3>,
    pub rest_hips_y: f32,
}

/// Clipe de animação `.vrma` carregado e pronto para retargeting em qualquer VRM.
#[derive(Debug, Clone)]
pub struct VrmaClip {
    pub name: String,
    pub duration: f32,
    pub rotation_tracks: HashMap<String, VrmaRotationTrack>,
    pub hips_translation: Option<VrmaTranslationTrack>,
}

/// Mapa de parentesco humanoide segundo a especificação VRMC_vrm_animation 1.0.
pub fn humanoid_parent_bone(bone: &str) -> Option<&'static str> {
    match bone {
        "hips" => None,
        "spine" => Some("hips"),
        "chest" => Some("spine"),
        "upperChest" => Some("chest"),
        "neck" => Some("upperChest"),
        "head" => Some("neck"),
        "leftEye" | "rightEye" | "jaw" => Some("head"),
        "leftShoulder" | "rightShoulder" => Some("upperChest"),
        "leftUpperArm" => Some("leftShoulder"),
        "leftLowerArm" => Some("leftUpperArm"),
        "leftHand" => Some("leftLowerArm"),
        "rightUpperArm" => Some("rightShoulder"),
        "rightLowerArm" => Some("rightUpperArm"),
        "rightHand" => Some("rightLowerArm"),
        "leftUpperLeg" | "rightUpperLeg" => Some("hips"),
        "leftLowerLeg" => Some("leftUpperLeg"),
        "leftFoot" => Some("leftLowerLeg"),
        "leftToes" => Some("leftFoot"),
        "rightLowerLeg" => Some("rightUpperLeg"),
        "rightFoot" => Some("rightLowerLeg"),
        "rightToes" => Some("rightFoot"),
        // Dedos da mão esquerda
        "leftThumbMetacarpal" | "leftThumbProximal" | "leftIndexProximal"
        | "leftMiddleProximal" | "leftRingProximal" | "leftLittleProximal" => Some("leftHand"),
        "leftThumbDistal" => Some("leftThumbProximal"),
        "leftIndexIntermediate" => Some("leftIndexProximal"),
        "leftIndexDistal" => Some("leftIndexIntermediate"),
        "leftMiddleIntermediate" => Some("leftMiddleProximal"),
        "leftMiddleDistal" => Some("leftMiddleIntermediate"),
        "leftRingIntermediate" => Some("leftRingProximal"),
        "leftRingDistal" => Some("leftRingIntermediate"),
        "leftLittleIntermediate" => Some("leftLittleProximal"),
        "leftLittleDistal" => Some("leftLittleIntermediate"),
        // Dedos da mão direita
        "rightThumbMetacarpal" | "rightThumbProximal" | "rightIndexProximal"
        | "rightMiddleProximal" | "rightRingProximal" | "rightLittleProximal" => Some("rightHand"),
        "rightThumbDistal" => Some("rightThumbProximal"),
        "rightIndexIntermediate" => Some("rightIndexProximal"),
        "rightIndexDistal" => Some("rightIndexIntermediate"),
        "rightMiddleIntermediate" => Some("rightMiddleProximal"),
        "rightMiddleDistal" => Some("rightMiddleIntermediate"),
        "rightRingIntermediate" => Some("rightRingProximal"),
        "rightRingDistal" => Some("rightRingIntermediate"),
        "rightLittleIntermediate" => Some("rightLittleProximal"),
        "rightLittleDistal" => Some("rightLittleIntermediate"),
        _ => None,
    }
}

pub fn is_finger_bone(bone: &str) -> bool {
    const FINGERS: [&str; 5] = ["Thumb", "Index", "Middle", "Ring", "Little"];
    FINGERS.iter().any(|f| bone.contains(f))
}

/// Extrai o JSON bruto de um arquivo `.glb` / `.vrm` / `.vrma` ou `.gltf`.
fn parse_raw_gltf_json(bytes: &[u8]) -> Result<Value> {
    if bytes.len() >= 20 && &bytes[0..4] == b"glTF" {
        let chunk_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let end = (20 + chunk_len).min(bytes.len());
        let json: Value = serde_json::from_slice(&bytes[20..end])
            .context("Falha ao decodificar chunk JSON do binário GLB/VRM")?;
        Ok(json)
    } else {
        let json: Value = serde_json::from_slice(bytes)
            .context("Falha ao decodificar arquivo JSON glTF")?;
        Ok(json)
    }
}

impl VrmModelData {
    /// Carrega um arquivo `.vrm`, `.glb` ou `.gltf` do disco.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("Não foi possível ler o arquivo 3D: {}", path.display()))?;
        Self::load_from_slice(&bytes, path.parent())
    }

    /// Carrega um modelo VRM/glTF a partir de bytes em memória.
    pub fn load_from_slice(bytes: &[u8], _base_dir: Option<&Path>) -> Result<Self> {
        let raw_json = parse_raw_gltf_json(bytes)?;
        let gltf = gltf::Gltf::from_slice(bytes).context("Falha ao importar slice glTF/VRM")?;
        let document = gltf.document;
        let blob_slice: &[u8] = gltf.blob.as_deref().unwrap_or(&[]);
        if blob_slice.is_empty() {
            bail!("Arquivo VRM/GLB não possui chunk binário de geometria");
        }

        // 1. Hierarquia de nós (SceneNode)
        let node_count = document.nodes().len();
        let mut parents = vec![None; node_count];
        let mut children_map = vec![Vec::new(); node_count];
        for node in document.nodes() {
            let p = node.index();
            for child in node.children() {
                parents[child.index()] = Some(p);
                children_map[p].push(child.index());
            }
        }

        let mut nodes = Vec::with_capacity(node_count);
        for node in document.nodes() {
            let idx = node.index();
            let (t, r, s) = node.transform().decomposed();
            nodes.push(SceneNode {
                index: idx,
                name: node.name().unwrap_or("").to_string(),
                parent: parents[idx],
                children: children_map[idx].clone(),
                rest_translation: Vec3::from_array(t),
                rest_rotation: Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
                rest_scale: Vec3::from_array(s),
                rest_world_matrix: Mat4::IDENTITY,
                rest_world_rotation: Quat::IDENTITY,
                rest_world_position: Vec3::ZERO,
            });
        }

        let root_nodes: Vec<usize> = (0..node_count).filter(|&i| parents[i].is_none()).collect();
        let mut topological_order = Vec::with_capacity(node_count);
        let mut stack: Vec<usize> = root_nodes.iter().copied().rev().collect();
        while let Some(idx) = stack.pop() {
            topological_order.push(idx);
            for &child in nodes[idx].children.iter().rev() {
                stack.push(child);
            }
        }

        // Calcula matrizes e rotações globais de repouso (rest pose)
        for &idx in &topological_order {
            let local_mat = Mat4::from_scale_rotation_translation(
                nodes[idx].rest_scale,
                nodes[idx].rest_rotation,
                nodes[idx].rest_translation,
            );
            if let Some(p) = nodes[idx].parent {
                let parent_mat = nodes[p].rest_world_matrix;
                let parent_rot = nodes[p].rest_world_rotation;
                let world_mat = parent_mat * local_mat;
                let world_rot = (parent_rot * nodes[idx].rest_rotation).normalize();
                let world_pos = world_mat.transform_point3(Vec3::ZERO);
                nodes[idx].rest_world_matrix = world_mat;
                nodes[idx].rest_world_rotation = world_rot;
                nodes[idx].rest_world_position = world_pos;
            } else {
                nodes[idx].rest_world_matrix = local_mat;
                nodes[idx].rest_world_rotation = nodes[idx].rest_rotation;
                nodes[idx].rest_world_position = nodes[idx].rest_translation;
            }
        }

        // 2. Skins (Inverse Bind Matrices + Joints)
        let mut skins = Vec::new();
        for skin in document.skins() {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let reader = skin.reader(|_| Some(blob_slice));
            let ibm: Vec<Mat4> = if let Some(iter) = reader.read_inverse_bind_matrices() {
                iter.map(|m| Mat4::from_cols_array_2d(&m)).collect()
            } else {
                vec![Mat4::IDENTITY; joints.len()]
            };
            skins.push(VrmSkin {
                joints,
                inverse_bind_matrices: ibm,
            });
        }

        // 3. Texturas decodificadas em paralelo em todos os núcleos da CPU
        let raw_img_slices: Vec<(String, &[u8])> = document
            .textures()
            .map(|tex| {
                let name = tex.name().unwrap_or("").to_string();
                match tex.source().source() {
                    gltf::image::Source::View { view, .. } => {
                        let start = view.offset();
                        let end = (start + view.length()).min(blob_slice.len());
                        (name, &blob_slice[start..end])
                    }
                    gltf::image::Source::Uri { .. } => (name, &[][..]),
                }
            })
            .collect();

        let textures: Vec<DecodedTexture> = std::thread::scope(|s| {
            let handles: Vec<_> = raw_img_slices
                .into_iter()
                .map(|(name, img_bytes)| {
                    s.spawn(move || {
                        if !img_bytes.is_empty() {
                            if let Ok(dyn_img) = image::load_from_memory(img_bytes) {
                                let rgba8 = dyn_img.into_rgba8();
                                let (w, h) = rgba8.dimensions();
                                return DecodedTexture {
                                    name,
                                    width: w,
                                    height: h,
                                    rgba: rgba8.into_raw(),
                                };
                            }
                        }
                        DecodedTexture {
                            name,
                            width: 1,
                            height: 1,
                            rgba: vec![255, 255, 255, 255],
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // 4. Materiais (glTF PBR + extensão VRM 0.x MToon)
        let vrm_ext = raw_json.get("extensions").and_then(|e| e.get("VRM"));
        let vrm1_ext = raw_json.get("extensions").and_then(|e| e.get("VRMC_vrm"));
        let is_vrm0 = vrm_ext.is_some();

        let vrm_mat_props = vrm_ext
            .and_then(|v| v.get("materialProperties"))
            .and_then(|m| m.as_array());

        let mut materials = Vec::new();
        for (mat_idx, mat) in document.materials().enumerate() {
            let mut vmat = VrmMaterial {
                name: mat.name().unwrap_or("material").to_string(),
                ..Default::default()
            };

            let pbr = mat.pbr_metallic_roughness();
            let bcf = pbr.base_color_factor();
            vmat.base_color = Vec4::from_array(bcf);
            vmat.shade_color = Vec4::new(bcf[0] * 0.86, bcf[1] * 0.80, bcf[2] * 0.85, bcf[3]);
            vmat.emissive_color = Vec3::from_array(mat.emissive_factor());
            if let Some(info) = pbr.base_color_texture() {
                vmat.base_texture = Some(info.texture().index());
            }
            vmat.alpha_mode = match mat.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };
            vmat.alpha_cutoff = mat.alpha_cutoff().unwrap_or(0.5);
            vmat.cull_mode = if mat.double_sided() {
                CullMode::None
            } else {
                CullMode::Back
            };
            vmat.render_queue = match vmat.alpha_mode {
                AlphaMode::Opaque => 2000,
                AlphaMode::Mask => 2450,
                AlphaMode::Blend => 3000,
            };

            // Sobrescreve com propriedades ricas do VRM/MToon quando disponíveis
            if let Some(props_arr) = vrm_mat_props {
                if let Some(prop) = props_arr.get(mat_idx) {
                    if prop.get("shader").and_then(|s| s.as_str()) == Some("VRM/MToon") {
                        vmat.is_mtoon = true;
                    }
                    if let Some(rq) = prop.get("renderQueue").and_then(|r| r.as_i64()) {
                        vmat.render_queue = rq as i32;
                    }
                    if let Some(vec_props) = prop.get("vectorProperties").and_then(|v| v.as_object()) {
                        if let Some(c) = read_vec4_json(vec_props.get("_Color")) {
                            vmat.base_color = c;
                        }
                        if let Some(sc) = read_vec4_json(vec_props.get("_ShadeColor")) {
                            vmat.shade_color = sc;
                        }
                        if let Some(oc) = read_vec4_json(vec_props.get("_OutlineColor")) {
                            vmat.outline_color = oc;
                        }
                        if let Some(rc) = read_vec4_json(vec_props.get("_RimColor")) {
                            vmat.rim_color = rc.truncate();
                        }
                    }
                    if let Some(tex_props) = prop.get("textureProperties").and_then(|t| t.as_object()) {
                        // Nota: em VRM 0.x `textureProperties._MainTex` referencia o índice em `textures`
                        if let Some(mt) = tex_props.get("_MainTex").and_then(|v| v.as_u64()) {
                            vmat.base_texture = Some(mt as usize);
                        }
                        if let Some(st) = tex_props.get("_ShadeTexture").and_then(|v| v.as_u64()) {
                            vmat.shade_texture = Some(st as usize);
                        }
                    }
                    if let Some(flt_props) = prop.get("floatProperties").and_then(|f| f.as_object()) {
                        if let Some(cutoff) = flt_props.get("_Cutoff").and_then(|v| v.as_f64()) {
                            vmat.alpha_cutoff = cutoff as f32;
                        }
                        if let Some(cull) = flt_props.get("_CullMode").and_then(|v| v.as_f64()) {
                            vmat.cull_mode = match cull as i32 {
                                0 => CullMode::None,
                                1 => CullMode::Front,
                                _ => CullMode::Back,
                            };
                        }
                        if let Some(ss) = flt_props.get("_ShadeShift").and_then(|v| v.as_f64()) {
                            vmat.shade_shift = ss as f32;
                        }
                        if let Some(st) = flt_props.get("_ShadeToony").and_then(|v| v.as_f64()) {
                            vmat.shade_toony = st as f32;
                        }
                        if let Some(rfp) = flt_props.get("_RimFresnelPower").and_then(|v| v.as_f64()) {
                            vmat.rim_fresnel_power = rfp as f32;
                        }
                        if let Some(rl) = flt_props.get("_RimLift").and_then(|v| v.as_f64()) {
                            vmat.rim_lift = rl as f32;
                        }
                        let outline_mode = flt_props
                            .get("_OutlineWidthMode")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as i32;
                        let outline_w = flt_props
                            .get("_OutlineWidth")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        if outline_mode > 0 && outline_w > 0.001 {
                            vmat.has_outline = true;
                            // Em VRM 0.x MToon, _OutlineWidth é dado em centímetros quando WorldCoordinates
                            vmat.outline_width = (outline_w * 0.0065).clamp(0.0004, 0.0035);
                        }
                    }
                }
            }

            materials.push(vmat);
        }
        if materials.is_empty() {
            materials.push(VrmMaterial::default());
        }

        // 5. Mapeia qual nó instancia cada mesh e qual skin usa
        let mut mesh_to_node_skin: HashMap<usize, (usize, Option<usize>)> = HashMap::new();
        for node in document.nodes() {
            if let Some(mesh) = node.mesh() {
                mesh_to_node_skin.insert(
                    mesh.index(),
                    (node.index(), node.skin().map(|s| s.index())),
                );
            }
        }

        // 6. Malhas (Meshes), Primitivas e Morph Targets
        let raw_meshes = raw_json.get("meshes").and_then(|m| m.as_array());
        let mut meshes = Vec::new();
        let mut bbox_min = Vec3::splat(f32::MAX);
        let mut bbox_max = Vec3::splat(f32::MIN);

        for mesh in document.meshes() {
            let mesh_idx = mesh.index();
            let (node_idx, skin_idx) = mesh_to_node_skin
                .get(&mesh_idx)
                .copied()
                .unwrap_or((0, None));
            let node_world = nodes.get(node_idx).map(|n| n.rest_world_matrix).unwrap_or(Mat4::IDENTITY);

            // Extrai nomes dos morph targets de extras.targetNames
            let target_names = extract_mesh_target_names(raw_meshes.and_then(|rm| rm.get(mesh_idx)));

            let mut base_positions = Vec::new();
            let mut base_normals = Vec::new();
            let mut uvs = Vec::new();
            let mut joints = Vec::new();
            let mut weights = Vec::new();
            let mut morph_targets = Vec::new();
            let mut primitives = Vec::new();
            let mut vertex_data_loaded = false;

            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|_| Some(blob_slice));

                if !vertex_data_loaded {
                    if let Some(pos_iter) = reader.read_positions() {
                        base_positions = pos_iter.map(Vec3::from_array).collect();
                    }
                    if base_positions.is_empty() {
                        continue;
                    }
                    let vcount = base_positions.len();

                    for &p in &base_positions {
                        let wp = if skin_idx.is_some() {
                            p
                        } else {
                            node_world.transform_point3(p)
                        };
                        bbox_min = bbox_min.min(wp);
                        bbox_max = bbox_max.max(wp);
                    }

                    base_normals = if let Some(norm_iter) = reader.read_normals() {
                        norm_iter.map(Vec3::from_array).collect()
                    } else {
                        vec![Vec3::Z; vcount]
                    };

                    uvs = if let Some(uv_iter) = reader.read_tex_coords(0) {
                        uv_iter.into_f32().map(Vec2::from_array).collect()
                    } else {
                        vec![Vec2::ZERO; vcount]
                    };

                    joints = if let Some(j_iter) = reader.read_joints(0) {
                        j_iter.into_u16().collect()
                    } else {
                        vec![[0, 0, 0, 0]; vcount]
                    };

                    weights = if let Some(w_iter) = reader.read_weights(0) {
                        w_iter
                            .into_f32()
                            .map(|w| {
                                let sum = w[0] + w[1] + w[2] + w[3];
                                if sum > 1e-5 {
                                    [w[0] / sum, w[1] / sum, w[2] / sum, w[3] / sum]
                                } else {
                                    [1.0, 0.0, 0.0, 0.0]
                                }
                            })
                            .collect()
                    } else {
                        vec![[1.0, 0.0, 0.0, 0.0]; vcount]
                    };

                    // Morph targets (armazenamento esparso apenas dos vértices que realmente se movem!)
                    for (t_idx, (pos_disp, norm_disp, _)) in reader.read_morph_targets().enumerate() {
                        let name = target_names
                            .get(t_idx)
                            .cloned()
                            .unwrap_or_else(|| format!("target_{t_idx}"));

                        let mut position_deltas = Vec::new();
                        if let Some(p_iter) = pos_disp {
                            for (vi, d) in p_iter.enumerate() {
                                let dv = Vec3::from_array(d);
                                if dv.length_squared() > 1e-10 {
                                    position_deltas.push((vi as u32, dv));
                                }
                            }
                        }

                        let mut normal_deltas = Vec::new();
                        if let Some(n_iter) = norm_disp {
                            for (vi, d) in n_iter.enumerate() {
                                let dn = Vec3::from_array(d);
                                if dn.length_squared() > 1e-8 {
                                    normal_deltas.push((vi as u32, dn));
                                }
                            }
                        }

                        morph_targets.push(MorphTargetDelta {
                            name,
                            position_deltas,
                            normal_deltas,
                        });
                    }

                    vertex_data_loaded = true;
                }

                let indices: Vec<u32> = if let Some(idx_iter) = reader.read_indices() {
                    idx_iter.into_u32().collect()
                } else {
                    (0..base_positions.len() as u32).collect()
                };

                let material_index = prim.material().index().unwrap_or(0);
                primitives.push(VrmPrimitive {
                    indices,
                    material_index,
                });
            }

            if !base_positions.is_empty() && !primitives.is_empty() {
                meshes.push(VrmMesh {
                    name: mesh.name().unwrap_or("").to_string(),
                    mesh_index: mesh_idx,
                    node_index: node_idx,
                    skin_index: skin_idx,
                    visible: true,
                    base_positions,
                    base_normals,
                    uvs,
                    joints,
                    weights,
                    morph_targets,
                    primitives,
                });
            }
        }

        // 7. Humanoid Bones (VRM 0.x + VRM 1.0 + fallback por nome)
        let mut humanoid_bones = HashMap::new();
        if let Some(hb_arr) = vrm_ext
            .and_then(|v| v.get("humanoid"))
            .and_then(|h| h.get("humanBones"))
            .and_then(|b| b.as_array())
        {
            for item in hb_arr {
                if let (Some(bone_name), Some(node_idx)) = (
                    item.get("bone").and_then(|b| b.as_str()),
                    item.get("node").and_then(|n| n.as_u64()),
                ) {
                    humanoid_bones.insert(bone_name.to_string(), node_idx as usize);
                }
            }
        } else if let Some(hb_obj) = vrm1_ext
            .and_then(|v| v.get("humanoid"))
            .and_then(|h| h.get("humanBones"))
            .and_then(|b| b.as_object())
        {
            for (bone_name, val) in hb_obj {
                if let Some(node_idx) = val.get("node").and_then(|n| n.as_u64()) {
                    humanoid_bones.insert(bone_name.clone(), node_idx as usize);
                }
            }
        }

        // 8. BlendShape Groups (VRM 0.x)
        let mut blend_shape_groups = Vec::new();
        if let Some(groups_arr) = vrm_ext
            .and_then(|v| v.get("blendShapeMaster"))
            .and_then(|bsm| bsm.get("blendShapeGroups"))
            .and_then(|g| g.as_array())
        {
            for g in groups_arr {
                let name = g.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let preset_name = g
                    .get("presetName")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let is_binary = g.get("isBinary").and_then(|b| b.as_bool()).unwrap_or(false);
                let mut binds = Vec::new();
                if let Some(binds_arr) = g.get("binds").and_then(|b| b.as_array()) {
                    for b in binds_arr {
                        if let (Some(m_idx), Some(t_idx), Some(w)) = (
                            b.get("mesh").and_then(|v| v.as_u64()),
                            b.get("index").and_then(|v| v.as_u64()),
                            b.get("weight").and_then(|v| v.as_f64()),
                        ) {
                            binds.push(BlendShapeBind {
                                mesh_index: m_idx as usize,
                                target_index: t_idx as usize,
                                weight: (w as f32 / 100.0).clamp(0.0, 1.0),
                            });
                        }
                    }
                }
                blend_shape_groups.push(BlendShapeGroup {
                    name,
                    preset_name,
                    is_binary,
                    binds,
                });
            }
        }

        // 9. SpringBone Secondary Animation (VRM 0.x)
        let mut spring_colliders = Vec::new();
        let mut collider_group_ranges: Vec<Vec<usize>> = Vec::new();
        if let Some(sec) = vrm_ext.and_then(|v| v.get("secondaryAnimation")) {
            if let Some(cg_arr) = sec.get("colliderGroups").and_then(|c| c.as_array()) {
                for cg in cg_arr {
                    let node_idx = cg.get("node").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
                    let mut group_indices = Vec::new();
                    if let Some(cols) = cg.get("colliders").and_then(|c| c.as_array()) {
                        for col in cols {
                            let off = col.get("offset");
                            let ox = off.and_then(|o| o.get("x")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            let oy = off.and_then(|o| o.get("y")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            let oz = off.and_then(|o| o.get("z")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            let radius = col.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.02) as f32;
                            let idx = spring_colliders.len();
                            spring_colliders.push(SpringCollider {
                                node_index: node_idx,
                                offset: Vec3::new(ox, oy, oz),
                                radius,
                            });
                            group_indices.push(idx);
                        }
                    }
                    collider_group_ranges.push(group_indices);
                }
            }
        }

        let mut spring_bone_groups = Vec::new();
        if let Some(sec) = vrm_ext.and_then(|v| v.get("secondaryAnimation")) {
            if let Some(bg_arr) = sec.get("boneGroups").and_then(|b| b.as_array()) {
                for bg in bg_arr {
                    let comment = bg.get("comment").and_then(|s| s.as_str()).unwrap_or("").to_string();
                    let stiffness = bg.get("stiffiness").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let gravity_power = bg.get("gravityPower").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let drag_force = bg.get("dragForce").and_then(|v| v.as_f64()).unwrap_or(0.4) as f32;
                    let hit_radius = bg.get("hitRadius").and_then(|v| v.as_f64()).unwrap_or(0.02) as f32;
                    let gdir = bg.get("gravityDir");
                    let gx = gdir.and_then(|o| o.get("x")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let gy = gdir.and_then(|o| o.get("y")).and_then(|v| v.as_f64()).unwrap_or(-1.0) as f32;
                    let gz = gdir.and_then(|o| o.get("z")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

                    let root_bones: Vec<usize> = bg
                        .get("bones")
                        .and_then(|b| b.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|u| u as usize))
                                .collect()
                        })
                        .unwrap_or_default();

                    let mut collider_indices = Vec::new();
                    if let Some(cg_refs) = bg.get("colliderGroups").and_then(|c| c.as_array()) {
                        for r in cg_refs {
                            if let Some(cg_idx) = r.as_u64() {
                                if let Some(list) = collider_group_ranges.get(cg_idx as usize) {
                                    collider_indices.extend_from_slice(list);
                                }
                            }
                        }
                    }

                    if !root_bones.is_empty() {
                        spring_bone_groups.push(SpringBoneGroup {
                            comment,
                            stiffness,
                            gravity_power,
                            gravity_dir: Vec3::new(gx, gy, gz).normalize_or_zero(),
                            drag_force,
                            hit_radius,
                            root_bones,
                            collider_indices,
                        });
                    }
                }
            }
        }

        // 10. Auto-escala e centralização idênticos ao Usami.jsx (altura alvo = 2.25m)
        let size = (bbox_max - bbox_min).max(Vec3::splat(0.001));
        let center = (bbox_min + bbox_max) * 0.5;
        let auto_scale = if bbox_max.y > bbox_min.y {
            2.25 / size.y.max(0.001)
        } else {
            1.0
        };
        let auto_offset = if bbox_max.y > bbox_min.y {
            Vec3::new(-center.x * auto_scale, -bbox_min.y * auto_scale, -center.z * auto_scale)
        } else {
            Vec3::ZERO
        };

        let title = vrm_ext
            .and_then(|v| v.get("meta"))
            .and_then(|m| m.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("Avatar 3D")
            .to_string();

        Ok(Self {
            title,
            is_vrm0,
            nodes,
            root_nodes,
            topological_order,
            skins,
            meshes,
            materials,
            textures,
            humanoid_bones,
            blend_shape_groups,
            spring_colliders,
            spring_bone_groups,
            auto_scale,
            auto_offset,
        })
    }
}

impl VrmaClip {
    /// Carrega um arquivo `.vrma` (VRMC_vrm_animation 1.0) e pré-calcula as rotações normalizadas T-pose.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let bytes = fs::read(path)
            .with_context(|| format!("Falha ao ler arquivo VRMA: {}", path.display()))?;
        Self::load_from_slice(&bytes, name)
    }

    pub fn load_from_slice(bytes: &[u8], name: String) -> Result<Self> {
        let raw_json = parse_raw_gltf_json(bytes)?;
        let gltf = gltf::Gltf::from_slice(bytes).context("Falha ao importar binário VRMA")?;
        let document = gltf.document;
        let blob_slice: &[u8] = gltf.blob.as_deref().unwrap_or(&[]);

        let vrma_ext = raw_json
            .get("extensions")
            .and_then(|e| e.get("VRMC_vrm_animation"))
            .context("Arquivo não possui extensão VRMC_vrm_animation")?;

        // Mapa node_index -> bone_name
        let mut node_to_bone: HashMap<usize, String> = HashMap::new();
        let mut bone_to_node: HashMap<String, usize> = HashMap::new();
        if let Some(hb_obj) = vrma_ext
            .get("humanoid")
            .and_then(|h| h.get("humanBones"))
            .and_then(|b| b.as_object())
        {
            for (bone_name, val) in hb_obj {
                if let Some(node_idx) = val.get("node").and_then(|n| n.as_u64()) {
                    node_to_bone.insert(node_idx as usize, bone_name.clone());
                    bone_to_node.insert(bone_name.clone(), node_idx as usize);
                }
            }
        }

        // Calcula rotações e posições de repouso (rest pose) dos nós do VRMA
        let node_count = document.nodes().len();
        let mut parents = vec![None; node_count];
        let mut local_trs = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); node_count];
        for node in document.nodes() {
            let idx = node.index();
            let (t, r, s) = node.transform().decomposed();
            local_trs[idx] = (
                Vec3::from_array(t),
                Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
                Vec3::from_array(s),
            );
            for child in node.children() {
                parents[child.index()] = Some(idx);
            }
        }

        let mut world_rot = vec![Quat::IDENTITY; node_count];
        let mut world_mat = vec![Mat4::IDENTITY; node_count];
        for idx in 0..node_count {
            let mut chain = Vec::new();
            let mut cur = Some(idx);
            while let Some(c) = cur {
                chain.push(c);
                cur = parents[c];
            }
            chain.reverse();
            let mut m = Mat4::IDENTITY;
            let mut q = Quat::IDENTITY;
            for c in chain {
                let (t, r, s) = local_trs[c];
                m = m * Mat4::from_scale_rotation_translation(s, r, t);
                q = (q * r).normalize();
            }
            world_rot[idx] = q;
            world_mat[idx] = m;
        }

        let hips_node = bone_to_node.get("hips").copied();
        let rest_hips_pos = hips_node
            .map(|h| world_mat[h].transform_point3(Vec3::ZERO))
            .unwrap_or(Vec3::Y);
        let hips_parent_mat = hips_node
            .and_then(|h| parents[h])
            .map(|p| world_mat[p])
            .unwrap_or(Mat4::IDENTITY);
        let hips_parent_rot = hips_node
            .and_then(|h| parents[h])
            .map(|p| world_rot[p])
            .unwrap_or(Quat::IDENTITY);

        let mut duration = 0.0f32;
        let mut rotation_tracks = HashMap::new();
        let mut hips_translation = None;

        if let Some(anim) = document.animations().next() {
            for channel in anim.channels() {
                let target_node = channel.target().node().index();
                let Some(bone_name) = node_to_bone.get(&target_node) else {
                    continue;
                };

                // Encontra o osso humanoide pai presente no arquivo VRMA
                let mut parent_bone = humanoid_parent_bone(bone_name);
                while let Some(pb) = parent_bone {
                    if bone_to_node.contains_key(pb) {
                        break;
                    }
                    parent_bone = humanoid_parent_bone(pb);
                }
                let parent_world_q = parent_bone
                    .and_then(|pb| bone_to_node.get(pb))
                    .map(|&p_idx| world_rot[p_idx])
                    .unwrap_or(hips_parent_rot);
                let bone_world_q_inv = world_rot[target_node].inverse();

                let reader = channel.reader(|_| Some(blob_slice));
                let Some(inputs) = reader.read_inputs() else {
                    continue;
                };
                let times: Vec<f32> = inputs.collect();
                if let Some(&last_t) = times.last() {
                    duration = duration.max(last_t);
                }

                let Some(outputs) = reader.read_outputs() else {
                    continue;
                };

                match outputs {
                    gltf::animation::util::ReadOutputs::Rotations(rot_iter) => {
                        let rotations: Vec<Quat> = rot_iter
                            .into_f32()
                            .map(|q| {
                                let raw_q = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
                                // Fórmula oficial @pixiv/three-vrm-animation:
                                // q_norm = Q_world(parent) * q_raw * Q_world(bone)^-1
                                (parent_world_q * raw_q * bone_world_q_inv).normalize()
                            })
                            .collect();

                        rotation_tracks.insert(
                            bone_name.clone(),
                            VrmaRotationTrack {
                                bone_name: bone_name.clone(),
                                times,
                                rotations,
                            },
                        );
                    }
                    gltf::animation::util::ReadOutputs::Translations(trans_iter) => {
                        if bone_name == "hips" {
                            let deltas: Vec<Vec3> = trans_iter
                                .map(|v| {
                                    let world_v =
                                        hips_parent_mat.transform_point3(Vec3::from_array(v));
                                    world_v - rest_hips_pos
                                })
                                .collect();
                            hips_translation = Some(VrmaTranslationTrack {
                                times,
                                deltas,
                                rest_hips_y: rest_hips_pos.y.max(0.1),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(Self {
            name,
            duration,
            rotation_tracks,
            hips_translation,
        })
    }
}

fn read_vec4_json(val: Option<&Value>) -> Option<Vec4> {
    let arr = val?.as_array()?;
    if arr.len() >= 4 {
        Some(Vec4::new(
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
            arr[3].as_f64()? as f32,
        ))
    } else if arr.len() == 3 {
        Some(Vec4::new(
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
            1.0,
        ))
    } else {
        None
    }
}

fn extract_mesh_target_names(raw_mesh: Option<&Value>) -> Vec<String> {
    let Some(m) = raw_mesh else {
        return Vec::new();
    };
    if let Some(arr) = m
        .get("extras")
        .and_then(|e| e.get("targetNames"))
        .and_then(|t| t.as_array())
    {
        return arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    if let Some(arr) = m
        .get("primitives")
        .and_then(|p| p.as_array())
        .and_then(|p| p.first())
        .and_then(|p0| p0.get("extras"))
        .and_then(|e| e.get("targetNames"))
        .and_then(|t| t.as_array())
    {
        return arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    Vec::new()
}
