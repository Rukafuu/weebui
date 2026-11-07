use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use gpui::RenderImage;
use image::{Frame, RgbaImage};
use smallvec::smallvec;

use crate::loader::{AlphaMode, CullMode, VrmModelData};
use crate::skeleton::SkeletonRuntime;

const MSAA_SAMPLES: u32 = 4;
const MAX_JOINTS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    sky_color: [f32; 4],
    ground_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    shade_color: [f32; 4],
    outline_color: [f32; 4],
    emissive_rim: [f32; 4], // xyz = emissive, w = rim_fresnel_power
    rim_color_lift: [f32; 4], // xyz = rim_color, w = rim_lift
    toon_params: [f32; 4], // x = shade_shift, y = shade_toony, z = alpha_cutoff, w = outline_width
    flags: [u32; 4], // x = alpha_mode (0=Opaque, 1=Mask, 2=Blend), y = is_outline_pass, z = is_mtoon, w = pad
}

struct GpuPrimitive {
    index_buffer: wgpu::Buffer,
    index_count: u32,
    material_index: usize,
    render_queue: i32,
    alpha_mode: AlphaMode,
    cull_mode: CullMode,
    has_outline: bool,
}

struct GpuMesh {
    mesh_index_in_model: usize,
    skin_index: Option<usize>,
    vertex_buffer: wgpu::Buffer,
    cpu_vertices: Vec<GpuVertex>,
    skin_bind_group: wgpu::BindGroup,
    skin_buffer: wgpu::Buffer,
    skin_matrices_scratch: Vec<[f32; 16]>,
    primitives: Vec<GpuPrimitive>,
}

struct GpuMaterialBind {
    surface_bind_group: wgpu::BindGroup,
    outline_bind_group: Option<wgpu::BindGroup>,
}

struct RenderTargets {
    width: u32,
    height: u32,
    msaa_view: wgpu::TextureView,
    resolve_texture: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

/// Câmera orbital/stage configurável (padrão alinhado com Usami.jsx).
#[derive(Debug, Clone)]
pub struct CameraState {
    pub fov_y_deg: f32,
    pub position: Vec3,
    pub target: Vec3,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        // Enquadramento padrão de busto/corpo da Usami no Stage Card
        Self {
            fov_y_deg: 28.0,
            position: Vec3::new(0.0, 1.38, 3.35),
            target: Vec3::new(0.0, 1.28, 0.0),
            near: 0.05,
            far: 100.0,
        }
    }
}

/// Motor de renderização 3D WGPU Offscreen com suporte a MToon, Outline, MSAA 4x e BlendShapes.
pub struct VrmRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    camera: CameraState,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    pipeline_opaque_back: wgpu::RenderPipeline,
    pipeline_opaque_none: wgpu::RenderPipeline,
    pipeline_outline_front: wgpu::RenderPipeline,
    pipeline_blend_back: wgpu::RenderPipeline,
    pipeline_blend_none: wgpu::RenderPipeline,

    gpu_meshes: Vec<GpuMesh>,
    gpu_materials: Vec<GpuMaterialBind>,
    draw_order: Vec<(usize, usize)>, // (gpu_mesh_idx, prim_idx) ordenados por render_queue
    targets: RenderTargets,
}

impl VrmRenderer {
    pub fn new(model: &VrmModelData, width: u32, height: u32) -> Result<Self> {
        let width = width.clamp(64, 2048);
        let height = height.clamp(64, 2048);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("Nenhum adaptador GPU compatível com WGPU (DirectX 12 / Vulkan) encontrado")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpui-vrm-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("Falha ao inicializar dispositivo lógico WGPU")?;

        // 1. Layouts de Bind Group
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let skin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 2. Shader WGSL MToon / PBR
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mtoon_shader"),
            source: wgpu::ShaderSource::Wgsl(MTOON_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vrm_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl, &skin_bgl, &material_bgl],
            push_constant_ranges: &[],
        });

        let pipeline_opaque_back =
            create_render_pipeline(&device, &pipeline_layout, &shader, Some(wgpu::Face::Back), false);
        let pipeline_opaque_none =
            create_render_pipeline(&device, &pipeline_layout, &shader, None, false);
        let pipeline_outline_front =
            create_render_pipeline(&device, &pipeline_layout, &shader, Some(wgpu::Face::Front), false);
        let pipeline_blend_back =
            create_render_pipeline(&device, &pipeline_layout, &shader, Some(wgpu::Face::Back), true);
        let pipeline_blend_none =
            create_render_pipeline(&device, &pipeline_layout, &shader, None, true);

        // 3. Camera Uniform
        let camera = CameraState::default();
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 4. Upload de texturas do modelo (+ textura branca fallback 1x1)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vrm_linear_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let white_tex_view = create_rgba_texture_view(&device, &queue, 1, 1, &[255, 255, 255, 255]);
        let mut tex_views = Vec::with_capacity(model.textures.len());
        for tex in &model.textures {
            let view = create_rgba_texture_view(&device, &queue, tex.width, tex.height, &tex.rgba);
            tex_views.push(view);
        }

        // 5. Bind Groups de Materiais (superfície + outline opcional)
        let mut gpu_materials = Vec::with_capacity(model.materials.len());
        for mat in &model.materials {
            let base_view = mat
                .base_texture
                .and_then(|idx| tex_views.get(idx))
                .unwrap_or(&white_tex_view);
            let shade_view = mat
                .shade_texture
                .or(mat.base_texture)
                .and_then(|idx| tex_views.get(idx))
                .unwrap_or(base_view);

            let surface_uni = build_material_uniform(mat, false);
            let surface_buf = create_init_buffer(
                &device,
                bytemuck::bytes_of(&surface_uni),
                wgpu::BufferUsages::UNIFORM,
            );
            let surface_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&mat.name),
                layout: &material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: surface_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(base_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(shade_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            let outline_bind_group = if mat.has_outline && mat.alpha_mode != AlphaMode::Blend {
                let outline_uni = build_material_uniform(mat, true);
                let outline_buf = create_init_buffer(
                    &device,
                    bytemuck::bytes_of(&outline_uni),
                    wgpu::BufferUsages::UNIFORM,
                );
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("outline_bg"),
                    layout: &material_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: outline_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(base_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(shade_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                }))
            } else {
                None
            };

            gpu_materials.push(GpuMaterialBind {
                surface_bind_group,
                outline_bind_group,
            });
        }

        // 6. Malhas e Primitivas na GPU
        let mut gpu_meshes = Vec::with_capacity(model.meshes.len());
        let mut draw_order = Vec::new();

        for (m_idx, mesh) in model.meshes.iter().enumerate() {
            let vcount = mesh.base_positions.len();
            let mut cpu_vertices = Vec::with_capacity(vcount);
            for i in 0..vcount {
                let j = mesh.joints[i];
                cpu_vertices.push(GpuVertex {
                    position: mesh.base_positions[i].to_array(),
                    normal: mesh.base_normals[i].to_array(),
                    uv: mesh.uvs[i].to_array(),
                    joints: [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32],
                    weights: mesh.weights[i],
                });
            }

            let vertex_buffer = create_init_buffer(
                &device,
                bytemuck::cast_slice(&cpu_vertices),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            );

            let skin_matrices_scratch = vec![Mat4::IDENTITY.to_cols_array(); MAX_JOINTS];
            let skin_buffer = create_init_buffer(
                &device,
                bytemuck::cast_slice(&skin_matrices_scratch),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let skin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skin_bg"),
                layout: &skin_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: skin_buffer.as_entire_binding(),
                }],
            });

            let mut primitives = Vec::with_capacity(mesh.primitives.len());
            for (p_idx, prim) in mesh.primitives.iter().enumerate() {
                let index_buffer = create_init_buffer(
                    &device,
                    bytemuck::cast_slice(&prim.indices),
                    wgpu::BufferUsages::INDEX,
                );
                let mat_idx = prim.material_index.min(model.materials.len() - 1);
                let mat = &model.materials[mat_idx];
                primitives.push(GpuPrimitive {
                    index_buffer,
                    index_count: prim.indices.len() as u32,
                    material_index: mat_idx,
                    render_queue: mat.render_queue,
                    alpha_mode: mat.alpha_mode,
                    cull_mode: mat.cull_mode,
                    has_outline: gpu_materials[mat_idx].outline_bind_group.is_some(),
                });
                draw_order.push((m_idx, p_idx));
            }

            gpu_meshes.push(GpuMesh {
                mesh_index_in_model: m_idx,
                skin_index: mesh.skin_index,
                vertex_buffer,
                cpu_vertices,
                skin_bind_group,
                skin_buffer,
                skin_matrices_scratch,
                primitives,
            });
        }

        // Ordena primitivas: Opaque -> Mask -> Blend, respeitando render_queue do MToon
        draw_order.sort_by_key(|&(m_idx, p_idx)| {
            let p = &gpu_meshes[m_idx].primitives[p_idx];
            let cat = match p.alpha_mode {
                AlphaMode::Opaque => 0,
                AlphaMode::Mask => 1,
                AlphaMode::Blend => 2,
            };
            (cat, p.render_queue)
        });

        let targets = create_render_targets(&device, width, height);

        Ok(Self {
            device,
            queue,
            camera,
            camera_buffer,
            camera_bind_group,
            pipeline_opaque_back,
            pipeline_opaque_none,
            pipeline_outline_front,
            pipeline_blend_back,
            pipeline_blend_none,
            gpu_meshes,
            gpu_materials,
            draw_order,
            targets,
        })
    }

    pub fn camera_mut(&mut self) -> &mut CameraState {
        &mut self.camera
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.clamp(64, 2048);
        let height = height.clamp(64, 2048);
        if self.targets.width != width || self.targets.height != height {
            self.targets = create_render_targets(&self.device, width, height);
        }
    }

    /// Atualiza os vértices deformados por BlendShapes/MorphTargets na GPU apenas para malhas com alvos ativos.
    pub fn update_morph_targets(&mut self, model: &VrmModelData, mesh_weights: &[Vec<f32>]) {
        for gpu_mesh in &mut self.gpu_meshes {
            let m_idx = gpu_mesh.mesh_index_in_model;
            let mesh = &model.meshes[m_idx];
            if mesh.morph_targets.is_empty() {
                continue;
            }
            let Some(weights) = mesh_weights.get(m_idx) else {
                continue;
            };

            // Restaura posições e normais base
            for (i, v) in gpu_mesh.cpu_vertices.iter_mut().enumerate() {
                v.position = mesh.base_positions[i].to_array();
                v.normal = mesh.base_normals[i].to_array();
            }

            // Acumula apenas os morph targets com peso não-zero usando a lista esparsa
            for (t_idx, &w) in weights.iter().enumerate() {
                if w <= 0.001 {
                    continue;
                }
                let mt = &mesh.morph_targets[t_idx];
                for &(vi, dp) in &mt.position_deltas {
                    let pos = &mut gpu_mesh.cpu_vertices[vi as usize].position;
                    pos[0] += dp.x * w;
                    pos[1] += dp.y * w;
                    pos[2] += dp.z * w;
                }
                for &(vi, dn) in &mt.normal_deltas {
                    let norm = &mut gpu_mesh.cpu_vertices[vi as usize].normal;
                    norm[0] += dn.x * w;
                    norm[1] += dn.y * w;
                    norm[2] += dn.z * w;
                }
            }

            self.queue.write_buffer(
                &gpu_mesh.vertex_buffer,
                0,
                bytemuck::cast_slice(&gpu_mesh.cpu_vertices),
            );
        }
    }

    /// Renderiza a cena completa em BGRA8 com fundo transparente e retorna um `Arc<RenderImage>` pronto para o GPUI.
    pub fn render_frame(
        &mut self,
        model: &VrmModelData,
        skeleton: &SkeletonRuntime,
    ) -> Result<Arc<RenderImage>> {
        let width = self.targets.width;
        let height = self.targets.height;
        let aspect = (width as f32) / (height.max(1) as f32);

        // 1. Atualiza Camera Uniform (iluminação fiel ao Usami.jsx: Hemisphere + Directional)
        let proj = Mat4::perspective_rh(
            self.camera.fov_y_deg.to_radians(),
            aspect,
            self.camera.near,
            self.camera.far,
        );
        let view = Mat4::look_at_rh(self.camera.position, self.camera.target, Vec3::Y);
        let view_proj = proj * view;

        let cam_uni = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.extend(1.0).to_array(),
            light_dir: Vec3::new(2.0, 4.0, 4.0).normalize().extend(0.0).to_array(),
            light_color: [1.0, 0.922, 0.847, 1.0], // 0xffebd8
            sky_color: [1.0, 1.0, 1.0, 0.55],
            ground_color: [0.463, 0.392, 0.522, 0.35], // 0x766485
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uni));

        // 2. Atualiza matrizes de Skinning para cada malha
        for gpu_mesh in &mut self.gpu_meshes {
            skeleton.compute_skin_matrices(
                model,
                gpu_mesh.skin_index,
                &mut gpu_mesh.skin_matrices_scratch,
            );
            let joint_count = gpu_mesh
                .skin_index
                .and_then(|s| model.skins.get(s))
                .map(|sk| sk.joints.len().min(MAX_JOINTS))
                .unwrap_or(1)
                .max(1);
            self.queue.write_buffer(
                &gpu_mesh.skin_buffer,
                0,
                bytemuck::cast_slice(&gpu_mesh.skin_matrices_scratch[..joint_count]),
            );
        }

        // 3. Codifica RenderPass MSAA 4x + Resolve para textura single-sample
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vrm_frame_encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrm_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.msaa_view,
                    resolve_target: Some(&self.targets.resolve_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_bind_group(0, &self.camera_bind_group, &[]);

            for &(m_idx, p_idx) in &self.draw_order {
                if !model.meshes[m_idx].visible {
                    continue;
                }
                let gpu_mesh = &self.gpu_meshes[m_idx];
                let prim = &gpu_mesh.primitives[p_idx];
                let gpu_mat = &self.gpu_materials[prim.material_index];

                rpass.set_bind_group(1, &gpu_mesh.skin_bind_group, &[]);
                rpass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                // Passada principal da superfície
                let pipeline = match (prim.alpha_mode, prim.cull_mode) {
                    (AlphaMode::Blend, CullMode::Back) => &self.pipeline_blend_back,
                    (AlphaMode::Blend, _) => &self.pipeline_blend_none,
                    (_, CullMode::Back) => &self.pipeline_opaque_back,
                    (_, _) => &self.pipeline_opaque_none,
                };
                rpass.set_pipeline(pipeline);
                rpass.set_bind_group(2, &gpu_mat.surface_bind_group, &[]);
                rpass.draw_indexed(0..prim.index_count, 0, 0..1);

                // Passada de contorno MToon (Backface Extrusion)
                if prim.has_outline {
                    if let Some(outline_bg) = &gpu_mat.outline_bind_group {
                        rpass.set_pipeline(&self.pipeline_outline_front);
                        rpass.set_bind_group(2, outline_bg, &[]);
                        rpass.draw_indexed(0..prim.index_count, 0, 0..1);
                    }
                }
            }
        }

        // 4. Copia a textura resolvida (BGRA8) para o staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.resolve_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.targets.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.targets.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // 5. Mapeia e extrai os bytes BGRA8 diretamente para `gpui::RenderImage`
        let slice = self.targets.staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        let unpadded_row = (width * 4) as usize;
        let padded_row = self.targets.padded_bytes_per_row as usize;
        let total_bytes = unpadded_row * (height as usize);
        let mut raw_bgra = vec![0u8; total_bytes];

        {
            let mapped = slice.get_mapped_range();
            if padded_row == unpadded_row {
                raw_bgra.copy_from_slice(&mapped[..total_bytes]);
            } else {
                for y in 0..(height as usize) {
                    let src_start = y * padded_row;
                    let dst_start = y * unpadded_row;
                    raw_bgra[dst_start..dst_start + unpadded_row]
                        .copy_from_slice(&mapped[src_start..src_start + unpadded_row]);
                }
            }
        }
        self.targets.staging_buffer.unmap();

        let img = RgbaImage::from_raw(width, height, raw_bgra)
            .context("Dimensões inválidas ao construir frame BGRA")?;
        let frame = Frame::new(img);
        Ok(Arc::new(RenderImage::new(smallvec![frame])))
    }
}

fn create_init_buffer(
    device: &wgpu::Device,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let aligned_size = ((contents.len() as u64 + 3) & !3).max(16);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: aligned_size,
        usage,
        mapped_at_creation: true,
    });
    buffer.slice(..contents.len() as u64).get_mapped_range_mut().copy_from_slice(contents);
    buffer.unmap();
    buffer
}

fn create_rgba_texture_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_tex"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width.max(1)),
            rows_per_image: Some(height.max(1)),
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_render_targets(device: &wgpu::Device, width: u32, height: u32) -> RenderTargets {
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let msaa_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_msaa_color"),
        size: extent,
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_resolve_color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm_msaa_depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded = width * 4;
    let padded_bytes_per_row = (unpadded + align - 1) & !(align - 1);
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vrm_staging_buf"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    RenderTargets {
        width,
        height,
        msaa_view: msaa_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        resolve_view: resolve_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        resolve_texture,
        depth_view: depth_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        staging_buffer,
        padded_bytes_per_row,
    }
}

fn build_material_uniform(mat: &crate::loader::VrmMaterial, is_outline: bool) -> MaterialUniform {
    let alpha_mode_u = match mat.alpha_mode {
        AlphaMode::Opaque => 0u32,
        AlphaMode::Mask => 1u32,
        AlphaMode::Blend => 2u32,
    };
    MaterialUniform {
        base_color: mat.base_color.to_array(),
        shade_color: mat.shade_color.to_array(),
        outline_color: mat.outline_color.to_array(),
        emissive_rim: Vec4::from((mat.emissive_color, mat.rim_fresnel_power)).to_array(),
        rim_color_lift: Vec4::from((mat.rim_color, mat.rim_lift)).to_array(),
        toon_params: [
            mat.shade_shift,
            mat.shade_toony,
            mat.alpha_cutoff,
            mat.outline_width,
        ],
        flags: [
            alpha_mode_u,
            u32::from(is_outline),
            u32::from(mat.is_mtoon),
            0,
        ],
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    cull_mode: Option<wgpu::Face>,
    alpha_blend: bool,
) -> wgpu::RenderPipeline {
    let blend_state = if alpha_blend {
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        })
    } else {
        Some(wgpu::BlendState::REPLACE)
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vrm_mtoon_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Uint32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 48,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: blend_state,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: !alpha_blend,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

const MTOON_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
};

struct MaterialUniform {
    base_color: vec4<f32>,
    shade_color: vec4<f32>,
    outline_color: vec4<f32>,
    emissive_rim: vec4<f32>,
    rim_color_lift: vec4<f32>,
    toon_params: vec4<f32>,
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var base_tex: texture_2d<f32>;
@group(2) @binding(2) var shade_tex: texture_2d<f32>;
@group(2) @binding(3) var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let skin_mat =
        in.weights.x * joint_matrices[in.joints.x] +
        in.weights.y * joint_matrices[in.joints.y] +
        in.weights.z * joint_matrices[in.joints.z] +
        in.weights.w * joint_matrices[in.joints.w];

    var world_pos4 = skin_mat * vec4<f32>(in.position, 1.0);
    let world_norm = normalize((skin_mat * vec4<f32>(in.normal, 0.0)).xyz);

    if (material.flags.y == 1u) {
        let outline_w = material.toon_params.w;
        world_pos4 = vec4<f32>(world_pos4.xyz + world_norm * outline_w, 1.0);
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_normal = world_norm;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let main_sample = textureSample(base_tex, tex_sampler, in.uv);
    let shade_sample = textureSample(shade_tex, tex_sampler, in.uv);

    let lit_col = main_sample.rgb * material.base_color.rgb;
    let shd_col = shade_sample.rgb * material.shade_color.rgb;
    let alpha = main_sample.a * material.base_color.a;

    // Alpha Mask (Cutout)
    if (material.flags.x == 1u && alpha < material.toon_params.z) {
        discard;
    }

    // Passada de Outline MToon
    if (material.flags.y == 1u) {
        let out_rgb = mix(material.outline_color.rgb, lit_col * material.outline_color.rgb, 0.55);
        return vec4<f32>(out_rgb, alpha);
    }

    var n = normalize(in.world_normal);
    if (!is_front) {
        n = -n;
    }

    let l = normalize(camera.light_dir.xyz);
    let v = normalize(camera.camera_pos.xyz - in.world_pos);

    let ndotl = dot(n, l);
    let shift = material.toon_params.x;
    let toony = clamp(material.toon_params.y, 0.05, 0.98);
    let feather = max(1.0 - toony, 0.08);
    let shade_factor = smoothstep(shift - feather * 0.5, shift + feather * 0.5, ndotl);

    // Iluminação Anime MToon suave (preserva fidelidade da arte original da Shiro)
    var color = mix(shd_col, lit_col, 0.35 + 0.65 * shade_factor);

    // Luz hemisférica suave
    let hemi_t = n.y * 0.5 + 0.5;
    let hemi_col = mix(camera.ground_color.rgb, camera.sky_color.rgb, hemi_t);
    color = color * mix(vec3<f32>(0.92), hemi_col * camera.light_color.rgb, 0.22);

    // Rim Light Fresnel anime sutil
    let rim_power = max(material.emissive_rim.w, 1.5);
    let rim_lift = material.rim_color_lift.w;
    let fresnel = pow(clamp(1.0 - max(dot(n, v), 0.0) + rim_lift, 0.0, 1.0), rim_power);
    color = color + material.rim_color_lift.rgb * fresnel * 0.18;

    // Emissive
    color = color + material.emissive_rim.rgb;

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(alpha, 0.0, 1.0));
}
"#;
