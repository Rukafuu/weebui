use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use glam::{Vec2, Vec3};
use gpui::prelude::*;
use gpui::*;
use parking_lot::Mutex;

use crate::expressions::{EmotionConfig, ExpressionDriver};
use crate::loader::{VrmModelData, VrmaClip};
use crate::physics::SpringBoneSimulator;
use crate::renderer::VrmRenderer;
use crate::skeleton::SkeletonRuntime;

/// Comandos enfileirados pelo `VrmController` para execução thread-safe no loop do `VrmView`.
#[derive(Debug, Clone)]
enum VrmCommand {
    RegisterEmotion {
        name: String,
        config: EmotionConfig,
    },
    SetEmotion {
        tag: String,
        ttl_sec: f32,
        trigger_vrma: bool,
    },
    SetExpression {
        name: String,
        weight: f32,
    },
    ClearExpressions,
    PlayVrma {
        name_or_path: String,
        looping: bool,
        crossfade: f32,
    },
    StopVrma {
        crossfade: f32,
    },
    SpeakText {
        text: String,
        wav_bytes: Option<Vec<u8>>,
    },
    SetSpeakingVolume {
        speaking: bool,
        volume: f32,
    },
    BeginSpeechStream,
    EndSpeechStream,
    LookAtNormalized {
        x: f32,
        y: f32,
    },
    SetCamera {
        position: Vec3,
        target: Vec3,
        fov_y_deg: f32,
    },
    OrbitCamera {
        yaw_delta: f32,
        pitch_delta: f32,
        zoom_delta: f32,
    },
    SetSpringBoneEnabled(bool),
    SetArmClearanceDeg(f32),
    ResetCamera,
}

#[derive(Default)]
struct ControllerSharedState {
    commands: Vec<VrmCommand>,
    active_emotion: String,
    active_vrma: String,
    is_speaking: bool,
}

/// Controlador reativo e thread-safe (`Clone + Send + Sync`) para comandar o avatar 3D VRM a partir
/// de qualquer tarefa assíncrona (chat HTTP, stream de áudio `rodio`, eventos de mouse/UI).
#[derive(Clone)]
pub struct VrmController {
    shared: Arc<Mutex<ControllerSharedState>>,
}

impl VrmController {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(ControllerSharedState {
                commands: Vec::new(),
                active_emotion: "neutral".into(),
                active_vrma: "idle".into(),
                is_speaking: false,
            })),
        }
    }

    /// Registra uma definição personalizada de emoção / morph targets em tempo de execução.
    pub fn register_emotion(&self, name: impl Into<String>, config: EmotionConfig) {
        self.shared.lock().commands.push(VrmCommand::RegisterEmotion {
            name: name.into(),
            config,
        });
    }

    /// Define a emoção facial do avatar (presets padrão VRM como `"happy"`, `"angry"`, `"sad"`, `"relaxed"` ou apelidos registrados).
    pub fn set_emotion(&self, tag: impl Into<String>) {
        self.set_emotion_with_options(tag, 5.0, true);
    }

    pub fn set_emotion_with_options(&self, tag: impl Into<String>, ttl_sec: f32, trigger_vrma: bool) {
        let tag = tag.into();
        let mut lock = self.shared.lock();
        lock.active_emotion = tag.clone();
        lock.commands.push(VrmCommand::SetEmotion {
            tag,
            ttl_sec,
            trigger_vrma,
        });
    }

    /// Define manualmente o peso (`0.0..=1.0`) de um BlendShape / MorphTarget específico.
    pub fn set_expression(&self, name: impl Into<String>, weight: f32) {
        self.shared.lock().commands.push(VrmCommand::SetExpression {
            name: name.into(),
            weight,
        });
    }

    pub fn clear_expressions(&self) {
        self.shared.lock().commands.push(VrmCommand::ClearExpressions);
    }

    /// Reproduz uma animação `.vrma` pelo nome de catálogo (`"idle"`, `"talking"`, `"thinking"`, `"angry"`, `"blush"`, `"goodbye"`, `"jump"`, `"sad"`, `"sleepy"`, `"surprised"`, `"clapping"`, `"nod"`) ou caminho em disco.
    pub fn play_vrma(&self, name_or_path: impl Into<String>, looping: bool) {
        let name_or_path = name_or_path.into();
        let mut lock = self.shared.lock();
        lock.active_vrma = name_or_path.clone();
        lock.commands.push(VrmCommand::PlayVrma {
            name_or_path,
            looping,
            crossfade: 0.45,
        });
    }

    pub fn play_idle(&self) {
        self.play_vrma("idle", true);
    }

    pub fn stop_vrma(&self) {
        self.shared
            .lock()
            .commands
            .push(VrmCommand::StopVrma { crossfade: 0.4 });
    }

    /// Inicia o Lip-Sync labial sincronizado com o texto da fala e (opcionalmente) o buffer WAV do TTS.
    pub fn speak(&self, text: impl Into<String>, wav_bytes: Option<Vec<u8>>) {
        let mut lock = self.shared.lock();
        lock.is_speaking = true;
        lock.commands.push(VrmCommand::SpeakText {
            text: text.into(),
            wav_bytes,
        });
    }

    /// Inicia a fala em streaming: o clipe `talking` começa no primeiro chunk
    /// de áudio e a boca segue `set_speaking` até `end_speech_stream`.
    pub fn begin_speech_stream(&self) {
        let mut lock = self.shared.lock();
        lock.is_speaking = true;
        lock.commands.push(VrmCommand::BeginSpeechStream);
    }

    /// Fecha a fala em streaming e devolve o avatar ao idle.
    pub fn end_speech_stream(&self) {
        let mut lock = self.shared.lock();
        lock.is_speaking = false;
        lock.commands.push(VrmCommand::EndSpeechStream);
    }

    /// Controla diretamente a abertura labial por volume (`0.0..=1.0`).
    pub fn set_speaking(&self, speaking: bool, volume: f32) {
        let mut lock = self.shared.lock();
        lock.is_speaking = speaking;
        lock.commands.push(VrmCommand::SetSpeakingVolume {
            speaking,
            volume,
        });
    }

    /// Orienta o olhar (olhos, pescoço e cabeça) para uma coordenada normalizada `-1.0..=1.0`.
    pub fn look_at_normalized(&self, x: f32, y: f32) {
        self.shared.lock().commands.push(VrmCommand::LookAtNormalized {
            x: x.clamp(-1.0, 1.0),
            y: y.clamp(-1.0, 1.0),
        });
    }

    pub fn set_camera(&self, position: Vec3, target: Vec3, fov_y_deg: f32) {
        self.shared.lock().commands.push(VrmCommand::SetCamera {
            position,
            target,
            fov_y_deg,
        });
    }

    pub fn orbit_camera(&self, yaw_delta: f32, pitch_delta: f32, zoom_delta: f32) {
        self.shared.lock().commands.push(VrmCommand::OrbitCamera {
            yaw_delta,
            pitch_delta,
            zoom_delta,
        });
    }

    /// Restaura a câmera orbital para a posição padrão de enquadramento de corpo inteiro.
    pub fn reset_camera(&self) {
        self.shared.lock().commands.push(VrmCommand::ResetCamera);
    }

    pub fn set_spring_bone_enabled(&self, enabled: bool) {
        self.shared
            .lock()
            .commands
            .push(VrmCommand::SetSpringBoneEnabled(enabled));
    }

    pub fn set_arm_clearance_deg(&self, deg: f32) {
        self.shared
            .lock()
            .commands
            .push(VrmCommand::SetArmClearanceDeg(deg));
    }

    pub fn active_emotion(&self) -> String {
        self.shared.lock().active_emotion.clone()
    }

    pub fn active_vrma(&self) -> String {
        self.shared.lock().active_vrma.clone()
    }

    pub fn is_speaking(&self) -> bool {
        self.shared.lock().is_speaking
    }
}

impl Default for VrmController {
    fn default() -> Self {
        Self::new()
    }
}

/// View GPUI nativa que hospeda e renderiza o avatar 3D VRM a 60 FPS com transparência alfa.
pub struct VrmView {
    controller: VrmController,
    model: VrmModelData,
    skeleton: SkeletonRuntime,
    physics: SpringBoneSimulator,
    expressions: ExpressionDriver,
    renderer: VrmRenderer,
    vrma_dir: Option<PathBuf>,
    vrma_cache: HashMap<String, Arc<VrmaClip>>,
    last_frame_instant: Instant,
    current_image: Option<Arc<RenderImage>>,
    prev_painted_image: Arc<Mutex<Option<Arc<RenderImage>>>>,
    last_bounds: Arc<Mutex<Option<Bounds<Pixels>>>>,
    was_speaking: bool,
    // Controle de órbita/zoom na câmera
    orbit_yaw: f32,
    orbit_pitch: f32,
    orbit_distance: f32,
    camera_target_y: f32,
    last_drag_pos: Option<Point<Pixels>>,
}

impl VrmView {
    /// Carrega um modelo `.vrm` / `.glb` e inicializa o pipeline WGPU + animações VRMA.
    pub fn new(
        model_path: impl AsRef<Path>,
        vrma_dir: Option<PathBuf>,
        controller: VrmController,
    ) -> Result<Self> {
        let model = VrmModelData::load_from_path(model_path.as_ref())?;
        let skeleton = SkeletonRuntime::new(&model);
        let physics = SpringBoneSimulator::new(&model, &skeleton);
        let expressions = ExpressionDriver::new(&model);
        let renderer = VrmRenderer::new(&model, 560, 620)?;

        let mut view = Self {
            controller,
            model,
            skeleton,
            physics,
            expressions,
            renderer,
            vrma_dir,
            vrma_cache: HashMap::new(),
            last_frame_instant: Instant::now(),
            current_image: None,
            prev_painted_image: Arc::new(Mutex::new(None)),
            last_bounds: Arc::new(Mutex::new(None)),
            was_speaking: false,
            orbit_yaw: 0.0,
            orbit_pitch: 0.0,
            orbit_distance: 4.4,
            camera_target_y: 1.15,
            last_drag_pos: None,
        };

        view.sync_orbit_camera();
        // Pré-carrega e inicia a animação de repouso (aidelle_Idle.vrma) se disponível
        view.play_vrma_internal("idle", true, 0.2);
        Ok(view)
    }

    /// Construtor compatível com `cx.new(|cx| VrmView::load(..., cx))`.
    pub fn load(
        model_path: impl AsRef<Path>,
        vrma_dir: Option<PathBuf>,
        controller: VrmController,
        _cx: &mut Context<Self>,
    ) -> Result<Self> {
        Self::new(model_path, vrma_dir, controller)
    }

    pub fn controller(&self) -> VrmController {
        self.controller.clone()
    }

    pub fn model_title(&self) -> &str {
        &self.model.title
    }

    fn sync_orbit_camera(&mut self) {
        let cam = self.renderer.camera_mut();
        let cos_p = self.orbit_pitch.cos();
        let sin_p = self.orbit_pitch.sin();
        let sin_y = self.orbit_yaw.sin();
        let cos_y = self.orbit_yaw.cos();

        let offset = Vec3::new(
            self.orbit_distance * cos_p * sin_y,
            self.orbit_distance * sin_p,
            self.orbit_distance * cos_p * cos_y,
        );
        let target = Vec3::new(0.0, self.camera_target_y, 0.0);
        cam.target = target;
        cam.position = target + offset;
    }

    fn resolve_vrma_clip(&mut self, name_or_path: &str) -> Option<Arc<VrmaClip>> {
        let key = name_or_path.trim().to_lowercase();
        if let Some(clip) = self.vrma_cache.get(&key) {
            return Some(clip.clone());
        }

        let filename = match key.as_str() {
            "idle" => "aidelle_Idle.vrma",
            "relax" | "idleoriginal" => "Relax.vrma",
            "talking" | "speech" => "aidelle_Talking.vrma",
            "thinking" | "tilt" => "Thinking.vrma",
            "aidelle_thinking" => "aidelle_Thinking.vrma",
            "wave" | "waving" => "aidelle_Waving.vrma",
            "goodbye" => "Goodbye.vrma",
            "nod" | "aidelle_head_nod_yes" => "aidelle_Head_Nod_Yes.vrma",
            "angry" | "tsundere" => "Angry.vrma",
            "blush" | "shy" => "Blush.vrma",
            "clap" | "clapping" => "Clapping.vrma",
            "jump" | "cheer" => "Jump.vrma",
            "lookaround" => "LookAround.vrma",
            "sad" => "Sad.vrma",
            "sleepy" | "sleeping" | "tired" => "Sleepy.vrma",
            "surprised" => "Surprised.vrma",
            other => other,
        };

        let candidate = if Path::new(filename).exists() {
            PathBuf::from(filename)
        } else if let Some(dir) = &self.vrma_dir {
            let p = dir.join(filename);
            if p.exists() {
                p
            } else {
                dir.join(format!("{name_or_path}.vrma"))
            }
        } else {
            PathBuf::from(filename)
        };

        if candidate.exists() {
            if let Ok(clip) = VrmaClip::load_from_path(&candidate) {
                let arc = Arc::new(clip);
                self.vrma_cache.insert(key, arc.clone());
                return Some(arc);
            }
        }
        None
    }

    fn play_vrma_internal(&mut self, name_or_path: &str, looping: bool, crossfade: f32) {
        if let Some(clip) = self.resolve_vrma_clip(name_or_path) {
            self.skeleton.play_vrma(clip, looping, crossfade);
            self.controller.shared.lock().active_vrma = name_or_path.to_string();
        }
    }

    fn process_commands(&mut self) {
        let cmds: Vec<VrmCommand> = {
            let mut lock = self.controller.shared.lock();
            std::mem::take(&mut lock.commands)
        };

        for cmd in cmds {
            match cmd {
                VrmCommand::RegisterEmotion { name, config } => {
                    self.expressions.register_emotion(name, config);
                }
                VrmCommand::SetEmotion {
                    tag,
                    ttl_sec,
                    trigger_vrma,
                } => {
                    let suggested = self.expressions.set_emotion(&tag, ttl_sec);
                    if trigger_vrma {
                        if let Some(vrma_name) = suggested {
                            self.play_vrma_internal(&vrma_name, false, 0.45);
                        }
                    }
                }
                VrmCommand::SetExpression { name, weight } => {
                    self.expressions.set_expression(&name, weight);
                }
                VrmCommand::ClearExpressions => {
                    self.expressions.clear_manual_expressions();
                }
                VrmCommand::PlayVrma {
                    name_or_path,
                    looping,
                    crossfade,
                } => {
                    self.play_vrma_internal(&name_or_path, looping, crossfade);
                }
                VrmCommand::StopVrma { crossfade } => {
                    self.skeleton.stop_vrma(crossfade);
                }
                VrmCommand::SpeakText { text, wav_bytes } => {
                    self.expressions.start_speech(&text, wav_bytes.as_deref());
                    self.play_vrma_internal("talking", true, 0.35);
                    self.was_speaking = true;
                }
                VrmCommand::SetSpeakingVolume { speaking, volume } => {
                    self.expressions.set_speaking_volume(speaking, volume);
                }
                VrmCommand::BeginSpeechStream => {
                    self.expressions.begin_speech_stream();
                    self.play_vrma_internal("talking", true, 0.35);
                    self.was_speaking = true;
                }
                VrmCommand::EndSpeechStream => {
                    self.expressions.end_speech_stream();
                    self.was_speaking = false;
                    self.play_vrma_internal("idle", true, 0.5);
                }
                VrmCommand::LookAtNormalized { x, y } => {
                    self.skeleton.look_target = Vec2::new(x, y);
                }
                VrmCommand::SetCamera {
                    position,
                    target,
                    fov_y_deg,
                } => {
                    let cam = self.renderer.camera_mut();
                    cam.position = position;
                    cam.target = target;
                    cam.fov_y_deg = fov_y_deg;
                }
                VrmCommand::OrbitCamera {
                    yaw_delta,
                    pitch_delta,
                    zoom_delta,
                } => {
                    self.orbit_yaw = (self.orbit_yaw + yaw_delta).clamp(-1.2, 1.2);
                    self.orbit_pitch = (self.orbit_pitch + pitch_delta).clamp(-0.45, 0.65);
                    self.orbit_distance = (self.orbit_distance + zoom_delta).clamp(1.2, 6.5);
                    self.sync_orbit_camera();
                }
                VrmCommand::SetSpringBoneEnabled(enabled) => {
                    self.physics.enabled = enabled;
                }
                VrmCommand::SetArmClearanceDeg(deg) => {
                    self.skeleton.arm_clearance_deg = deg.clamp(0.0, 45.0);
                }
                VrmCommand::ResetCamera => {
                    self.orbit_yaw = 0.0;
                    self.orbit_pitch = 0.0;
                    self.orbit_distance = 4.4;
                    self.camera_target_y = 1.15;
                    self.sync_orbit_camera();
                }
            }
        }
    }

    fn step_and_render(&mut self) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_instant)
            .as_secs_f32()
            .clamp(0.001, 0.05);
        self.last_frame_instant = now;

        self.process_commands();

        // Se estava falando e terminou o áudio/visemas, retorna suavemente para a animação de repouso (idle)
        let speaking_now = self.expressions.is_speaking();
        self.controller.shared.lock().is_speaking = speaking_now;
        if self.was_speaking && !speaking_now {
            self.was_speaking = false;
            self.play_vrma_internal("idle", true, 0.5);
        }
        // Se uma animação one-shot terminou e o mixer ficou vazio, retoma o idle
        if self.skeleton.current_clip_name().is_none() {
            self.play_vrma_internal("idle", true, 0.5);
        }

        // 1. Atualiza pose esquelética (VRMA + Respiração + Look-At IK)
        self.skeleton.update_pose(&self.model, dt);

        // 2. Atualiza simulação física Verlet de SpringBones (cabelos, fitas, mangas, saia)
        self.physics.step(&self.model, &mut self.skeleton, dt);

        // 3. Atualiza pesos de expressões, piscadas e visemas e aplica morph targets na GPU se mudaram
        if self.expressions.update(dt) {
            self.renderer
                .update_morph_targets(&self.model, &self.expressions.mesh_target_weights);
        }

        // 4. Ajusta resolução de renderização ao tamanho real do elemento no layout do GPUI
        if let Some(b) = *self.last_bounds.lock() {
            let w: f32 = b.size.width.into();
            let h: f32 = b.size.height.into();
            if w >= 64.0 && h >= 64.0 {
                self.renderer.resize(w as u32, h as u32);
            }
        }

        // 5. Renderiza o frame 3D WGPU para `gpui::RenderImage`
        if let Ok(frame_img) = self.renderer.render_frame(&self.model, &self.skeleton) {
            self.current_image = Some(frame_img);
        }
    }
}

impl Render for VrmView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.step_and_render();
        window.request_animation_frame();

        let current_img = self.current_image.clone();
        let prev_slot = self.prev_painted_image.clone();
        let bounds_slot = self.last_bounds.clone();
        let bounds_for_mouse = self.last_bounds.clone();
        let ctrl_for_move = self.controller.clone();

        div()
            .id("gpui-vrm-viewport")
            .size_full()
            .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _win, _cx| {
                if let Some(b) = *bounds_for_mouse.lock() {
                    let w: f32 = b.size.width.into();
                    let h: f32 = b.size.height.into();
                    let ox: f32 = b.origin.x.into();
                    let oy: f32 = b.origin.y.into();
                    let mx: f32 = ev.position.x.into();
                    let my: f32 = ev.position.y.into();

                    if w > 1.0 && h > 1.0 {
                        let nx = (((mx - ox) / w) * 2.0 - 1.0).clamp(-1.0, 1.0);
                        let ny = (((my - oy) / h) * 2.0 - 1.0).clamp(-1.0, 1.0);
                        ctrl_for_move.look_at_normalized(nx, ny);
                    }
                }

                // Arrastar com o botão esquerdo orbita suavemente o avatar
                if ev.pressed_button == Some(MouseButton::Left) {
                    if let Some(last) = this.last_drag_pos {
                        let dx: f32 = (ev.position.x - last.x).into();
                        let dy: f32 = (ev.position.y - last.y).into();
                        this.orbit_yaw = (this.orbit_yaw - dx * 0.008).clamp(-1.4, 1.4);
                        this.orbit_pitch = (this.orbit_pitch + dy * 0.005).clamp(-0.45, 0.65);
                        this.sync_orbit_camera();
                    }
                    this.last_drag_pos = Some(ev.position);
                } else {
                    this.last_drag_pos = None;
                }
            }))
            .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, _win, _cx| {
                let delta_y: f32 = match ev.delta {
                    ScrollDelta::Pixels(p) => p.y.into(),
                    ScrollDelta::Lines(l) => l.y * 24.0,
                };
                this.orbit_distance = (this.orbit_distance - delta_y * 0.003).clamp(1.2, 6.5);
                this.sync_orbit_camera();
            }))
            .child(
                canvas(
                    move |bounds, _window, _cx| {
                        *bounds_slot.lock() = Some(bounds);
                    },
                    move |bounds, _, window, _cx| {
                        if let Some(img) = current_img {
                            if let Some(old) = prev_slot.lock().replace(img.clone()) {
                                if old.id != img.id {
                                    let _ = window.drop_image(old);
                                }
                            }
                            let _ =
                                window.paint_image(bounds, Corners::default(), img, 0, false);
                        }
                    },
                )
                .size_full(),
            )
    }
}
