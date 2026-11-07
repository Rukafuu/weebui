//! # `gpui-vrm`
//!
//! Biblioteca exclusiva 100% nativa em Rust para renderização de avatares 3D VRM (0.x / 1.0),
//! modelos glTF 2.0 (`.glb` / `.gltf`), animações humanoides `.vrma`, MToon Cel-Shading,
//! BlendShapes / Visemas de Lip-Sync e física elástica de SpringBones dentro do framework **GPUI**.

pub mod expressions;
pub mod loader;
pub mod physics;
pub mod renderer;
pub mod skeleton;
pub mod view;

pub use expressions::{EmotionConfig, ExpressionDriver, VisemeSlot, plan_visemes, resolve_emotion_config};
pub use loader::{
    AlphaMode, BlendShapeBind, BlendShapeGroup, CullMode, DecodedTexture, MorphTargetDelta,
    SceneNode, SpringBoneGroup, SpringCollider, VrmMaterial, VrmMesh, VrmModelData, VrmPrimitive,
    VrmSkin, VrmaClip, VrmaRotationTrack, VrmaTranslationTrack,
};
pub use physics::SpringBoneSimulator;
pub use renderer::{CameraState, GpuVertex, VrmRenderer};
pub use skeleton::SkeletonRuntime;
pub use view::{VrmController, VrmView};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn test_plan_visemes_and_emotions() {
        let slots = plan_visemes("Hmph, não vá se achando, baka!");
        assert!(!slots.is_empty());
        assert_eq!(slots[0].vowel, 'a');

        let tsundere = resolve_emotion_config("tsundere");
        assert!(!tsundere.weights.is_empty());
        assert_eq!(tsundere.suggested_vrma.as_deref(), Some("Angry"));

        let joy = resolve_emotion_config("joy");
        assert!(!joy.weights.is_empty());

        let happy = resolve_emotion_config("happy");
        assert_eq!(happy.suggested_vrma.as_deref(), Some("Clapping"));
    }

    #[test]
    fn test_load_shiro_vrm_and_render_headless_frame() {
        let vrm_path = PathBuf::from(r"A:\Usami Project\usami\public\models\shiro.vrm");
        if !vrm_path.exists() {
            return;
        }

        let model = VrmModelData::load_from_path(&vrm_path).expect("Falha ao carregar shiro.vrm");
        assert!(model.is_vrm0);
        assert_eq!(model.meshes.len(), 12);
        assert!(model.humanoid_bones.contains_key("hips"));
        assert!(model.humanoid_bones.contains_key("head"));
        assert!(!model.spring_bone_groups.is_empty());

        let mut skeleton = SkeletonRuntime::new(&model);
        let mut physics = SpringBoneSimulator::new(&model, &skeleton);
        let mut expressions = ExpressionDriver::new(&model);

        let vrma_path =
            PathBuf::from(r"A:\Usami Project\usami\public\animations\vrma\aidelle_Idle.vrma");
        if vrma_path.exists() {
            let clip = Arc::new(
                VrmaClip::load_from_path(&vrma_path).expect("Falha ao carregar aidelle_Idle.vrma"),
            );
            assert!(clip.duration > 0.5);
            skeleton.play_vrma(clip, true, 0.2);
        }

        expressions.set_emotion("tsundere", 4.0);
        expressions.start_speech("Baka!", None);

        skeleton.update_pose(&model, 0.016);
        physics.step(&model, &mut skeleton, 0.016);
        let changed = expressions.update(0.016);
        assert!(changed);

        let mut renderer =
            VrmRenderer::new(&model, 256, 256).expect("Falha ao inicializar VrmRenderer WGPU");
        renderer.update_morph_targets(&model, &expressions.mesh_target_weights);
        let render_img = renderer
            .render_frame(&model, &skeleton)
            .expect("Falha ao renderizar frame WGPU");

        let bytes = render_img.as_bytes(0).expect("Frame vazio");
        assert_eq!(bytes.len(), 256 * 256 * 4);
        // Confirma que o avatar foi efetivamente desenhado (existem pixels opacos e pixels transparentes)
        let opaque_pixels = bytes.chunks_exact(4).filter(|px| px[3] > 128).count();
        let transparent_pixels = bytes.chunks_exact(4).filter(|px| px[3] == 0).count();
        assert!(
            opaque_pixels > 1000,
            "Esperava >1000 pixels desenhados do avatar, obteve {opaque_pixels}"
        );
        assert!(
            transparent_pixels > 1000,
            "Esperava fundo transparente ao redor do avatar, obteve {transparent_pixels}"
        );
    }
}
