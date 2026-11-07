//! # GPUI VRM Avatar Viewer Example
//!
//! A standalone GPUI 3D avatar viewer demonstrating:
//! - Offscreen WGPU 3D rendering in GPUI
//! - MToon cel-shading with outlines & rim lighting
//! - VRM 0.x / 1.0 blendshape expressions
//! - VRMA skeletal animation retargeting
//! - SpringBone hair & cloth physics
//! - Real-time Look-At mouse tracking and camera orbit/zoom
//!
//! ## Usage:
//! ```bash
//! cargo run --example avatar_viewer -- path/to/avatar.vrm [path/to/vrma_folder]
//! ```

use std::path::{Path, PathBuf};
use gpui::prelude::*;
use gpui::*;
use gpui_vrm::{VrmController, VrmView};

struct AvatarViewerApp {
    vrm_controller: VrmController,
    vrm_view: Option<Entity<VrmView>>,
    model_name: SharedString,
    current_emotion: SharedString,
    status_message: SharedString,
}

impl AvatarViewerApp {
    fn new(vrm_path: &Path, vrma_dir: Option<PathBuf>, cx: &mut Context<Self>) -> Self {
        let controller = VrmController::new();
        controller.set_emotion("joy");

        let view_res = VrmView::new(vrm_path, vrma_dir, controller.clone());
        let (view_entity, name) = match view_res {
            Ok(v) => {
                let title = if v.model_title().is_empty() {
                    vrm_path.file_name().unwrap_or_default().to_string_lossy().to_string()
                } else {
                    v.model_title().to_string()
                };
                (Some(cx.new(|_cx| v)), title)
            }
            Err(e) => {
                eprintln!("[avatar_viewer] Error loading VRM model: {:?}", e);
                (None, format!("Failed to load: {e}"))
            }
        };

        Self {
            vrm_controller: controller,
            vrm_view: view_entity,
            model_name: name.into(),
            current_emotion: "joy".into(),
            status_message: "Left drag: Orbit | Scroll: Zoom | Mouse move: Look-At".into(),
        }
    }

    fn set_emotion(&mut self, emotion: &'static str, cx: &mut Context<Self>) {
        self.vrm_controller.set_emotion(emotion);
        self.current_emotion = emotion.into();
        self.status_message = format!("Emotion set to: {emotion}").into();
        cx.notify();
    }

    fn speak_text(&mut self, text: &str, cx: &mut Context<Self>) {
        self.vrm_controller.speak(text, None);
        self.status_message = format!("Speaking: \"{text}\"").into();
        cx.notify();
    }
}

impl Render for AvatarViewerApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model_label = self.model_name.clone();
        let status = self.status_message.clone();
        let current_emo = self.current_emotion.clone();

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x12141a))
            .text_color(rgb(0xe8ebf2))
            // Top Bar
            .child(
                div()
                    .h(px(48.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .bg(rgb(0x1a1d26))
                    .border_b_1()
                    .border_color(rgb(0x2d3240))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(0x8da2fb))
                                    .child("GPUI-VRM"),
                            )
                            .child(div().text_color(rgb(0x606880)).child("|"))
                            .child(div().font_weight(FontWeight::SEMIBOLD).child(model_label)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x8f99b3))
                            .child(format!("Active Expression: {current_emo}")),
                    ),
            )
            // Center Viewport
            .child(
                div()
                    .flex_1()
                    .relative()
                    .overflow_hidden()
                    .child(
                        if let Some(view) = &self.vrm_view {
                            div().size_full().child(view.clone()).into_any_element()
                        } else {
                            div()
                                .size_full()
                                .flex()
                                .items_center()
                                .justify_center()
                                .text_color(rgb(0xfa6464))
                                .child("Failed to load avatar model.")
                                .into_any_element()
                        },
                    ),
            )
            // Bottom Controls Toolbar
            .child(
                div()
                    .p_3()
                    .bg(rgb(0x181b24))
                    .border_t_1()
                    .border_color(rgb(0x282c39))
                    .flex()
                    .flex_col()
                    .gap_2()
                    // Buttons Row
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(0x737d99))
                                    .child("EXPRESSIONS:"),
                            )
                            .child(make_button("btn-neutral", "Neutral", cx.listener(|this, _, _win, cx| this.set_emotion("neutral", cx))))
                            .child(make_button("btn-joy", "Joy (Happy)", cx.listener(|this, _, _win, cx| this.set_emotion("joy", cx))))
                            .child(make_button("btn-angry", "Angry", cx.listener(|this, _, _win, cx| this.set_emotion("angry", cx))))
                            .child(make_button("btn-sorrow", "Sorrow (Sad)", cx.listener(|this, _, _win, cx| this.set_emotion("sorrow", cx))))
                            .child(make_button("btn-relaxed", "Relaxed", cx.listener(|this, _, _win, cx| this.set_emotion("relaxed", cx))))
                            .child(make_button("btn-surprised", "Surprised", cx.listener(|this, _, _win, cx| this.set_emotion("surprised", cx))))
                            .child(make_button("btn-blink", "Blink", cx.listener(|this, _, _win, cx| this.set_emotion("blink", cx))))
                            .child(div().w(px(8.0)))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(0x737d99))
                                    .child("LIP-SYNC:"),
                            )
                            .child(make_button("btn-hello", "Say Hello", cx.listener(|this, _, _win, cx| this.speak_text("Hello from GPUI and Rust!", cx))))
                            .child(make_button("btn-vowels", "Vowels A-E-I-O-U", cx.listener(|this, _, _win, cx| this.speak_text("A E I O U", cx)))),
                    )
                    // Status bar
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x717c99))
                            .child(status),
                    ),
            )
    }
}

fn make_button<F>(id: &'static str, label: &'static str, on_click: F) -> impl IntoElement
where
    F: Fn(&ClickEvent, &mut Window, &mut App) + 'static,
{
    div()
        .id(id)
        .px_3()
        .py_1()
        .bg(rgb(0x282e3d))
        .hover(|s| s.bg(rgb(0x373e52)))
        .rounded(px(4.0))
        .text_xs()
        .font_weight(FontWeight::MEDIUM)
        .cursor_pointer()
        .on_click(on_click)
        .child(label)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Fallbacks to explore local directory or companion project
        [
            PathBuf::from("model.vrm"),
            PathBuf::from("avatar.vrm"),
            PathBuf::from(r"A:\Usami Project\usami\public\models\shiro.vrm"),
            PathBuf::from("../usami/public/models/shiro.vrm"),
        ]
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("model.vrm"))
    };

    let vrma_dir = if args.len() > 2 {
        Some(PathBuf::from(&args[2]))
    } else {
        [
            PathBuf::from("animations"),
            PathBuf::from(r"A:\Usami Project\usami\public\animations\vrma"),
            PathBuf::from("../usami/public/animations/vrma"),
        ]
        .into_iter()
        .find(|p| p.exists())
    };

    println!("[gpui-vrm] Starting Avatar Viewer");
    println!("  Model: {:?}", model_path);
    println!("  Animations Dir: {:?}", vrma_dir);

    Application::new().run(move |cx: &mut App| {
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds {
                origin: Point::new(px(100.0), px(80.0)),
                size: Size {
                    width: px(880.0),
                    height: px(800.0),
                },
            })),
            titlebar: Some(TitlebarOptions {
                title: Some("GPUI VRM Avatar Viewer".into()),
                appears_transparent: false,
                traffic_light_position: None,
            }),
            ..Default::default()
        };

        cx.open_window(options, |_window, cx| {
            cx.new(|cx| AvatarViewerApp::new(&model_path, vrma_dir, cx))
        })
        .expect("Failed to open GPUI window");
    });
}
