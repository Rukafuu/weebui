use std::collections::HashMap;

use crate::loader::VrmModelData;

/// Tempo de vida de uma expressão manual, na mesma janela usada pela web (~4,5 s).
/// Sem isso, cada `set_expression` de um turno ficava ligado para sempre e os
/// símbolos da cabeça se acumulavam.
const MANUAL_EXPRESSION_TTL_SEC: f32 = 4.5;
/// Últimos segundos em que o peso decai até zero, para não "piscar" de volta.
const MANUAL_EXPRESSION_FADE_SEC: f32 = 0.4;

/// Item de agendamento de visema labial (portado de `usami/src/visemes.js`).
#[derive(Debug, Clone)]
pub struct VisemeSlot {
    pub vowel: char,
    pub start_sec: f32,
    pub duration_sec: f32,
}

/// Definição de uma emoção (pesos de expressões + flag de olhos fechados).
#[derive(Debug, Clone)]
pub struct EmotionConfig {
    pub weights: Vec<(String, f32)>,
    pub fallback_preset: Option<(String, f32)>,
    pub eyes_closed: bool,
    pub suggested_vrma: Option<String>,
}

impl EmotionConfig {
    pub fn new(weights: &[(&str, f32)], eyes_closed: bool) -> Self {
        Self {
            weights: weights.iter().map(|&(s, w)| (s.to_string(), w)).collect(),
            fallback_preset: None,
            eyes_closed,
            suggested_vrma: None,
        }
    }

    pub fn preset(preset_name: &str, weight: f32) -> Self {
        Self {
            weights: vec![(preset_name.to_string(), weight)],
            fallback_preset: Some((preset_name.to_string(), weight)),
            eyes_closed: false,
            suggested_vrma: None,
        }
    }

    pub fn with_fallback(mut self, preset: &str, weight: f32) -> Self {
        self.fallback_preset = Some((preset.to_string(), weight));
        self
    }

    pub fn with_suggested_vrma(mut self, vrma: &str) -> Self {
        self.suggested_vrma = Some(vrma.to_string());
        self
    }
}

/// Driver de Expressões Faciais, Emoções, Piscar Natural e Lip-Sync (Visemas + Envelope de Áudio).
pub struct ExpressionDriver {
    /// Pesos atuais de cada morph target por malha: `mesh_target_weights[mesh_idx][target_idx]`.
    pub mesh_target_weights: Vec<Vec<f32>>,
    prev_mesh_weights: Vec<Vec<f32>>,

    /// Mapa de nome (lowercase) -> lista de `(mesh_idx, target_idx, weight_scale)`.
    name_to_binds: HashMap<String, Vec<(usize, usize, f32)>>,

    /// Emoções personalizadas registradas dinamicamente
    custom_emotions: HashMap<String, EmotionConfig>,

    // Estado da emoção atual e interpolação
    active_emotion_targets: HashMap<String, f32>,
    current_emotion_weights: HashMap<String, f32>,
    emotion_eyes_closed: bool,
    emotion_ttl: f32,

    // Overrides manuais de expressões
    manual_weights: HashMap<String, f32>,
    /// Tempo restante (segundos) de cada expressão manual até ser liberada.
    manual_ttl: HashMap<String, f32>,

    // Sistema de piscar natural (Auto-Blink)
    blink_timer: f32,
    blink_phase: f32,
    is_blinking: bool,
    rng_seed: u64,

    // Sistema de Lip-Sync (cronograma de vogais + envelope RMS de áudio WAV)
    viseme_schedule: Vec<VisemeSlot>,
    audio_envelope_60hz: Vec<f32>,
    speech_elapsed: f32,
    speech_duration: f32,
    speaking_manual: bool,
    speaking_volume: f32,
}

impl ExpressionDriver {
    pub fn new(model: &VrmModelData) -> Self {
        let mut mesh_target_weights = Vec::with_capacity(model.meshes.len());
        let mut prev_mesh_weights = Vec::with_capacity(model.meshes.len());
        let mut name_to_binds: HashMap<String, Vec<(usize, usize, f32)>> = HashMap::new();

        for (m_idx, mesh) in model.meshes.iter().enumerate() {
            let t_len = mesh.morph_targets.len();
            mesh_target_weights.push(vec![0.0; t_len]);
            prev_mesh_weights.push(vec![-1.0; t_len]);

            for (t_idx, mt) in mesh.morph_targets.iter().enumerate() {
                let key = mt.name.to_lowercase();
                name_to_binds.entry(key).or_default().push((m_idx, t_idx, 1.0));
            }
        }

        for group in &model.blend_shape_groups {
            let binds: Vec<(usize, usize, f32)> = group
                .binds
                .iter()
                .filter_map(|b| {
                    let mesh_pos = model
                        .meshes
                        .iter()
                        .position(|m| m.mesh_index == b.mesh_index)?;
                    if b.target_index < model.meshes[mesh_pos].morph_targets.len() {
                        Some((mesh_pos, b.target_index, b.weight))
                    } else {
                        None
                    }
                })
                .collect();

            if !binds.is_empty() {
                if !group.name.is_empty() {
                    name_to_binds.insert(group.name.to_lowercase(), binds.clone());
                }
                if !group.preset_name.is_empty() && group.preset_name != "unknown" {
                    name_to_binds.insert(group.preset_name.to_lowercase(), binds);
                }
            }
        }

        Self {
            mesh_target_weights,
            prev_mesh_weights,
            name_to_binds,
            custom_emotions: HashMap::new(),
            active_emotion_targets: HashMap::new(),
            current_emotion_weights: HashMap::new(),
            emotion_eyes_closed: false,
            emotion_ttl: 0.0,
            manual_weights: HashMap::new(),
            manual_ttl: HashMap::new(),
            blink_timer: 2.6,
            blink_phase: 0.0,
            is_blinking: false,
            rng_seed: 0x9E37_79B9_7F4A_7C15,
            viseme_schedule: Vec::new(),
            audio_envelope_60hz: Vec::new(),
            speech_elapsed: 0.0,
            speech_duration: 0.0,
            speaking_manual: false,
            speaking_volume: 0.0,
        }
    }

    /// Registra uma definição personalizada de emoção / morph targets em tempo de execução.
    pub fn register_emotion(&mut self, name: impl Into<String>, config: EmotionConfig) {
        self.custom_emotions.insert(name.into().to_lowercase(), config);
    }

    pub fn has_expression(&self, name: &str) -> bool {
        self.name_to_binds.contains_key(&name.to_lowercase())
    }

    /// Define uma expressão individual manualmente (ex.: `"happy"`, `"blink"`, `"aa"`).
    pub fn set_expression(&mut self, name: &str, weight: f32) {
        let key = name.to_lowercase();
        if weight.abs() < 1e-4 {
            self.manual_weights.remove(&key);
            self.manual_ttl.remove(&key);
        } else {
            self.manual_weights.insert(key.clone(), weight.clamp(0.0, 1.0));
            self.manual_ttl.insert(key, MANUAL_EXPRESSION_TTL_SEC);
        }
    }

    pub fn clear_manual_expressions(&mut self) {
        self.manual_weights.clear();
        self.manual_ttl.clear();
    }

    /// Aplica uma emoção semântica (suporta presets VRM 0.x / 1.0 e apelidos customizados).
    /// Retorna a sugestão de clipe VRMA correspondente à emoção, se houver.
    pub fn set_emotion(&mut self, raw_tag: &str, ttl_sec: f32) -> Option<String> {
        let tag = raw_tag.trim().to_lowercase();
        let cfg = self
            .custom_emotions
            .get(&tag)
            .cloned()
            .unwrap_or_else(|| resolve_emotion_config(&tag));

        self.active_emotion_targets.clear();

        let mut matched_any = false;
        for (expr_name, w) in &cfg.weights {
            let key = expr_name.to_lowercase();
            if self.name_to_binds.contains_key(&key) {
                self.active_emotion_targets.insert(key, *w);
                matched_any = true;
            }
        }
        if !matched_any {
            if let Some((preset, w)) = &cfg.fallback_preset {
                let key = preset.to_lowercase();
                if self.name_to_binds.contains_key(&key) {
                    self.active_emotion_targets.insert(key, *w);
                }
            }
        }

        self.emotion_eyes_closed = cfg.eyes_closed;
        self.emotion_ttl = ttl_sec.max(0.5);
        cfg.suggested_vrma
    }

    /// Inicia o Lip-Sync a partir do texto da fala e (opcionalmente) dos bytes WAV do TTS.
    pub fn start_speech(&mut self, text: &str, wav_bytes: Option<&[u8]>) {
        self.viseme_schedule = plan_visemes(text);
        let text_dur = self
            .viseme_schedule
            .last()
            .map(|s| s.start_sec + s.duration_sec)
            .unwrap_or(0.0);

        if let Some(wav) = wav_bytes {
            let (env, wav_dur) = extract_wav_envelope_60hz(wav);
            if wav_dur > 0.1 && text_dur > 0.1 {
                // Escala o cronograma de vogais do texto para coincidir exatamente com a duração do áudio WAV
                let ratio = wav_dur / text_dur;
                for slot in &mut self.viseme_schedule {
                    slot.start_sec *= ratio;
                    slot.duration_sec *= ratio;
                }
            }
            self.audio_envelope_60hz = env;
            self.speech_duration = wav_dur.max(text_dur);
        } else {
            self.audio_envelope_60hz.clear();
            self.speech_duration = text_dur;
        }
        self.speech_elapsed = 0.0;
    }

    pub fn set_speaking_volume(&mut self, speaking: bool, volume: f32) {
        self.speaking_manual = speaking;
        self.speaking_volume = volume.clamp(0.0, 1.0);
    }

    /// Fala em streaming: a duração só é conhecida no fim, então a boca segue
    /// o volume enviado a cada chunk de PCM em vez do cronograma de visemas.
    /// Zera o cronograma anterior para o ramo manual de `update` valer desde o
    /// primeiro chunk, mesmo que a fala anterior ainda estivesse no ar.
    pub fn begin_speech_stream(&mut self) {
        self.viseme_schedule.clear();
        self.audio_envelope_60hz.clear();
        self.speech_elapsed = 0.0;
        self.speech_duration = 0.0;
        self.speaking_manual = true;
        self.speaking_volume = 0.0;
    }

    pub fn end_speech_stream(&mut self) {
        self.speaking_manual = false;
        self.speaking_volume = 0.0;
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking_manual || self.speech_elapsed < self.speech_duration
    }

    /// Atualiza todos os pesos de expressões, piscadas e visemas para o frame atual.
    /// Retorna `true` se os pesos de morph targets mudaram (exigindo atualização do buffer de vértices).
    pub fn update(&mut self, dt: f32) -> bool {
        let dt = dt.clamp(0.0, 0.1);

        // 1. Decaimento de TTL da emoção
        if self.emotion_ttl > 0.0 {
            self.emotion_ttl -= dt;
            if self.emotion_ttl <= 0.0 {
                self.active_emotion_targets.clear();
                self.emotion_eyes_closed = false;
            }
        }

        // 1.5. Decaimento de TTL das expressões manuais (evita acúmulo entre turnos)
        if !self.manual_ttl.is_empty() {
            let weights = &mut self.manual_weights;
            self.manual_ttl.retain(|key, ttl| {
                *ttl -= dt;
                if *ttl <= 0.0 {
                    weights.remove(key);
                    false
                } else {
                    true
                }
            });
        }

        // Interpolação suave dos pesos de emoção (attack rápido, release suave)
        let mut all_emotion_keys: Vec<String> = self
            .current_emotion_weights
            .keys()
            .chain(self.active_emotion_targets.keys())
            .cloned()
            .collect();
        all_emotion_keys.sort();
        all_emotion_keys.dedup();

        for key in all_emotion_keys {
            let target = self.active_emotion_targets.get(&key).copied().unwrap_or(0.0);
            let cur = self.current_emotion_weights.get(&key).copied().unwrap_or(0.0);
            let speed = if target > cur { 10.0 } else { 4.5 };
            let next = cur + (target - cur) * (dt * speed).clamp(0.0, 1.0);
            if next < 0.002 && target == 0.0 {
                self.current_emotion_weights.remove(&key);
            } else {
                self.current_emotion_weights.insert(key, next);
            }
        }

        // 2. Piscar automático natural (suprimido se a emoção atual já fecha os olhos)
        let mut blink_weight = 0.0f32;
        if !self.emotion_eyes_closed {
            if self.is_blinking {
                self.blink_phase += dt / 0.15;
                if self.blink_phase >= 1.0 {
                    self.is_blinking = false;
                    self.blink_phase = 0.0;
                    let r = self.next_rand_f32();
                    // 18% de chance de piscada dupla rápida
                    self.blink_timer = if r < 0.18 { 0.18 } else { 2.2 + r * 3.2 };
                } else {
                    // Curva triangular suave fechando em 40% do tempo e abrindo em 60%
                    blink_weight = if self.blink_phase < 0.4 {
                        self.blink_phase / 0.4
                    } else {
                        1.0 - (self.blink_phase - 0.4) / 0.6
                    };
                }
            } else {
                self.blink_timer -= dt;
                if self.blink_timer <= 0.0 {
                    self.is_blinking = true;
                    self.blink_phase = 0.0;
                }
            }
        }

        // 3. Lip-Sync (Visemas a, i, u, e, o)
        let mut vowel_weights = [('a', 0.0f32), ('i', 0.0), ('u', 0.0), ('e', 0.0), ('o', 0.0)];
        if self.speech_elapsed < self.speech_duration {
            self.speech_elapsed += dt;
            let t = self.speech_elapsed;

            // Envelope de amplitude do áudio WAV em 60Hz (quando presente)
            let env_gain = if !self.audio_envelope_60hz.is_empty() {
                let frame_f = t * 60.0;
                let idx0 = (frame_f as usize).min(self.audio_envelope_60hz.len() - 1);
                let idx1 = (idx0 + 1).min(self.audio_envelope_60hz.len() - 1);
                let frac = frame_f.fract();
                let amp = self.audio_envelope_60hz[idx0] * (1.0 - frac)
                    + self.audio_envelope_60hz[idx1] * frac;
                (amp * 1.35).clamp(0.15, 1.0)
            } else {
                0.85
            };

            for slot in &self.viseme_schedule {
                if t >= slot.start_sec && t <= slot.start_sec + slot.duration_sec {
                    let local_p = (t - slot.start_sec) / slot.duration_sec.max(0.01);
                    let shape = (local_p * std::f32::consts::PI).sin().clamp(0.25, 1.0);
                    for (v, w) in &mut vowel_weights {
                        if *v == slot.vowel {
                            *w = (shape * env_gain).clamp(0.0, 0.95);
                        }
                    }
                    break;
                }
            }
        } else if self.speaking_manual && self.speaking_volume > 0.02 {
            // Fallback por volume direto
            vowel_weights[0].1 = (self.speaking_volume * 0.85).clamp(0.0, 0.9);
            vowel_weights[4].1 = (self.speaking_volume * 0.35).clamp(0.0, 0.5);
        }

        // 4. Zera matriz de pesos por malha e acumula todas as camadas
        for weights in &mut self.mesh_target_weights {
            weights.fill(0.0);
        }

        for (key, &w) in &self.current_emotion_weights {
            Self::apply_weight_to_meshes(&self.name_to_binds, &mut self.mesh_target_weights, key, w);
        }
        for (key, &w) in &self.manual_weights {
            // Decai suavemente nos últimos instantes antes do vencimento.
            let ttl = self.manual_ttl.get(key).copied().unwrap_or(0.0);
            let fade = (ttl / MANUAL_EXPRESSION_FADE_SEC).clamp(0.0, 1.0);
            Self::apply_weight_to_meshes(
                &self.name_to_binds,
                &mut self.mesh_target_weights,
                key,
                w * fade,
            );
        }
        if blink_weight > 0.001 {
            if !Self::apply_weight_to_meshes(
                &self.name_to_binds,
                &mut self.mesh_target_weights,
                "blink",
                blink_weight,
            ) {
                Self::apply_weight_to_meshes(
                    &self.name_to_binds,
                    &mut self.mesh_target_weights,
                    "blinking",
                    blink_weight,
                );
            }
        }
        for (vowel, w) in vowel_weights {
            if w > 0.001 {
                let mut buf = [0u8; 4];
                let s = vowel.encode_utf8(&mut buf);
                // Tenta tanto o padrão VRM 0.0 ("a", "i", "u", "e", "o") quanto VRM 1.0 ("aa", "ih", "ou", "ee", "oh")
                if !Self::apply_weight_to_meshes(&self.name_to_binds, &mut self.mesh_target_weights, s, w) {
                    let vrm1_name = match vowel {
                        'a' => "aa",
                        'i' => "ih",
                        'u' => "ou",
                        'e' => "ee",
                        'o' => "oh",
                        _ => s,
                    };
                    Self::apply_weight_to_meshes(&self.name_to_binds, &mut self.mesh_target_weights, vrm1_name, w);
                }
            }
        }

        // Verifica se houve mudança perceptível para evitar upload desnecessário na GPU
        let mut changed = false;
        for (m_idx, weights) in self.mesh_target_weights.iter().enumerate() {
            for (t_idx, &w) in weights.iter().enumerate() {
                if (w - self.prev_mesh_weights[m_idx][t_idx]).abs() > 0.002 {
                    self.prev_mesh_weights[m_idx][t_idx] = w;
                    changed = true;
                }
            }
        }
        changed
    }

    fn apply_weight_to_meshes(
        name_to_binds: &HashMap<String, Vec<(usize, usize, f32)>>,
        mesh_target_weights: &mut [Vec<f32>],
        name_lower: &str,
        weight: f32,
    ) -> bool {
        let Some(binds) = name_to_binds.get(name_lower) else {
            return false;
        };
        for &(m_idx, t_idx, scale) in binds {
            if let Some(m) = mesh_target_weights.get_mut(m_idx) {
                if let Some(slot) = m.get_mut(t_idx) {
                    *slot = (*slot + weight * scale).clamp(0.0, 1.0);
                }
            }
        }
        true
    }

    fn next_rand_f32(&mut self) -> f32 {
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 7;
        self.rng_seed ^= self.rng_seed << 17;
        ((self.rng_seed & 0x00FF_FFFF) as f32) / (0x0100_0000 as f32)
    }
}

/// Planejador de visemas a partir de texto em português/inglês (portado de `usami/src/visemes.js`).
pub fn plan_visemes(text: &str) -> Vec<VisemeSlot> {
    let char_sec = 0.055f32;
    let gap_sec = 0.040f32;
    let mut at = 0.0f32;
    let mut slots: Vec<VisemeSlot> = Vec::new();
    let mut prev_merged = false;

    for ch in text.chars() {
        let is_speakable = ch.is_alphanumeric();
        let vowel = if is_speakable {
            classify_vowel(ch)
        } else {
            None
        };

        if let Some(v) = vowel {
            if prev_merged {
                if let Some(last) = slots.last_mut() {
                    if last.vowel == v {
                        last.duration_sec += char_sec;
                        at += char_sec;
                        continue;
                    }
                }
            }
            slots.push(VisemeSlot {
                vowel: v,
                start_sec: at,
                duration_sec: char_sec,
            });
            prev_merged = true;
        } else {
            prev_merged = false;
        }

        at += if is_speakable { char_sec } else { gap_sec };
    }
    slots
}

fn classify_vowel(c: char) -> Option<char> {
    match c.to_lowercase().next()? {
        'a' | 'á' | 'à' | 'â' | 'ã' | 'ä' => Some('a'),
        'e' | 'é' | 'è' | 'ê' | 'ë' => Some('e'),
        'i' | 'í' | 'ì' | 'î' | 'ï' => Some('i'),
        'o' | 'ó' | 'ò' | 'ô' | 'õ' | 'ö' => Some('o'),
        'u' | 'ú' | 'ù' | 'û' | 'ü' => Some('u'),
        _ => None,
    }
}

/// Extrai o envelope RMS de amplitude em 60Hz diretamente de um buffer WAV PCM 16-bit/float.
fn extract_wav_envelope_60hz(wav: &[u8]) -> (Vec<f32>, f32) {
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return (Vec::new(), 0.0);
    }

    let mut pos = 12usize;
    let mut channels = 1u16;
    let mut sample_rate = 24000u32;
    let mut bits_per_sample = 16u16;
    let mut audio_format = 1u16;
    let mut data_slice: &[u8] = &[];

    while pos + 8 <= wav.len() {
        let chunk_id = &wav[pos..pos + 4];
        let chunk_len =
            u32::from_le_bytes([wav[pos + 4], wav[pos + 5], wav[pos + 6], wav[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_len).min(wav.len());

        if chunk_id == b"fmt " && chunk_len >= 16 {
            audio_format = u16::from_le_bytes([wav[body_start], wav[body_start + 1]]);
            channels = u16::from_le_bytes([wav[body_start + 2], wav[body_start + 3]]).max(1);
            sample_rate = u32::from_le_bytes([
                wav[body_start + 4],
                wav[body_start + 5],
                wav[body_start + 6],
                wav[body_start + 7],
            ])
            .max(8000);
            bits_per_sample = u16::from_le_bytes([wav[body_start + 14], wav[body_start + 15]]);
        } else if chunk_id == b"data" {
            data_slice = &wav[body_start..body_end];
            break;
        }
        pos = body_end + (chunk_len & 1);
    }

    if data_slice.is_empty() {
        return (Vec::new(), 0.0);
    }

    let bytes_per_sample = (bits_per_sample / 8).max(1) as usize;
    let frame_stride = bytes_per_sample * (channels as usize);
    let total_frames = data_slice.len() / frame_stride;
    if total_frames == 0 {
        return (Vec::new(), 0.0);
    }

    let duration_sec = (total_frames as f32) / (sample_rate as f32);
    let frames_per_window = ((sample_rate as usize) / 60).max(1);
    let num_windows = (total_frames / frames_per_window).max(1);
    let mut env = Vec::with_capacity(num_windows);

    for w in 0..num_windows {
        let start_f = w * frames_per_window;
        let end_f = ((w + 1) * frames_per_window).min(total_frames);
        let mut sum_sq = 0.0f32;
        let count = (end_f - start_f).max(1);

        for f in start_f..end_f {
            let offset = f * frame_stride;
            let sample = if audio_format == 3 && bits_per_sample == 32 && offset + 4 <= data_slice.len() {
                f32::from_le_bytes([
                    data_slice[offset],
                    data_slice[offset + 1],
                    data_slice[offset + 2],
                    data_slice[offset + 3],
                ])
            } else if bits_per_sample == 16 && offset + 2 <= data_slice.len() {
                let s = i16::from_le_bytes([data_slice[offset], data_slice[offset + 1]]);
                (s as f32) / 32768.0
            } else {
                0.0
            };
            sum_sq += sample * sample;
        }

        let rms = (sum_sq / (count as f32)).sqrt();
        env.push((rms * 3.2).clamp(0.0, 1.0));
    }

    (env, duration_sec)
}

/// Resolve presets padrão do VRM 0.x e 1.0 (happy, angry, sad, relaxed, surprised, blink, etc.)
/// assim como apelidos semânticos e mapeamentos expressivos comuns.
pub fn resolve_emotion_config(raw_tag: &str) -> EmotionConfig {
    let tag = raw_tag.trim().to_lowercase();
    match tag.as_str() {
        // VRM 0.x / 1.0 Standard Presets
        "neutral" => EmotionConfig {
            weights: vec![("neutral".into(), 1.0)],
            fallback_preset: None,
            eyes_closed: false,
            suggested_vrma: None,
        },
        "joy" | "happy" | "casual_happy" | "polite_happy" | "intimate_happy" => EmotionConfig {
            weights: vec![("happy".into(), 1.0), ("joy".into(), 1.0), ("02-prefab_happy02".into(), 1.0)],
            fallback_preset: Some(("joy".into(), 0.9)),
            eyes_closed: false,
            suggested_vrma: Some("Clapping".into()),
        },
        "angry" | "casual_angry" | "serious_angry" => EmotionConfig {
            weights: vec![("angry".into(), 0.9), ("08-prefab_angry02".into(), 0.9)],
            fallback_preset: Some(("angry".into(), 0.9)),
            eyes_closed: false,
            suggested_vrma: Some("Angry".into()),
        },
        "sorrow" | "sad" | "polite_sad" | "casual_sad" | "intimate_sad" | "whisper_sad" => EmotionConfig {
            weights: vec![("sad".into(), 0.85), ("sorrow".into(), 0.85), ("12-prefab_sad01".into(), 0.85)],
            fallback_preset: Some(("sorrow".into(), 0.85)),
            eyes_closed: false,
            suggested_vrma: Some("Sad".into()),
        },
        "fun" | "relaxed" | "smile" | "polite_smile" | "casual_smile" | "intimate_smile" | "whisper_smile"
        | "whisper_happy" => EmotionConfig {
            weights: vec![("relaxed".into(), 0.8), ("fun".into(), 0.8), ("01-prefab_happy01".into(), 0.75)],
            fallback_preset: Some(("fun".into(), 0.75)),
            eyes_closed: false,
            suggested_vrma: None,
        },
        "surprised" | "polite_surprise" | "casual_surprise" | "serious_surprise" => EmotionConfig {
            weights: vec![("surprised".into(), 0.9), ("21-prefab_emoji".into(), 0.9)],
            fallback_preset: Some(("surprised".into(), 0.9)),
            eyes_closed: false,
            suggested_vrma: Some("Surprised".into()),
        },
        "blink" => EmotionConfig {
            weights: vec![("blink".into(), 1.0)],
            fallback_preset: Some(("blink".into(), 1.0)),
            eyes_closed: true,
            suggested_vrma: None,
        },
        "blink_l" | "blinkleft" => EmotionConfig {
            weights: vec![("blink_l".into(), 1.0), ("blinkleft".into(), 1.0)],
            fallback_preset: None,
            eyes_closed: false,
            suggested_vrma: None,
        },
        "blink_r" | "blinkright" => EmotionConfig {
            weights: vec![("blink_r".into(), 1.0), ("blinkright".into(), 1.0)],
            fallback_preset: None,
            eyes_closed: false,
            suggested_vrma: None,
        },

        // Emoções estendidas de avatares / Tsundere / Expressões ricas
        "tsundere" | "casual_jealous" | "intimate_jealous" => EmotionConfig {
            weights: vec![("08-prefab_angry02".into(), 0.75), ("15-prefab_awkward01".into(), 0.45)],
            fallback_preset: Some(("angry".into(), 0.7)),
            eyes_closed: false,
            suggested_vrma: Some("Angry".into()),
        },
        "very_happy" | "casual_very_happy" | "intimate_very_happy" | "cheer" => EmotionConfig {
            weights: vec![("happy".into(), 1.0), ("05-prefab_happy05".into(), 1.0)],
            fallback_preset: Some(("joy".into(), 1.0)),
            eyes_closed: false,
            suggested_vrma: Some("Jump".into()),
        },
        "playful" | "casual_playful" | "intimate_playful" => EmotionConfig {
            weights: vec![("fun".into(), 0.95), ("06-prefab_happy06".into(), 0.95)],
            fallback_preset: Some(("fun".into(), 0.95)),
            eyes_closed: false,
            suggested_vrma: None,
        },
        "laughing" | "casual_laughing" | "intimate_relieved" | "whisper_relieved" => EmotionConfig {
            weights: vec![("happy".into(), 0.95), ("04-prefab_happy04".into(), 0.95)],
            fallback_preset: Some(("joy".into(), 1.0)),
            eyes_closed: true,
            suggested_vrma: Some("Clapping".into()),
        },
        "caring" | "polite_caring" | "casual_caring" | "intimate_caring"
        | "intimate_very_caring" | "whisper_caring" | "casual_relieved" => EmotionConfig {
            weights: vec![("relaxed".into(), 0.8), ("03-prefab_happy03".into(), 0.85)],
            fallback_preset: Some(("fun".into(), 0.7)),
            eyes_closed: false,
            suggested_vrma: Some("aidelle_Head_Nod_Yes".into()),
        },
        "shy" | "blush" | "polite_embarrassed" | "casual_shy" | "intimate_shy" | "whisper_shy" => {
            EmotionConfig {
                weights: vec![("15-prefab_awkward01".into(), 0.85)],
                fallback_preset: Some(("fun".into(), 0.4)),
                eyes_closed: false,
                suggested_vrma: Some("Blush".into()),
            }
        }
        "embarrassed" | "casual_embarrassed" | "intimate_very_shy" => EmotionConfig {
            weights: vec![("16-prefab_awkward02".into(), 1.0)],
            fallback_preset: Some(("sorrow".into(), 0.4)),
            eyes_closed: true,
            suggested_vrma: Some("Blush".into()),
        },
        "frustrated" | "casual_frustrated" => EmotionConfig {
            weights: vec![("07-prefab_angry01".into(), 0.85)],
            fallback_preset: Some(("angry".into(), 0.8)),
            eyes_closed: false,
            suggested_vrma: Some("Angry".into()),
        },
        "very_angry" | "serious_very_angry" => EmotionConfig {
            weights: vec![("angry".into(), 1.0), ("10-prefab_angry04".into(), 1.0)],
            fallback_preset: Some(("angry".into(), 1.0)),
            eyes_closed: true,
            suggested_vrma: Some("Angry".into()),
        },
        "cold" | "polite_cold" | "polite_angry" | "serious_cold" | "serious_neutral"
        | "serious_protective" => EmotionConfig {
            weights: vec![("11-prefab_angry05".into(), 0.75)],
            fallback_preset: Some(("angry".into(), 0.5)),
            eyes_closed: false,
            suggested_vrma: None,
        },
        "tearful" | "crying" | "intimate_tearful" | "intimate_very_sad" | "whisper_tearful" => {
            EmotionConfig {
                weights: vec![("sad".into(), 0.95), ("13-prefab_sad02".into(), 0.95)],
                fallback_preset: Some(("sorrow".into(), 1.0)),
                eyes_closed: false,
                suggested_vrma: Some("Sad".into()),
            }
        }
        "tired" | "sleepy" | "casual_tired" | "serious_disappointed" | "serious_sad"
        | "intimate_sleepy" | "intimate_tired" | "whisper_sleepy" => EmotionConfig {
            weights: vec![("14-prefab_sad03".into(), 0.75)],
            fallback_preset: Some(("sorrow".into(), 0.6)),
            eyes_closed: false,
            suggested_vrma: Some("Sleepy".into()),
        },
        "worried" | "anxious" | "polite_worried" | "casual_worried" | "serious_worried"
        | "serious_anxious" | "intimate_worried" | "whisper_worried" => EmotionConfig {
            weights: vec![("17-prefab_awkward03".into(), 0.8), ("19-prefab_doubt".into(), 0.4)],
            fallback_preset: Some(("sorrow".into(), 0.5)),
            eyes_closed: false,
            suggested_vrma: Some("Thinking".into()),
        },
        "doubt" | "thinking" | "mumbling" | "casual_mumbling" | "intimate_mumbling" => {
            EmotionConfig {
                weights: vec![("19-prefab_doubt".into(), 0.85)],
                fallback_preset: None,
                eyes_closed: false,
                suggested_vrma: Some("Thinking".into()),
            }
        }
        "fear" | "startled" | "polite_fear" | "casual_fear" | "serious_fear" => EmotionConfig {
            weights: vec![("20-prefab_startled".into(), 0.9)],
            fallback_preset: Some(("surprised".into(), 0.9)),
            eyes_closed: false,
            suggested_vrma: Some("Surprised".into()),
        },
        _ => EmotionConfig {
            weights: Vec::new(),
            fallback_preset: None,
            eyes_closed: false,
            suggested_vrma: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_model() -> VrmModelData {
        VrmModelData {
            title: String::new(),
            is_vrm0: false,
            nodes: Vec::new(),
            root_nodes: Vec::new(),
            topological_order: Vec::new(),
            skins: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            humanoid_bones: HashMap::new(),
            blend_shape_groups: Vec::new(),
            spring_colliders: Vec::new(),
            spring_bone_groups: Vec::new(),
            auto_scale: 1.0,
            auto_offset: glam::Vec3::ZERO,
        }
    }

    fn advance(driver: &mut ExpressionDriver, seconds: f32) {
        // `update` limita o dt a 0.1 s, então avançamos em passos.
        let steps = (seconds / 0.1).round() as usize;
        for _ in 0..steps {
            driver.update(0.1);
        }
    }

    #[test]
    fn manual_expression_expires_after_ttl() {
        let mut driver = ExpressionDriver::new(&empty_model());
        driver.set_expression("happy", 1.0);

        advance(&mut driver, 4.0);
        assert!(driver.manual_weights.contains_key("happy"));

        advance(&mut driver, 1.0);
        assert!(driver.manual_weights.is_empty());
        assert!(driver.manual_ttl.is_empty());
    }

    #[test]
    fn setting_again_refreshes_ttl() {
        let mut driver = ExpressionDriver::new(&empty_model());
        driver.set_expression("happy", 1.0);

        advance(&mut driver, 4.0);
        driver.set_expression("happy", 1.0);

        advance(&mut driver, 4.0);
        assert!(driver.manual_weights.contains_key("happy"));

        advance(&mut driver, 1.0);
        assert!(driver.manual_weights.is_empty());
    }

    #[test]
    fn clearing_removes_weights_and_ttls() {
        let mut driver = ExpressionDriver::new(&empty_model());
        driver.set_expression("happy", 1.0);
        driver.clear_manual_expressions();

        assert!(driver.manual_weights.is_empty());
        assert!(driver.manual_ttl.is_empty());
    }
}
