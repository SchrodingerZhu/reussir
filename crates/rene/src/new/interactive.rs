//! The `rene new -i` wizard: the scaffold choices as a stepped modal dialog.
//!
//! The wizard draws on **stderr** (stdout stays machine-readable, matching
//! the rest of the crate) in raw mode on the alternate screen, restored on
//! every exit path including panics. Five steps — package, targets, build
//! target, version control, review — shown one at a time in a centered
//! modal: Enter validates the step and moves forward (from the review step,
//! it creates), Esc moves back (from the first step, it cancels), ↑/↓ or Tab
//! move within a step, Space toggles the focused checkbox, printable keys
//! edit the focused text field, and Ctrl-C cancels from anywhere.
//!
//! The dialog is headed by the Sentinel mark, drawn through whatever picture
//! protocol the terminal answers ratatui-image's stdio query with (kitty
//! graphics, iTerm2 inline images, sixel). A terminal without one gets a
//! gradient wordmark instead — the half-block fallback is too coarse for the
//! mark's detail to read.
//!
//! The key handling and validation live in [`Wizard`], a plain state machine
//! the unit tests drive with synthetic key events — no terminal involved;
//! only [`wizard`] touches the real one.

use std::io::IsTerminal;

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui_image::StatefulImage;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;

use super::{Options, Plan, Vcs, check_literal, check_name};

/// Run the wizard on the real terminal and return the submitted plan.
/// Cancelling is an error — the caller must not scaffold anything.
pub(super) fn wizard(opts: &Options, default_name: &str) -> Result<Plan, String> {
    let mut wizard = Wizard::new(opts, default_name);
    let _screen = RawScreen::enter()?;
    // Per ratatui-image's contract the graphics query must run after
    // entering the alternate screen and before the first event read.
    let mut logo = Logo::detect();
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stderr()))
        .map_err(|e| format!("cannot start the terminal UI: {e}"))?;
    loop {
        terminal
            .draw(|frame| wizard.render(frame, logo.as_mut()))
            .map_err(|e| format!("cannot draw the wizard: {e}"))?;
        if let Event::Key(key) = event::read().map_err(|e| format!("cannot read a key: {e}"))? {
            match wizard.handle_key(&key) {
                Outcome::Continue => {}
                Outcome::Cancel => return Err("cancelled; nothing was created".to_owned()),
                Outcome::Submit => return Ok(wizard.into_plan()),
            }
        }
    }
}

/// Raw mode + alternate screen on stderr, undone on drop — the unwind path
/// included, so a panic never strands the user's terminal in raw mode.
struct RawScreen;

impl RawScreen {
    fn enter() -> Result<Self, String> {
        enable_raw_mode().map_err(|e| format!("cannot enter raw mode: {e}"))?;
        if let Err(e) = execute!(std::io::stderr(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(format!("cannot enter the alternate screen: {e}"));
        }
        Ok(Self)
    }
}

impl Drop for RawScreen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stderr(), LeaveAlternateScreen);
    }
}

/// The Sentinel mark (a distribution copy from reussir-lang/artwork; see
/// `misc/logos/README.md`), embedded so the binary stays self-contained.
const SENTINEL_PNG: &[u8] =
    include_bytes!("../../../../misc/logos/reussir-sentinel-color-1024.png");

/// Rows the mark spans in the dialog when a picture protocol draws it.
const LOGO_ROWS: u16 = 8;

/// The mark, ready to draw through the terminal's picture protocol.
struct Logo {
    state: StatefulProtocol,
    /// Columns the square mark needs to span [`LOGO_ROWS`] rows, from the
    /// terminal's reported cell geometry — for centering; the protocol
    /// itself keeps the aspect ratio.
    cols: u16,
}

impl Logo {
    /// Query the terminal and prepare the mark, or `None` for the wordmark
    /// fallback: no picture protocol (half-block art is too coarse for the
    /// mark), a piped stdout (the query round-trips on stdout/stdin, which
    /// must stay clean for machine consumers), or an undecodable PNG.
    fn detect() -> Option<Self> {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        let picker = Picker::from_query_stdio().ok()?;
        if picker.protocol_type() == ProtocolType::Halfblocks {
            return None;
        }
        let mark = image::load_from_memory(SENTINEL_PNG).ok()?;
        let font = picker.font_size();
        let cols = u32::from(LOGO_ROWS) * u32::from(font.height) / u32::from(font.width.max(1));
        Some(Logo {
            state: picker.new_resize_protocol(mark),
            cols: cols.min(u32::from(u16::MAX)) as u16,
        })
    }
}

/// The Sentinel palette (reussir-lang/artwork, `BRAND.md`) as gradient
/// stops: rose → violet → sky.
const STOPS: [(u8, u8, u8); 3] = [(0xFF, 0x7A, 0x8C), (0x9B, 0x7B, 0xFF), (0x55, 0xD6, 0xFF)];
const ROSE: Color = Color::Rgb(0xFF, 0x7A, 0x8C);
const VIOLET: Color = Color::Rgb(0x9B, 0x7B, 0xFF);
const SKY: Color = Color::Rgb(0x55, 0xD6, 0xFF);

/// The color at position `i` of `n` along the [`STOPS`] gradient.
fn gradient(i: usize, n: usize) -> Color {
    let t = i as f64 / (n.max(2) - 1) as f64;
    let (from, to, t) = if t < 0.5 {
        (STOPS[0], STOPS[1], t * 2.0)
    } else {
        (STOPS[1], STOPS[2], (t - 0.5) * 2.0)
    };
    let lerp = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * t).round() as u8;
    Color::Rgb(lerp(from.0, to.0), lerp(from.1, to.1), lerp(from.2, to.2))
}

/// The steps, in order.
const STEP_PACKAGE: usize = 0;
const STEP_TARGETS: usize = 1;
const STEP_BUILD: usize = 2;
const STEP_VCS: usize = 3;
const STEP_REVIEW: usize = 4;
const STEPS: usize = 5;

const STEP_TITLES: [&str; STEPS] = [
    "package",
    "targets",
    "build target",
    "version control",
    "review",
];

/// Focusable fields per step (the review step has none).
const FIELDS: [usize; STEPS] = [2, 3, 1, 1, 0];

/// What a key press did to the wizard.
enum Outcome {
    Continue,
    Submit,
    Cancel,
}

/// The wizard's state: three text buffers, four checkboxes, the step, the
/// focus within it, and the last validation failure (shown until the next
/// edit or step change).
struct Wizard {
    name: String,
    version: String,
    bin: bool,
    lib: bool,
    staticlib: bool,
    target: String,
    git: bool,
    step: usize,
    focus: usize,
    error: Option<String>,
}

impl Wizard {
    fn new(opts: &Options, default_name: &str) -> Self {
        let no_kind_flag = !(opts.bin || opts.lib || opts.staticlib);
        Wizard {
            name: default_name.to_owned(),
            version: "0.1.0".to_owned(),
            bin: opts.bin || no_kind_flag,
            lib: opts.lib,
            staticlib: opts.staticlib,
            target: opts.target.clone().unwrap_or_default(),
            git: opts.vcs.unwrap_or(Vcs::Git) == Vcs::Git,
            step: STEP_PACKAGE,
            focus: 0,
            error: None,
        }
    }

    fn handle_key(&mut self, key: &KeyEvent) -> Outcome {
        // Windows terminals report releases too; act on presses only.
        if key.kind == KeyEventKind::Release {
            return Outcome::Continue;
        }
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Outcome::Cancel;
            }
            KeyCode::Esc => {
                if self.step == STEP_PACKAGE {
                    return Outcome::Cancel;
                }
                self.step -= 1;
                self.focus = 0;
                self.error = None;
            }
            KeyCode::Enter => match self.validate_step() {
                Err(reason) => self.error = Some(reason),
                Ok(()) if self.step == STEP_REVIEW => return Outcome::Submit,
                Ok(()) => {
                    self.step += 1;
                    self.focus = 0;
                    self.error = None;
                }
            },
            KeyCode::Up | KeyCode::BackTab if FIELDS[self.step] > 0 => {
                self.focus = (self.focus + FIELDS[self.step] - 1) % FIELDS[self.step];
            }
            KeyCode::Down | KeyCode::Tab if FIELDS[self.step] > 0 => {
                self.focus = (self.focus + 1) % FIELDS[self.step];
            }
            KeyCode::Char(' ') if self.toggle().is_some() => {
                let toggle = self.toggle().expect("guarded by the match arm");
                *toggle = !*toggle;
            }
            KeyCode::Char(c) if self.buffer().is_some() && !c.is_control() => {
                self.buffer().expect("guarded by the match arm").push(c);
                self.error = None;
            }
            KeyCode::Backspace if self.buffer().is_some() => {
                self.buffer().expect("guarded by the match arm").pop();
                self.error = None;
            }
            _ => {}
        }
        Outcome::Continue
    }

    /// The focused text buffer, if the focus is on a text field.
    fn buffer(&mut self) -> Option<&mut String> {
        match (self.step, self.focus) {
            (STEP_PACKAGE, 0) => Some(&mut self.name),
            (STEP_PACKAGE, 1) => Some(&mut self.version),
            (STEP_BUILD, 0) => Some(&mut self.target),
            _ => None,
        }
    }

    /// The focused checkbox, if the focus is on one.
    fn toggle(&mut self) -> Option<&mut bool> {
        match (self.step, self.focus) {
            (STEP_TARGETS, 0) => Some(&mut self.bin),
            (STEP_TARGETS, 1) => Some(&mut self.lib),
            (STEP_TARGETS, 2) => Some(&mut self.staticlib),
            (STEP_VCS, 0) => Some(&mut self.git),
            _ => None,
        }
    }

    /// The same checks the non-interactive path runs, applied to the step
    /// they belong to so a mistake surfaces before the step is left. Every
    /// step between an edit and the review step re-validates on the way, so
    /// reaching the end means the whole plan is sound.
    fn validate_step(&self) -> Result<(), String> {
        match self.step {
            STEP_PACKAGE => {
                check_name(&self.name)?;
                check_literal(&self.version, "version")
            }
            STEP_BUILD if !self.target.is_empty() => check_literal(&self.target, "target triple"),
            _ => Ok(()),
        }
    }

    /// The submitted plan; call only after every step validated.
    fn into_plan(self) -> Plan {
        Plan {
            // All three kinds unchecked still scaffolds a package: one with
            // no targets, the libdir-printing workflow `build` supports.
            bin: self.bin,
            lib: self.lib,
            staticlib: self.staticlib,
            target: (!self.target.is_empty()).then_some(self.target),
            vcs: if self.git { Vcs::Git } else { Vcs::None },
            name: self.name,
            version: self.version,
        }
    }

    fn render(&self, frame: &mut ratatui::Frame, logo: Option<&mut Logo>) {
        // The wordmark header is two rows; the mark, LOGO_ROWS — drop the
        // picture when the terminal is too short for the dialog around it.
        let head_rows = match &logo {
            Some(_) if frame.area().height >= LOGO_ROWS + 13 => LOGO_ROWS,
            _ => 2,
        };
        let modal = centered(frame.area(), 66, head_rows + 13);
        frame.render_widget(Clear, modal);
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(VIOLET))
            .title(Line::from(" rene · new package ").centered());
        let inner = block.inner(modal);
        frame.render_widget(block, modal);
        let rows = Layout::vertical([
            Constraint::Length(head_rows), // the mark or the wordmark
            Constraint::Length(1),
            Constraint::Length(1), // step title
            Constraint::Length(1), // step dots
            Constraint::Length(1),
            Constraint::Length(4), // the step's fields
            Constraint::Length(1),
            Constraint::Length(1), // validation failure
            Constraint::Length(1), // key hints
        ])
        .split(inner);

        match logo {
            Some(logo) if head_rows == LOGO_ROWS => {
                let cols = logo.cols.clamp(1, rows[0].width);
                let at = Rect::new(
                    rows[0].x + (rows[0].width - cols) / 2,
                    rows[0].y,
                    cols,
                    rows[0].height,
                );
                frame.render_stateful_widget(StatefulImage::default(), at, &mut logo.state);
            }
            _ => frame.render_widget(self.wordmark(), rows[0]),
        }

        frame.render_widget(
            Line::from(STEP_TITLES[self.step])
                .style(Style::default().add_modifier(Modifier::BOLD))
                .centered(),
            rows[2],
        );
        frame.render_widget(self.dots(), rows[3]);

        let fields =
            Layout::vertical([Constraint::Length(1); 4]).split(rows[5].inner(Margin::new(3, 0)));
        match self.step {
            STEP_PACKAGE => {
                frame.render_widget(self.text_row(0, "package name", &self.name), fields[0]);
                frame.render_widget(self.text_row(1, "version", &self.version), fields[1]);
            }
            STEP_TARGETS => {
                frame.render_widget(self.check_row(0, self.bin, "executable"), fields[0]);
                frame.render_widget(self.check_row(1, self.lib, "dynamic library"), fields[1]);
                frame.render_widget(
                    self.check_row(2, self.staticlib, "static library"),
                    fields[2],
                );
            }
            STEP_BUILD => {
                frame.render_widget(self.text_row(0, "machine triple", &self.target), fields[0]);
                frame.render_widget(
                    Line::from("  an empty triple builds for the host")
                        .style(Style::default().add_modifier(Modifier::DIM)),
                    fields[2],
                );
            }
            STEP_VCS => {
                frame.render_widget(
                    self.check_row(0, self.git, "initialize a git repository"),
                    fields[0],
                );
            }
            _ => {
                let kinds = [
                    (self.bin, "executable"),
                    (self.lib, "dynamic library"),
                    (self.staticlib, "static library"),
                ]
                .iter()
                .filter(|(on, _)| *on)
                .map(|(_, kind)| *kind)
                .collect::<Vec<_>>();
                let kinds = if kinds.is_empty() {
                    "none (a target-less package)".to_owned()
                } else {
                    kinds.join(" + ")
                };
                let machine = if self.target.is_empty() {
                    "host"
                } else {
                    &self.target
                };
                let vcs = if self.git { "git" } else { "none" };
                let package = format!("{} v{}", self.name, self.version);
                frame.render_widget(summary_row("package", &package), fields[0]);
                frame.render_widget(summary_row("targets", &kinds), fields[1]);
                frame.render_widget(summary_row("machine", machine), fields[2]);
                frame.render_widget(summary_row("vcs", vcs), fields[3]);
            }
        }

        if let Some(error) = &self.error {
            frame.render_widget(
                Line::from(error.as_str())
                    .style(Style::default().fg(ROSE))
                    .centered(),
                rows[7],
            );
        }
        frame.render_widget(
            Line::from(self.hints())
                .style(Style::default().add_modifier(Modifier::DIM))
                .centered(),
            rows[8],
        );
    }

    /// The two-row header for terminals without a picture protocol: the
    /// wordmark along the Sentinel gradient.
    fn wordmark(&self) -> Paragraph<'static> {
        let letters = "REUSSIR";
        let mark = letters
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                [
                    Span::styled(
                        c.to_string(),
                        Style::default()
                            .fg(gradient(i, letters.len()))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                ]
            })
            .collect::<Vec<_>>();
        Paragraph::new(vec![
            Line::from(mark),
            Line::from("package wizard").style(Style::default().add_modifier(Modifier::DIM)),
        ])
        .centered()
    }

    /// The wizard's progress: a filled dot per step reached, the current one
    /// in the accent color.
    fn dots(&self) -> Line<'static> {
        let mut spans = Vec::new();
        for step in 0..STEPS {
            let (dot, style) = if step == self.step {
                ("●", Style::default().fg(SKY))
            } else if step < self.step {
                ("●", Style::default().fg(VIOLET).add_modifier(Modifier::DIM))
            } else {
                ("○", Style::default().add_modifier(Modifier::DIM))
            };
            spans.push(Span::styled(dot, style));
            if step + 1 < STEPS {
                spans.push(Span::raw("  "));
            }
        }
        Line::from(spans).centered()
    }

    fn text_row(&self, index: usize, label: &str, value: &str) -> Line<'_> {
        let focused = self.focus == index;
        let value_style = if focused {
            Style::default().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default()
        };
        Line::from(vec![
            self.marker(focused),
            Span::raw(format!("{label:<18}")),
            Span::styled(format!("{value}▏"), value_style),
        ])
    }

    fn check_row(&self, index: usize, checked: bool, label: &str) -> Line<'_> {
        let focused = self.focus == index;
        let style = if focused {
            Style::default().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default()
        };
        let mark = if checked { "[x]" } else { "[ ]" };
        Line::from(vec![
            self.marker(focused),
            Span::styled(format!("{mark} {label}"), style),
        ])
    }

    fn marker(&self, focused: bool) -> Span<'static> {
        if focused {
            Span::styled("▸ ", Style::default().fg(SKY))
        } else {
            Span::raw("  ")
        }
    }

    /// The footer keys, phrased for the current step.
    fn hints(&self) -> String {
        let mut hints = Vec::new();
        if FIELDS[self.step] > 1 {
            hints.push("↑↓ move");
        }
        if matches!(self.step, STEP_TARGETS | STEP_VCS) {
            hints.push("space toggle");
        }
        hints.push(if self.step == STEP_REVIEW {
            "enter create"
        } else {
            "enter next"
        });
        hints.push(if self.step == STEP_PACKAGE {
            "esc cancel"
        } else {
            "esc back"
        });
        hints.join(" · ")
    }
}

fn summary_row<'a>(label: &'a str, value: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("  {label:<10}"),
            Style::default().add_modifier(Modifier::DIM),
        ),
        Span::raw(value.to_owned()),
    ])
}

/// `width` × `height` (clamped to `area`) in `area`'s middle.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn options() -> Options {
        Options {
            path: PathBuf::from("pkg"),
            name: None,
            bin: false,
            lib: true,
            staticlib: false,
            target: Some("wasm32-wasip1".to_owned()),
            vcs: None,
            interactive: true,
        }
    }

    fn key(wizard: &mut Wizard, code: KeyCode) -> Outcome {
        wizard.handle_key(&KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn type_str(wizard: &mut Wizard, text: &str) {
        for c in text.chars() {
            key(wizard, KeyCode::Char(c));
        }
    }

    /// The flags seed the fields; walking the steps with edits and toggles
    /// shapes the plan, and only the review step's Enter submits.
    #[test]
    fn walking_the_steps_shapes_the_plan() {
        let mut wizard = Wizard::new(&options(), "pkg");
        // `--lib` was given, so the executable default is off and dynlib on.
        assert!(!wizard.bin);
        assert!(wizard.lib);

        type_str(&mut wizard, "_v2"); // append to the name
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));
        assert_eq!(wizard.step, STEP_TARGETS);

        key(&mut wizard, KeyCode::Char(' ')); // executable on
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));

        assert_eq!(wizard.step, STEP_BUILD);
        for _ in 0.."wasm32-wasip1".len() {
            key(&mut wizard, KeyCode::Backspace); // clear: host build
        }
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));

        assert_eq!(wizard.step, STEP_VCS);
        key(&mut wizard, KeyCode::Char(' ')); // decline git
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));

        assert_eq!(wizard.step, STEP_REVIEW);
        assert!(matches!(key(&mut wizard, KeyCode::Enter), Outcome::Submit));

        let plan = wizard.into_plan();
        assert_eq!(plan.name, "pkg_v2");
        assert_eq!(plan.version, "0.1.0");
        assert!(plan.bin && plan.lib && !plan.staticlib);
        assert_eq!(plan.target, None);
        assert_eq!(plan.vcs, Vcs::None);
    }

    /// A bad name blocks the first step with the reason on screen; fixing it
    /// clears the error and moves on.
    #[test]
    fn validation_blocks_the_step_until_fixed() {
        let mut wizard = Wizard::new(&options(), "my-pkg");
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));
        assert_eq!(wizard.step, STEP_PACKAGE, "an invalid name pins the step");
        assert!(wizard.error.as_deref().unwrap().contains("cannot name"));

        for _ in 0.."my-pkg".len() {
            key(&mut wizard, KeyCode::Backspace);
        }
        assert!(wizard.error.is_none(), "edits clear the error");
        type_str(&mut wizard, "my_pkg");
        assert!(matches!(
            key(&mut wizard, KeyCode::Enter),
            Outcome::Continue
        ));
        assert_eq!(wizard.step, STEP_TARGETS);
    }

    /// Esc backtracks a step and only cancels from the first; Ctrl-C cancels
    /// anywhere; releases are ignored (Windows reports both).
    #[test]
    fn esc_backtracks_and_cancels_only_at_the_start() {
        let mut wizard = Wizard::new(&options(), "pkg");
        key(&mut wizard, KeyCode::Enter);
        assert_eq!(wizard.step, STEP_TARGETS);
        assert!(matches!(key(&mut wizard, KeyCode::Esc), Outcome::Continue));
        assert_eq!(wizard.step, STEP_PACKAGE);
        assert!(matches!(key(&mut wizard, KeyCode::Esc), Outcome::Cancel));

        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(wizard.handle_key(&ctrl_c), Outcome::Cancel));

        let mut release = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        wizard.handle_key(&release);
        assert_eq!(wizard.name, "pkg", "a release must not edit the buffer");
    }

    /// Every step renders its fields (no layout panics, all labels present),
    /// with the footer phrased for the step.
    #[test]
    fn every_step_renders_its_labels() {
        let needles: [&[&str]; STEPS] = [
            &["package name", "version", "esc cancel"],
            &[
                "executable",
                "dynamic library",
                "static library",
                "space toggle",
            ],
            &["machine triple", "builds for the host", "esc back"],
            &["git repository"],
            &["targets", "machine", "vcs", "enter create"],
        ];
        let backend = ratatui::backend::TestBackend::new(70, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut wizard = Wizard::new(&options(), "pkg");
        wizard.error = Some("boom".to_owned());
        for (step, step_needles) in needles.iter().enumerate() {
            wizard.step = step;
            terminal.draw(|frame| wizard.render(frame, None)).unwrap();
            let screen = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            for needle in [
                &["rene", "new package", "R E U S S I R", "boom"],
                *step_needles,
            ]
            .concat()
            {
                assert!(
                    screen.contains(needle),
                    "step {step}: missing `{needle}` in:\n{screen}"
                );
            }
        }
    }
}
