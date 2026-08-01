//! The `rene new -i` form: one ratatui screen over every scaffold choice.
//!
//! The form draws on **stderr** (stdout stays machine-readable, matching the
//! rest of the crate) in raw mode on the alternate screen, restored on every
//! exit path including panics. The flags provide the initial values; ↑/↓ or
//! Tab move between fields, printable keys edit the focused text field,
//! Space toggles the focused checkbox, Enter validates and submits, and Esc
//! (or Ctrl-C) cancels without creating anything.
//!
//! The key handling and validation live in [`Form`], a plain state machine
//! the unit tests drive with synthetic key events — no terminal involved;
//! only [`form`] touches the real one.

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use super::{Options, Plan, Vcs, check_literal, check_name};

/// Run the form on the real terminal and return the submitted plan.
/// Cancelling is an error — the caller must not scaffold anything.
pub(super) fn form(opts: &Options, default_name: &str) -> Result<Plan, String> {
    let mut form = Form::new(opts, default_name);
    let _screen = RawScreen::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stderr()))
        .map_err(|e| format!("cannot start the terminal UI: {e}"))?;
    loop {
        terminal
            .draw(|frame| form.render(frame))
            .map_err(|e| format!("cannot draw the form: {e}"))?;
        if let Event::Key(key) = event::read().map_err(|e| format!("cannot read a key: {e}"))? {
            match form.handle_key(&key) {
                Outcome::Continue => {}
                Outcome::Cancel => return Err("cancelled; nothing was created".to_owned()),
                Outcome::Submit => return Ok(form.into_plan()),
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

/// The fields, in focus order.
const NAME: usize = 0;
const VERSION: usize = 1;
const BIN: usize = 2;
const LIB: usize = 3;
const STATICLIB: usize = 4;
const TARGET: usize = 5;
const GIT: usize = 6;
const FIELDS: usize = 7;

/// What a key press did to the form.
enum Outcome {
    Continue,
    Submit,
    Cancel,
}

/// The form's state: three text buffers, four checkboxes, the focus, and
/// the last validation failure (shown until the next edit).
struct Form {
    name: String,
    version: String,
    bin: bool,
    lib: bool,
    staticlib: bool,
    target: String,
    git: bool,
    focus: usize,
    error: Option<String>,
}

impl Form {
    fn new(opts: &Options, default_name: &str) -> Self {
        let no_kind_flag = !(opts.bin || opts.lib || opts.staticlib);
        Form {
            name: default_name.to_owned(),
            version: "0.1.0".to_owned(),
            bin: opts.bin || no_kind_flag,
            lib: opts.lib,
            staticlib: opts.staticlib,
            target: opts.target.clone().unwrap_or_default(),
            git: opts.vcs.unwrap_or(Vcs::Git) == Vcs::Git,
            focus: NAME,
            error: None,
        }
    }

    fn handle_key(&mut self, key: &KeyEvent) -> Outcome {
        // Windows terminals report releases too; act on presses only.
        if key.kind == KeyEventKind::Release {
            return Outcome::Continue;
        }
        match key.code {
            KeyCode::Esc => return Outcome::Cancel,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Outcome::Cancel;
            }
            KeyCode::Enter => match self.validate() {
                Ok(()) => return Outcome::Submit,
                Err(reason) => self.error = Some(reason),
            },
            KeyCode::Up | KeyCode::BackTab => self.focus = (self.focus + FIELDS - 1) % FIELDS,
            KeyCode::Down | KeyCode::Tab => self.focus = (self.focus + 1) % FIELDS,
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
        match self.focus {
            NAME => Some(&mut self.name),
            VERSION => Some(&mut self.version),
            TARGET => Some(&mut self.target),
            _ => None,
        }
    }

    /// The focused checkbox, if the focus is on one.
    fn toggle(&mut self) -> Option<&mut bool> {
        match self.focus {
            BIN => Some(&mut self.bin),
            LIB => Some(&mut self.lib),
            STATICLIB => Some(&mut self.staticlib),
            GIT => Some(&mut self.git),
            _ => None,
        }
    }

    /// The same checks the non-interactive path runs, before anything is
    /// written.
    fn validate(&self) -> Result<(), String> {
        check_name(&self.name)?;
        check_literal(&self.version, "version")?;
        if !self.target.is_empty() {
            check_literal(&self.target, "target triple")?;
        }
        Ok(())
    }

    /// The submitted plan; call only after [`Self::validate`] passed.
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

    fn render(&self, frame: &mut ratatui::Frame) {
        let block = Block::bordered().title(" rene new ");
        let inner = block.inner(frame.area());
        frame.render_widget(block, frame.area());
        let rows = Layout::vertical([Constraint::Length(1); 10]).split(inner);

        frame.render_widget(self.text_row(NAME, "package name", &self.name), rows[0]);
        frame.render_widget(self.text_row(VERSION, "version", &self.version), rows[1]);
        frame.render_widget(self.check_row(BIN, self.bin, "executable target"), rows[2]);
        frame.render_widget(
            self.check_row(LIB, self.lib, "dynamic library target"),
            rows[3],
        );
        frame.render_widget(
            self.check_row(STATICLIB, self.staticlib, "static library target"),
            rows[4],
        );
        frame.render_widget(
            self.text_row(TARGET, "cross target (empty = the host)", &self.target),
            rows[5],
        );
        frame.render_widget(
            self.check_row(GIT, self.git, "initialize a git repository"),
            rows[6],
        );
        if let Some(error) = &self.error {
            frame.render_widget(
                Paragraph::new(error.as_str()).style(Style::default().fg(Color::Red)),
                rows[8],
            );
        }
        frame.render_widget(
            Paragraph::new("↑/↓ move · space toggle · enter create · esc cancel")
                .style(Style::default().add_modifier(Modifier::DIM)),
            rows[9],
        );
    }

    fn text_row(&self, index: usize, label: &str, value: &str) -> Paragraph<'_> {
        let style = self.focus_style(index);
        Paragraph::new(Line::from(vec![
            Span::raw(format!("{label:<34}")),
            Span::styled(format!("{value}▏"), style),
        ]))
    }

    fn check_row(&self, index: usize, checked: bool, label: &str) -> Paragraph<'_> {
        let mark = if checked { "[x]" } else { "[ ]" };
        Paragraph::new(Line::from(Span::styled(
            format!("{mark} {label}"),
            self.focus_style(index),
        )))
    }

    fn focus_style(&self, index: usize) -> Style {
        if self.focus == index {
            Style::default().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default()
        }
    }
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

    fn key(form: &mut Form, code: KeyCode) -> Outcome {
        form.handle_key(&KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn type_str(form: &mut Form, text: &str) {
        for c in text.chars() {
            key(form, KeyCode::Char(c));
        }
    }

    /// The flags seed the fields; edits and toggles land in the plan.
    #[test]
    fn editing_and_toggling_shape_the_plan() {
        let mut form = Form::new(&options(), "pkg");
        // `--lib` was given, so the executable default is off and dynlib on.
        assert!(!form.bin);
        assert!(form.lib);

        type_str(&mut form, "_v2"); // append to the name
        key(&mut form, KeyCode::Down); // version
        key(&mut form, KeyCode::Down); // executable
        key(&mut form, KeyCode::Char(' ')); // turn it on
        key(&mut form, KeyCode::Down); // dynlib
        key(&mut form, KeyCode::Down); // staticlib
        key(&mut form, KeyCode::Down); // cross target
        for _ in 0.."wasm32-wasip1".len() {
            key(&mut form, KeyCode::Backspace); // clear: host build
        }
        key(&mut form, KeyCode::Down); // git
        key(&mut form, KeyCode::Char(' ')); // decline it
        assert!(matches!(key(&mut form, KeyCode::Enter), Outcome::Submit));

        let plan = form.into_plan();
        assert_eq!(plan.name, "pkg_v2");
        assert_eq!(plan.version, "0.1.0");
        assert!(plan.bin && plan.lib && !plan.staticlib);
        assert_eq!(plan.target, None);
        assert_eq!(plan.vcs, Vcs::None);
    }

    /// A bad name blocks submission with the reason on screen; fixing it
    /// clears the error and submits.
    #[test]
    fn validation_blocks_submit_until_fixed() {
        let mut form = Form::new(&options(), "my-pkg");
        assert!(matches!(key(&mut form, KeyCode::Enter), Outcome::Continue));
        assert!(form.error.as_deref().unwrap().contains("cannot name"));

        for _ in 0.."my-pkg".len() {
            key(&mut form, KeyCode::Backspace);
        }
        assert!(form.error.is_none(), "edits clear the error");
        type_str(&mut form, "my_pkg");
        assert!(matches!(key(&mut form, KeyCode::Enter), Outcome::Submit));
        assert_eq!(form.into_plan().name, "my_pkg");
    }

    /// Esc and Ctrl-C cancel; releases are ignored (Windows reports both).
    #[test]
    fn cancel_keys_and_key_releases() {
        let mut form = Form::new(&options(), "pkg");
        assert!(matches!(key(&mut form, KeyCode::Esc), Outcome::Cancel));
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(matches!(form.handle_key(&ctrl_c), Outcome::Cancel));

        let mut release = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        form.handle_key(&release);
        assert_eq!(form.name, "pkg", "a release must not edit the buffer");
    }

    /// The form renders every field into the buffer (no layout panics, all
    /// labels present).
    #[test]
    fn the_form_renders_every_field() {
        let backend = ratatui::backend::TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut form = Form::new(&options(), "pkg");
        form.error = Some("boom".to_owned());
        terminal.draw(|frame| form.render(frame)).unwrap();

        let screen = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        for needle in [
            "rene new",
            "package name",
            "version",
            "executable target",
            "dynamic library target",
            "static library target",
            "cross target",
            "git repository",
            "boom",
            "esc cancel",
        ] {
            assert!(screen.contains(needle), "missing `{needle}` in:\n{screen}");
        }
    }
}
