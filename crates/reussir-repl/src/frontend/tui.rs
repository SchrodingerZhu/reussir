//! The interactive terminal UI: a scrollback pane over a multiline input
//! editor (ratatui + ratatui-textarea).
//!
//! Input submission is *smart Enter*: Enter submits when the buffer is a
//! command or lexically complete ([`crate::input::is_incomplete`]), and
//! inserts a newline otherwise — so a pasted or half-typed `fn f(x: i64) {`
//! simply continues on the next line. Alt+Enter (and Shift+Enter where the
//! terminal reports it) always inserts a newline; Ctrl+J always submits.
//! GHCi-style `:{` … `}:` blocks are still honored for muscle-memory and
//! paste compatibility: a buffer whose first line is `:{` submits only when
//! its last line is `}:`.
//!
//! Up/Down browse history when the cursor is on the first/last line
//! (otherwise they move within the editor); history persists across
//! sessions. Ctrl+C clears the buffer (quit hint on empty), Ctrl+D on an
//! empty buffer quits, PageUp/PageDown scroll the scrollback.

use std::io::Write as _;
use std::path::PathBuf;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui_textarea::{CursorMove, Input, TextArea};

use reussir_core::semi::{Report, Severity, render_reports_to};
use reussir_syntax::diagnostics::{self, SourceMap};

use reussir_jit::OptLevel;

use crate::session::{self, Config, Exit, Outcome, ReplSession};

/// Run the TUI: owns terminal setup/teardown and the `:clear` session loop.
pub fn run(config: Config) -> Result<(), String> {
    // `ratatui::init` enters the alternate screen + raw mode and installs a
    // panic hook that restores the terminal.
    let terminal = ratatui::init();
    let _ = ratatui::crossterm::execute!(
        std::io::stdout(),
        ratatui::crossterm::event::EnableBracketedPaste
    );
    let mut tui = Tui::new(terminal, config);
    tui.load_history();

    let result = loop {
        let exit = match session::run(config, |session| tui.drive(session)) {
            Ok(exit) => exit,
            Err(message) => break Err(message),
        };
        match exit {
            Exit::Quit => break Ok(()),
            Exit::Clear => {
                tui.out_counter = 0;
                tui.push_line(Line::from("Context cleared".italic()));
                continue;
            }
        }
    };

    tui.save_history();
    let _ = ratatui::crossterm::execute!(
        std::io::stdout(),
        ratatui::crossterm::event::DisableBracketedPaste
    );
    ratatui::restore();
    result
}

struct Tui {
    terminal: ratatui::DefaultTerminal,
    config: Config,
    scrollback: Vec<Line<'static>>,
    /// Lines scrolled up from the bottom (0 = stick to newest output).
    scroll_up: u16,
    textarea: TextArea<'static>,
    history: Vec<String>,
    /// `Some(i)` while browsing history entry `i`; `None` while editing.
    history_pos: Option<usize>,
    /// The in-progress input stashed while browsing history.
    stash: Vec<String>,
    /// Consecutive Ctrl+C presses on an empty buffer (2 quits).
    interrupts: u8,
    /// Evaluating flag rendered in the status line during `eval`.
    busy: bool,
    /// Jupyter-style `Out[n]` numbering for evaluated values; resets with
    /// the session on `:clear`.
    out_counter: usize,
}

impl Tui {
    fn new(terminal: ratatui::DefaultTerminal, config: Config) -> Self {
        let mut tui = Tui {
            terminal,
            config,
            scrollback: vec![
                Line::from(format!(
                    "Reussir REPL v{} (Rust pipeline)",
                    env!("CARGO_PKG_VERSION")
                )),
                Line::from(
                    "Type :help for commands · Enter submits · Alt+Enter newline · :q quits"
                        .italic()
                        .dark_gray(),
                ),
            ],
            scroll_up: 0,
            textarea: TextArea::default(),
            history: Vec::new(),
            history_pos: None,
            stash: Vec::new(),
            interrupts: 0,
            busy: false,
            out_counter: 0,
        };
        tui.reset_textarea();
        tui
    }

    // ----- the session drive loop -----

    fn drive(&mut self, session: &mut ReplSession<'_, '_>) -> Exit {
        loop {
            let defs = session.definition_count();
            let opt = session.opt;
            if self.draw(defs, opt).is_err() {
                return Exit::Quit;
            }
            match event::read() {
                Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => {
                    if let Some(exit) = self.handle_key(key, session) {
                        return exit;
                    }
                }
                Ok(Event::Paste(text)) => {
                    self.textarea
                        .insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
                }
                Ok(_) => {} // resize triggers redraw; ignore the rest
                Err(_) => return Exit::Quit,
            }
        }
    }

    fn handle_key(&mut self, key: KeyEvent, session: &mut ReplSession<'_, '_>) -> Option<Exit> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        if !matches!(key.code, KeyCode::Char('c')) || !ctrl {
            self.interrupts = 0;
        }

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.buffer().is_empty() {
                    self.interrupts += 1;
                    if self.interrupts >= 2 {
                        return Some(Exit::Quit);
                    }
                    self.push_line(Line::from(
                        "(press Ctrl+C again or Ctrl+D to quit)".dark_gray(),
                    ));
                } else {
                    self.reset_textarea();
                    self.history_pos = None;
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.buffer().is_empty() {
                    return Some(Exit::Quit);
                }
            }
            KeyCode::Char('j') if ctrl => return self.submit(session),
            KeyCode::Enter if alt || shift => {
                self.textarea.insert_newline();
            }
            KeyCode::Enter => {
                if self.wants_submit() {
                    return self.submit(session);
                }
                self.textarea.insert_newline();
            }
            KeyCode::Up if self.textarea.cursor().0 == 0 => self.history_prev(),
            KeyCode::Down if self.textarea.cursor().0 + 1 == self.textarea.lines().len() => {
                self.history_next()
            }
            KeyCode::PageUp => {
                self.scroll_up = self.scroll_up.saturating_add(5);
            }
            KeyCode::PageDown => {
                self.scroll_up = self.scroll_up.saturating_sub(5);
            }
            _ => {
                self.textarea.input(Input::from(key));
            }
        }
        None
    }

    // ----- submission -----

    fn buffer(&self) -> String {
        self.textarea.lines().join("\n")
    }

    /// Should Enter submit the current buffer (as opposed to adding a line)?
    fn wants_submit(&self) -> bool {
        let text = self.buffer();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return true; // submitting a blank buffer is a no-op
        }
        // An explicit `:{` block submits only when closed by `}:`.
        if let Some(first) = text.lines().next()
            && first.trim() == ":{"
        {
            return text.lines().last().is_some_and(|l| l.trim() == "}:");
        }
        // Commands are one-liners.
        if trimmed.starts_with(':') {
            return true;
        }
        !crate::input::is_incomplete(&text)
    }

    fn submit(&mut self, session: &mut ReplSession<'_, '_>) -> Option<Exit> {
        let mut text = self.buffer();
        // Strip an explicit `:{` … `}:` wrapper.
        if text.lines().next().is_some_and(|l| l.trim() == ":{")
            && text.lines().last().is_some_and(|l| l.trim() == "}:")
        {
            let lines: Vec<&str> = text.lines().collect();
            text = lines[1..lines.len() - 1].join("\n");
        }

        // Echo the input into the scrollback.
        for (i, line) in self.buffer().lines().enumerate() {
            let prompt = if i == 0 { "λ> " } else { ".. " };
            self.push_line(Line::from(vec![
                Span::styled(prompt, Style::default().fg(Color::Cyan)),
                Span::raw(line.to_string()),
            ]));
        }

        if !text.trim().is_empty() && self.history.last().map(String::as_str) != Some(text.as_str())
        {
            self.history.push(text.clone());
        }
        self.history_pos = None;
        self.reset_textarea();
        self.scroll_up = 0;

        if text.trim().is_empty() {
            return None;
        }

        // Repaint with the busy marker before a (synchronous) evaluation;
        // commands are instant and skip the flash.
        if !text.trim_start().starts_with(':') {
            self.busy = true;
            let _ = self.draw(session.definition_count(), session.opt);
        }
        let outcome = session.eval(&text);
        self.busy = false;
        // Keep the restart config in step with `:set opt`, so a `:clear`
        // rebuilds the session at the level the user chose, not the CLI's.
        self.config.opt = session.opt;

        self.render_outcome(&outcome, &text)
    }

    fn render_outcome(&mut self, outcome: &Outcome, input: &str) -> Option<Exit> {
        match outcome {
            Outcome::Empty => {}
            Outcome::Quit => return Some(Exit::Quit),
            Outcome::ClearRequested => return Some(Exit::Clear),
            Outcome::Text(text) => {
                for line in text.lines() {
                    self.push_line(Line::from(line.to_string()));
                }
            }
            Outcome::Definitions { count, warnings } => {
                self.render_reports(warnings, input);
                for _ in 0..*count {
                    self.push_line(Line::from("Definition added.".green()));
                }
            }
            Outcome::Value {
                value,
                ty,
                warnings,
            } => {
                self.render_reports(warnings, input);
                self.out_counter += 1;
                let title = format!("Out[{}]", self.out_counter);
                let content = if ty == "()" {
                    vec![Span::styled(
                        "()".to_string(),
                        Style::default().add_modifier(Modifier::BOLD),
                    )]
                } else {
                    vec![
                        Span::styled(value.clone(), Style::default().add_modifier(Modifier::BOLD)),
                        Span::raw(" : "),
                        Span::styled(ty.clone(), Style::default().fg(Color::Magenta)),
                    ]
                };
                self.push_boxed(&title, content);
            }
            Outcome::ParseErrors(errors) => {
                let map = SourceMap::new(input);
                let mut rendered = Vec::new();
                let _ =
                    diagnostics::render_errors("<repl>", input, &map, errors, false, &mut rendered);
                for line in String::from_utf8_lossy(&rendered).lines() {
                    self.push_line(Line::from(line.to_string().red()));
                }
            }
            Outcome::Reports(reports) => self.render_reports(reports, input),
            Outcome::Backend(message) => {
                self.push_line(Line::from(format!("error: {message}").red()));
            }
        }
        None
    }

    /// Render elaboration/mono reports with the shared ariadne caret
    /// renderer (the same one `rrc` and the plain frontend use), one report
    /// at a time so each block can be styled by its severity.
    fn render_reports(&mut self, reports: &[Report], input: &str) {
        for report in reports {
            let mut rendered = Vec::new();
            let _ = render_reports_to(
                "<repl>",
                input,
                std::slice::from_ref(report),
                false,
                &mut rendered,
            );
            let style = match report.severity {
                Severity::Error => Style::default().fg(Color::Red),
                Severity::Warning => Style::default().fg(Color::Yellow),
            };
            for line in String::from_utf8_lossy(&rendered).lines() {
                self.push_line(Line::from(Span::styled(line.to_string(), style)));
            }
        }
    }

    // ----- history -----

    fn history_prev(&mut self) {
        let next = match self.history_pos {
            None if self.history.is_empty() => return,
            None => {
                self.stash = self.textarea.lines().to_vec();
                self.history.len() - 1
            }
            Some(0) => return,
            Some(i) => i - 1,
        };
        self.history_pos = Some(next);
        self.set_buffer_text(&self.history[next].clone());
    }

    fn history_next(&mut self) {
        match self.history_pos {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.history_pos = Some(i + 1);
                self.set_buffer_text(&self.history[i + 1].clone());
            }
            Some(_) => {
                self.history_pos = None;
                let stash = std::mem::take(&mut self.stash);
                self.reset_textarea();
                for (i, line) in stash.iter().enumerate() {
                    if i > 0 {
                        self.textarea.insert_newline();
                    }
                    self.textarea.insert_str(line);
                }
            }
        }
    }

    fn history_path() -> Option<PathBuf> {
        // XDG_DATA_HOME/~/.local/share on Linux, %LOCALAPPDATA% on Windows
        // (a HOME-based fallback would silently disable history there),
        // ~/Library/Application Support on macOS.
        Some(
            dirs::data_local_dir()?
                .join("reussir")
                .join("rrepl_history"),
        )
    }

    fn load_history(&mut self) {
        let Some(path) = Self::history_path() else {
            return;
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            return;
        };
        // One entry per line; embedded newlines escaped (see save_history).
        self.history = content
            .lines()
            .filter(|l| !l.is_empty())
            .map(unescape_entry)
            .collect();
    }

    fn save_history(&self) {
        let Some(path) = Self::history_path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        const KEEP: usize = 1000;
        let start = self.history.len().saturating_sub(KEEP);
        let mut out = String::new();
        for entry in &self.history[start..] {
            out.push_str(&escape_entry(entry));
            out.push('\n');
        }
        let _ = std::fs::File::create(&path).and_then(|mut f| f.write_all(out.as_bytes()));
    }

    // ----- rendering -----

    fn reset_textarea(&mut self) {
        let mut textarea = TextArea::default();
        textarea.set_cursor_line_style(Style::default());
        textarea.set_placeholder_text("enter an expression or definition (:help for commands)");
        textarea.set_block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" λ "),
        );
        self.textarea = textarea;
    }

    fn set_buffer_text(&mut self, text: &str) {
        self.reset_textarea();
        for (i, line) in text.lines().enumerate() {
            if i > 0 {
                self.textarea.insert_newline();
            }
            self.textarea.insert_str(line);
        }
        self.textarea.move_cursor(CursorMove::Bottom);
        self.textarea.move_cursor(CursorMove::End);
    }

    fn push_line(&mut self, line: Line<'static>) {
        self.scrollback.push(line);
        self.scroll_up = 0;
    }

    /// Render a one-line result in a titled box (a Jupyter-style output
    /// cell):
    ///
    /// ```text
    /// ╭─ Out[1] ─╮
    /// │ 42 : i64 │
    /// ╰──────────╯
    /// ```
    fn push_boxed(&mut self, title: &str, content: Vec<Span<'static>>) {
        let border = Style::default().fg(Color::DarkGray);
        let content_w: usize = content.iter().map(|s| s.content.chars().count()).sum();
        let head_w = title.chars().count() + 3; // "─ <title> "
        let inner_w = (content_w + 2).max(head_w + 1);

        self.push_line(Line::from(vec![
            Span::styled("╭─ ", border),
            Span::styled(
                title.to_string(),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {}╮", "─".repeat(inner_w - head_w)), border),
        ]));
        let mut mid = vec![Span::styled("│ ", border)];
        mid.extend(content);
        mid.push(Span::styled(
            format!("{}│", " ".repeat(inner_w - content_w - 1)),
            border,
        ));
        self.push_line(Line::from(mid));
        self.push_line(Line::from(Span::styled(
            format!("╰{}╯", "─".repeat(inner_w)),
            border,
        )));
    }

    fn draw(&mut self, defs: usize, opt: OptLevel) -> std::io::Result<()> {
        let scrollback = &self.scrollback;
        let scroll_up = &mut self.scroll_up;
        let textarea = &self.textarea;
        let busy = self.busy;

        self.terminal.draw(|frame| {
            let input_height = (textarea.lines().len() as u16 + 2)
                .min(frame.area().height * 2 / 5)
                .max(3);
            let [output_area, input_area, status_area] = Layout::vertical([
                Constraint::Min(1),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .areas(frame.area());

            // Scrollback pinned to the bottom, offset by the scroll position.
            // Long lines wrap, so the pinning math must count *visual* rows —
            // `line_count` applies the paragraph's own wrapping rules.
            let paragraph =
                Paragraph::new(Text::from(scrollback.clone())).wrap(Wrap { trim: false });
            let total = paragraph.line_count(output_area.width) as u16;
            let view = output_area.height;
            let max_up = total.saturating_sub(view);
            *scroll_up = (*scroll_up).min(max_up);
            let offset = max_up - *scroll_up;
            frame.render_widget(paragraph.scroll((offset, 0)), output_area);

            frame.render_widget(textarea, input_area);

            let status = Line::from(vec![
                Span::styled(
                    if busy { " evaluating… " } else { " ready " },
                    if busy {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default().fg(Color::Green)
                    },
                ),
                Span::raw(format!("│ opt: {opt:?} │ defs: {defs} ")).dark_gray(),
                Span::raw("│ Enter submit · Alt+Enter newline · Ctrl+J force · PgUp/PgDn scroll ")
                    .dark_gray(),
            ]);
            frame.render_widget(Paragraph::new(status), status_area);
        })?;
        Ok(())
    }
}

// History entries are one line each; embedded newlines/backslashes escaped.
fn escape_entry(entry: &str) -> String {
    entry.replace('\\', "\\\\").replace('\n', "\\n")
}

fn unescape_entry(entry: &str) -> String {
    let mut out = String::with_capacity(entry.len());
    let mut chars = entry.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}
