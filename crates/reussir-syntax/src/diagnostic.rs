use crate::syntax::SyntaxError;
use ariadne::{Color, Label, Report, ReportKind, Source};

pub fn render_syntax_errors(file_name: &str, input: &str, errors: &[SyntaxError]) -> String {
    let mut rendered = String::new();
    for error in errors {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(&render_syntax_error(file_name, input, error));
    }
    rendered
}

pub fn render_syntax_error(file_name: &str, input: &str, error: &SyntaxError) -> String {
    let mut out = Vec::new();
    let start = error.start.min(input.len());
    let end = error
        .end
        .max(start.saturating_add(1))
        .min(input.len().max(1));
    Report::build(ReportKind::Error, (file_name, start..end))
        .with_message("syntax error")
        .with_label(
            Label::new((file_name, start..end))
                .with_message(error.message.clone())
                .with_color(Color::Red),
        )
        .finish()
        .write((file_name, Source::from(input)), &mut out)
        .expect("writing Ariadne report to Vec cannot fail");
    String::from_utf8(out).expect("Ariadne rendered non-UTF-8 diagnostics")
}
