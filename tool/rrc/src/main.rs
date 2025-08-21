use std::path::PathBuf;

use palc::Parser as CommandParser;
use reussir_bridge::{CompileOptions, LogLevel, OptOption, OutputTarget};
use reussir_core::ariadne::Source;
use reussir_core::{CGContext, Path};
use reussir_front::chumsky::Parser;
use reussir_front::lexer::Token;
use reussir_sema::populate_module;
use unicode_ident::{is_xid_continue, is_xid_start};

pub fn validate_module_prefix(prefix: &str) -> Vec<String> {
    let parts: Vec<&str> = prefix.split("::").collect();

    // Validate each part is a valid identifier
    for part in &parts {
        if part.is_empty() {
            panic!("Module prefix part cannot be empty");
        }

        // Check first character is a valid identifier start
        if !is_xid_start(part.chars().next().unwrap()) {
            panic!(
                "Module prefix part '{}' must start with a valid identifier character",
                part
            );
        }

        // Check remaining characters are valid identifier continue
        for ch in part.chars().skip(1) {
            if !is_xid_continue(ch) {
                panic!(
                    "Module prefix part '{}' contains invalid identifier character '{}'",
                    part, ch
                );
            }
        }
    }

    parts
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>()
}

#[derive(CommandParser)]
struct Options {
    /// The input file to compile. If not provided, the input will be read from stdin.
    #[arg(short, long)]
    input_file: Option<PathBuf>,
    /// The output file to write the compiled code to.
    #[arg(short, long)]
    output_file: PathBuf,
    /// The optimization level to use.
    #[arg(short = 'O', long, default_value = "default")]
    opt_level: OptOption,
    /// The log level to use.
    #[arg(short, long, default_value = "info")]
    log_level: LogLevel,
    /// module prefix
    #[arg(short, long)]
    module_prefix: Option<String>,

    /// The target to compile to.
    #[arg(short, long, default_value = "object")]
    target: OutputTarget,
}

fn main() {
    let options = Options::parse();

    let tracing_level = match options.log_level {
        LogLevel::Error => tracing::Level::ERROR,
        LogLevel::Warning => tracing::Level::WARN,
        LogLevel::Info => tracing::Level::INFO,
        LogLevel::Debug => tracing::Level::DEBUG,
        LogLevel::Trace => tracing::Level::TRACE,
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .try_init()
        .unwrap();

    // split :: in prefix and check that each of them is a valid identifier
    let module_prefix = options
        .module_prefix
        .map(|p| validate_module_prefix(&p))
        .unwrap_or_default();

    let basename = options
        .input_file
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|p| p.to_str())
        .map(|s| s.trim_end_matches(".rr"))
        .unwrap_or("main");

    let input_file = options
        .input_file
        .as_ref()
        .and_then(|p| p.to_str())
        .unwrap_or("<stdin>");

    let output_file = options.output_file.to_str().unwrap();

    let module_path = Path::new(basename.into(), module_prefix.iter().map(|s| s.into()));

    let mut ctx = reussir_core::Context::new(module_path.clone());
    let mut parser_state = reussir_front::ParserState::new(module_path.clone(), input_file);
    let source = if options.input_file.is_some() {
        std::fs::read_to_string(input_file).unwrap()
    } else {
        std::io::read_to_string(std::io::stdin()).unwrap()
    };
    let parser = reussir_front::module();
    let token_stream = Token::stream(input_file.into(), &source);
    let res = parser
        .parse_with_state(token_stream, &mut parser_state)
        .unwrap();
    let module = populate_module(&mut ctx, &res, &source);
    let module = module.unwrap();
    let mut buffer = Vec::new();
    let mut codegen = CGContext::new(Source::from(source.to_owned()), &mut buffer);
    module.codegen(&mut codegen).unwrap();
    let texture = String::from_utf8(buffer).unwrap();
    tracing::trace!("{}", texture);
    let mut compile_options = CompileOptions::default();
    compile_options.target = options.target;
    compile_options.opt = options.opt_level;
    compile_options.log_level = options.log_level;
    reussir_bridge::compile_for_native_machine(&texture, input_file, &output_file, compile_options);
}
