use cxx::{ExternType, type_id};

pub mod string_view;

#[derive(Copy, Clone)]
#[repr(C)]
pub enum OutputTarget {
    LLVMIR,
    ASM,
    Object,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum OptOption {
    None,
    Default,
    Aggressive,
    Size,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}

unsafe impl ExternType for OutputTarget {
    type Id = type_id!("reussir::OutputTarget");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for OptOption {
    type Id = type_id!("reussir::OptOption");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for LogLevel {
    type Id = type_id!("reussir::LogLevel");
    type Kind = cxx::kind::Trivial;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CompileOptions {
    pub target: OutputTarget,
    pub opt: OptOption,
    pub log_level: LogLevel,
    backend_log: Option<extern "C" fn(string_view::StringView, LogLevel)>,
}

unsafe impl ExternType for CompileOptions {
    type Id = type_id!("reussir::CompileOptions");
    type Kind = cxx::kind::Trivial;
}

pub extern "C" fn log<'a>(message: string_view::StringView<'a>, level: LogLevel) {
    let message_str = match message.to_str() {
        Ok(s) => s,
        Err(_) => "<non-UTF8 log message>",
    };
    match level {
        LogLevel::Error => tracing::error!("{}", message_str),
        LogLevel::Warning => tracing::warn!("{}", message_str),
        LogLevel::Info => tracing::info!("{}", message_str),
        LogLevel::Debug => tracing::debug!("{}", message_str),
        LogLevel::Trace => tracing::trace!("{}", message_str),
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            target: OutputTarget::LLVMIR,
            opt: OptOption::Default,
            log_level: LogLevel::Info,
            backend_log: Some(log),
        }
    }
}

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("Reussir/Bridge.h");

        #[namespace = "reussir"]
        type OutputTarget = super::OutputTarget;
        #[namespace = "reussir"]
        type OptOption = super::OptOption;
        #[namespace = "reussir"]
        type LogLevel = super::LogLevel;
        #[namespace = "reussir"]
        type CompileOptions = super::CompileOptions;

        #[namespace = "std"]
        #[cxx_name = "string_view"]
        type StringView<'a> = super::string_view::StringView<'a>;

        #[namespace = "reussir"]
        #[cxx_name = "compileForNativeMachine"]
        pub fn compile_for_native_machine(
            mlirTextureModule: StringView,
            sourceName: StringView,
            outputFile: StringView,
            options: CompileOptions,
        );
    }
}

pub use ffi::compile_for_native_machine;

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::Level;
    use tracing_subscriber;
    use tracing_subscriber::EnvFilter;

    #[test]
    fn test_compile_for_native_machine() {
        // Initialize tracing subscriber for logging
        let filter = EnvFilter::from_default_env();
        let level = filter
            .max_level_hint()
            .and_then(|hint| hint.into_level())
            .unwrap_or(Level::DEBUG);

        _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

        let mlir_module = string_view::StringView::new("module {}");
        let output_file = string_view::StringView::new("/tmp/output.ll");
        let source_name = string_view::StringView::new("test.mlir");

        let options = CompileOptions {
            target: OutputTarget::LLVMIR,
            opt: OptOption::Default,
            log_level: match level {
                Level::ERROR => LogLevel::Error,
                Level::WARN => LogLevel::Warning,
                Level::INFO => LogLevel::Info,
                Level::DEBUG => LogLevel::Debug,
                Level::TRACE => LogLevel::Trace,
            },
            backend_log: Some(log),
        };

        compile_for_native_machine(mlir_module, source_name, output_file, options);
    }
}
