mod embedded;
mod semantic;
mod server;

use std::process::ExitCode;

use async_lsp::client_monitor::ClientProcessMonitorLayer;
use async_lsp::concurrency::ConcurrencyLayer;
use async_lsp::panic::CatchUnwindLayer;
use async_lsp::server::LifecycleLayer;
use async_lsp::tracing::TracingLayer;
use palc::Parser;
use tower::ServiceBuilder;
use tracing::Level;

/// Semantic-token language server for Reussir.
///
/// The server communicates with one LSP client over stdin/stdout.
#[derive(Parser)]
#[command(name = "reussir-lsp", version)]
struct Cli {}

#[tokio::main]
async fn main() -> ExitCode {
    // `palc` represents `--help` as a parse error. Keep the conventional CLI
    // contract used by `rrc` and `rene`: help goes to stdout with exit 0,
    // while genuine usage errors go to stderr with exit 2.
    let Cli {} = match Cli::try_parse_from(std::env::args_os()) {
        Ok(cli) => cli,
        Err(err) => match err.try_into_help() {
            Ok(help) => {
                println!("{help}");
                return ExitCode::SUCCESS;
            }
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        },
    };

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    let (main_loop, _) = async_lsp::MainLoop::new_server(|client| {
        ServiceBuilder::new()
            .layer(TracingLayer::default())
            .layer(LifecycleLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::default())
            .layer(ClientProcessMonitorLayer::new(client))
            .service(server::ServerState::router())
    });

    #[cfg(unix)]
    let (stdin, stdout) = (
        async_lsp::stdio::PipeStdin::lock_tokio().expect("stdin must be a pipe"),
        async_lsp::stdio::PipeStdout::lock_tokio().expect("stdout must be a pipe"),
    );
    #[cfg(not(unix))]
    let (stdin, stdout) = (
        tokio_util::compat::TokioAsyncReadCompatExt::compat(tokio::io::stdin()),
        tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(tokio::io::stdout()),
    );

    if let Err(error) = main_loop.run_buffered(stdin, stdout).await {
        tracing::error!(%error, "language server stopped");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
