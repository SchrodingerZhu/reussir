mod embedded;
mod semantic;
mod server;

use async_lsp::client_monitor::ClientProcessMonitorLayer;
use async_lsp::concurrency::ConcurrencyLayer;
use async_lsp::panic::CatchUnwindLayer;
use async_lsp::server::LifecycleLayer;
use async_lsp::tracing::TracingLayer;
use tower::ServiceBuilder;
use tracing::Level;

const HELP: &str = "reussir-lsp — semantic-token language server for Reussir\n\nUSAGE:\n    reussir-lsp\n\nThe server communicates with one LSP client over stdin/stdout.\n";

#[tokio::main(flavor = "current_thread")]
async fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("-h" | "--help") => {
            print!("{HELP}");
            return;
        }
        Some("-V" | "--version") => {
            println!("reussir-lsp {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some(argument) => {
            eprintln!("reussir-lsp: unexpected argument: {argument}\n\n{HELP}");
            std::process::exit(2);
        }
        None => {}
    }

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
        std::process::exit(1);
    }
}
