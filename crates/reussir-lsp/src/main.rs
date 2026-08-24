// The async-lsp main loop lands with the server routing; until then the
// binary only anchors the crate so the encoding module and its tests build.
#[allow(dead_code)]
mod semantic;

fn main() {
    eprintln!("reussir-lsp: the LSP main loop is not wired up yet");
    std::process::exit(2);
}
