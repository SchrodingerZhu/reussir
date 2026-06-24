//! Generates the textual-IR parser from the lalrpop grammar at build time.

fn main() {
    lalrpop::process_root().unwrap();
}
