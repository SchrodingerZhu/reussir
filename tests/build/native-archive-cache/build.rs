#[allow(dead_code)]
#[path = "../../../crates/reussir-backend-sys/native_archive.rs"]
mod native_archive;

use std::env;
use std::path::PathBuf;

fn main() {
    let value = env::var("REUSSIR_CACHE_PROBE_VALUE").expect("probe value is set");
    println!("cargo:rerun-if-env-changed=REUSSIR_CACHE_PROBE_VALUE");

    cc::Build::new()
        .file("value.c")
        .define("REUSSIR_CACHE_PROBE_VALUE", Some(value.as_str()))
        .compile("reussir_cache_probe");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    native_archive::track(&out_dir, &["reussir_cache_probe"])
        .expect("failed to fingerprint cache-probe archive");
}
