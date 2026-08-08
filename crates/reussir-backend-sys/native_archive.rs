use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

fn archive_candidates(lib_dir: &Path, archive: &str) -> [PathBuf; 3] {
    [
        lib_dir.join(format!("lib{archive}.a")),
        lib_dir.join(format!("{archive}.lib")),
        lib_dir.join(format!("lib{archive}.lib")),
    ]
}

pub fn archive_exists(lib_dir: &Path, archive: &str) -> bool {
    archive_candidates(lib_dir, archive)
        .into_iter()
        .any(|path| path.is_file())
}

fn hash_file(hasher: &mut blake3::Hasher, path: &Path) -> io::Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
    }
}

fn fingerprint(lib_dir: &Path, archives: &[&str]) -> io::Result<blake3::Hash> {
    let mut hasher = blake3::Hasher::new();
    for archive in archives {
        for path in archive_candidates(lib_dir, archive) {
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .expect("archive candidate has a file name")
                .as_encoded_bytes();
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name);
            hasher.update(&path.metadata()?.len().to_le_bytes());
            hash_file(&mut hasher, &path)?;
        }
    }
    Ok(hasher.finalize())
}

pub fn track(lib_dir: &Path, archives: &[&str]) -> io::Result<()> {
    let fingerprint = fingerprint(lib_dir, archives)?;

    // Cargo notices the archive changes below and invokes rustc again, but a
    // compiler cache can otherwise reuse the old final-link result because the
    // rustc command line only names each archive; it does not contain its
    // contents. Put the contents in rustc's arguments so that cache key changes
    // whenever any linked native archive does.
    println!("cargo:rustc-cfg=reussir_native_archive_fingerprint_{fingerprint}");

    // Only the candidates that exist, mirroring `fingerprint()`. Cargo cannot
    // stat a path that is not there and treats it as changed, so declaring all
    // three re-runs this script on *every* build — and at least two of the
    // three are always absent, on every platform (`lib<x>.a` off Windows,
    // `<x>.lib`/`lib<x>.lib` off Unix). Measured on Windows before this guard:
    // a second build with no source changes re-ran the script and relinked
    // rrc, 56s of work for nothing.
    //
    // The trade: an archive appearing where none existed at this script's last
    // run is not noticed by itself. That is fine here because CMake builds the
    // native archives before it invokes cargo, so an archive that is going to
    // exist already does — and if one is later added to `archives`, the
    // fingerprint in the emitted cfg changes and re-runs the link anyway.
    for archive in archives {
        for path in archive_candidates(lib_dir, archive) {
            if !path.is_file() {
                continue;
            }
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    Ok(())
}
