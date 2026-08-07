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

    for archive in archives {
        for path in archive_candidates(lib_dir, archive) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    Ok(())
}
