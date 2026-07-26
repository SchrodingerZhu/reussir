//! Toolchain resolution and the driver's own link steps: locating the rustc
//! that links (and the bundled lld to pin on ELF), the runtime archive and
//! polyffi library directories, the scratch archive members a packaged or
//! linked target collects, and [`link_product`] — the rustc invocation that
//! turns objects plus the runtime into an executable or dynamic library.

use std::path::{Path, PathBuf};

use reussir_backend::llvm::PolyffiPaths;

use crate::write_archive;

use super::cli::Cli;
use super::stage::{Lto, RuntimeLinkage, Stage};

/// The per-unit intermediates a packaged or linked target consumes — archive
/// members or link inputs — in a scratch directory that lives until the
/// deliverable is written.
///
/// The members are intermediates, not artifacts: naming them after the
/// product (`libfoo.0.o`) keeps the archive's member names — and so linker
/// diagnostics — meaningful, while the directory's removal on drop keeps
/// them out of the user's output directory.
pub(crate) struct ScratchMembers {
    dir: tempfile::TempDir,
    stem: String,
    extension: &'static str,
    members: Vec<PathBuf>,
}

impl ScratchMembers {
    pub(crate) fn new(output: &Path, lto: Lto) -> Result<Self, String> {
        let dir = tempfile::Builder::new()
            .prefix("rrc-scratch-")
            .tempdir()
            .map_err(|e| format!("cannot create a scratch directory: {e}"))?;
        let stem = output
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "lib".to_owned());
        Ok(ScratchMembers {
            dir,
            stem,
            extension: if lto == Lto::None { "o" } else { "bc" },
            members: Vec::new(),
        })
    }

    /// The scratch directory itself, for the link step's generated sources.
    pub(crate) fn dir(&self) -> &Path {
        self.dir.path()
    }

    /// Reserve and return the path of the next member.
    pub(crate) fn next_member(&mut self) -> PathBuf {
        let path = self.dir.path().join(format!(
            "{}.{}.{}",
            self.stem,
            self.members.len(),
            self.extension
        ));
        self.members.push(path.clone());
        path
    }

    /// Archive the members into `output`, then drop the scratch directory.
    pub(crate) fn finish(self, output: &Path, triple: &str) -> Result<(), String> {
        write_archive(output, &self.members, triple)
    }
}

/// The launcher a Reussir executable is built around, embedded at compile
/// time: a real Rust `bin` whose `main` calls `__reussir_main`, so rustc
/// emits the C entry that runs `lang_start` and the program starts under the
/// full Rust runtime. See the file's own docs for why nothing less will do.
pub(crate) const LAUNCHER_SOURCE: &str = include_str!("../../../reussir-rt/launcher/main.rs");

/// Link `--emit executable`/`dynlib`: compile the embedded launcher (or an
/// empty cdylib shim) with rustc, handing it the program's objects, the
/// runtime, and the platform's system libraries — the link line
/// `tests/integration/frontend/main_attribute.rr` used to spell by hand.
///
/// rustc rather than a C toolchain because the launcher *is* Rust — that is
/// how the program gets the real Rust runtime — and because rustc already
/// knows each target's linker and system-library baseline.
pub(crate) fn link_product(
    cli: &Cli,
    target: Stage,
    triple: &str,
    exports: Option<&[String]>,
    scratch: &ScratchMembers,
    output: &Path,
) -> Result<(), String> {
    let msvc = triple.contains("-windows") && triple.contains("msvc");
    let apple = triple.contains("-apple");

    // An executable needs the entry point; say so up front when the frontend
    // recorded the surface, rather than as a linker's undefined-symbol error.
    // (`None` — a backend-stage input — leaves the check to the linker.)
    let main_symbol = reussir_core::semi::Elaborator::MAIN_SYMBOL;
    if target == Stage::Executable
        && let Some(exports) = exports
        && !exports.iter().any(|e| e == main_symbol)
    {
        return Err(format!(
            "an executable needs an entry point: no function is marked `#[main]` \
             (and no trampoline exports `{main_symbol}`)"
        ));
    }

    let rustc = resolve_link_rustc(cli)?;
    let libdirs = runtime_libdirs(cli)?;
    let linkage = cli.runtime_linkage.unwrap_or(RuntimeLinkage::Static);

    let mut cmd = std::process::Command::new(&rustc);
    cmd.arg("--edition").arg("2024");
    match target {
        Stage::Executable => {
            let launcher = scratch.dir().join("launcher.rs");
            std::fs::write(&launcher, LAUNCHER_SOURCE)
                .map_err(|e| format!("cannot write the launcher source: {e}"))?;
            cmd.arg(launcher);
        }
        Stage::Dynlib => {
            // The shim gives rustc a crate to build the shared library
            // around; every symbol of substance arrives through the link
            // inputs below.
            let shim = scratch.dir().join("shim.rs");
            std::fs::write(&shim, "")
                .map_err(|e| format!("cannot write the cdylib shim source: {e}"))?;
            cmd.arg(shim).arg("--crate-type").arg("cdylib");
        }
        _ => unreachable!("link_product only links executable/dynlib"),
    }
    cmd.arg("-o").arg(output);
    if let Some(linker) = &cli.linker {
        if !linker.is_file() {
            return Err(format!("--linker `{}` does not exist", linker.display()));
        }
        cmd.arg(format!("-Clinker={}", linker.display()));
    }
    for member in &scratch.members {
        cmd.arg(format!("-Clink-arg={}", member.display()));
    }

    // The runtime. The static archive is named by *path*, not `-l static=`:
    // rustc lowers the latter to a bare `-l` on ELF and Mach-O, and the
    // linker then prefers a shared library sitting in the same directory
    // (see main_attribute.rr's history for the load-time failure that buys).
    match linkage {
        RuntimeLinkage::Static => {
            let archive = if msvc {
                "reussir_rt.lib"
            } else {
                "libreussir_rt.a"
            };
            cmd.arg(format!(
                "-Clink-arg={}",
                find_in_libdirs(&libdirs, archive)?.display()
            ));
            // What the archive's own code needs beyond what rustc links
            // anyway — invisible to rustc inside a *native* archive. (The
            // shared runtime records its own dependencies.)
            let sys_libs: &[&str] = if apple {
                &["framework=Security", "framework=CoreFoundation"]
            } else if msvc {
                &["ntdll", "ws2_32", "advapi32", "bcrypt", "userenv"]
            } else {
                &[]
            };
            for lib in sys_libs {
                cmd.arg("-l").arg(lib);
            }
        }
        RuntimeLinkage::Dynamic => {
            if msvc {
                // The DLL is linked through its import library, whose
                // `reussir_rt.dll.lib` name `-l dylib=` would not find (it
                // would find the *static* `reussir_rt.lib` instead).
                cmd.arg(format!(
                    "-Clink-arg={}",
                    find_in_libdirs(&libdirs, "reussir_rt.dll.lib")?.display()
                ));
            } else {
                let shared = if apple {
                    "libreussir_rt.dylib"
                } else {
                    "libreussir_rt.so"
                };
                // Presence check only — the link itself goes through `-L`/`-l`
                // so the product records the library's soname, not a path.
                find_in_libdirs(&libdirs, shared)?;
                for dir in &libdirs {
                    cmd.arg("-L").arg(dir);
                }
                cmd.arg("-l").arg("dylib=reussir_rt");
            }
        }
    }

    // A dynamic library's exports: rustc's cdylib machinery localizes
    // everything the *crate* does not export — which is everything here, the
    // shim being empty — so the trampoline surface is spelled to the linker
    // directly, in each platform's dialect.
    if target == Stage::Dynlib {
        let Some(exports) = exports else {
            unreachable!("dynlib from a backend-stage input is refused up front");
        };
        if msvc {
            // COFF exports nothing by default; /EXPORT also grows the
            // import library the consumer links.
            for export in exports {
                cmd.arg(format!("-Clink-arg=/EXPORT:{export}"));
            }
        } else if apple {
            // ld64 takes the union of every -exported_symbols_list given.
            let list = scratch.dir().join("exports.list");
            let text: String = exports.iter().map(|e| format!("_{e}\n")).collect();
            std::fs::write(&list, text)
                .map_err(|e| format!("cannot write the export list: {e}"))?;
            cmd.arg(format!(
                "-Clink-arg=-Wl,-exported_symbols_list,{}",
                list.display()
            ));
        } else if !exports.is_empty() {
            // A second version script unions with the `local: *` one rustc
            // passes. That union semantics is lld's: GNU bfd rejects the
            // second script outright ("anonymous version tag cannot be
            // combined with other version tags"), and bfd is what `cc` picks
            // on the Linux targets where lld is not yet rustc's default
            // (aarch64). Steer this link to the toolchain's bundled rust-lld
            // — the same linker the x86_64 default resolves to — through the
            // stable `cc` arguments rustc's own switch expands to, since the
            // rustc driving the link need not be a nightly (`rene` hands rrc
            // whatever toolchain its environment resolves, and the opt-in
            // `-Clink-self-contained=+linker` is nightly-gated). A rustc
            // shipped without the component falls back to a system lld.
            if triple.contains("-linux") {
                if let Some(gcc_ld) = bundled_lld_dir(&rustc) {
                    cmd.arg(format!("-Clink-arg=-B{}", gcc_ld.display()));
                }
                cmd.arg("-Clink-arg=-fuse-ld=lld");
            }
            let script = scratch.dir().join("exports.ver");
            let mut text = String::from("{ global:\n");
            for export in exports {
                text.push_str("  ");
                text.push_str(export);
                text.push_str(";\n");
            }
            text.push_str("};\n");
            std::fs::write(&script, text)
                .map_err(|e| format!("cannot write the export version script: {e}"))?;
            cmd.arg(format!(
                "-Clink-arg=-Wl,--version-script={}",
                script.display()
            ));
        }
    }

    for arg in &cli.link_arg {
        cmd.arg(format!("-Clink-arg={arg}"));
    }

    // rustc's diagnostics stream straight through to stderr.
    let status = cmd
        .status()
        .map_err(|e| format!("cannot run `{rustc}` for the link step: {e}"))?;
    if !status.success() {
        return Err(format!(
            "linking `{}` failed: `{rustc}` exited with {status}",
            output.display()
        ));
    }
    Ok(())
}

/// The directory of the toolchain's bundled rust-lld shims (`gcc-ld`, next
/// to the target libdir), asked of the rustc driving the link. `None` when
/// that distribution ships no rust-lld component.
pub(crate) fn bundled_lld_dir(rustc: &str) -> Option<PathBuf> {
    let out = std::process::Command::new(rustc)
        .args(["--print", "target-libdir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let libdir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let dir = libdir.parent()?.join("bin").join("gcc-ld");
    dir.is_dir().then_some(dir)
}

/// The rustc driving the link step: `--polyffi-rust-path`, then the
/// `REUSSIR_RUSTC` environment variable, then `rustc` on `PATH` — the same
/// order the polyffi texture compiles resolve in.
pub(crate) fn resolve_link_rustc(cli: &Cli) -> Result<String, String> {
    if let Some(path) = &cli.polyffi_rust_path {
        return resolve_rustc(path);
    }
    if let Some(env) = std::env::var_os("REUSSIR_RUSTC") {
        let path = PathBuf::from(&env);
        if !path.is_file() {
            return Err(format!("REUSSIR_RUSTC `{}` does not exist", path.display()));
        }
        return Ok(path.to_string_lossy().into_owned());
    }
    resolve_rustc(Path::new("rustc")).map_err(|_| {
        "cannot find `rustc` for the link step: pass --polyffi-rust-path or set \
         REUSSIR_RUSTC"
            .to_owned()
    })
}

/// The directories searched for the runtime library: `--polyffi-libdir`, then
/// the path-separated `REUSSIR_RUSTC_DEPS` environment variable.
pub(crate) fn runtime_libdirs(cli: &Cli) -> Result<Vec<PathBuf>, String> {
    if !cli.polyffi_libdir.is_empty() {
        return Ok(cli.polyffi_libdir.clone());
    }
    if let Some(env) = std::env::var_os("REUSSIR_RUSTC_DEPS") {
        let dirs: Vec<PathBuf> = std::env::split_paths(&env).collect();
        if !dirs.is_empty() {
            return Ok(dirs);
        }
    }
    Err(
        "cannot locate the Reussir runtime for the link step: pass --polyffi-libdir \
         or set REUSSIR_RUSTC_DEPS"
            .to_owned(),
    )
}

/// The first libdir containing `name`, or an error naming everywhere it
/// looked.
pub(crate) fn find_in_libdirs(libdirs: &[PathBuf], name: &str) -> Result<PathBuf, String> {
    for dir in libdirs {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "cannot find `{name}` in {}",
        libdirs
            .iter()
            .map(|d| format!("`{}`", d.display()))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The back leg for one module, writing to `output` (per-unit paths in a
/// partitioned build; `-o` itself otherwise).
#[allow(clippy::too_many_arguments)]
/// Resolve the explicit polyffi toolchain locations from the command line.
/// Both are validated here so a typo fails with a clear driver diagnostic
/// instead of a texture-compile error deep in the pipeline.
pub(crate) fn polyffi_paths(cli: &Cli) -> Result<PolyffiPaths, String> {
    let rust_path = cli
        .polyffi_rust_path
        .as_deref()
        .map(resolve_rustc)
        .transpose()?;
    let libdirs = cli
        .polyffi_libdir
        .iter()
        .map(|dir| {
            if !dir.is_dir() {
                return Err(format!(
                    "--polyffi-libdir `{}` is not a directory",
                    dir.display()
                ));
            }
            Ok(dir.to_string_lossy().into_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PolyffiPaths { rust_path, libdirs })
}

/// A bare `--polyffi-rust-path` name (no separator) searches `PATH`; anything
/// else is used as given. Either way the executable must exist.
pub(crate) fn resolve_rustc(path: &Path) -> Result<String, String> {
    let is_bare = path.components().count() == 1 && !path.is_absolute();
    if is_bare && !path.exists() {
        let name = path.as_os_str().to_owned();
        let found = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths).find_map(|dir| {
                let candidate = dir.join(&name);
                if candidate.is_file() {
                    return Some(candidate);
                }
                if cfg!(windows) {
                    let with_exe = dir.join(format!("{}.exe", name.to_string_lossy()));
                    if with_exe.is_file() {
                        return Some(with_exe);
                    }
                }
                None
            })
        });
        if let Some(found) = found {
            return Ok(found.to_string_lossy().into_owned());
        }
    }
    if !path.is_file() {
        return Err(format!(
            "--polyffi-rust-path `{}` does not exist",
            path.display()
        ));
    }
    Ok(path.to_string_lossy().into_owned())
}
