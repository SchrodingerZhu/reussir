import { mkdir } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(extensionRoot, '../..');
const cargoTarget = path.resolve(process.env.CARGO_TARGET_DIR || path.join(repositoryRoot, 'target'));
const rustflags = [process.env.RUSTFLAGS, '-Cpanic=abort'].filter(Boolean).join(' ');

// The JS glue wasm-bindgen emits only loads modules produced by the same
// crate version, so fail fast on version skew instead of at runtime. The
// workspace pins wasm-bindgen exactly (Cargo.toml) and flake.nix supplies
// the matching CLI.
const lock = readFileSync(path.join(repositoryRoot, 'Cargo.lock'), 'utf8');
const lockedVersion = lock.match(/name = "wasm-bindgen"\nversion = "([^"]+)"/)?.[1];
const cli = spawnSync('wasm-bindgen', ['--version'], { encoding: 'utf8' });
if (cli.status !== 0) {
  console.error('wasm-bindgen CLI not found on PATH; enter the dev shell (flake.nix provides it)');
  process.exit(1);
}
const cliVersion = cli.stdout.trim().split(' ').pop();
if (lockedVersion && cliVersion !== lockedVersion) {
  console.error(`wasm-bindgen CLI ${cliVersion} does not match the Cargo.lock pin ${lockedVersion}`);
  process.exit(1);
}

const build = spawnSync(
  'cargo',
  [
    'build',
    '--locked',
    '--target',
    'wasm32-unknown-unknown',
    '--release',
    '--package',
    'reussir-vscode-wasm'
  ],
  {
    cwd: repositoryRoot,
    env: { ...process.env, CARGO_TARGET_DIR: cargoTarget, RUSTFLAGS: rustflags },
    stdio: 'inherit'
  }
);

if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

const source = path.join(
  cargoTarget,
  'wasm32-unknown-unknown',
  'release',
  'reussir_vscode_wasm.wasm'
);
const destinationDirectory = path.join(extensionRoot, 'dist', 'wasm');
await mkdir(destinationDirectory, { recursive: true });

const bindgen = spawnSync(
  'wasm-bindgen',
  ['--target', 'nodejs', '--out-dir', destinationDirectory, '--out-name', 'reussir_vscode_wasm', source],
  { stdio: 'inherit' }
);

if (bindgen.status !== 0) {
  process.exit(bindgen.status ?? 1);
}
