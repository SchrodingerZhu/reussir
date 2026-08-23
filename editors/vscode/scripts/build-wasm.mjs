import { copyFile, mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(extensionRoot, '../..');
const cargoTarget = path.resolve(process.env.CARGO_TARGET_DIR || path.join(repositoryRoot, 'target'));
const rustflags = [process.env.RUSTFLAGS, '-Cpanic=abort'].filter(Boolean).join(' ');

const result = spawnSync(
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

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}

const source = path.join(
  cargoTarget,
  'wasm32-unknown-unknown',
  'release',
  'reussir_vscode_wasm.wasm'
);
const destinationDirectory = path.join(extensionRoot, 'dist');
await mkdir(destinationDirectory, { recursive: true });
await copyFile(source, path.join(destinationDirectory, 'reussir_vscode_wasm.wasm'));
