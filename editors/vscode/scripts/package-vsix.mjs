import { copyFile, mkdir, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(extensionRoot, '../..');
const output = path.resolve(
  process.env.REUSSIR_VSIX_OUT || path.join(repositoryRoot, 'build', 'reussir-vscode.vsix')
);

await mkdir(path.dirname(output), { recursive: true });
const stagedLicense = path.join(extensionRoot, 'LICENSE');
await copyFile(path.join(repositoryRoot, 'LICENSE'), stagedLicense);
const executable = path.join(
  extensionRoot,
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'vsce.cmd' : 'vsce'
);
let status = 1;
try {
  const packaged = spawnSync(executable, ['package', '--no-dependencies', '--out', output], {
    cwd: extensionRoot,
    env: process.env,
    stdio: 'inherit'
  });
  status = packaged.status ?? 1;
} finally {
  await rm(stagedLicense, { force: true });
}
process.exitCode = status;
