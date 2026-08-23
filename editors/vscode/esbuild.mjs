import * as esbuild from 'esbuild';

const production = process.argv.includes('--production');

await Promise.all([
  esbuild.build({
    entryPoints: ['src/extension.ts'],
    bundle: true,
    external: ['vscode'],
    format: 'cjs',
    platform: 'node',
    target: 'node20',
    outfile: 'dist/extension.js',
    sourcemap: production ? false : 'linked',
    minify: production,
    logLevel: 'info'
  }),
  esbuild.build({
    entryPoints: ['test/protocol-smoke.ts'],
    bundle: true,
    format: 'cjs',
    platform: 'node',
    target: 'node20',
    outfile: 'dist/test/protocol-smoke.cjs',
    sourcemap: production ? false : 'linked',
    minify: false,
    logLevel: 'info'
  })
]);
