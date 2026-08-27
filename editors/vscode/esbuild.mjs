import * as esbuild from 'esbuild';

const production = process.argv.includes('--production');

await esbuild.build({
  entryPoints: ['src/extension.ts'],
  bundle: true,
  // The wasm-bindgen module must stay a runtime require: it locates its
  // .wasm next to its own file (__dirname), which bundling would break.
  // The specifier resolves from dist/extension.js back into dist/wasm/.
  external: ['vscode', '../dist/wasm/reussir_vscode_wasm'],
  format: 'cjs',
  platform: 'node',
  target: 'node20',
  outfile: 'dist/extension.js',
  sourcemap: production ? false : 'linked',
  minify: production,
  logLevel: 'info'
});
