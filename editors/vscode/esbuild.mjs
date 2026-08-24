import * as esbuild from 'esbuild';

const production = process.argv.includes('--production');

// The wasm-bindgen module must stay a runtime require: it locates its .wasm
// next to its own file (__dirname), which bundling would break. Each bundle
// rewrites the import specifier so the require resolves from that bundle's
// output directory back into dist/wasm/.
const externalWasm = runtimePath => ({
  name: 'external-wasm-bindings',
  setup(build) {
    build.onResolve({ filter: /\/reussir_vscode_wasm$/ }, () => ({
      path: runtimePath,
      external: true
    }));
  }
});

await Promise.all([
  esbuild.build({
    entryPoints: ['src/extension.ts'],
    bundle: true,
    external: ['vscode'],
    plugins: [externalWasm('./wasm/reussir_vscode_wasm')],
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
    plugins: [externalWasm('../wasm/reussir_vscode_wasm')],
    format: 'cjs',
    platform: 'node',
    target: 'node20',
    outfile: 'dist/test/protocol-smoke.cjs',
    sourcemap: production ? false : 'linked',
    minify: false,
    logLevel: 'info'
  })
]);
