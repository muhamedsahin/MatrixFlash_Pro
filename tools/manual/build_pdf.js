// tools/manual/build_pdf.js
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const styles = require('./styles');
const coverAndToc = require('./chapters/cover_and_toc');
const ch01 = require('./chapters/ch01_hardware_microarchitecture');
const ch02 = require('./chapters/ch02_tensor_engine');
const ch03 = require('./chapters/ch03_math_and_operators');
const ch04 = require('./chapters/ch04_linear_algebra_gemm');
const ch05 = require('./chapters/ch05_decompositions_and_solvers');
const ch06 = require('./chapters/ch06_memory_engine');
const ch07 = require('./chapters/ch07_backend_and_heterogeneous');
const ch08 = require('./chapters/ch08_execution_and_cuda_graphs');
const ch09 = require('./chapters/ch09_sparse_engine');
const ch10 = require('./chapters/ch10_autograd_engine');
const ch11 = require('./chapters/ch11_random_engine');
const ch12 = require('./chapters/ch12_serialization_engine');
const ch13 = require('./chapters/ch13_api_reference');
const ch14 = require('./chapters/ch14_production_examples');
const ch15 = require('./chapters/ch15_benchmark_verification');
const ch16 = require('./chapters/ch16_ptx_and_sass_analysis');
const ch17 = require('./chapters/ch17_numerical_proofs_stability');
const ch18 = require('./chapters/ch18_enterprise_integration_deployment');

const completeHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MatrixFlash-Pro Technical Reference Manual</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <style>
${styles}
  </style>
</head>
<body>
${coverAndToc}
${ch01}
${ch02}
${ch03}
${ch04}
${ch05}
${ch06}
${ch07}
${ch08}
${ch09}
${ch10}
${ch11}
${ch12}
${ch13}
${ch14}
${ch15}
${ch16}
${ch17}
${ch18}
</body>
</html>`;

const outHtml = path.resolve(__dirname, 'manual_complete.html');
const outPdf = path.resolve(__dirname, '../../MatrixFlash_Pro_Technical_Manual.pdf');
const publicPdf = path.resolve(__dirname, '../../doc/public/MatrixFlash_Pro_Technical_Manual.pdf');
const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

fs.writeFileSync(outHtml, completeHtml, 'utf8');
console.log('Assembled HTML size:', (fs.statSync(outHtml).size / 1024).toFixed(1), 'KB');

console.log('Rendering PDF via headless Microsoft Edge...');
const args = [
  '--headless=new',
  '--disable-gpu',
  '--no-pdf-header-footer',
  `--print-to-pdf=${outPdf}`,
  `file:///${outHtml.replace(/\\/g, '/')}`
];

const res = spawnSync(edgePath, args, { encoding: 'utf8' });
if (res.status !== 0) {
  console.error('Edge execution failed:', res.stderr || res.stdout);
  process.exit(1);
}

if (!fs.existsSync(outPdf)) {
  console.error('PDF file was not created!');
  process.exit(1);
}

const pdfBuf = fs.readFileSync(outPdf);
console.log('Generated PDF size:', (pdfBuf.length / 1024).toFixed(1), 'KB');

// Detect page count
const pageMatches = pdfBuf.toString('latin1').match(/\/Type\s*\/Page[^s]/g);
const pageCount = pageMatches ? pageMatches.length : 'unknown';
console.log('>>> PDF Page Count:', pageCount, 'pages <<<');

// Copy to doc/public so visitors can download it
fs.copyFileSync(outPdf, publicPdf);
console.log('Copied to:', publicPdf);
