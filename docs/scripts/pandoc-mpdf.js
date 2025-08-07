const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function listMdFiles(root) {
  const files = [];
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (['node_modules', 'build', 'static', 'src', '.docusaurus'].includes(entry.name)) continue;
        walk(full);
      } else if (/\.(md|mdx)$/i.test(entry.name)) {
        files.push(full);
      }
    }
  }
  walk(root);
  return files.sort();
}

function writeCombined(root, files, outPath) {
  const parts = [];
  for (const f of files) {
    let md = fs.readFileSync(f, 'utf8');
    // Normalize internal links to headings only; let Pandoc handle TOC
    md = md.replace(/\]\((https?:[^)]+)\)/g, (m, url) => {
      try {
        const u = new URL(url);
        if (u.hostname === 'localhost' || u.hostname === '127.0.0.1' || /xafron\.nl$/.test(u.hostname)) {
          return '](#)';
        }
        return m;
      } catch {
        return m;
      }
    });
    parts.push(`\n\n<!-- ${path.relative(root, f)} -->\n\n${md}\n`);
  }
  fs.writeFileSync(outPath, parts.join('\n'), 'utf8');
}

function runPandoc(docsRoot, inputMd, outPdf) {
  const filters = ['--filter', 'mermaid-filter-kroki'];
  // Try to call pandoc available in PATH
  const cmd = [
    'pandoc', inputMd,
    '--from', 'gfm',
    '--toc', '--toc-depth=3',
    '--pdf-engine=xelatex',
    '--metadata', 'title=Xafron Documentation',
    '--output', outPdf,
    ...filters
  ].join(' ');
  console.log('Running:', cmd);
  execSync(cmd, { stdio: 'inherit', cwd: docsRoot });
}

async function main() {
  const docsRoot = path.join(__dirname, '..');
  const combinedMd = path.join(docsRoot, 'build', 'combined.md');
  const outPdf = path.join(docsRoot, 'static', 'pdf', 'xafron-documentation.pdf');
  const outDir = path.dirname(outPdf);
  if (!fs.existsSync(path.dirname(combinedMd))) {
    fs.mkdirSync(path.dirname(combinedMd), { recursive: true });
  }
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const files = listMdFiles(docsRoot);
  writeCombined(docsRoot, files, combinedMd);
  runPandoc(docsRoot, combinedMd, outPdf);
  console.log('PDF generated at:', outPdf);
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}