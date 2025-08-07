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

function filePathToRoute(root, filePath) {
  const rel = path.relative(root, filePath).replace(/\\/g, '/');
  // Remove leading docs/ if present (we are already in docs root)
  let p = rel;
  // Drop extension
  p = p.replace(/\.(md|mdx)$/i, '');
  // README or index represent folder index
  if (p.endsWith('/README') || p.endsWith('/index')) {
    p = p.replace(/\/(README|index)$/i, '/');
  }
  if (p === 'README' || p === 'index') p = '';
  if (!p.startsWith('/')) p = '/' + p;
  // Collapse //
  p = p.replace(/\/+/g, '/');
  return p;
}

function routeToAnchor(route) {
  return `sec_${route.replace(/[^a-z0-9]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'home'}`;
}

async function renderMermaidPngBase64(code) {
  const res = await fetch('https://kroki.io/mermaid/png', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: code,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Kroki PNG failed: ${res.status} ${res.statusText} - ${t}`);
  }
  const ab = await res.arrayBuffer();
  const buf = Buffer.from(ab);
  return buf.toString('base64');
}

async function transformMarkdown(md, anchor, routeToAnchorMap) {
  // Replace Mermaid code fences with embedded PNG data URIs
  const fence = /```mermaid\n([\s\S]*?)```/g;
  let out = '';
  let lastIdx = 0;
  let m;
  while ((m = fence.exec(md)) !== null) {
    out += md.slice(lastIdx, m.index);
    const code = m[1];
    try {
      const b64 = await renderMermaidPngBase64(code);
      out += `\n\n![Mermaid](data:image/png;base64,${b64})\n\n`;
    } catch (e) {
      out += `\n\n${'```'}mermaid\n${code}\n${'```'}\n\n`;
    }
    lastIdx = m.index + m[0].length;
  }
  out += md.slice(lastIdx);

  // Add id to first H1
  out = out.replace(/^(#\s+)(.+)$/m, (match, hashes, title) => {
    // Avoid duplicating id if already present
    if (/\{#.+\}\s*$/.test(match)) return match;
    return `${hashes}${title} {#${anchor}}`;
  });

  // Rewrite root-relative links to anchors when possible
  out = out.replace(/\]\((\/[^)#\s]+)\)/g, (match, href) => {
    const a = routeToAnchorMap.get(href.replace(/\/$/, ''));
    if (a) return `](#${a})`;
    return match;
  });

  // Normalize absolute links for localhost/docs host to plain text anchors
  out = out.replace(/\]\((https?:[^)]+)\)/g, (match, url) => {
    try {
      const u = new URL(url);
      if (u.hostname === 'localhost' || u.hostname === '127.0.0.1' || /xafron\.nl$/i.test(u.hostname)) {
        const pathOnly = u.pathname.replace(/\/$/, '');
        const a = routeToAnchorMap.get(pathOnly);
        return a ? `](#${a})` : '](#)';
      }
      return match;
    } catch {
      return match;
    }
  });

  return out;
}

async function writeCombined(root, files, outPath) {
  // Build route->anchor map
  const routeToAnchorMap = new Map();
  const fileRouteAnchor = new Map();
  for (const f of files) {
    const route = filePathToRoute(root, f).replace(/\/$/, '');
    const anchor = routeToAnchor(route);
    routeToAnchorMap.set(route, anchor);
    fileRouteAnchor.set(f, { route, anchor });
  }

  const parts = [];
  for (const f of files) {
    let md = fs.readFileSync(f, 'utf8');
    const { anchor } = fileRouteAnchor.get(f);
    md = await transformMarkdown(md, anchor, routeToAnchorMap);
    parts.push(`\n\n<!-- ${path.relative(root, f)} -->\n\n${md}\n`);
  }
  fs.writeFileSync(outPath, parts.join('\n'), 'utf8');
}

function runPandoc(docsRoot, inputMd, outPdf) {
  const cmd = [
    'pandoc',
    `'${inputMd}'`,
    '--from', 'gfm',
    '--toc', '--toc-depth=3',
    '--pdf-engine=xelatex',
    `--metadata=title:"Xafron Documentation"`,
    '--output', `'${outPdf}'`
  ].join(' ');
  console.log('Running:', cmd);
  execSync(cmd, { stdio: 'inherit', cwd: docsRoot, shell: '/bin/bash' });
}

async function main() {
  if (typeof fetch !== 'function') {
    console.error('Global fetch API not available. Node 18+ required.');
    process.exit(1);
  }
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
  await writeCombined(docsRoot, files, combinedMd);
  runPandoc(docsRoot, combinedMd, outPdf);
  console.log('PDF generated at:', outPdf);

  // Also copy into the built site so it is deployed
  const buildPdfDir = path.join(docsRoot, 'build', 'pdf');
  const buildPdfPath = path.join(buildPdfDir, 'xafron-documentation.pdf');
  try {
    fs.mkdirSync(buildPdfDir, { recursive: true });
    fs.copyFileSync(outPdf, buildPdfPath);
    console.log('PDF copied to build output at:', buildPdfPath);
  } catch (err) {
    console.warn('Warning: failed to copy PDF into build directory:', err.message);
  }
}

if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}