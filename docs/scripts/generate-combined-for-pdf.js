const fs = require('fs');
const path = require('path');

async function ensureDir(dir) {
  await fs.promises.mkdir(dir, { recursive: true });
}

function readInlineStyles(docsRoot, buildDir) {
  let styles = '';
  const printCss = path.join(docsRoot, 'print.css');
  if (fs.existsSync(printCss)) {
    styles += fs.readFileSync(printCss, 'utf8');
  }
  // Optionally include built styles if available
  const cssDir = path.join(buildDir, 'assets', 'css');
  if (fs.existsSync(cssDir)) {
    const files = fs.readdirSync(cssDir).filter(f => f.endsWith('.css'));
    for (const f of files) {
      try {
        styles += '\n' + fs.readFileSync(path.join(cssDir, f), 'utf8');
      } catch (_) {}
    }
  }
  return styles;
}

function listDocPathsFromFile(listFilePath) {
  const raw = fs.readFileSync(listFilePath, 'utf8');
  return raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((url) => {
      try {
        const u = new URL(url);
        return u.pathname;
      } catch {
        return url.startsWith('/') ? url : `/${url}`;
      }
    });
}

function mapRouteToSource(docsRoot, routePath) {
  // Map routes to source markdown files
  let p = routePath;
  if (!p.startsWith('/')) p = `/${p}`;
  if (p.endsWith('/')) {
    const candidates = [
      path.join(docsRoot, p, 'README.md'),
      path.join(docsRoot, p, 'index.md'),
      path.join(docsRoot, p.slice(0, -1) + '.md'),
      path.join(docsRoot, p, 'README.mdx'),
      path.join(docsRoot, p, 'index.mdx'),
      path.join(docsRoot, p.slice(0, -1) + '.mdx'),
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) return c;
    }
  } else {
    const candidates = [
      path.join(docsRoot, `${p}.md`),
      path.join(docsRoot, `${p}.mdx`),
      path.join(docsRoot, p, 'index.md'),
      path.join(docsRoot, p, 'README.md'),
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) return c;
    }
  }
  return null;
}

function sha1(content) {
  const crypto = require('crypto');
  return crypto.createHash('sha1').update(content).digest('hex');
}

async function renderMermaidToSvg(mermaidCode) {
  const res = await fetch('https://kroki.io/mermaid/svg', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: mermaidCode,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Kroki failed: ${res.status} ${res.statusText} - ${t}`);
  }
  return await res.text();
}

function extractTitleFromMarkdown(md) {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : '';
}

function convertMarkdownToHtml(marked, md) {
  return marked.parse(md);
}

function routeToAnchor(route) {
  return `__pdf_doc_${route.replace(/[^a-z0-9]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'home'}`;
}

function rewriteLinksToAnchors(html, routes) {
  // Rewrite root-relative or relative links that match known routes to anchors
  const set = new Set(routes.map(r => r.replace(/\/$/, '')));
  return html.replace(/href=\"([^\"]+)\"/g, (m, href) => {
    if (/^(mailto:|tel:|javascript:|https?:)/i.test(href)) return m;
    // normalize
    let p = href;
    if (!p.startsWith('/')) {
      // relative like ../path or path
      p = '/' + p.replace(/^\.\//, '');
    }
    p = p.replace(/\/$/, '');
    if (set.has(p)) {
      return `href="#${routeToAnchor(p)}"`;
    }
    return m;
  });
}

async function buildCombinedHtml() {
  const docsRoot = path.join(__dirname, '..');
  const buildDir = path.join(docsRoot, 'build');
  const listFile = path.join(docsRoot, 'localhost.txt');
  const outputDir = buildDir;
  const imagesDir = path.join(buildDir, 'mermaid');
  await ensureDir(imagesDir);

  if (!fs.existsSync(buildDir)) {
    console.error('Build directory not found. Run the Docusaurus build first.');
    process.exit(1);
  }

  // Lazy import marked
  let marked;
  try {
    marked = require('marked');
  } catch (e) {
    console.error('Missing dependency: marked. Install it with `npm i -D marked`.');
    process.exit(1);
  }

  const routes = listDocPathsFromFile(listFile);
  const inlineStyles = readInlineStyles(docsRoot, buildDir);

  let bodyHtml = '';
  bodyHtml += '<nav><h1>Documentation</h1><ol>';
  for (const route of routes) {
    const anchor = routeToAnchor(route);
    bodyHtml += `<li><a href="#${anchor}">${route}</a></li>`;
  }
  bodyHtml += '</ol></nav>';

  for (const route of routes) {
    const srcPath = mapRouteToSource(docsRoot, route);
    if (!srcPath) continue;
    let md = fs.readFileSync(srcPath, 'utf8');

    // Replace Mermaid fences with image tags, render and save SVG
    const mermaidFence = /```mermaid\n([\s\S]*?)```/g;
    let match;
    const renders = [];
    while ((match = mermaidFence.exec(md)) !== null) {
      const code = match[1];
      const id = sha1(code);
      renders.push({ code, id, start: match.index, end: match.index + match[0].length });
    }
    for (const r of renders) {
      const svgPath = path.join(imagesDir, `${r.id}.svg`);
      if (!fs.existsSync(svgPath)) {
        try {
          const svg = await renderMermaidToSvg(r.code);
          fs.writeFileSync(svgPath, svg, 'utf8');
        } catch (_) {}
      }
    }
    let offset = 0;
    for (const r of renders) {
      const imgTag = `\n<img src="/mermaid/${r.id}.svg" alt="Mermaid diagram" />\n`;
      const s = r.start + offset;
      const e = r.end + offset;
      md = md.slice(0, s) + imgTag + md.slice(e);
      offset += imgTag.length - (r.end - r.start);
    }

    const title = extractTitleFromMarkdown(md);
    const anchor = routeToAnchor(route);
    let sectionHtml = convertMarkdownToHtml(marked, md);
    sectionHtml = rewriteLinksToAnchors(sectionHtml, routes);

    bodyHtml += `\n<section>\n<a name="${anchor}"></a>\n<h1>${title || route}</h1>\n${sectionHtml}\n</section>\n`;
  }

  const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Xafron Documentation (Combined)</title>
<style>
${inlineStyles}
</style>
</head>
<body>
${bodyHtml}
</body>
</html>`;

  const outPath = path.join(outputDir, 'combined.html');
  fs.writeFileSync(outPath, html, 'utf8');
  console.log('Combined HTML generated at:', outPath);
}

if (require.main === module) {
  if (typeof fetch !== 'function') {
    console.error('Global fetch API not available. Node 18+ required.');
    process.exit(1);
  }
  buildCombinedHtml().catch((e) => {
    console.error('Error generating combined HTML:', e);
    process.exit(1);
  });
}

module.exports = { buildCombinedHtml };