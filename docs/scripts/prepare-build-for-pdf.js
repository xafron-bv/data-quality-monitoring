const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

async function ensureDir(dirPath) {
  await fs.promises.mkdir(dirPath, { recursive: true });
}

function walkFiles(dirPath, filePredicate) {
  const result = [];
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      result.push(...walkFiles(fullPath, filePredicate));
    } else if (entry.isFile() && filePredicate(fullPath)) {
      result.push(fullPath);
    }
  }
  return result;
}

function sha1(content) {
  return crypto.createHash('sha1').update(content).digest('hex');
}

function decodeHtmlEntities(str) {
  return str
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

async function renderMermaidToSvg(mermaidCode) {
  // Use Kroki API to render Mermaid to SVG without Puppeteer
  const response = await fetch('https://kroki.io/mermaid/svg', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: mermaidCode,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Kroki render failed: ${response.status} ${response.statusText} - ${text}`);
  }
  return await response.text();
}

function isDocRoutePath(p) {
  // Ignore obvious assets
  return !/\.(css|js|png|jpg|jpeg|gif|svg|webp|ico|pdf)$/i.test(p) && !p.startsWith('/mermaid/');
}

function toDocAnchorIdFromPath(docPath) {
  // Normalize path like "/getting-started/installation/" -> "__pdf_doc_getting-started-installation"
  let normalized = docPath.replace(/index\.html$/i, '')
    .replace(/\/index\/?$/i, '/')
    .replace(/\.html$/i, '')
    .replace(/\/$/, '');
  if (!normalized.startsWith('/')) normalized = `/${normalized}`;
  const slug = normalized.replace(/[^a-z0-9]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase();
  return `__pdf_doc_${slug || 'home'}`;
}

function injectDocAnchor(html, anchorId) {
  const anchorHtml = `<a name="${anchorId}"><span style="display:none">.</span></a>`;
  const bodyOpen = html.match(/<body[^>]*>/i);
  if (bodyOpen) {
    const idx = bodyOpen.index + bodyOpen[0].length;
    return html.slice(0, idx) + anchorHtml + html.slice(idx);
  }
  return anchorHtml + html;
}

function rewriteSiteLinksToInternalAnchors(html, knownHostnames) {
  // Replace hrefs that point to site pages with internal anchors
  return html.replace(/href="([^"]+)"/g, (match, href) => {
    try {
      // Ignore javascript/mailto/tel and fragments
      if (/^(#|mailto:|tel:|javascript:)/i.test(href)) return match;

      let urlObj;
      let pathPart = href;

      // Absolute URL
      if (/^https?:\/\//i.test(href)) {
        urlObj = new URL(href);
        if (!knownHostnames.has(urlObj.hostname.toLowerCase())) return match;
        pathPart = `${urlObj.pathname}`; // ignore query; keep only hash later
      }

      // Root-relative path
      if (pathPart.startsWith('/')) {
        if (!isDocRoutePath(pathPart)) return match;
        const anchorId = toDocAnchorIdFromPath(pathPart);
        return `href="#${anchorId}"`;
      }

      // Relative links (e.g., ../foo)
      if (!/^([a-z]+:)?\//i.test(pathPart)) {
        return match;
      }

      return match;
    } catch (_) {
      return match;
    }
  });
}

function rewriteInternalLinks(html) {
  // Convert absolute site links to site-relative so wkhtmltopdf can keep them internal
  // Matches href="http(s)://host[:port]/path[?q][#h]"
  return html.replace(/href="(https?:\/\/[^"?#]+[^\"]*)"/g, (match, url) => {
    try {
      const u = new URL(url);
      const host = (u.hostname || '').toLowerCase();
      if (
        host === 'localhost' ||
        host === '127.0.0.1' ||
        host === 'docs.xafron.nl'
      ) {
        const pathWithHash = `${u.pathname}${u.hash || ''}`;
        return `href="${pathWithHash}"`;
      }
      return match;
    } catch (_) {
      return match;
    }
  });
}

function splitHeadBody(html) {
  const headMatch = html.match(/<head>[\s\S]*?<\/head>/i);
  const bodyMatch = html.match(/<body[\s\S]*<\/body>/i);
  if (!headMatch || !bodyMatch) return { head: '', body: html };
  return { head: headMatch[0], body: bodyMatch[0] };
}

function extractPagination(html) {
  const start = html.indexOf('<nav class="docusaurus-mt-lg pagination-nav');
  if (start === -1) return null;
  const end = html.indexOf('</nav>', start);
  if (end === -1) return null;
  return html.slice(start, end + '</nav>'.length);
}

function restorePagination(processedBody, originalPagination) {
  if (!originalPagination) return processedBody;
  // Replace any modified pagination nav with the original one
  return processedBody.replace(/<nav class="docusaurus-mt-lg pagination-nav[\s\S]*?<\/nav>/, originalPagination);
}

async function processHtmlFile(filePath, outputImagesDir, buildDir) {
  let html = await fs.promises.readFile(filePath, 'utf8');

  // 0) Work only on body for link rewriting, keep <head> intact
  const { head, body } = splitHeadBody(html);
  let workingBody = body || html;

  // 1) Replace Mermaid <div class="mermaid"> blocks with static SVG images
  const replacements = [];
  const mermaidDivRegex = /<div class="mermaid">([\s\S]*?)<\/div>/g;
  let match;
  while ((match = mermaidDivRegex.exec(workingBody)) !== null) {
    const mermaidCode = decodeHtmlEntities(match[1]);
    const id = sha1(mermaidCode);
    replacements.push({ start: match.index, end: match.index + match[0].length, code: mermaidCode, id });
  }

  // 2) Also handle SSR output that may still contain code fences rendered as <pre><code class="language-mermaid">...</code></pre>
  const mermaidPreCodeRegex = /<pre[^>]*>\s*<code[^>]*class="[^"]*language-mermaid[^"]*"[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/g;
  while ((match = mermaidPreCodeRegex.exec(workingBody)) !== null) {
    const mermaidCode = decodeHtmlEntities(match[1]);
    const id = sha1(mermaidCode);
    replacements.push({ start: match.index, end: match.index + match[0].length, code: mermaidCode, id });
  }

  if (replacements.length > 0) {
    await ensureDir(outputImagesDir);

    // Render and replace sequentially to avoid overwhelming the API
    let offset = 0;
    let processedHtml = workingBody;
    for (const { start, end, code, id } of replacements) {
      const svgPath = path.join(outputImagesDir, `${id}.svg`);
      if (!fs.existsSync(svgPath)) {
        const svg = await renderMermaidToSvg(code);
        await fs.promises.writeFile(svgPath, svg, 'utf8');
      }
      const imgTag = `<img src="/mermaid/${id}.svg" alt="Mermaid diagram" />`;
      const replaceStart = start + offset;
      const replaceEnd = end + offset;
      processedHtml = processedHtml.slice(0, replaceStart) + imgTag + processedHtml.slice(replaceEnd);
      offset += imgTag.length - (end - start);
    }

    workingBody = processedHtml;
  }

  // 3) Inject an internal anchor per page for cross-page linking
  const relPath = path.relative(buildDir, filePath).replace(/\\/g, '/');
  // Map .../index.html to route path
  let routePath = '/' + relPath.replace(/^index\.html$/i, '')
    .replace(/\/index\.html$/i, '/')
    .replace(/\.html$/i, '')
    .replace(/\/+/g, '/');
  const anchorId = toDocAnchorIdFromPath(routePath);
  workingBody = injectDocAnchor(workingBody, anchorId);

  // Preserve original pagination for crawler
  const originalPagination = extractPagination(workingBody);

  // 4) Rewrite absolute production/localhost links to site-relative first (body only)
  workingBody = rewriteInternalLinks(workingBody);

  // 5) Rewrite site links to internal anchors so they stay within the PDF (body only)
  const knownHostnames = new Set(['localhost', '127.0.0.1', 'docs.xafron.nl']);
  workingBody = rewriteSiteLinksToInternalAnchors(workingBody, knownHostnames);

  // 6) Restore pagination nav unchanged
  workingBody = restorePagination(workingBody, originalPagination);

  // Reassemble document
  if (head) {
    html = html.replace(/<head>[\s\S]*?<\/head>/i, head).replace(/<body[\s\S]*<\/body>/i, workingBody);
  } else {
    html = workingBody;
  }

  await fs.promises.writeFile(filePath, html, 'utf8');
}

async function main() {
  const buildDir = path.join(__dirname, '..', 'build');
  const imagesDir = path.join(buildDir, 'mermaid');

  if (!fs.existsSync(buildDir)) {
    console.error('Build directory not found. Run the Docusaurus build before this script.');
    process.exit(1);
  }

  const htmlFiles = walkFiles(buildDir, (p) => p.endsWith('.html'));
  console.log(`Post-processing ${htmlFiles.length} HTML files for PDF rendering...`);

  for (const file of htmlFiles) {
    await processHtmlFile(file, imagesDir, buildDir);
  }

  console.log('Build post-processing complete. Mermaid diagrams rendered and internal links normalized.');
}

if (require.main === module) {
  // Node 18+ has global fetch
  if (typeof fetch !== 'function') {
    console.error('Global fetch API not available in this Node runtime. Node 18+ is required.');
    process.exit(1);
  }

  main().catch((err) => {
    console.error('Error during build post-processing:', err);
    process.exit(1);
  });
}