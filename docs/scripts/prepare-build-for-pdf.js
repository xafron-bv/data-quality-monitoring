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

function rewriteAbsoluteSiteLinksToRoot(html) {
  // Normalize docs.xafron.nl and localhost URLs to root-relative so wkhtmltopdf treats them as internal
  return html.replace(/href="(https?:\/\/[^"?#]+[^\"]*)"/g, (match, url) => {
    try {
      const u = new URL(url);
      const host = (u.hostname || '').toLowerCase();
      if (host === 'localhost' || host === '127.0.0.1' || host === 'docs.xafron.nl') {
        return `href="${u.pathname}${u.hash || ''}"`;
      }
      return match;
    } catch (_) {
      return match;
    }
  });
}

async function processHtmlFile(filePath, outputImagesDir) {
  let html = await fs.promises.readFile(filePath, 'utf8');

  // Replace Mermaid <div class="mermaid"> blocks with static SVG images (fallback for PDF)
  const replacements = [];
  const mermaidDivRegex = /<div class="mermaid">([\s\S]*?)<\/div>/g;
  let match;
  while ((match = mermaidDivRegex.exec(html)) !== null) {
    const mermaidCode = decodeHtmlEntities(match[1]);
    const id = sha1(mermaidCode);
    replacements.push({ start: match.index, end: match.index + match[0].length, code: mermaidCode, id });
  }

  // Handle <pre><code class="language-mermaid">...</code></pre>
  const mermaidPreCodeRegex = /<pre[^>]*>\s*<code[^>]*class="[^"]*language-mermaid[^"]*"[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/g;
  while ((match = mermaidPreCodeRegex.exec(html)) !== null) {
    const mermaidCode = decodeHtmlEntities(match[1]);
    const id = sha1(mermaidCode);
    replacements.push({ start: match.index, end: match.index + match[0].length, code: mermaidCode, id });
  }

  if (replacements.length > 0) {
    await ensureDir(outputImagesDir);

    let offset = 0;
    let processedHtml = html;
    for (const { start, end, code, id } of replacements) {
      const svgPath = path.join(outputImagesDir, `${id}.svg`);
      if (!fs.existsSync(svgPath)) {
        try {
          const svg = await renderMermaidToSvg(code);
          await fs.promises.writeFile(svgPath, svg, 'utf8');
        } catch (e) {
          // If rendering fails, skip replacing this diagram
          continue;
        }
      }
      const imgTag = `<img src="/mermaid/${id}.svg" alt="Mermaid diagram" />`;
      const replaceStart = start + offset;
      const replaceEnd = end + offset;
      processedHtml = processedHtml.slice(0, replaceStart) + imgTag + processedHtml.slice(replaceEnd);
      offset += imgTag.length - (end - start);
    }

    html = processedHtml;
  }

  // Normalize absolute site links to root-relative
  html = rewriteAbsoluteSiteLinksToRoot(html);

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
    await processHtmlFile(file, imagesDir);
  }

  console.log('Build post-processing complete. Mermaid fallback rendered where possible, and absolute links normalized.');
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