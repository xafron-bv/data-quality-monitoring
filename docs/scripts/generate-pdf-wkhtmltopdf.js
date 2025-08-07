const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForServer(url, timeoutMs = 60000, intervalMs = 1000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch (_) {
      // ignore until ready
    }
    await sleep(intervalMs);
  }
  return false;
}

async function generateWithCrawler(docsRoot) {
  console.log('Generating PDF with docusaurus-wkhtmltopdf...');
  const wkArgs = [
    '--user-style-sheet print.css',
    '--enable-internal-links',
    '--enable-javascript',
    '--javascript-delay 5000',
    '--no-stop-slow-scripts',
    '--print-media-type',
    '--load-error-handling ignore'
  ].join(' ');
  const command = `npx docusaurus-wkhtmltopdf -u http://localhost:3001 --dest . --output xafron-documentation.pdf --toc --wkhtmltopdf-args "${wkArgs}"`;
  console.log('Command:', command);
  execSync(command, { stdio: 'inherit', cwd: docsRoot });
}

async function generateFromCombined(docsRoot) {
  console.log('Generating PDF from combined.html...');
  const combinedPath = path.join(docsRoot, 'build', 'combined.html');
  if (!fs.existsSync(combinedPath)) {
    throw new Error('combined.html not found');
  }
  const outPath = path.join(docsRoot, 'xafron-documentation.pdf');
  const wkhtmlcmd = [
    'wkhtmltopdf',
    '--enable-internal-links',
    '--enable-javascript',
    '--javascript-delay', '5000',
    '--no-stop-slow-scripts',
    '--print-media-type',
    '--load-error-handling', 'ignore',
    'build/combined.html',
    'xafron-documentation.pdf'
  ];
  console.log('Command:', wkhtmlcmd.join(' '));
  execSync(wkhtmlcmd.join(' '), { stdio: 'inherit', cwd: docsRoot });
  return outPath;
}

async function generatePDF() {
  console.log('Starting PDF generation...');
  
  const docsRoot = path.join(__dirname, '..');
  const buildDir = path.join(docsRoot, 'build');
  const outputDir = path.join(buildDir, 'pdf');
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Try to add wkhtmltopdf binary from npm installer to PATH if available
  try {
    const wkhtmltopdfInstaller = require('wkhtmltopdf-installer');
    if (wkhtmltopdfInstaller && wkhtmltopdfInstaller.path) {
      const wkBinDir = path.dirname(wkhtmltopdfInstaller.path);
      process.env.PATH = `${wkBinDir}:${process.env.PATH || ''}`;
      console.log('wkhtmltopdf binary added to PATH from npm installer.');
    }
  } catch (_) {
    // ignore if not installed; will rely on system wkhtmltopdf
  }

  // Build combined single HTML
  try {
    const { buildCombinedHtml } = require('./generate-combined-for-pdf');
    await buildCombinedHtml();
  } catch (e) {
    console.warn('Failed to generate combined HTML:', e.message);
  }

  // Try combined first; fall back to crawler
  let sourcePdf = null;
  try {
    sourcePdf = await generateFromCombined(docsRoot);
  } catch (e) {
    console.warn('Combined generation failed; falling back to crawler. Reason:', e.message);

    // Start the Docusaurus server for crawler fallback
    console.log('Starting Docusaurus server...');
    const serverProcess = spawn('npx', ['docusaurus', 'serve', '--port', '3001'], {
      cwd: docsRoot,
      detached: true,
      stdio: 'ignore'
    });
    serverProcess.unref();
    console.log('Waiting for server to start...');
    const serverReady = await waitForServer('http://localhost:3001');
    if (!serverReady) {
      console.error('Docusaurus server did not become ready in time.');
      process.exit(1);
    }
    try {
      await generateWithCrawler(docsRoot);
      const candidates = [
        path.join(docsRoot, 'xafron-documentation.pdf'),
        path.join(docsRoot, 'pdf', 'xafron-documentation.pdf'),
      ];
      for (const p of candidates) {
        if (fs.existsSync(p)) {
          sourcePdf = p; break;
        }
      }
    } finally {
      try {
        if (process.platform === 'win32') {
          execSync('taskkill /F /IM node.exe /FI "COMMANDLINE like *docusaurus*serve*"', { stdio: 'ignore' });
        } else {
          execSync('pkill -f "docusaurus serve"', { stdio: 'ignore' });
        }
      } catch (_) {}
    }
  }

  if (!sourcePdf || !fs.existsSync(sourcePdf)) {
    console.error('PDF generation failed.');
    process.exit(1);
  }

  const targetPdf = path.join(outputDir, 'xafron-documentation.pdf');
  fs.copyFileSync(sourcePdf, targetPdf);
  console.log(`PDF generated successfully at: ${targetPdf}`);

  // Also copy to static directory
  const staticPdfDir = path.join(docsRoot, 'static', 'pdf');
  if (!fs.existsSync(staticPdfDir)) {
    fs.mkdirSync(staticPdfDir, { recursive: true });
  }
  const staticTarget = path.join(staticPdfDir, 'xafron-documentation.pdf');
  fs.copyFileSync(sourcePdf, staticTarget);
  console.log(`PDF copied to static directory: ${staticTarget}`);
}

// Run if called directly
if (require.main === module) {
  generatePDF().catch(error => {
    console.error('PDF generation failed:', error);
    process.exit(1);
  });
}

module.exports = { generatePDF };