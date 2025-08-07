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

async function generatePDF() {
  console.log('Starting PDF generation with wkhtmltopdf...');
  
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
  
  // Start the docusaurus server
  console.log('Starting Docusaurus server...');
  const serverProcess = spawn('npx', ['docusaurus', 'serve', '--port', '3001'], {
    cwd: docsRoot,
    detached: true,
    stdio: 'ignore'
  });
  
  serverProcess.unref();
  
  // Wait for server to start (poll HTTP rather than fixed sleep)
  console.log('Waiting for server to start...');
  const serverReady = await waitForServer('http://localhost:3001');
  if (!serverReady) {
    console.error('Docusaurus server did not become ready in time.');
    process.exit(1);
  }
  
  try {
    // Generate PDF using docusaurus-wkhtmltopdf
    console.log('Generating PDF with docusaurus-wkhtmltopdf...');
    
    const command = 'npx docusaurus-wkhtmltopdf -u http://localhost:3001 --dest . --output xafron-documentation.pdf --toc --wkhtmltopdf-args "--user-style-sheet print.css"';
    
    console.log('Command:', command);
    
    execSync(command, {
      stdio: 'inherit',
      cwd: docsRoot
    });
    
    // Determine where the generated PDF landed
    const candidatePaths = [
      path.join(docsRoot, 'xafron-documentation.pdf'),
      path.join(docsRoot, 'pdf', 'xafron-documentation.pdf'),
    ];

    let sourcePdf = null;
    for (const p of candidatePaths) {
      if (fs.existsSync(p)) {
        sourcePdf = p;
        break;
      }
    }

    if (sourcePdf) {
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
    } else {
      console.error('PDF generation failed - file not found at any expected location:', candidatePaths.join(', '));
      process.exit(1);
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    console.error('This might be due to missing wkhtmltopdf dependency');
    
    // Don't create a placeholder file - fail the build instead
    console.error('\nPDF generation failed! Please ensure wkhtmltopdf is available.');
    console.error('Optionally add devDependency: wkhtmltopdf-installer');
    
    // Exit with error code to fail the build
    process.exit(1);
  } finally {
    // Kill the server
    try {
      if (process.platform === 'win32') {
        execSync('taskkill /F /IM node.exe /FI "COMMANDLINE like *docusaurus*serve*"', { stdio: 'ignore' });
      } else {
        execSync('pkill -f "docusaurus serve"', { stdio: 'ignore' });
      }
    } catch (e) {
      // Server might have already stopped
    }
  }
}

// Run if called directly
if (require.main === module) {
  generatePDF().catch(error => {
    console.error('PDF generation failed:', error);
    process.exit(1);
  });
}

module.exports = { generatePDF };