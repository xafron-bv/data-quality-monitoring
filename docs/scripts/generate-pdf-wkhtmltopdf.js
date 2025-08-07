const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testServerReady(url, maxAttempts = 20) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      execSync(`curl -s -o /dev/null -w "%{http_code}" ${url}`, { timeout: 5000 });
      console.log(`Server is ready at ${url}`);
      return true;
    } catch (error) {
      console.log(`Waiting for server... attempt ${i + 1}/${maxAttempts}`);
      await sleep(2000);
    }
  }
  throw new Error(`Server not ready after ${maxAttempts} attempts`);
}

async function generatePDF() {
  console.log('Starting PDF generation with wkhtmltopdf...');
  
  const buildDir = path.join(__dirname, '..', 'build');
  const outputDir = path.join(buildDir, 'pdf');
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Start the docusaurus server
  console.log('Starting Docusaurus server...');
  const serverProcess = spawn('npx', ['docusaurus', 'serve', '--port', '3001'], {
    cwd: path.join(__dirname, '..'),
    detached: true,
    stdio: 'ignore'
  });
  
  serverProcess.unref();
  
  // Wait for server to start
  console.log('Waiting for server to start...');
  await sleep(3000);
  
  // Test server connectivity
  await testServerReady('http://localhost:3001');
  
  try {
    // Generate PDF using direct wkhtmltopdf with Mermaid detection
    console.log('Generating PDF with Mermaid detection...');
    
    const outputPdf = path.join(__dirname, '..', 'xafron-documentation.pdf');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    const mermaidScriptPath = path.join(__dirname, 'mermaid-wait.js');
    
    // Enhanced wkhtmltopdf command for proper Mermaid detection  
    const wkhtmltopdfCmd = [
      'wkhtmltopdf',
      '--page-size', 'A4',
      '--encoding', 'UTF-8',
      '--enable-javascript',
      '--debug-javascript',
      '--load-error-handling', 'ignore',
      '--no-stop-slow-scripts',
      '--javascript-delay', '8000', // 8 second delay for Mermaid rendering
      '--window-status', 'mermaid-ready', // Wait for this window status
      '--print-media-type',
      '--user-style-sheet', printCssPath,
      '--run-script', mermaidScriptPath,
      'http://localhost:3001',
      outputPdf
    ];
    
    const command = wkhtmltopdfCmd.join(' ');
    
    console.log('Command:', command);
    
    execSync(command, {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
    
    // Check if PDF was generated
    if (fs.existsSync(outputPdf)) {
      console.log(`PDF generated successfully at: ${outputPdf}`);
      
      // Move to build directory
      const targetPdf = path.join(outputDir, 'xafron-documentation.pdf');
      if (outputPdf !== targetPdf) {
        fs.copyFileSync(outputPdf, targetPdf);
        console.log(`PDF copied to build directory: ${targetPdf}`);
      }
      
      // Also copy to static directory
      const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
      if (!fs.existsSync(staticPdfDir)) {
        fs.mkdirSync(staticPdfDir, { recursive: true });
      }
      fs.copyFileSync(targetPdf, path.join(staticPdfDir, 'xafron-documentation.pdf'));
      console.log(`PDF copied to static directory: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    } else {
      console.error('PDF generation failed - file not found at:', outputPdf);
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    console.error('This might be due to missing wkhtmltopdf or ghostscript dependencies');
    
    // Don't create a placeholder file - fail the build instead
    console.error('\nPDF generation failed! Please ensure wkhtmltopdf and ghostscript are installed.');
    console.error('On Ubuntu/Debian: sudo apt-get install wkhtmltopdf ghostscript');
    console.error('On macOS: brew install wkhtmltopdf ghostscript');
    
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