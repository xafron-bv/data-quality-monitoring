const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testServerReady(url, maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const { execSync } = require('child_process');
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

async function generatePDFAlternative() {
  console.log('Starting alternative PDF generation with enhanced Mermaid support...');
  
  const buildDir = path.join(__dirname, '..', 'build');
  const outputDir = path.join(buildDir, 'pdf');
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Kill any existing servers
  try {
    execSync('pkill -f "docusaurus serve" || true', { stdio: 'ignore' });
  } catch (e) {
    // Ignore errors
  }
  
  // Start the docusaurus server
  console.log('Starting Docusaurus server...');
  const serverProcess = spawn('npx', ['docusaurus', 'serve', '--port', '3001', '--host', '0.0.0.0'], {
    cwd: path.join(__dirname, '..'),
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  
  serverProcess.unref();
  
  try {
    // Wait for server to be ready
    await testServerReady('http://localhost:3001');
    
    // Additional wait for Mermaid to initialize
    console.log('Waiting additional time for Mermaid initialization...');
    await sleep(5000);
    
    // Generate PDF using wkhtmltopdf directly
    console.log('Generating PDF with wkhtmltopdf directly...');
    
    const outputPdf = path.join(outputDir, 'xafron-documentation.pdf');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    
    // Enhanced wkhtmltopdf command for better Mermaid support
    const wkhtmltopdfCmd = [
      'wkhtmltopdf',
      '--page-size', 'A4',
      '--orientation', 'Portrait',
      '--margin-top', '1in',
      '--margin-bottom', '1in',
      '--margin-left', '0.75in',
      '--margin-right', '0.75in',
      '--encoding', 'UTF-8',
      '--enable-javascript',
      '--javascript-delay', '5000', // 5 second delay for JavaScript
      '--no-stop-slow-scripts',
      '--debug-javascript',
      '--load-error-handling', 'ignore',
      '--load-media-error-handling', 'ignore',
      '--viewport-size', '1280x1024',
      '--disable-smart-shrinking',
      '--print-media-type',
      '--user-style-sheet', printCssPath,
      '--toc',
      '--toc-header-text', 'Table of Contents',
      '--toc-text-size-shrink', '0.8',
      'http://localhost:3001',
      outputPdf
    ];
    
    console.log('Running wkhtmltopdf with enhanced options...');
    console.log('Command:', wkhtmltopdfCmd.join(' '));
    
    execSync(wkhtmltopdfCmd.join(' '), {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
    
    if (fs.existsSync(outputPdf)) {
      console.log(`PDF generated successfully at: ${outputPdf}`);
      
      // Also copy to static directory
      const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
      if (!fs.existsSync(staticPdfDir)) {
        fs.mkdirSync(staticPdfDir, { recursive: true });
      }
      fs.copyFileSync(outputPdf, path.join(staticPdfDir, 'xafron-documentation.pdf'));
      console.log(`PDF copied to static directory: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    } else {
      throw new Error('PDF file was not generated');
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    
    // Provide diagnostic information
    console.error('\nDiagnostic Information:');
    try {
      console.error('wkhtmltopdf version:', execSync('wkhtmltopdf --version', { encoding: 'utf8' }));
    } catch (e) {
      console.error('wkhtmltopdf not found or not working');
    }
    
    try {
      console.error('Server status:', execSync('curl -s -I http://localhost:3001', { encoding: 'utf8' }));
    } catch (e) {
      console.error('Server not accessible');
    }
    
    throw error;
  } finally {
    // Kill the server
    try {
      execSync('pkill -f "docusaurus serve"', { stdio: 'ignore' });
    } catch (e) {
      // Server might have already stopped
    }
  }
}

// Run if called directly
if (require.main === module) {
  generatePDFAlternative().catch(error => {
    console.error('PDF generation failed:', error);
    process.exit(1);
  });
}

module.exports = { generatePDFAlternative };