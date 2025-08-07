const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testServerReady(url, maxAttempts = 20) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const result = execSync(`curl -s -o /dev/null -w "%{http_code}" ${url}`, { 
        timeout: 5000,
        encoding: 'utf8'
      });
      if (result.trim() === '200') {
        console.log(`Server is ready at ${url}`);
        return true;
      }
    } catch (error) {
      console.log(`Waiting for server... attempt ${i + 1}/${maxAttempts}`);
      await sleep(2000);
    }
  }
  throw new Error(`Server not ready after ${maxAttempts} attempts`);
}

async function generatePDF() {
  console.log('Starting PDF generation...');
  
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
    stdio: 'ignore'
  });
  
  serverProcess.unref();
  
  try {
    // Wait for server to be ready
    await testServerReady('http://localhost:3001');
    
    // Additional wait for page to fully load and render
    console.log('Waiting for page content to load...');
    await sleep(10000); // 10-second wait for Mermaid rendering
    
    // Generate PDF using wkhtmltopdf
    console.log('Generating PDF with wkhtmltopdf...');
    
    const outputPdf = path.join(outputDir, 'xafron-documentation.pdf');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    
    // wkhtmltopdf command with enhanced Mermaid support
    const wkhtmltopdfCmd = [
      'wkhtmltopdf',
      '--page-size', 'A4',
      '--margin-top', '20mm',
      '--margin-bottom', '20mm',
      '--margin-left', '15mm',
      '--margin-right', '15mm',
      '--encoding', 'UTF-8',
      '--enable-javascript',
      '--javascript-delay', '5000', // JavaScript delay for Mermaid rendering
      '--load-error-handling', 'ignore',
      '--print-media-type',
      '--user-style-sheet', printCssPath,
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
      
      // Also copy to static directory
      const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
      if (!fs.existsSync(staticPdfDir)) {
        fs.mkdirSync(staticPdfDir, { recursive: true });
      }
      fs.copyFileSync(outputPdf, path.join(staticPdfDir, 'xafron-documentation.pdf'));
      console.log(`PDF copied to static directory: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    } else {
      console.error('PDF generation failed - file not found at:', outputPdf);
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
  generatePDF().catch(error => {
    console.error('PDF generation failed:', error);
    process.exit(1);
  });
}

module.exports = { generatePDF };