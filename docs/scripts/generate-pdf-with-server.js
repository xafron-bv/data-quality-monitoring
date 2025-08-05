const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function generatePDF() {
  console.log('Starting PDF generation process...');
  
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
  await sleep(5000);
  
  try {
    // Generate PDF using docs-to-pdf
    const command = `npx docs-to-pdf docusaurus --initialDocURLs="http://localhost:3001/README" --outputPDFFilename="xafron-documentation.pdf" --coverTitle="Xafron Documentation" --coverSub="Data Quality Detection System<br/>Version ${new Date().getFullYear()}" --disableTOC=false --pdfMargin="20,20,20,20"`;
    
    console.log('Generating PDF...');
    
    execSync(command, {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
    
    // Move the generated PDF to the correct location
    const generatedPdf = path.join(__dirname, '..', 'xafron-documentation.pdf');
    const targetPdf = path.join(outputDir, 'xafron-documentation.pdf');
    
    if (fs.existsSync(generatedPdf)) {
      fs.renameSync(generatedPdf, targetPdf);
      console.log(`PDF generated successfully at: ${targetPdf}`);
      
      // Also copy to static directory
      const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
      if (!fs.existsSync(staticPdfDir)) {
        fs.mkdirSync(staticPdfDir, { recursive: true });
      }
      fs.copyFileSync(targetPdf, path.join(staticPdfDir, 'xafron-documentation.pdf'));
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
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
    process.exit(0);
  });
}

module.exports = { generatePDF };