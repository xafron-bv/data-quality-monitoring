const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
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
  await sleep(5000);
  
  try {
    // Generate PDF using docusaurus-wkhtmltopdf
    console.log('Generating PDF with docusaurus-wkhtmltopdf...');
    
    const command = 'npx docusaurus-wkhtmltopdf -u http://localhost:3001 --output xafron-documentation.pdf --compress --toc --wkhtmltopdf-args "--user-style-sheet print.css"';
    
    console.log('Command:', command);
    
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
      console.log(`PDF copied to static directory: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    } else {
      console.error('PDF generation failed - file not found at:', generatedPdf);
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    console.error('This might be due to missing wkhtmltopdf or ghostscript dependencies');
    
    // Create a placeholder file to prevent 404 errors
    const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
    if (!fs.existsSync(staticPdfDir)) {
      fs.mkdirSync(staticPdfDir, { recursive: true });
    }
    
    const placeholderPath = path.join(staticPdfDir, 'xafron-documentation.pdf');
    const placeholderContent = `PDF generation failed during build.
Please check the build logs for more details.
Visit the online documentation at: https://docs.xafron.nl/

Note: This requires wkhtmltopdf and ghostscript to be installed on the system.`;
    
    fs.writeFileSync(placeholderPath, placeholderContent);
    console.log('Created placeholder file at:', placeholderPath);
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