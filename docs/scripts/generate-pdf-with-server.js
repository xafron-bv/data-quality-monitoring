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
    // Generate PDF using docs-to-pdf with explicit Docusaurus v2 parameters
    // Added baseUrl parameter to handle link conversion
    const baseUrl = 'http://localhost:3001';
    const command = `npx docs-to-pdf --initialDocURLs="${baseUrl}/" --contentSelector="article" --paginationSelector="a.pagination-nav__link.pagination-nav__link--next" --excludeSelectors=".margin-vert--xl a,[class^='tocCollapsible'],.breadcrumbs,.theme-edit-this-page" --outputPDFFilename="xafron-documentation.pdf" --coverTitle="Xafron Documentation" --coverSub="Data Quality Detection System<br/>Version ${new Date().getFullYear()}" --pdfMargin="20,20,20,20" --baseUrl="${baseUrl}"`;
    
    console.log('Generating PDF...');
    console.log('Command:', command);
    
    // Set up environment for Puppeteer
    const env = {
      ...process.env,
      PUPPETEER_ARGS: process.env.PUPPETEER_ARGS || '--no-sandbox,--disable-setuid-sandbox',
      PUPPETEER_EXECUTABLE_PATH: process.env.PUPPETEER_EXECUTABLE_PATH || undefined
    };
    
    execSync(command, {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..'),
      env: env
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
    console.error('This might be due to missing browser dependencies in CI environment');
    
    // Create a placeholder file to prevent 404 errors
    const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
    if (!fs.existsSync(staticPdfDir)) {
      fs.mkdirSync(staticPdfDir, { recursive: true });
    }
    
    // Create a simple text file explaining the issue
    const placeholderPath = path.join(staticPdfDir, 'xafron-documentation.pdf');
    const placeholderContent = `PDF generation failed during build.
Please check the GitHub Actions logs for more details.
Visit the online documentation at: https://docs.xafron.nl/`;
    
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
    process.exit(0);
  });
}

module.exports = { generatePDF };