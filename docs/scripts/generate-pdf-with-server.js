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
    // Add CSS to handle links in PDF - making them relative or removing localhost references
    const cssStyle = `
      @media print {
        /* Convert localhost links to relative links for PDF */
        a[href^="http://localhost:3001"] {
          color: #0066cc !important;
          text-decoration: underline !important;
        }
        
        /* Hide navigation links that don't work in PDF */
        .navbar, .theme-doc-sidebar-container, .pagination-nav {
          display: none !important;
        }
        
        /* Ensure content links are visible */
        article a {
          color: #0066cc !important;
          text-decoration: underline !important;
        }
        
        /* Remove link URLs from being printed after the link text */
        a[href]:after {
          content: none !important;
        }
      }
    `.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
    
    const command = `npx docs-to-pdf --initialDocURLs="http://localhost:3001/" --contentSelector="article" --paginationSelector="a.pagination-nav__link.pagination-nav__link--next" --excludeSelectors=".margin-vert--xl a,[class^='tocCollapsible'],.breadcrumbs,.theme-edit-this-page" --outputPDFFilename="xafron-documentation.pdf" --coverTitle="Xafron Documentation" --coverSub="Data Quality Detection System<br/>Version ${new Date().getFullYear()}" --pdfMargin="20,20,20,20" --cssStyle="${cssStyle}"`;
    
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