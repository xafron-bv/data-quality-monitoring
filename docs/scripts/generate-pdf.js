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

function getAllDocumentationPages() {
  // Extract all page paths from the sidebar configuration
  const sidebarPath = path.join(__dirname, '..', 'sidebars.js');
  const sidebarContent = fs.readFileSync(sidebarPath, 'utf8');
  
  // Parse the sidebar to extract all page paths
  const pages = [
    "README", // Homepage
    // Getting Started
    "getting-started/",
    "getting-started/installation",
    "getting-started/basic-usage", 
    "getting-started/quick-start",
    // Architecture
    "architecture/overview",
    "architecture/detection-methods",
    // User Guides
    "user-guides/running-detection",
    "user-guides/analyzing-results",
    "user-guides/optimization",
    // Reference
    "reference/cli",
    "reference/configuration",
    "reference/interfaces",
    // Development
    "development/adding-fields",
    "development/contributing",
    // Deployment
    "deployment/examples"
  ];
  
  return pages;
}

async function generatePDF() {
  console.log('Starting comprehensive PDF generation...');
  
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
    
    // Get all documentation pages
    const pages = getAllDocumentationPages();
    console.log(`Found ${pages.length} pages to include in PDF`);
    
    // Generate PDF using wkhtmltopdf with all pages
    console.log('Generating comprehensive PDF with wkhtmltopdf...');
    
    const outputPdf = path.join(outputDir, 'xafron-documentation.pdf');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    
    // Create URLs for all pages
    const baseUrl = 'http://localhost:3001';
    const pageUrls = pages.map(page => {
      if (page === 'README') return baseUrl + '/';
      return baseUrl + '/' + page;
    });
    
    console.log('Pages to include:');
    pageUrls.forEach((url, index) => {
      console.log(`  ${index + 1}. ${url}`);
    });
    
    // wkhtmltopdf command with all pages
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
      '--footer-center', '[page] of [topage]', // Add page numbers
      '--footer-font-size', '8',
      'toc', // Add table of contents
      '--toc-header-text', 'Table of Contents',
      ...pageUrls, // Add all page URLs
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
      const stats = fs.statSync(outputPdf);
      console.log(`PDF generated successfully at: ${outputPdf}`);
      console.log(`File size: ${(stats.size / 1024).toFixed(1)} KB`);
      
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