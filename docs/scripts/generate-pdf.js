const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testServerReady(url, maxAttempts = 10) {
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
  // Dynamically extract all page paths from the sidebar configuration
  const sidebarPath = path.join(__dirname, '..', 'sidebars.js');
  
  try {
    // Import the sidebar configuration
    delete require.cache[require.resolve(sidebarPath)];
    const sidebarModule = require(sidebarPath);
    const sidebars = sidebarModule.default || sidebarModule;
    
    const pages = [];
    
    function extractPages(items) {
      for (const item of items) {
        if (typeof item === 'string') {
          pages.push(item);
        } else if (item.type === 'category' && item.items) {
          extractPages(item.items);
        }
      }
    }
    
    // Extract pages from the main sidebar
    if (sidebars.tutorialSidebar) {
      extractPages(sidebars.tutorialSidebar);
    }
    
    console.log(`Dynamically found ${pages.length} pages from sidebar configuration`);
    return pages;
    
  } catch (error) {
    console.warn('Could not parse sidebar configuration, using fallback pages');
    
    // Minimal fallback
    return ['README'];
  }
}

async function generateCompletePDF() {
  console.log('Starting complete PDF generation...');
  
  const buildDir = path.join(__dirname, '..', 'build');
  const outputDir = path.join(buildDir, 'pdf');
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  try {
    // Test if server is already running
    await testServerReady('http://localhost:3001');
    
    // Additional wait for page to fully load and render
    console.log('Waiting for page content to load...');
    await sleep(8000); // 8-second wait for Mermaid rendering
    
    // Get all documentation pages
    const pages = getAllDocumentationPages();
    console.log(`Found ${pages.length} pages to include in PDF`);
    
    // Create URLs for valid pages only
    const baseUrl = 'http://localhost:3001';
    const validUrls = [];
    
    for (const page of pages) {
      let url;
      if (page === 'README') {
        url = baseUrl + '/';
      } else {
        url = baseUrl + '/' + page;
      }
      
      // Test if the URL exists
      try {
        const testResult = execSync(`curl -s -o /dev/null -w "%{http_code}" "${url}"`, { 
          encoding: 'utf8',
          timeout: 5000 
        });
        if (testResult.trim() === '200') {
          validUrls.push(url);
          console.log(`  ✓ ${url}`);
        } else {
          console.log(`  ✗ Skipping ${url} (HTTP ${testResult.trim()})`);
        }
      } catch (error) {
        console.log(`  ✗ Skipping ${url} (connection error)`);
      }
    }
    
    console.log(`\nGenerating PDF from ${validUrls.length} valid pages...`);
    
    // Generate PDF using simple wkhtmltopdf approach
    const outputPdf = path.join(outputDir, 'xafron-documentation.pdf');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    
    // Simple wkhtmltopdf command that works reliably
    const wkhtmltopdfCmd = [
      'wkhtmltopdf',
      '--page-size', 'A4',
      '--margin-top', '20mm',
      '--margin-bottom', '20mm', 
      '--margin-left', '15mm',
      '--margin-right', '15mm',
      '--encoding', 'UTF-8',
      '--enable-javascript',
      '--javascript-delay', '5000',
      '--load-error-handling', 'ignore',
      '--print-media-type',
      '--user-style-sheet', printCssPath,
      ...validUrls,
      outputPdf
    ];
    
    const command = wkhtmltopdfCmd.join(' ');
    console.log('Executing PDF generation...');
    
    execSync(command, {
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
    
    // Check if PDF was generated
    if (fs.existsSync(outputPdf)) {
      const stats = fs.statSync(outputPdf);
      console.log(`\n✅ PDF generated successfully!`);
      console.log(`   Location: ${outputPdf}`);
      console.log(`   Size: ${(stats.size / 1024).toFixed(1)} KB`);
      console.log(`   Pages included: ${validUrls.length}`);
      
      // Also copy to static directory
      const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
      if (!fs.existsSync(staticPdfDir)) {
        fs.mkdirSync(staticPdfDir, { recursive: true });
      }
      fs.copyFileSync(outputPdf, path.join(staticPdfDir, 'xafron-documentation.pdf'));
      console.log(`   Copied to: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    } else {
      console.error('❌ PDF generation failed - file not found');
    }
    
  } catch (error) {
    console.error('❌ Error generating PDF:', error.message);
    throw error;
  }
}

// Run if called directly
if (require.main === module) {
  generateCompletePDF().catch(error => {
    console.error('PDF generation failed:', error.message);
    process.exit(1);
  });
}

module.exports = { generatePDF: generateCompletePDF };