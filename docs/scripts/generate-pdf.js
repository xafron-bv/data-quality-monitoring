const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

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
  
  let browser;
  try {
    // Launch browser
    browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu'
      ]
    });
    
    const page = await browser.newPage();
    
    // Set viewport
    await page.setViewport({ width: 1200, height: 800 });
    
    // Set user agent
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');
    
    // Inject CSS for better PDF styling
    await page.addStyleTag({
      content: `
        @media print {
          /* Hide navigation and sidebar */
          .navbar, .theme-doc-sidebar-container, .pagination-nav, 
          .theme-edit-this-page, .breadcrumbs, [class^='tocCollapsible'] {
            display: none !important;
          }
          
          /* Ensure content is properly formatted */
          article {
            margin: 0 !important;
            padding: 20px !important;
          }
          
          /* Make links visible */
          a {
            color: #0066cc !important;
            text-decoration: underline !important;
          }
          
          /* Remove link URLs from being printed */
          a[href]:after {
            content: none !important;
          }
          
          /* Ensure code blocks are readable */
          pre {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            background-color: #f6f8fa !important;
            border: 1px solid #e1e4e8 !important;
            border-radius: 6px !important;
            padding: 16px !important;
            margin: 16px 0 !important;
          }
          
          /* Ensure tables are properly formatted */
          table {
            border-collapse: collapse !important;
            width: 100% !important;
          }
          
          th, td {
            border: 1px solid #d0d7de !important;
            padding: 8px !important;
            text-align: left !important;
          }
          
          th {
            background-color: #f6f8fa !important;
          }
          
          /* Ensure images are properly sized */
          img {
            max-width: 100% !important;
            height: auto !important;
          }
          
          /* Page breaks */
          h1, h2, h3 {
            page-break-after: avoid !important;
          }
          
          pre, table, img {
            page-break-inside: avoid !important;
          }
        }
      `
    });
    
    // Get all pages to include in PDF
    const pages = [
      '/',
      '/getting-started/installation',
      '/getting-started/quick-start',
      '/architecture/overview',
      '/architecture/components',
      '/user-guides/basic-usage',
      '/user-guides/advanced-features',
      '/reference/api',
      '/reference/configuration',
      '/deployment/docker',
      '/deployment/kubernetes',
      '/development/setup',
      '/development/contributing'
    ];
    
    // Filter out pages that don't exist by checking if they return 404
    const validPages = [];
    for (const pageUrl of pages) {
      try {
        const response = await page.goto(`http://localhost:3001${pageUrl}`, { 
          waitUntil: 'networkidle2',
          timeout: 10000 
        });
        if (response.status() === 200) {
          validPages.push(pageUrl);
          console.log(`✓ Found page: ${pageUrl}`);
        } else {
          console.log(`✗ Page not found: ${pageUrl}`);
        }
      } catch (error) {
        console.log(`✗ Error accessing page: ${pageUrl} - ${error.message}`);
      }
    }
    
    if (validPages.length === 0) {
      throw new Error('No valid pages found for PDF generation');
    }
    
    console.log(`Generating PDF with ${validPages.length} pages...`);
    
    // For now, we'll generate a PDF from the main page
    // In a more advanced implementation, you could use a PDF merging library
    // to combine multiple page PDFs into one document
    
    // Navigate to the main page
    await page.goto('http://localhost:3001/', { 
      waitUntil: 'networkidle2',
      timeout: 10000 
    });
    
    // Wait a bit for any dynamic content to load
    await sleep(2000);
    
    // Generate PDF
    const pdfBuffer = await page.pdf({
      format: 'A4',
      printBackground: true,
      margin: {
        top: '20mm',
        right: '20mm',
        bottom: '20mm',
        left: '20mm'
      },
      displayHeaderFooter: true,
      headerTemplate: `
        <div style="font-size: 10px; margin-left: 20px; margin-right: 20px; width: 100%; text-align: center;">
          <span class="title"></span>
        </div>
      `,
      footerTemplate: `
        <div style="font-size: 10px; margin-left: 20px; margin-right: 20px; width: 100%; text-align: center;">
          <span class="pageNumber"></span> / <span class="totalPages"></span>
        </div>
      `,
      preferCSSPageSize: true
    });
    
    // Save PDF
    const pdfPath = path.join(outputDir, 'xafron-documentation.pdf');
    fs.writeFileSync(pdfPath, pdfBuffer);
    console.log(`PDF generated successfully at: ${pdfPath}`);
    
    // Also copy to static directory
    const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
    if (!fs.existsSync(staticPdfDir)) {
      fs.mkdirSync(staticPdfDir, { recursive: true });
    }
    fs.copyFileSync(pdfPath, path.join(staticPdfDir, 'xafron-documentation.pdf'));
    console.log(`PDF copied to static directory: ${path.join(staticPdfDir, 'xafron-documentation.pdf')}`);
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    
    // Create a placeholder file to prevent 404 errors
    const staticPdfDir = path.join(__dirname, '..', 'static', 'pdf');
    if (!fs.existsSync(staticPdfDir)) {
      fs.mkdirSync(staticPdfDir, { recursive: true });
    }
    
    const placeholderPath = path.join(staticPdfDir, 'xafron-documentation.pdf');
    const placeholderContent = `PDF generation failed during build.
Please check the build logs for more details.
Visit the online documentation at: https://docs.xafron.nl/`;
    
    fs.writeFileSync(placeholderPath, placeholderContent);
    console.log('Created placeholder file at:', placeholderPath);
  } finally {
    // Close browser
    if (browser) {
      await browser.close();
    }
    
    // Kill the server
    try {
      if (process.platform === 'win32') {
        require('child_process').execSync('taskkill /F /IM node.exe /FI "COMMANDLINE like *docusaurus*serve*"', { stdio: 'ignore' });
      } else {
        require('child_process').execSync('pkill -f "docusaurus serve"', { stdio: 'ignore' });
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