const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

async function generatePDF() {
  console.log('Starting PDF generation from built documentation...');
  
  const buildDir = path.join(__dirname, '..', 'build');
  const staticDir = path.join(__dirname, '..', 'static', 'pdf');
  
  // Ensure the static PDF directory exists
  if (!fs.existsSync(staticDir)) {
    fs.mkdirSync(staticDir, { recursive: true });
  }
  
  // Start a local server to serve the built files
  console.log('Starting local server...');
  const serverProcess = execSync('npx serve -s build -p 5000', {
    cwd: path.join(__dirname, '..'),
    detached: true,
    stdio: 'ignore'
  });
  
  // Wait a bit for server to start
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  try {
    const page = await browser.newPage();
    
    // Set viewport
    await page.setViewport({ width: 1200, height: 800 });
    
    // Navigate to the main documentation page
    console.log('Navigating to documentation...');
    await page.goto('http://localhost:5000', { 
      waitUntil: 'networkidle0',
      timeout: 30000 
    });
    
    // Click on the first documentation link to enter docs
    await page.click('a[href="/README"]').catch(() => {
      console.log('Could not find README link, trying alternative...');
      return page.click('a[href*="docs"]').catch(() => {
        console.log('Using current page for PDF generation');
      });
    });
    
    await page.waitForTimeout(2000);
    
    // Remove navigation elements for cleaner PDF
    await page.evaluate(() => {
      // Remove navbar
      const navbar = document.querySelector('.navbar');
      if (navbar) navbar.remove();
      
      // Remove footer
      const footer = document.querySelector('footer');
      if (footer) footer.remove();
      
      // Remove sidebar
      const sidebar = document.querySelector('.theme-doc-sidebar-container');
      if (sidebar) sidebar.remove();
      
      // Remove pagination
      const pagination = document.querySelector('.pagination-nav');
      if (pagination) pagination.remove();
      
      // Expand main content
      const mainContent = document.querySelector('.theme-doc-markdown');
      if (mainContent && mainContent.parentElement) {
        mainContent.parentElement.style.maxWidth = '100%';
        mainContent.parentElement.style.margin = '0 auto';
        mainContent.parentElement.style.padding = '20px';
      }
    });
    
    // Generate the PDF
    const pdfPath = path.join(staticDir, 'xafron-documentation.pdf');
    await page.pdf({
      path: pdfPath,
      format: 'A4',
      margin: {
        top: '20mm',
        right: '20mm',
        bottom: '20mm',
        left: '20mm'
      },
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%; color: #666;">Xafron Documentation</div>',
      footerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%; color: #666;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'
    });
    
    console.log(`PDF generated successfully at: ${pdfPath}`);
    
    // Copy to build directory as well
    const buildPdfDir = path.join(buildDir, 'pdf');
    if (!fs.existsSync(buildPdfDir)) {
      fs.mkdirSync(buildPdfDir, { recursive: true });
    }
    fs.copyFileSync(pdfPath, path.join(buildPdfDir, 'xafron-documentation.pdf'));
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    
    // Create a placeholder PDF info file
    const placeholderHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>PDF Documentation</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
      max-width: 600px;
      margin: 50px auto;
      padding: 20px;
      text-align: center;
    }
    .info {
      background-color: #f0f7ff;
      border: 2px solid #0066cc;
      border-radius: 8px;
      padding: 30px;
    }
    h1 { color: #0066cc; }
    code {
      background-color: #f5f5f5;
      padding: 2px 4px;
      border-radius: 3px;
    }
  </style>
</head>
<body>
  <div class="info">
    <h1>📄 PDF Documentation</h1>
    <p>To generate a PDF of the documentation:</p>
    <ol style="text-align: left; max-width: 400px; margin: 20px auto;">
      <li>Open any documentation page</li>
      <li>Press <code>Ctrl+P</code> (Windows/Linux) or <code>Cmd+P</code> (Mac)</li>
      <li>Select "Save as PDF" in the print dialog</li>
    </ol>
    <p>Alternatively, you can install Prince or wkhtmltopdf for automated PDF generation.</p>
  </div>
</body>
</html>`;
    
    fs.writeFileSync(path.join(staticDir, 'index.html'), placeholderHtml);
    const buildPdfDir = path.join(buildDir, 'pdf');
    if (!fs.existsSync(buildPdfDir)) {
      fs.mkdirSync(buildPdfDir, { recursive: true });
    }
    fs.writeFileSync(path.join(buildPdfDir, 'index.html'), placeholderHtml);
  } finally {
    await browser.close();
    
    // Try to stop the server
    try {
      if (process.platform === 'win32') {
        execSync('taskkill /F /IM node.exe /FI "WINDOWTITLE eq serve*"', { stdio: 'ignore' });
      } else {
        execSync('pkill -f "serve -s build"', { stdio: 'ignore' });
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
    // Don't fail the build
    process.exit(0);
  });
}

module.exports = { generatePDF };