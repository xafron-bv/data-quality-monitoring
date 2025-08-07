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
  const sidebarPath = path.join(__dirname, '..', 'sidebars.js');
  
  try {
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
    
    if (sidebars.tutorialSidebar) {
      extractPages(sidebars.tutorialSidebar);
    }
    
    console.log(`Dynamically found ${pages.length} pages from sidebar configuration`);
    return pages;
    
  } catch (error) {
    console.warn('Could not parse sidebar configuration, using fallback pages');
    return ['README'];
  }
}

async function generatePDFWithBrowser() {
  console.log('Starting browser-based PDF generation with Mermaid support...');
  
  const buildDir = path.join(__dirname, '..', 'build');
  const outputDir = path.join(buildDir, 'pdf');
  
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  try {
    await testServerReady('http://localhost:3001');
    
    console.log('Getting all documentation pages...');
    const pages = getAllDocumentationPages();
    
    const baseUrl = 'http://localhost:3001';
    const validUrls = [];
    
    for (const page of pages) {
      let url;
      if (page === 'README') {
        url = baseUrl + '/';
      } else {
        url = baseUrl + '/' + page;
      }
      
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
    
    console.log(`\nGenerating PDF from ${validUrls.length} valid pages using Chrome...`);
    
    const outputPdf = path.join(outputDir, 'xafron-documentation.pdf');
    
    // Check if Chrome/Chromium is available
    let chromeCmd = null;
    const chromePaths = [
      'google-chrome',
      'google-chrome-stable', 
      'chromium-browser',
      'chromium',
      '/usr/bin/google-chrome',
      '/usr/bin/chromium-browser'
    ];
    
    for (const cmd of chromePaths) {
      try {
        execSync(`which ${cmd}`, { stdio: 'ignore' });
        chromeCmd = cmd;
        break;
      } catch (e) {
        // Try next
      }
    }
    
    if (!chromeCmd) {
      console.log('Chrome/Chromium not found. Installing chromium-browser...');
      execSync('sudo apt-get update && sudo apt-get install -y chromium-browser', { stdio: 'inherit' });
      chromeCmd = 'chromium-browser';
    }
    
    console.log(`Using Chrome: ${chromeCmd}`);
    
    // Create a temporary HTML file that loads all pages with proper Mermaid rendering
    const tempHtmlPath = path.join(outputDir, 'combined-docs.html');
    const printCssPath = path.join(__dirname, '..', 'print.css');
    
    // Read print CSS
    const printCss = fs.readFileSync(printCssPath, 'utf8');
    
    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Xafron Documentation</title>
    <style>
        ${printCss}
        
        /* Additional styles for combined PDF */
        .page-break {
            page-break-before: always;
        }
        
        .iframe-container {
            width: 100%;
            min-height: 100vh;
            border: none;
            margin: 0;
            padding: 0;
        }
        
        iframe {
            width: 100%;
            min-height: 100vh;
            border: none;
            margin: 0;
            padding: 0;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'neutral',
            securityLevel: 'loose'
        });
        
        // Wait for all iframes to load and Mermaid to render
        window.addEventListener('load', function() {
            setTimeout(() => {
                console.log('Page fully loaded with Mermaid rendered');
                window.mermaidReady = true;
            }, 5000);
        });
    </script>
</head>
<body>
    <h1>Xafron Documentation</h1>
    ${validUrls.map((url, index) => `
        ${index > 0 ? '<div class="page-break"></div>' : ''}
        <div class="iframe-container">
            <iframe src="${url}" onload="console.log('Loaded: ${url}')"></iframe>
        </div>
    `).join('\\n')}
</body>
</html>`;
    
    fs.writeFileSync(tempHtmlPath, htmlContent);
    
    // Generate PDF with Chrome
    const chromeArgs = [
      '--headless',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-extensions',
      '--disable-plugins',
      '--virtual-time-budget=30000', // Wait 30 seconds for content to load
      '--run-all-compositor-stages-before-draw',
      '--print-to-pdf=' + outputPdf,
      '--print-to-pdf-no-header',
      `file://${tempHtmlPath}`
    ];
    
    console.log('Generating PDF with Chrome...');
    console.log(`Command: ${chromeCmd} ${chromeArgs.join(' ')}`);
    
    execSync(`${chromeCmd} ${chromeArgs.join(' ')}`, {
      stdio: 'inherit',
      timeout: 60000 // 60 second timeout
    });
    
    // Clean up temporary file
    fs.unlinkSync(tempHtmlPath);
    
    if (fs.existsSync(outputPdf)) {
      const stats = fs.statSync(outputPdf);
      console.log(`\n✅ PDF generated successfully with Chrome!`);
      console.log(`   Location: ${outputPdf}`);
      console.log(`   Size: ${(stats.size / 1024).toFixed(1)} KB`);
      console.log(`   Pages included: ${validUrls.length}`);
      
      // Copy to static directory
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

if (require.main === module) {
  generatePDFWithBrowser().catch(error => {
    console.error('PDF generation failed:', error.message);
    process.exit(1);
  });
}

module.exports = { generatePDFWithBrowser };