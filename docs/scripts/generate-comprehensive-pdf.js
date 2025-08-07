const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function generateComprehensivePDF() {
  console.log('Generating comprehensive PDF with all documentation...');
  
  const outputPath = path.join(__dirname, '..', 'static', 'pdf', 'xafron-documentation.pdf');
  
  // Ensure the directory exists
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Define the order of files to include in the PDF
  const fileOrder = [
    'README.md',
    'getting-started/README.md',
    'getting-started/installation.md',
    'getting-started/basic-usage.md',
    'getting-started/quick-start.md',
    'architecture/overview.md',
    'architecture/detection-methods.md',
    'user-guides/running-detection.md',
    'user-guides/analyzing-results.md',
    'user-guides/optimization.md',
    'reference/cli.md',
    'reference/configuration.md',
    'reference/interfaces.md',
    'development/adding-fields.md',
    'development/contributing.md',
    'deployment/examples.md'
  ];
  
  // Create a combined markdown file
  const combinedContent = [];
  
  // Add title page
  combinedContent.push(`# Xafron Documentation
## Data Quality Detection System

*Generated on: ${new Date().toLocaleDateString()}*

---

`);
  
  // Process each file in order
  fileOrder.forEach((filePath, index) => {
    const fullPath = path.join(__dirname, '..', filePath);
    
    if (fs.existsSync(fullPath)) {
      console.log(`Processing: ${filePath}`);
      
      let content = fs.readFileSync(fullPath, 'utf8');
      
      // Remove frontmatter if present
      content = content.replace(/^---[\s\S]*?---\n/, '');
      
      // Add section header
      const sectionName = filePath.replace('.md', '').replace(/\//g, ' > ');
      combinedContent.push(`\n\n## ${sectionName}\n\n`);
      
      // Add the content
      combinedContent.push(content);
      
      // Add page break between major sections
      if (index < fileOrder.length - 1) {
        combinedContent.push('\n\n---\n\n');
      }
    } else {
      console.warn(`Warning: File not found: ${filePath}`);
    }
  });
  
  // Add footer
  combinedContent.push(`\n\n---\n\n*End of Xafron Documentation*\n\nFor the latest version and interactive features, visit: https://docs.xafron.nl`);
  
  const combinedMarkdown = combinedContent.join('');
  
  // Write combined markdown to temporary file
  const tempFile = path.join(__dirname, '..', 'temp-combined.md');
  fs.writeFileSync(tempFile, combinedMarkdown);
  
  console.log('Combined markdown created, generating PDF...');
  
  // Convert markdown to HTML using a simple converter
  const htmlContent = convertMarkdownToHTML(combinedMarkdown);
  
  // Write HTML to temporary file
  const tempHtmlFile = path.join(__dirname, '..', 'temp-combined.html');
  fs.writeFileSync(tempHtmlFile, htmlContent);
  
  // Generate PDF using puppeteer
  generatePDFFromHTML(tempHtmlFile, outputPath)
    .then(() => {
      // Clean up temporary files
      fs.unlinkSync(tempFile);
      fs.unlinkSync(tempHtmlFile);
      
      // Get file size
      const stats = fs.statSync(outputPath);
      const fileSizeInKB = (stats.size / 1024).toFixed(2);
      console.log(`PDF generated successfully at: ${outputPath}`);
      console.log(`PDF file size: ${fileSizeInKB} KB`);
      
      // Copy to build directory
      const buildPdfPath = path.join(__dirname, '..', 'build', 'pdf', 'xafron-documentation.pdf');
      fs.copyFileSync(outputPath, buildPdfPath);
      console.log(`PDF copied to build directory: ${buildPdfPath}`);
    })
    .catch(error => {
      console.error('Error generating PDF:', error);
      // Clean up temporary files
      if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
      if (fs.existsSync(tempHtmlFile)) fs.unlinkSync(tempHtmlFile);
    });
}

function convertMarkdownToHTML(markdown) {
  // Simple markdown to HTML converter
  let html = markdown
    // Headers
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // Code blocks
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    // Inline code
    .replace(/`(.*?)`/g, '<code>$1</code>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    // Line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  
  // Wrap in HTML structure
  return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Xafron Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 2cm;
            font-size: 12px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            page-break-after: avoid;
        }
        h2 {
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            page-break-after: avoid;
        }
        h3 {
            color: #7f8c8d;
            page-break-after: avoid;
        }
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
            page-break-inside: avoid;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        p {
            margin-bottom: 10px;
        }
        ul, ol {
            margin-bottom: 10px;
        }
        li {
            margin-bottom: 5px;
        }
        @page {
            margin: 2cm;
        }
        .page-break {
            page-break-before: always;
        }
    </style>
</head>
<body>
    <p>${html}</p>
</body>
</html>`;
}

async function generatePDFFromHTML(htmlFile, outputPath) {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Read the HTML file
  const htmlContent = fs.readFileSync(htmlFile, 'utf8');
  
  // Set content and wait for it to load
  await page.setContent(htmlContent, { waitUntil: 'networkidle0' });
  
  // Generate PDF
  await page.pdf({
    path: outputPath,
    format: 'A4',
    margin: {
      top: '2cm',
      right: '2cm',
      bottom: '2cm',
      left: '2cm'
    },
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%;">Xafron Documentation</div>',
    footerTemplate: '<div style="font-size: 10px; text-align: center; width: 100%;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
  });
  
  await browser.close();
}

// Run if called directly
if (require.main === module) {
  generateComprehensivePDF();
}

module.exports = { generateComprehensivePDF };