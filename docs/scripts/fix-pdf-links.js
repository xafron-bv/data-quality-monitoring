const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

/**
 * Fix localhost links in HTML content to work properly in PDF
 * @param {string} htmlContent - The HTML content to process
 * @param {string} baseUrl - The base URL to remove (e.g., 'http://localhost:3001')
 * @returns {string} - The processed HTML content
 */
function fixLinksForPDF(htmlContent, baseUrl = 'http://localhost:3001') {
  const dom = new JSDOM(htmlContent);
  const document = dom.window.document;
  
  // Find all anchor tags
  const links = document.querySelectorAll('a[href]');
  
  links.forEach(link => {
    const href = link.getAttribute('href');
    
    // Skip external links and already processed links
    if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) {
      return;
    }
    
    // Handle absolute localhost links
    if (href.startsWith(baseUrl)) {
      // Remove the base URL to make it relative
      const relativePath = href.substring(baseUrl.length);
      
      // Convert internal links to anchors for PDF navigation
      // This assumes the PDF will have all content in a single document
      if (relativePath.startsWith('/') && !relativePath.includes('://')) {
        // Extract the page identifier from the path
        // For example: /docs/getting-started -> #getting-started
        const anchor = relativePath
          .split('/')
          .filter(part => part && part !== 'docs')
          .join('-');
        
        link.setAttribute('href', anchor ? `#${anchor}` : '#');
      }
    }
    
    // Handle relative links that might still point to localhost
    else if (href.startsWith('/') && !href.includes('://')) {
      // Convert to anchor for internal navigation
      const anchor = href
        .split('/')
        .filter(part => part && part !== 'docs')
        .join('-');
      
      link.setAttribute('href', anchor ? `#${anchor}` : '#');
    }
  });
  
  return dom.serialize();
}

/**
 * Process an HTML file and fix its links
 * @param {string} inputPath - Path to the input HTML file
 * @param {string} outputPath - Path to save the processed HTML file
 * @param {string} baseUrl - The base URL to remove
 */
function processHTMLFile(inputPath, outputPath, baseUrl = 'http://localhost:3001') {
  try {
    const htmlContent = fs.readFileSync(inputPath, 'utf8');
    const fixedContent = fixLinksForPDF(htmlContent, baseUrl);
    fs.writeFileSync(outputPath, fixedContent, 'utf8');
    console.log(`✅ Processed HTML file saved to: ${outputPath}`);
  } catch (error) {
    console.error(`❌ Error processing HTML file: ${error.message}`);
    throw error;
  }
}

module.exports = {
  fixLinksForPDF,
  processHTMLFile
};

// If run directly from command line
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('Usage: node fix-pdf-links.js <input-html> <output-html> [base-url]');
    process.exit(1);
  }
  
  const [inputPath, outputPath, baseUrl] = args;
  processHTMLFile(inputPath, outputPath, baseUrl);
}