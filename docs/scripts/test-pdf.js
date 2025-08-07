const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function testPDFGeneration() {
  console.log('Testing basic PDF generation...');
  
  const outputPath = path.join(__dirname, '..', 'test-output.pdf');
  
  try {
    // Simple test with a single page
    const command = [
      'wkhtmltopdf',
      '--page-size', 'A4',
      '--enable-javascript',
      '--javascript-delay', '3000',
      '--load-error-handling', 'ignore',
      'http://localhost:3001',
      outputPath
    ].join(' ');
    
    console.log('Running command:', command);
    
    execSync(command, { stdio: 'inherit' });
    
    if (fs.existsSync(outputPath)) {
      console.log('✅ PDF generated successfully!');
      console.log('File size:', fs.statSync(outputPath).size, 'bytes');
      return true;
    } else {
      console.log('❌ PDF file not found');
      return false;
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
    return false;
  }
}

// Start server first
console.log('Make sure Docusaurus server is running on port 3001');
console.log('Run: npm run serve -- --port 3001');
console.log('Then run this test script');

if (require.main === module) {
  testPDFGeneration();
}