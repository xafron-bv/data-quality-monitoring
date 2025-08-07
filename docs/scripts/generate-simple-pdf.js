const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

function generateSimplePDF() {
  console.log('Generating simple PDF with pdfkit...');
  
  const outputPath = path.join(__dirname, '..', 'static', 'pdf', 'xafron-documentation.pdf');
  
  // Ensure the directory exists
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Create a new PDF document
  const doc = new PDFDocument({
    size: 'A4',
    margin: 50
  });
  
  // Pipe the PDF to a file
  doc.pipe(fs.createWriteStream(outputPath));
  
  // Add content to the PDF
  doc.fontSize(24)
     .text('Xafron Documentation', { align: 'center' })
     .moveDown(2);
  
  doc.fontSize(16)
     .text('Data Quality Detection System', { align: 'center' })
     .moveDown(3);
  
  doc.fontSize(12)
     .text('This is a simplified PDF version of the Xafron documentation.')
     .moveDown()
     .text('For the complete, interactive documentation with all features, diagrams, and examples, please visit:')
     .moveDown()
     .text('https://docs.xafron.nl', { underline: true })
     .moveDown(2);
  
  doc.fontSize(14)
     .text('Documentation Sections:', { underline: true })
     .moveDown();
  
  const sections = [
    'Getting Started',
    'Architecture',
    'User Guides', 
    'Reference',
    'Development',
    'Deployment'
  ];
  
  sections.forEach((section, index) => {
    doc.fontSize(12)
       .text(`${index + 1}. ${section}`)
       .moveDown(0.5);
  });
  
  doc.moveDown(2)
     .fontSize(10)
     .text('Generated on: ' + new Date().toLocaleDateString(), { align: 'center' })
     .moveDown()
     .text('Xafron - Data Quality Detection System', { align: 'center' });
  
  // Finalize the PDF
  doc.end();
  
  console.log(`PDF generated successfully at: ${outputPath}`);
}

// Run if called directly
if (require.main === module) {
  generateSimplePDF();
}

module.exports = { generateSimplePDF };