const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function findMermaidDiagrams(dir, diagrams = []) {
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && 
        !file.startsWith('.') && 
        file !== 'node_modules' && 
        file !== 'build' && 
        file !== 'static' &&
        file !== 'scripts') {
      findMermaidDiagrams(fullPath, diagrams);
    } else if (file.endsWith('.md')) {
      const content = fs.readFileSync(fullPath, 'utf8');
      const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
      let match;
      let index = 0;
      
      while ((match = mermaidRegex.exec(content)) !== null) {
        diagrams.push({
          file: fullPath,
          content: match[1].trim(),
          index: index++,
          id: `${path.basename(file, '.md')}_${index}`
        });
      }
    }
  }
  
  return diagrams;
}

async function prerenderMermaidDiagrams() {
  console.log('Pre-rendering Mermaid diagrams...');
  
  const docsDir = path.join(__dirname, '..');
  const outputDir = path.join(docsDir, 'static', 'mermaid-svg');
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Find all Mermaid diagrams
  const diagrams = findMermaidDiagrams(docsDir);
  console.log(`Found ${diagrams.length} Mermaid diagrams`);
  
  const renderedDiagrams = [];
  
  for (const diagram of diagrams) {
    try {
      const inputFile = path.join(outputDir, `${diagram.id}.mmd`);
      const outputFile = path.join(outputDir, `${diagram.id}.svg`);
      
      // Write mermaid source to temporary file
      fs.writeFileSync(inputFile, diagram.content);
      
      // Render with mermaid CLI
      console.log(`Rendering ${diagram.id}...`);
      execSync(`npx mmdc -i "${inputFile}" -o "${outputFile}" -t neutral -b white`, {
        stdio: 'inherit'
      });
      
      if (fs.existsSync(outputFile)) {
        renderedDiagrams.push({
          ...diagram,
          svgPath: outputFile,
          svgUrl: `/mermaid-svg/${diagram.id}.svg`
        });
        console.log(`  ✓ Rendered ${diagram.id}.svg`);
      } else {
        console.log(`  ✗ Failed to render ${diagram.id}`);
      }
      
      // Clean up temporary file
      fs.unlinkSync(inputFile);
      
    } catch (error) {
      console.error(`  ✗ Error rendering ${diagram.id}:`, error.message);
    }
  }
  
  console.log(`Successfully rendered ${renderedDiagrams.length}/${diagrams.length} diagrams`);
  
  // Create a mapping file for reference
  const mappingFile = path.join(outputDir, 'diagram-mapping.json');
  fs.writeFileSync(mappingFile, JSON.stringify(renderedDiagrams, null, 2));
  
  return renderedDiagrams;
}

// Run if called directly
if (require.main === module) {
  prerenderMermaidDiagrams().catch(error => {
    console.error('Pre-rendering failed:', error);
    process.exit(1);
  });
}

module.exports = { prerenderMermaidDiagrams };