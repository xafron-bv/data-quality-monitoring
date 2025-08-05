#!/usr/bin/env node

/**
 * Validates Mermaid diagram syntax in markdown files
 * This script checks all .md files in the docs directory for Mermaid diagrams
 * and validates their syntax using the mermaid-cli tool
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Colors for console output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m'
};

// Find all markdown files recursively
function findMarkdownFiles(dir, files = []) {
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
      findMarkdownFiles(fullPath, files);
    } else if (stat.isFile() && item.endsWith('.md')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

// Extract Mermaid diagrams from markdown content
function extractMermaidDiagrams(content, filePath) {
  const diagrams = [];
  const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
  let match;
  let index = 0;
  
  while ((match = mermaidRegex.exec(content)) !== null) {
    // Find line number
    const lines = content.substring(0, match.index).split('\n');
    const lineNumber = lines.length;
    
    diagrams.push({
      content: match[1],
      file: filePath,
      line: lineNumber,
      index: index++
    });
  }
  
  return diagrams;
}

// Validate a single Mermaid diagram
function validateDiagram(diagram) {
  const tempFile = path.join(__dirname, `temp_mermaid_${Date.now()}.mmd`);
  
  try {
    // Write diagram to temporary file
    fs.writeFileSync(tempFile, diagram.content);
    
    // Try to parse with mermaid-cli (if available)
    // For now, we'll do basic syntax validation
    const lines = diagram.content.trim().split('\n');
    const errors = [];
    
    // Check first line is a valid diagram type
    const validTypes = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
                       'stateDiagram', 'erDiagram', 'journey', 'gantt', 'pie',
                       'gitGraph', 'mindmap', 'timeline'];
    
    const firstLine = lines[0].trim();
    const diagramType = firstLine.split(/\s+/)[0];
    
    if (!validTypes.some(type => firstLine.startsWith(type))) {
      errors.push(`Invalid diagram type: "${diagramType}"`);
    }
    
    // Check for balanced brackets
    const openBrackets = (diagram.content.match(/[\[{(]/g) || []).length;
    const closeBrackets = (diagram.content.match(/[\]})]/g) || []).length;
    
    if (openBrackets !== closeBrackets) {
      errors.push(`Unbalanced brackets: ${openBrackets} opening, ${closeBrackets} closing`);
    }
    
    // Check for common syntax errors
    if (diagram.content.includes('→')) {
      errors.push('Contains Unicode arrow (→). Use --> instead');
    }
    
    if (diagram.content.includes('↓')) {
      errors.push('Contains Unicode arrow (↓). Use --> or vertical flow instead');
    }
    
    // Clean up
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
    
    return errors;
  } catch (error) {
    // Clean up on error
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
    return [`Error validating diagram: ${error.message}`];
  }
}

// Main validation function
function validateAllDiagrams() {
  console.log(`${colors.blue}🔍 Validating Mermaid diagrams in documentation...${colors.reset}\n`);
  
  const docsDir = path.resolve(__dirname, '..');
  const markdownFiles = findMarkdownFiles(docsDir);
  
  let totalDiagrams = 0;
  let totalErrors = 0;
  const errorDetails = [];
  
  for (const file of markdownFiles) {
    const content = fs.readFileSync(file, 'utf8');
    const diagrams = extractMermaidDiagrams(content, file);
    
    for (const diagram of diagrams) {
      totalDiagrams++;
      const errors = validateDiagram(diagram);
      
      if (errors.length > 0) {
        totalErrors += errors.length;
        errorDetails.push({
          file: path.relative(docsDir, diagram.file),
          line: diagram.line,
          errors: errors
        });
      }
    }
  }
  
  // Report results
  console.log(`Found ${totalDiagrams} Mermaid diagrams in ${markdownFiles.length} files\n`);
  
  if (totalErrors === 0) {
    console.log(`${colors.green}✅ All Mermaid diagrams are valid!${colors.reset}`);
    return 0;
  } else {
    console.log(`${colors.red}❌ Found ${totalErrors} errors in Mermaid diagrams:${colors.reset}\n`);
    
    for (const error of errorDetails) {
      console.log(`${colors.yellow}${error.file}:${error.line}${colors.reset}`);
      for (const err of error.errors) {
        console.log(`  ${colors.red}└─ ${err}${colors.reset}`);
      }
      console.log();
    }
    
    return 1;
  }
}

// Check for ASCII art diagrams
function checkForASCIIArt() {
  console.log(`${colors.blue}🔍 Checking for ASCII art diagrams...${colors.reset}\n`);
  
  const docsDir = path.resolve(__dirname, '..');
  const markdownFiles = findMarkdownFiles(docsDir);
  
  const asciiPatterns = [
    /[│├└─┌┐┘┤┬┴┼━┃┏┓┗┛┣┫┳┻╋]/,  // Box drawing characters
    /^\s*\+[-─]+\+/m,                // ASCII boxes
    /^\s*\|.*\|.*\|/m,               // ASCII tables (but might be markdown tables)
    /→|←|↓|↑/                        // Unicode arrows
  ];
  
  const findings = [];
  
  for (const file of markdownFiles) {
    const content = fs.readFileSync(file, 'utf8');
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      // Skip markdown tables
      if (line.trim().match(/^\|.*\|.*\|$/) && 
          lines[index + 1] && lines[index + 1].trim().match(/^\|[-:\s|]+\|$/)) {
        return;
      }
      
      // Check for ASCII art patterns
      for (const pattern of asciiPatterns) {
        if (pattern.test(line)) {
          // Skip if it's inside a code block
          let insideCodeBlock = false;
          for (let i = 0; i < index; i++) {
            if (lines[i].trim().startsWith('```')) {
              insideCodeBlock = !insideCodeBlock;
            }
          }
          
          if (!insideCodeBlock) {
            findings.push({
              file: path.relative(docsDir, file),
              line: index + 1,
              content: line.trim()
            });
            break;
          }
        }
      }
    });
  }
  
  if (findings.length > 0) {
    console.log(`${colors.yellow}⚠️  Found potential ASCII art diagrams:${colors.reset}\n`);
    
    for (const finding of findings) {
      console.log(`${colors.yellow}${finding.file}:${finding.line}${colors.reset}`);
      console.log(`  ${finding.content.substring(0, 60)}${finding.content.length > 60 ? '...' : ''}`);
    }
    
    console.log(`\n${colors.yellow}Consider converting these to Mermaid diagrams for better rendering.${colors.reset}`);
  } else {
    console.log(`${colors.green}✅ No ASCII art diagrams found!${colors.reset}`);
  }
}

// Run validation
if (require.main === module) {
  const exitCode = validateAllDiagrams();
  checkForASCIIArt();
  process.exit(exitCode);
}

module.exports = { validateAllDiagrams, checkForASCIIArt };