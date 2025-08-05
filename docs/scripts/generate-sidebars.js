#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Configuration for sidebar generation
const config = {
  docsDir: path.join(__dirname, '..'),
  outputFile: path.join(__dirname, '..', 'sidebars.js'),
  ignore: ['node_modules', 'build', 'scripts', 'static', 'src', '.docusaurus'],
  categoryOrder: [
    'getting-started',
    'architecture', 
    'detection-methods',
    'api',
    'configuration',
    'development',
    'operations',
    'reference'
  ],
  categoryLabels: {
    'getting-started': 'Getting Started',
    'architecture': 'Architecture',
    'detection-methods': 'Detection Methods',
    'api': 'API Reference',
    'configuration': 'Configuration',
    'development': 'Development',
    'operations': 'Operations',
    'reference': 'Reference'
  }
};

// Get all markdown files in a directory recursively
function getMarkdownFiles(dir, baseDir = dir) {
  const files = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !config.ignore.includes(item)) {
      files.push(...getMarkdownFiles(fullPath, baseDir));
    } else if (stat.isFile() && item.endsWith('.md')) {
      const relativePath = path.relative(baseDir, fullPath);
      const docPath = relativePath.replace(/\.md$/, '').replace(/\\/g, '/');
      if (docPath !== 'README') {
        files.push(docPath);
      }
    }
  }
  
  return files;
}

// Group files by category
function groupFilesByCategory(files) {
  const grouped = {};
  
  for (const file of files) {
    const parts = file.split('/');
    if (parts.length > 1) {
      const category = parts[0];
      if (!grouped[category]) {
        grouped[category] = [];
      }
      grouped[category].push(file);
    }
  }
  
  return grouped;
}

// Generate sidebar configuration
function generateSidebar() {
  const files = getMarkdownFiles(config.docsDir);
  const grouped = groupFilesByCategory(files);
  
  const sidebar = ['README'];
  
  // Add categories in specified order
  for (const category of config.categoryOrder) {
    if (grouped[category] && grouped[category].length > 0) {
      sidebar.push({
        type: 'category',
        label: config.categoryLabels[category] || category,
        items: grouped[category].sort()
      });
    }
  }
  
  // Add any remaining categories not in the order
  for (const category in grouped) {
    if (!config.categoryOrder.includes(category)) {
      sidebar.push({
        type: 'category',
        label: config.categoryLabels[category] || category,
        items: grouped[category].sort()
      });
    }
  }
  
  return sidebar;
}

// Generate the sidebars.js content
function generateSidebarsJs() {
  const sidebar = generateSidebar();
  
  const content = `/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Auto-generated from docs folder structure
  tutorialSidebar: ${JSON.stringify(sidebar, null, 2).replace(/"([^"]+)":/g, '$1:')},
};

export default sidebars;
`;

  return content;
}

// Main execution
function main() {
  try {
    const content = generateSidebarsJs();
    fs.writeFileSync(config.outputFile, content);
    console.log(`✅ Successfully generated ${config.outputFile}`);
    
    // Log what was found
    const files = getMarkdownFiles(config.docsDir);
    console.log(`📄 Found ${files.length} documentation files`);
    
    const grouped = groupFilesByCategory(files);
    for (const category in grouped) {
      console.log(`  📁 ${category}: ${grouped[category].length} files`);
    }
  } catch (error) {
    console.error('❌ Error generating sidebars:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { generateSidebar, generateSidebarsJs };