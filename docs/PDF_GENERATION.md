# PDF Generation for Docusaurus Documentation

This document explains the PDF generation approach used for the Xafron documentation.

## Overview

The PDF generation system uses **wkhtmltopdf** to render the Docusaurus site and generate high-quality PDFs. This approach is based on the [docusaurus-wkhtmltopdf](https://github.com/nuxnik/docusaurus-wkhtmltopdf) project, which provides an open-source alternative to closed-source PDF generators.

## How It Works

1. **Server Startup**: The script starts a local Docusaurus server on port 3001
2. **Site Crawling**: The docusaurus-wkhtmltopdf tool crawls the site to discover all documentation pages
3. **HTML to PDF Conversion**: Each page is converted to PDF using wkhtmltopdf
4. **PDF Merging**: All individual PDFs are merged into a single comprehensive document
5. **Compression**: The final PDF is compressed using Ghostscript (optional)
6. **Table of Contents**: A table of contents is automatically generated

## Files

- `scripts/generate-pdf-wkhtmltopdf.js` - Main PDF generation script
- `print.css` - Custom CSS for PDF styling
- `scripts/generate-pdf-advanced.js` - Previous Puppeteer implementation (backup)
- `scripts/generate-pdf-with-server.js` - Old implementation using docs-to-pdf

## Dependencies

- `docusaurus-wkhtmltopdf` - The main PDF generation tool
- `wkhtmltopdf` - HTML to PDF converter (system dependency)
- `ghostscript` - PDF compression tool (optional, system dependency)

## System Requirements

### Required
- **wkhtmltopdf**: HTML to PDF converter
  - Ubuntu/Debian: `sudo apt-get install wkhtmltopdf`
  - macOS: `brew install wkhtmltopdf`
  - Windows: Download from [wkhtmltopdf.org](https://wkhtmltopdf.org/)

### Optional
- **Ghostscript**: For PDF compression
  - Ubuntu/Debian: `sudo apt-get install ghostscript`
  - macOS: `brew install ghostscript`
  - Windows: Download from [ghostscript.com](https://www.ghostscript.com/)

## Usage

The PDF generation runs automatically after the build process:

```bash
npm run build
```

This will:
1. Build the Docusaurus site
2. Generate the PDF automatically via the `postbuild` script
3. Place the final PDF in `static/pdf/xafron-documentation.pdf`

## Manual PDF Generation

To generate a PDF manually:

```bash
npm run generate-pdf
```

## Configuration

### Pages Included

The tool automatically discovers and includes all accessible documentation pages by crawling the site structure.

### Styling

The PDF uses custom CSS (`print.css`) that:

- Hides navigation elements (navbar, sidebar, pagination)
- Ensures proper formatting for code blocks and tables
- Makes links visible and removes URL suffixes
- Prevents page breaks in inappropriate places
- Optimizes images and content layout
- Provides consistent typography and spacing

### PDF Settings

- Format: A4
- Compression: Enabled (if Ghostscript is available)
- Table of Contents: Automatically generated
- Custom CSS: Applied via `print.css`

## Features

### Automatic Features
- **Site Crawling**: Automatically discovers all documentation pages
- **Table of Contents**: Generates a clickable TOC
- **PDF Compression**: Reduces file size (requires Ghostscript)
- **Custom Styling**: Applies print-optimized CSS

### Customization Options
- **Target Specific Sections**: Can focus on specific documentation areas
- **Custom Output Filename**: Configurable output file name
- **Custom Working Directory**: Configurable output location
- **Additional wkhtmltopdf Options**: Pass custom arguments to wkhtmltopdf

## Troubleshooting

### Common Issues

1. **wkhtmltopdf not found**: Install wkhtmltopdf on your system
2. **Ghostscript not found**: Install ghostscript for compression (optional)
3. **Server not starting**: Check if port 3001 is available
4. **PDF generation fails**: Check system dependencies and permissions

### CI/CD Environment

For CI/CD environments, you can use the Docker image:

```bash
docker run --rm -v /tmp/pdf:/d2p/pdf nuxnik/docusaurus-to-pdf -u http://localhost:3001 --compress --toc
```

### Debugging

To debug PDF generation issues:

1. Check the console output for page discovery results
2. Verify that the Docusaurus server starts successfully
3. Ensure wkhtmltopdf is installed and accessible
4. Check for any system dependency issues

## Customization

### Modifying Styling

To change PDF styling, edit the `print.css` file:

```css
/* Your custom print styles here */
@media print {
  /* Custom styles */
}
```

### Adding Custom wkhtmltopdf Options

To add custom wkhtmltopdf arguments, modify the script:

```javascript
const command = 'npx docusaurus-wkhtmltopdf -u http://localhost:3001 --output xafron-documentation.pdf --compress --toc --wkhtmltopdf-args "--your-custom-option"';
```

### Targeting Specific Sections

To generate PDFs for specific sections only:

```bash
npx docusaurus-wkhtmltopdf -u http://localhost:3001/getting-started --output getting-started.pdf
```

## Performance

The PDF generation process typically takes 30-60 seconds depending on:

- Number of pages discovered
- Content complexity
- System resources
- Whether compression is enabled

## Output

The final PDF is saved to:
- `build/pdf/xafron-documentation.pdf` (build output)
- `static/pdf/xafron-documentation.pdf` (static assets)

The PDF includes:
- All discovered documentation pages
- Automatic table of contents
- Proper page numbering
- Consistent styling optimized for printing
- Compressed file size (if Ghostscript is available)

## Advantages of This Approach

1. **Open Source**: Uses open-source tools (wkhtmltopdf, Ghostscript)
2. **Mature**: Based on well-established tools
3. **Flexible**: Highly configurable with custom CSS and options
4. **Efficient**: Fast generation and good compression
5. **Reliable**: Stable and well-tested approach
6. **Docker Support**: Complete containerized solution available