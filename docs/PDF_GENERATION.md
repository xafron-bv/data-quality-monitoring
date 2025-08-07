# PDF Generation for Docusaurus Documentation

This document explains the PDF generation approach used for the Xafron documentation.

## Overview

The PDF generation system uses Puppeteer to render the Docusaurus site and generate high-quality PDFs. This approach provides better control over styling and formatting compared to the previous `docs-to-pdf` package.

## How It Works

1. **Server Startup**: The script starts a local Docusaurus server on port 3001
2. **Page Discovery**: It checks which documentation pages exist and are accessible
3. **Individual PDF Generation**: Each page is rendered and converted to a separate PDF
4. **PDF Merging**: All individual PDFs are merged into a single comprehensive document
5. **Cleanup**: Temporary files are removed and the final PDF is placed in the static directory

## Files

- `scripts/generate-pdf-advanced.js` - Main PDF generation script
- `scripts/generate-pdf.js` - Simple single-page PDF generation (backup)
- `scripts/generate-pdf-with-server.js` - Old implementation using docs-to-pdf

## Dependencies

- `puppeteer` - For browser automation and PDF generation
- `pdf-lib` - For merging multiple PDFs into one document

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

The script automatically includes the following pages (if they exist):

- Home page (`/`)
- Getting Started guides
- Architecture documentation
- User guides
- API reference
- Configuration documentation
- Deployment guides
- Development setup

### Styling

The PDF uses custom CSS for print media that:

- Hides navigation elements (navbar, sidebar, pagination)
- Ensures proper formatting for code blocks and tables
- Makes links visible and removes URL suffixes
- Prevents page breaks in inappropriate places
- Optimizes images and content layout

### PDF Settings

- Format: A4
- Margins: 20mm on all sides
- Background: Printed
- Headers/Footers: Page numbers and titles
- Quality: High resolution

## Troubleshooting

### Common Issues

1. **Server not starting**: Check if port 3001 is available
2. **Pages not found**: Verify that the documentation pages exist
3. **PDF generation fails**: Check browser dependencies in CI environment
4. **Memory issues**: The script includes memory optimization flags for Puppeteer

### CI/CD Environment

For CI/CD environments, the script includes:

- Sandbox disabling for containerized environments
- GPU acceleration disabling
- Memory optimization flags
- Graceful fallback to placeholder file if generation fails

### Debugging

To debug PDF generation issues:

1. Check the console output for page discovery results
2. Verify that the Docusaurus server starts successfully
3. Ensure all required pages are accessible
4. Check for any JavaScript errors in the browser console

## Customization

### Adding New Pages

To include additional pages, modify the `pages` array in `generate-pdf-advanced.js`:

```javascript
const pages = [
  { url: '/your-new-page', title: 'Your New Page' },
  // ... existing pages
];
```

### Modifying Styling

To change PDF styling, modify the CSS in the `addStyleTag` call:

```javascript
await page.addStyleTag({
  content: `
    @media print {
      /* Your custom styles here */
    }
  `
});
```

### Changing PDF Settings

To modify PDF generation settings, update the `page.pdf()` options:

```javascript
const pdfBuffer = await page.pdf({
  format: 'A4',
  printBackground: true,
  margin: { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' },
  // ... other options
});
```

## Performance

The PDF generation process typically takes 30-60 seconds depending on:

- Number of pages included
- Content complexity
- System resources
- Network speed (for external resources)

## Output

The final PDF is saved to:
- `build/pdf/xafron-documentation.pdf` (build output)
- `static/pdf/xafron-documentation.pdf` (static assets)

The PDF includes:
- All accessible documentation pages
- Proper page numbering
- Consistent styling
- Optimized layout for printing