# PDF Generation

This directory contains the generated PDF version of the documentation.

## How it works

1. During the build process (`npm run build`), the `postbuild` script runs
2. This executes `scripts/generate-comprehensive-pdf.js`
3. The script combines all markdown documentation files into a single comprehensive PDF
4. The PDF is generated using Puppeteer and includes all documentation content
5. The PDF is placed in `static/pdf/xafron-documentation.pdf` and copied to `build/pdf/`

## PDF Content

The generated PDF contains all documentation sections:
- Getting Started (Installation, Basic Usage, Quick Start)
- Architecture (Overview, Detection Methods)
- User Guides (Running Detection, Analyzing Results, Optimization)
- Reference (CLI, Configuration, Interfaces)
- Development (Adding Fields, Contributing)
- Deployment (Examples)

## File Size

The comprehensive PDF is approximately 1.4MB and contains all documentation content.

## Access

The PDF should be accessible at: `/pdf/xafron-documentation.pdf`

## Requirements

PDF generation requires:
- Node.js with the `puppeteer` npm package (listed in dependencies)
- Chrome/Chromium browser (automatically installed with Puppeteer)

## Features

- Professional formatting with proper headers and styling
- Page numbers and headers/footers
- Code syntax highlighting
- Proper page breaks between sections
- All documentation content included

## Note

This is a comprehensive PDF version containing all documentation. For the interactive version with search, navigation, and diagrams, please visit the online documentation at https://docs.xafron.nl