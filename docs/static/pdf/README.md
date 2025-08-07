# PDF Generation

This directory contains the generated PDF version of the documentation.

## Build Process

1. During the build process (`npm run build`), a PDF is automatically generated
2. This executes `scripts/generate-pdf-wkhtmltopdf.js`
3. The script uses wkhtmltopdf to render the built Docusaurus site into a PDF
4. The PDF includes all documentation content including rendered mermaid diagrams
5. The PDF is placed in `static/pdf/xafron-documentation.pdf` and copied to `build/pdf/`

## PDF Content

The generated PDF contains all documentation sections:
- Getting Started guides
- Architecture documentation  
- User Guides
- Reference documentation
- Development guides
- Deployment examples

## Access

The comprehensive PDF is generated during the build process.

## Download

The PDF should be accessible at: `/pdf/xafron-documentation.pdf`

## Requirements

PDF generation requires:
- Node.js (v18+)
- wkhtmltopdf system package
- ghostscript for PDF compression

## Note

The PDF file is not tracked in git. It is generated fresh during each build to ensure it contains the latest documentation content.

## Viewing Online

This is a comprehensive PDF version containing all documentation. For the interactive version with search, navigation, and diagrams, please visit the online documentation at https://docs.xafron.nl