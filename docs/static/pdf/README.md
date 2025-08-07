# PDF Generation

This directory contains the generated PDF version of the documentation.

## How it works

1. During the build process (`npm run build`), the `postbuild` script runs
2. This executes `scripts/generate-simple-pdf.js`
3. The script uses the `pdfkit` library to generate a simplified PDF version
4. The PDF is placed in `static/pdf/xafron-documentation.pdf`

## PDF Content

The generated PDF contains:
- Title and description of the Xafron system
- Link to the full online documentation
- Overview of documentation sections
- Generation date

## Access

The PDF should be accessible at: `/pdf/xafron-documentation.pdf`

## Requirements

PDF generation requires:
- Node.js with the `pdfkit` npm package (listed in dependencies)

## Note

This is a simplified PDF version. For the complete, interactive documentation with all features, diagrams, and examples, please visit the online documentation at https://docs.xafron.nl