# PDF Generation

This directory contains the generated PDF version of the documentation.

## How it works

1. During the build process (`npm run build`), the `postbuild` script runs
2. This executes `scripts/generate-pdf-with-server.js`
3. The script starts a local Docusaurus server and uses `docs-to-pdf` to generate a PDF
4. The PDF is placed in both `build/pdf/` and copied to `static/pdf/`

## Troubleshooting

If the PDF generation fails (which can happen in CI environments due to missing browser dependencies), a placeholder file will be created instead.

The PDF should be accessible at: `/pdf/xafron-documentation.pdf`

## Requirements

PDF generation requires:
- Chrome/Chromium browser
- Various system libraries (automatically installed in GitHub Actions)
- The `docs-to-pdf` npm package (listed in devDependencies)