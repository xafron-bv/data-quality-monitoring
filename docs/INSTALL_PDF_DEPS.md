# PDF Generation Dependencies Installation Guide

This guide explains how to install the system dependencies required for PDF generation.

## Required Dependencies

### wkhtmltopdf

**wkhtmltopdf** is the main HTML to PDF converter used by the PDF generation system.

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install wkhtmltopdf
```

#### macOS
```bash
brew install wkhtmltopdf
```

#### Windows
1. Download the installer from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)
2. Run the installer and follow the setup wizard
3. Add wkhtmltopdf to your system PATH

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum install wkhtmltopdf

# Fedora
sudo dnf install wkhtmltopdf
```

## Optional Dependencies

### Ghostscript

**Ghostscript** is used for PDF compression, which reduces the final file size.

#### Ubuntu/Debian
```bash
sudo apt-get install ghostscript
```

#### macOS
```bash
brew install ghostscript
```

#### Windows
1. Download from [ghostscript.com](https://www.ghostscript.com/download/gsdnld.html)
2. Run the installer
3. Add Ghostscript to your system PATH

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum install ghostscript

# Fedora
sudo dnf install ghostscript
```

## Verification

After installation, verify that the tools are available:

```bash
# Check wkhtmltopdf
wkhtmltopdf --version

# Check ghostscript (if installed)
gs --version
```

## Docker Alternative

If you prefer not to install system dependencies, you can use the Docker image:

```bash
# Pull the Docker image
docker pull nuxnik/docusaurus-to-pdf

# Run PDF generation
docker run --rm -v /tmp/pdf:/d2p/pdf nuxnik/docusaurus-to-pdf -u http://localhost:3001 --compress --toc
```

## Troubleshooting

### Common Issues

1. **Command not found**: Ensure the tools are in your system PATH
2. **Permission denied**: You may need to run installation commands with sudo
3. **Version conflicts**: Some systems may have older versions; consider using the Docker approach

### Windows-Specific Notes

- Ensure both wkhtmltopdf and Ghostscript are added to your system PATH
- You may need to restart your terminal/IDE after installation
- Some antivirus software may flag these tools; add them to your whitelist if needed

### macOS-Specific Notes

- If using Homebrew, ensure it's up to date: `brew update`
- You may need to install Xcode command line tools: `xcode-select --install`

### Linux-Specific Notes

- Some distributions may require additional dependencies
- For headless servers, ensure X11 libraries are available (though wkhtmltopdf can run headless)