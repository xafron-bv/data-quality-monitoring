// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Xafron Documentation',
  tagline: 'Data Quality Detection System',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://docs.xafron.nl',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'xafron', // Usually your GitHub org/user name.
  projectName: 'xafron-docs', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '.', // Use current directory as docs
          routeBasePath: '/', // Serve docs at the root
          sidebarPath: './sidebars.js',
          exclude: [
            '**/node_modules/**', 
            '**/.docusaurus/**', 
            '**/build/**', 
            '**/.git/**', 
            '**/src/**', 
            '**/static/**',
            '**/.gitbook.yaml',
            '**/book.json',
            '**/SUMMARY.md',
            '**/package.json',
            '**/package-lock.json',
            '**/*.js',
            '**/*.json',
            '**/*.css'
          ],
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/xafron/xafron-docs/tree/main/docs/',
        },
        blog: false, // Disable the blog plugin
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
  
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Xafron',
        logo: {
          alt: 'Xafron Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://github.com/xafron/xafron',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/getting-started/installation',
              },
              {
                label: 'Architecture',
                to: '/architecture/overview',
              },
              {
                label: 'API Reference',
                to: '/api/interfaces',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/xafron/xafron',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Xafron. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      mermaid: {
        theme: {light: 'neutral', dark: 'dark'},
      },
    }),
  plugins: [
    [
      'docusaurus-plugin-papersaurus',
      {
        autoBuildPdfs: true,
        keepDebugHtmls: false,
        sidebarNames: ['tutorialSidebar'],
        ignoreDocs: [],
        author: 'Xafron Team',
        coverPageHeader: `
          <div style="text-align: center; padding: 2cm 0;">
            <h1 style="color:#005479;font-size:36px;font-family:sans-serif;font-weight:bold;margin:0;">Xafron Documentation</h1>
            <h2 style="color:#005479;font-size:20px;font-family:sans-serif;font-weight:normal;margin-top:10px;">Data Quality Detection System</h2>
          </div>
        `,
        coverPageFooter: `
          <div style="text-align: center; padding: 1cm 0; color: #666; font-size: 12px;">
            <p>© ${new Date().getFullYear()} Xafron. All rights reserved.</p>
          </div>
        `,
        getPdfPageHeader: (siteConfig, pluginConfig, pageTitle) => {
          return `
            <div style="justify-content: space-between;align-items: center;height:2.5cm;display:flex;margin: 0 1.5cm;color: #005479;font-size:10px;font-family:sans-serif;width:100%;">
              <span style="flex-grow: 1; width: 50%; text-align:left;">Xafron Documentation</span>
              <span style="flex-grow: 1; width: 50%; text-align:right;">${pageTitle}</span>
            </div>
          `;
        },
        getPdfPageFooter: (siteConfig, pluginConfig, pageTitle) => {
          return `
            <div style="height:1cm;display:flex;margin: 0 1.5cm;color: #005479;font-size:9px;font-family:sans-serif;width:100%;align-items:center;">
              <span style="flex-grow: 1; width: 33%;">© ${new Date().getFullYear()} Xafron</span>
              <span style="flex-grow: 1; width: 33%; text-align:center;">${new Date().toISOString().substring(0,10)}</span>
              <span style="flex-grow: 1; width: 33%; text-align:right;">Page <span class='pageNumber'></span> / <span class='totalPages'></span></span>
            </div>`;
        },
        footerParser: /© \d{4} Xafron\d{4}-\d{2}-\d{2}Page \d* \/ \d*/g,
        stylesheets: [],
        addDownloadButton: true,
        downloadButtonText: '📄 Download PDF'
      },
    ],
  ],
};

export default config;