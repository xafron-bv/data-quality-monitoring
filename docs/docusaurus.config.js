// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
import rehypeRaw from 'rehype-raw';
import { remarkKroki } from 'remark-kroki';

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
          remarkPlugins: [
            [
              remarkKroki,
              {
                server: 'https://kroki.io',
                alias: ['mermaid'],
                target: 'mdx3',
                output: 'inline-svg'
              }
            ],
          ],
          rehypePlugins: [
            [
              rehypeRaw,
              {
                passThrough: [
                  'mdxFlowExpression',
                  'mdxJsxFlowElement',
                  'mdxJsxTextElement',
                  'mdxTextExpression',
                  'mdxjsEsm'
                ]
              }
            ]
          ]
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
            href: '/pdf/xafron-documentation.pdf',
            label: '📄 Download PDF',
            position: 'left',
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
                to: '/reference/interfaces',
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
};

export default config;