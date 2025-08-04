/**
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
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'README',
    'DETECTION_SYSTEM_ANALYSIS',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/basic-usage',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
      ],
    },
    {
      type: 'category',
      label: 'Detection Methods',
      items: [
        'detection-methods/overview',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/interfaces',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/brand-config',
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/new-fields',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/deployment',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli',
      ],
    },
  ],
};

export default sidebars;