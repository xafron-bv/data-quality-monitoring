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
        'architecture/core-components',
        'architecture/detection-methods',
        'architecture/data-flow',
      ],
    },
    {
      type: 'category',
      label: 'Detection Methods',
      items: [
        'detection-methods/overview',
        'detection-methods/validation',
        'detection-methods/pattern-based',
        'detection-methods/ml-based',
        'detection-methods/llm-based',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/interfaces',
        'api/validators',
        'api/anomaly-detectors',
        'api/reporters',
        'api/utilities',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/brand-config',
        'configuration/field-mappings',
        'configuration/thresholds',
        'configuration/models',
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/new-fields',
        'development/validators',
        'development/training',
        'development/testing',
        'development/contributing',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/deployment',
        'operations/performance',
        'operations/monitoring',
        'operations/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli',
        'reference/config-files',
        'reference/error-codes',
        'reference/glossary',
      ],
    },
  ],
};

export default sidebars;