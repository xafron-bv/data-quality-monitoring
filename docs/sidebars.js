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
  // Main documentation sidebar
  tutorialSidebar: [
    "README",
    {
      type: "category",
      label: "Getting Started",
      items: [
        "getting-started/README",
        "getting-started/installation",
        "getting-started/basic-usage",
        "getting-started/quick-start"
      ]
    },
    {
      type: "category",
      label: "Architecture",
      items: [
        "architecture/overview",
        "architecture/detection-methods"
      ]
    },
    {
      type: "category",
      label: "User Guides",
      items: [
        "user-guides/running-detection",
        "user-guides/analyzing-results",
        "user-guides/optimization"
      ]
    },
    {
      type: "category",
      label: "Reference",
      items: [
        "reference/cli",
        "reference/configuration",
        "reference/interfaces"
      ]
    },
    {
      type: "category",
      label: "Development",
      items: [
        "development/adding-fields",
        "development/contributing"
      ]
    },
    {
      type: "category",
      label: "Deployment",
      items: [
        "deployment/examples"
      ]
    }
  ],
};

export default sidebars;
