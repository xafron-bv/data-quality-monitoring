/**
 * Client-side script to wait for all Mermaid diagrams to be fully rendered
 * This script runs in the browser and sets a flag when all diagrams are ready
 */
(function() {
  'use strict';
  
  console.log('Mermaid wait script started');
  
  function waitForMermaidDiagrams() {
    return new Promise((resolve) => {
      let checkCount = 0;
      const maxChecks = 50; // Maximum 25 seconds (50 * 500ms)
      
      function checkMermaidStatus() {
        checkCount++;
        console.log(`Checking Mermaid status... attempt ${checkCount}`);
        
        // Find all mermaid containers
        const mermaidContainers = document.querySelectorAll('.mermaid');
        console.log(`Found ${mermaidContainers.length} Mermaid containers`);
        
        if (mermaidContainers.length === 0) {
          console.log('No Mermaid diagrams found - proceeding');
          resolve(true);
          return;
        }
        
        let renderedCount = 0;
        let totalDiagrams = mermaidContainers.length;
        
        // Check each mermaid container
        mermaidContainers.forEach((container, index) => {
          const svg = container.querySelector('svg');
          if (svg) {
            // Check if SVG has actual content (not just empty)
            const hasContent = svg.querySelector('g') || svg.children.length > 0;
            if (hasContent) {
              renderedCount++;
              console.log(`Diagram ${index + 1} is rendered`);
            } else {
              console.log(`Diagram ${index + 1} has SVG but no content yet`);
            }
          } else {
            console.log(`Diagram ${index + 1} has no SVG yet`);
          }
        });
        
        console.log(`${renderedCount}/${totalDiagrams} diagrams rendered`);
        
        if (renderedCount === totalDiagrams) {
          console.log('All Mermaid diagrams are rendered!');
          // Set window status for wkhtmltopdf
          window.status = 'mermaid-ready';
          resolve(true);
        } else if (checkCount >= maxChecks) {
          console.warn('Timeout waiting for Mermaid diagrams - proceeding anyway');
          // Set window status even on timeout
          window.status = 'mermaid-ready';
          resolve(false);
        } else {
          // Check again in 500ms
          setTimeout(checkMermaidStatus, 500);
        }
      }
      
      // Start checking after a brief initial delay
      setTimeout(checkMermaidStatus, 1000);
    });
  }
  
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      setTimeout(() => {
        waitForMermaidDiagrams().then((success) => {
          // Set a global flag that can be checked by wkhtmltopdf
          window.mermaidReady = true;
          window.mermaidSuccess = success;
          window.status = 'mermaid-ready';
          console.log('Mermaid rendering complete. Ready for PDF generation.');
          
          // Also add a visible indicator for debugging
          const indicator = document.createElement('div');
          indicator.id = 'mermaid-ready-indicator';
          indicator.style.display = 'none';
          indicator.textContent = 'MERMAID_READY';
          document.body.appendChild(indicator);
        });
      }, 500);
    });
  } else {
    setTimeout(() => {
      waitForMermaidDiagrams().then((success) => {
        window.mermaidReady = true;
        window.mermaidSuccess = success;
        window.status = 'mermaid-ready';
        console.log('Mermaid rendering complete. Ready for PDF generation.');
        
        const indicator = document.createElement('div');
        indicator.id = 'mermaid-ready-indicator';
        indicator.style.display = 'none';
        indicator.textContent = 'MERMAID_READY';
        document.body.appendChild(indicator);
      });
    }, 500);
  }
})();