#!/bin/bash
while true; do
    clear
    echo "=== Multi-Sample Evaluation Progress ==="
    echo "Time: $(date)"
    echo ""
    
    # Check if process is still running
    if pgrep -f "multi_sample_evaluation.*material_multisample_full" > /dev/null; then
        echo "Status: RUNNING"
        echo ""
        
        # Show sample progress
        echo "Sample Progress:"
        grep -o "Processing sample [0-9]*/[0-9]*" material_multisample_full.log | tail -1
        
        # Show detection counts
        echo ""
        echo "Recent Activity:"
        tail -5 material_multisample_full.log | grep -v "FutureWarning" | grep -v "encoder_attention_mask"
        
        # Check generated files
        echo ""
        echo "Samples Generated:"
        ls material_multisample_full/sample_*.csv 2>/dev/null | wc -l
        
    else
        echo "Status: COMPLETED"
        echo ""
        
        # Show summary if available
        if grep -q "SUMMARY" material_multisample_full.log; then
            echo "=== FINAL RESULTS ==="
            grep -A 100 "SUMMARY" material_multisample_full.log | head -50
        fi
        
        break
    fi
    
    sleep 30
done
