package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// MaterialValidator holds the logic for validating material strings.
type MaterialValidator struct {
	tokenizerRegex *regexp.Regexp
	validationCache map[string]bool
}

// NewMaterialValidator creates and initializes a new validator.
func NewMaterialValidator() *MaterialValidator {
	// A compiled regex for high-speed tokenization.
	regex := regexp.MustCompile(`(\d+(?:\.\d+)?|[a-zA-Z]+|[^a-zA-Z0-9\s])`)
	return &MaterialValidator{
		tokenizerRegex: regex,
		validationCache: make(map[string]bool),
	}
}

// tokenize breaks a raw string into its fundamental components.
func (v *MaterialValidator) tokenize(s string) []string {
	return v.tokenizerRegex.FindAllString(s, -1)
}

// isNumeric checks if a string can be parsed as a number.
func isNumeric(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

// validatePart checks if a given sequence of tokens sums to 100.
func (v *MaterialValidator) validatePart(tokens []string) bool {
	var total float64
	hasNumbers := false
	for _, t := range tokens {
		if isNumeric(t) {
			hasNumbers = true
			num, err := strconv.ParseFloat(t, 64)
			if err == nil {
				total += num
			}
		}
	}
	return hasNumbers && math.Abs(total-100.0) < 1e-6
}

// findValidPartition hypothesizes and tests different partitions of the token list.
func (v *MaterialValidator) findValidPartition(tokens []string) [][]string {
	// Hypothesis 1: The entire string is a single part.
	if v.validatePart(tokens) {
		return [][]string{tokens}
	}

	// Hypothesis 2: Split by common delimiters.
	potentialDelimiters := []string{"//", "|"}
	for _, delim := range potentialDelimiters {
		if contains(tokens, delim) {
			parts := splitTokens(tokens, delim)
			if len(parts) > 1 && allPartsValid(parts, v.validatePart) {
				return parts
			}
		}
	}

	// Hypothesis 3: Split by "KEYWORD:" patterns.
	parts, isMultiPart := splitByKeywordColon(tokens)
	if isMultiPart && len(parts) > 1 && allPartsValid(parts, v.validatePart) {
		return parts
	}

	return nil
}

// Validate checks a single material string, using a cache for speed.
func (v *MaterialValidator) Validate(materialString string) bool {
	if cachedResult, found := v.validationCache[materialString]; found {
		return cachedResult
	}

	tokens := v.tokenize(materialString)
	if len(tokens) == 0 {
		v.validationCache[materialString] = false
		return false
	}

	result := v.findValidPartition(tokens) != nil
	v.validationCache[materialString] = result
	return result
}

// --- Helper functions to make the logic cleaner ---

func contains(slice []string, item string) bool {
	for _, a := range slice {
		if a == item {
			return true
		}
	}
	return false
}

func splitTokens(tokens []string, delim string) [][]string {
	var parts [][]string
	currentPart := []string{}
	for _, token := range tokens {
		if token == delim {
			if len(currentPart) > 0 {
				parts = append(parts, currentPart)
			}
			currentPart = []string{}
		} else {
			currentPart = append(currentPart, token)
		}
	}
	if len(currentPart) > 0 {
		parts = append(parts, currentPart)
	}
	return parts
}

func allPartsValid(parts [][]string, validateFunc func([]string) bool) bool {
	for _, p := range parts {
		if !validateFunc(p) {
			return false
		}
	}
	return true
}

func splitByKeywordColon(tokens []string) (parts [][]string, isMultiPart bool) {
	currentPart := []string{}
	i := 0
	for i < len(tokens) {
		if i+1 < len(tokens) && regexp.MustCompile(`^[a-zA-Z]+$`).MatchString(tokens[i]) && tokens[i+1] == ":" {
			isMultiPart = true
			if len(currentPart) > 0 {
				parts = append(parts, currentPart)
			}
			currentPart = []string{tokens[i], tokens[i+1]}
			i += 2
		} else {
			currentPart = append(currentPart, tokens[i])
			i++
		}
	}
	if len(currentPart) > 0 {
		parts = append(parts, currentPart)
	}
	return parts, isMultiPart
}


func main() {
	// --- 1. Load Data ---
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <path_to_csv_file>")
		os.Exit(1)
	}
	filePath := os.Args[1]

	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Error: '%s' not found.", filePath)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		log.Fatalf("Error reading header from '%s': %v", filePath, err)
	}

	materialColIndex := -1
	for i, colName := range header {
		if strings.ToLower(colName) == "material" {
			materialColIndex = i
			break
		}
	}

	if materialColIndex == -1 {
		log.Fatal("Error: 'material' column not found.")
	}

	uniqueMaterials := make(map[string]struct{})
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // Skip rows with parsing errors
		}
		if len(record) > materialColIndex && record[materialColIndex] != "" {
			uniqueMaterials[record[materialColIndex]] = struct{}{}
		}
	}
	
	materialsList := make([]string, 0, len(uniqueMaterials))
	for material := range uniqueMaterials {
		materialsList = append(materialsList, material)
	}
	fmt.Printf("Loaded %d unique material strings for analysis from '%s'.\n\n", len(materialsList), filePath)
	
	// --- 2. Initialize and Run Validator ---
	validator := NewMaterialValidator()
	startTime := time.Now()

	var valid, invalid []string
	for _, material := range materialsList {
		if validator.Validate(material) {
			valid = append(valid, material)
		} else {
			invalid = append(invalid, material)
		}
	}
	
	duration := time.Since(startTime)

	// --- 3. Report the Results ---
	fmt.Printf("Validation completed in %v seconds.\n\n", duration.Seconds())
	
	sort.Strings(valid)
	sort.Strings(invalid)
	
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("          VALID Material Strings")
	fmt.Println(strings.Repeat("=", 50))
	if len(valid) == 0 {
		fmt.Println("No valid materials found.")
	} else {
		fmt.Printf("Found %d valid material formats:\n\n", len(valid))
		for _, item := range valid {
			fmt.Printf("  - '%s'\n", item)
		}
	}

	fmt.Println("\n\n" + strings.Repeat("=", 50))
	fmt.Println("         INVALID Material Strings")
	fmt.Println(strings.Repeat("=", 50))
	if len(invalid) == 0 {
		fmt.Println("No invalid materials found.")
	} else {
		fmt.Printf("Found %d materials that failed validation:\n\n", len(invalid))
		for _, item := range invalid {
			fmt.Printf("  - '%s'\n", item)
		}
	}
}
