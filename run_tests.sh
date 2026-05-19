#!/usr/bin/env bash
# AgentMem Test Runner
# ====================
# Quick commands for running different test suites

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AgentMem Test Suite Runner          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

case "${1:-all}" in
  unit)
    echo -e "${GREEN}Running unit tests...${NC}"
    uv run pytest tests/unit/ -v --tb=short
    ;;
  
  integration)
    echo -e "${GREEN}Running integration tests...${NC}"
    uv run pytest tests/integration/ -v --tb=short
    ;;
  
  benchmark)
    echo -e "${GREEN}Running performance benchmarks...${NC}"
    uv run pytest tests/benchmark/ -v --benchmark-verbose
    ;;
  
  heat)
    echo -e "${GREEN}Running heat scoring tests...${NC}"
    uv run pytest tests/unit/test_heat.py -v
    ;;
  
  api)
    echo -e "${GREEN}Running API integration tests...${NC}"
    uv run pytest tests/integration/test_api.py -v --tb=short
    ;;
  
  scale)
    echo -e "${GREEN}Running scale benchmarks...${NC}"
    uv run pytest tests/benchmark/test_scale.py -v --benchmark-verbose
    ;;
  
  fast)
    echo -e "${GREEN}Running fast tests only (< 60s)...${NC}"
    uv run pytest tests/unit/ -q -x
    ;;
  
  coverage)
    echo -e "${GREEN}Running tests with coverage...${NC}"
    uv pip install pytest-cov > /dev/null 2>&1 || true
    uv run pytest --cov=core --cov-report=html --cov-report=term-missing
    echo -e "\n${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
    ;;
  
  compare)
    echo -e "${GREEN}Comparing against baseline...${NC}"
    if [ -z "$2" ]; then
      echo -e "${RED}Usage: $0 compare <baseline-id>${NC}"
      echo -e "Example: $0 compare 0001_baseline"
      exit 1
    fi
    uv run pytest tests/benchmark/ --benchmark-compare="$2"
    ;;
  
  save)
    echo -e "${GREEN}Saving benchmark baseline...${NC}"
    uv run pytest tests/benchmark/ --benchmark-save="${2:-baseline}"
    ;;
  
  all)
    echo -e "${GREEN}Running complete test suite...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}\n"
    
    echo -e "${BLUE}1. Unit Tests${NC}"
    uv run pytest tests/unit/ -q || true
    echo ""
    
    echo -e "${BLUE}2. Integration Tests${NC}"
    uv run pytest tests/integration/ -q || true
    echo ""
    
    echo -e "${BLUE}3. Benchmarks${NC}"
    uv run pytest tests/benchmark/ -q --benchmark-disable || true
    ;;
  
  help|*)
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [command] [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  unit        Run unit tests only"
    echo "  integration Run integration tests only"
    echo "  benchmark   Run performance benchmarks"
    echo "  heat        Run heat scoring tests"
    echo "  api         Run API integration tests"
    echo "  scale       Run scale benchmarks"
    echo "  fast        Run fast tests only (< 60s)"
    echo "  coverage    Run with code coverage"
    echo "  compare     Compare against baseline (requires baseline ID)"
    echo "  save        Save current as baseline"
    echo "  all         Run complete test suite (default)"
    echo "  help        Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 unit"
    echo "  $0 benchmark"
    echo "  $0 save my-baseline"
    echo "  $0 compare 0001_my-baseline"
    echo "  $0 coverage"
    ;;
esac

echo ""
echo -e "${GREEN}✓ Test run complete!${NC}"
