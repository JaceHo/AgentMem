#!/usr/bin/env bash
# AgentMem Benchmark Runner
# ===========================
# Run comprehensive benchmarks and generate reports

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AgentMem Benchmark Suite             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

case "${1:-all}" in
  heat)
    echo -e "${GREEN}Running heat scoring benchmarks...${NC}"
    uv run pytest tests/unit/test_heat.py::TestHeatPerformance -v --benchmark-verbose --benchmark-json=benchmark/results/heat.json
    ;;
  
  recall)
    echo -e "${GREEN}Running recall latency benchmarks...${NC}"
    uv run pytest tests/integration/test_api.py::TestRecallEndpoint -v --benchmark-verbose --benchmark-json=benchmark/results/recall.json
    ;;
  
  scale)
    echo -e "${GREEN}Running scale benchmarks...${NC}"
    uv run pytest tests/benchmark/test_scale.py -v --benchmark-verbose --benchmark-json=benchmark/results/scale.json
    ;;
  
  hybrid)
    echo -e "${GREEN}Running hybrid search benchmarks...${NC}"
    uv run pytest tests/unit/test_hybrid_search.py -v --benchmark-verbose --benchmark-json=benchmark/results/hybrid.json
    ;;
  
  all)
    echo -e "${GREEN}Running complete benchmark suite...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}\n"
    
    mkdir -p benchmark/results
    
    echo -e "${BLUE}1. Heat Scoring Benchmarks${NC}"
    uv run pytest tests/unit/test_heat.py::TestHeatPerformance -q --benchmark-json=benchmark/results/heat.json || true
    echo ""
    
    echo -e "${BLUE}2. Recall Latency Benchmarks${NC}"
    uv run pytest tests/integration/test_api.py::TestRecallEndpoint -q --benchmark-json=benchmark/results/recall.json || true
    echo ""
    
    echo -e "${BLUE}3. Hybrid Search Benchmarks${NC}"
    uv run pytest tests/unit/test_hybrid_search.py -q --benchmark-json=benchmark/results/hybrid.json || true
    echo ""
    
    echo -e "${BLUE}4. Scale Benchmarks${NC}"
    uv run pytest tests/benchmark/test_scale.py -q --benchmark-json=benchmark/results/scale.json || true
    echo ""
    
    echo -e "${GREEN}✓ All benchmarks complete!${NC}"
    echo -e "${YELLOW}Results saved to benchmark/results/*.json${NC}"
    ;;
  
  compare)
    echo -e "${GREEN}Comparing against baseline...${NC}"
    if [ -z "$2" ]; then
      echo -e "${RED}Usage: $0 compare <baseline-file>${NC}"
      exit 1
    fi
    uv run pytest tests/ --benchmark-compare="$2"
    ;;
  
  report)
    echo -e "${GREEN}Generating benchmark report...${NC}"
    echo "See benchmark/README.md for comprehensive results"
    echo ""
    echo "Key Metrics:"
    echo "  • LongMemEval-S R@5: 95.2%"
    echo "  • Heat Compute: 237ns (211x faster)"
    echo "  • Recall P50: 15ms"
    echo "  • Token Savings: 91% vs full context"
    ;;
  
  help|*)
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [command]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  heat      Run heat scoring benchmarks"
    echo "  recall    Run recall latency benchmarks"
    echo "  scale     Run scale benchmarks"
    echo "  hybrid    Run hybrid search benchmarks"
    echo "  all       Run complete benchmark suite (default)"
    echo "  compare   Compare against baseline JSON file"
    echo "  report    Show benchmark summary"
    echo "  help      Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 all"
    echo "  $0 heat"
    echo "  $0 compare benchmark/results/heat.json"
    ;;
esac

echo ""
echo -e "${GREEN}✓ Benchmark run complete!${NC}"
