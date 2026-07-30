# 
# NexusCache-Engine GPU & System Memory Profiler
#
# Usage:
#   ./benchmarks/profile_memory.sh [OPTIONS]
#
# Examples:
#   ./benchmarks/profile_memory.sh --concurrency 64 --num-requests 200
#   ./benchmarks/profile_memory.sh --output-dir ./results/profile_run_1


set -euo pipefail

#
# Default Configuration Variables
TARGET_URL="${TARGET_URL:-http://localhost:8000/v1/chat/completions}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
QPS="${QPS:-0}" # 0 = Max speed
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.2}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmarks/results/profile_$(date +%Y%m%m_%H%M%S)}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color


# Parse Command Line Arguments

while [[ $# -gt 0 ]]; do
  case $1 in
    --url)
      TARGET_URL="$2"; shift 2 ;;
    --concurrency)
      CONCURRENCY="$2"; shift 2 ;;
    --num-requests)
      NUM_REQUESTS="$2"; shift 2 ;;
    --qps)
      QPS="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --engine)
      CONTAINER_ENGINE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--url URL] [--concurrency INT] [--num-requests INT] [--qps FLOAT] [--output-dir DIR] [--engine podman|docker]"
      exit 0 ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}" >&2; exit 1 ;;
  esac
done

# Initialization & Environment Checks
mkdir -p "$OUTPUT_DIR"
GPU_LOG="$OUTPUT_DIR/gpu_memory.csv"
SUMMARY_LOG="$OUTPUT_DIR/summary_report.txt"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}        NexusCache Memory Profiling Engine            ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Target URL      : ${YELLOW}${TARGET_URL}${NC}"
echo -e "Concurrency     : ${YELLOW}${CONCURRENCY}${NC}"
echo -e "Total Requests  : ${YELLOW}${NUM_REQUESTS}${NC}"
echo -e "Sample Interval : ${YELLOW}${SAMPLE_INTERVAL_SEC}s${NC}"
echo -e "Output Directory: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "------------------------------------------------------"

# Verify nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: 'nvidia-smi' could not be found. Ensure GPU drivers are accessible.${NC}" >&2
    exit 1
fi


# Background GPU Telemetry Daemon
echo "timestamp_ms,gpu_index,gpu_name,vram_used_mb,vram_total_mb,gpu_util_pct,power_draw_w" > "$GPU_LOG"

start_gpu_profiling() {
    echo -e "${GREEN}[+] Starting GPU telemetry background monitor...${NC}"
    (
        while true; do
            TS=$(date +%s%3N)
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
                       --format=csv,noheader,nounits | \
            awk -v ts="$TS" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4 "," $5 "," $6}' >> "$GPU_LOG"
            sleep "$SAMPLE_INTERVAL_SEC"
        done
    ) &
    PROFILER_PID=$!
}

cleanup() {
    if [[ -n "${PROFILER_PID:-}" ]]; then
        echo -e "\n${YELLOW}[!] Stopping telemetry monitor (PID: $PROFILER_PID)...${NC}"
        kill "$PROFILER_PID" 2>/dev/null || true
        wait "$PROFILER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT SIGINT SIGTERM


# Execution & Profiling Phase

# 1. Warm-up / Baseline measurement
start_gpu_profiling
echo -e "${GREEN}[+] Sampling baseline idle memory for 3 seconds...${NC}"
sleep 3

# 2. Trigger Benchmark Suite
echo -e "${GREEN}[+] Launching Benchmark Load Generator...${NC}"
BENCHMARK_START=$(date +%s)

if command -v $CONTAINER_ENGINE &> /dev/null; then
    $CONTAINER_ENGINE compose -f benchmarks/docker-compose.yaml exec serving-engine python benchmarks/load_generator.py \
      --url "$TARGET_URL" \
      --qps "$QPS" \
      --num-requests "$NUM_REQUESTS" \
      --concurrency "$CONCURRENCY" \
      --output-json "$OUTPUT_DIR/benchmark_metrics.json"
else
    python benchmarks/load_generator.py \
      --url "$TARGET_URL" \
      --qps "$QPS" \
      --num-requests "$NUM_REQUESTS" \
      --concurrency "$CONCURRENCY" \
      --output-json "$OUTPUT_DIR/benchmark_metrics.json"
fi

BENCHMARK_END=$(date +%s)
echo -e "${GREEN}[+] Sampling post-run memory stabilization for 3 seconds...${NC}"
sleep 3

# Kill background profiler loop
cleanup


# Post-Processing & Metrics Reporting
echo -e "${CYAN}------------------------------------------------------${NC}"
echo -e "${CYAN}             Memory Profile Summary                   ${NC}"
echo -e "${CYAN}------------------------------------------------------${NC}"

python3 - << EOF
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("$GPU_LOG")
    if df.empty:
        print("No telemetry data captured.")
        exit(0)

    baseline_vram = df['vram_used_mb'].iloc[:5].mean()
    peak_vram = df['vram_used_mb'].max()
    max_total_vram = df['vram_total_mb'].max()
    avg_gpu_util = df['gpu_util_pct'].mean()
    peak_power = df['power_draw_w'].max()

    delta_vram = peak_vram - baseline_vram
    vram_headroom = max_total_vram - peak_vram

    report = f"""
Baseline Idle VRAM : {baseline_vram:.2f} MB
Peak Active VRAM   : {peak_vram:.2f} MB / {max_total_vram:.2f} MB ({peak_vram/max_total_vram*100:.1f}%)
Dynamic KV Allocation: +{delta_vram:.2f} MB
Remaining Headroom : {vram_headroom:.2f} MB
Average GPU Util   : {avg_gpu_util:.1f}%
Max Power Consumption: {peak_power:.2f} W
"""
    print(report)
    with open("$SUMMARY_LOG", "w") as f:
        f.write(report)

except Exception as e:
    print(f"Error analyzing log data: {e}")
EOF

echo -e "${GREEN}Telemetry logs saved to:${NC} ${OUTPUT_DIR}/"