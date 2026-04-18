# Docker Containerization Guide for Drone Swarm Coordination

## Overview
This guide containerizes the entire drone swarm simulation environment for CPU-only systems (Intel integrated graphics compatible), ensuring performance is maintained while providing reproducible, quantifiable testing.

## Prerequisites

### System Requirements
- **CPU**: Multi-core processor (2+ cores recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Graphics**: Intel integrated graphics or any standard display output
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

### Installation

#### 1. Install Docker (if not already installed)

**Windows (Docker Desktop):**
- Download: https://www.docker.com/products/docker-desktop
- Run installer and follow setup
- Restart system if prompted

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Logout and login for group changes to take effect
```

**macOS:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop
- Run installer and follow setup

#### 2. Install Docker Compose
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Verify Docker Installation
```bash
docker --version
docker-compose --version
docker run hello-world  # Should see "Hello from Docker!"
```

## Building the Container

### Build with Caching
```bash
docker build -t drone-swarm:latest .
```

### Build with No Cache (force rebuild)
```bash
docker build --no-cache -t drone-swarm:latest .
```

### Check Image Size
```bash
docker images | grep drone-swarm
```

## Running the Simulation

### Interactive GUI Mode (with Display)
```bash
# Linux/WSL2: Set DISPLAY variable
export DISPLAY=:0

# Run simulation
docker-compose up drone-swarm
```

### Run with Docker Run (Alternative)
```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/benchmarks:/app/benchmarks \
  --mem 2g \
  --cpus 3 \
  drone-swarm:latest
```

### Headless Mode (No Display)
```bash
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/benchmarks:/app/benchmarks \
  --mem 2g \
  --cpus 3 \
  drone-swarm:latest python3 Milestone\ 2/src/main.py
```

## Performance Benchmarking

### Run Headless Benchmark
```bash
# Create benchmark volumes
mkdir -p logs benchmarks results

# Run benchmark service
docker-compose --profile benchmark up drone-swarm-benchmark
```

### Post-Benchmark Analysis
Benchmark results are saved to `benchmarks/benchmark_*.json` with metrics including:
- FPS (average, min, max, std dev)
- Frame time (ms)
- Compute time (ms)
- CPU usage (%)
- Memory usage (MiB)
- GPU usage (%)
- GPU memory usage (%)

### Compare Native vs Containerized Performance
```bash
# Collect native performance
python3 benchmark_native.py

# Run containerized benchmark
docker-compose --profile benchmark up drone-swarm-benchmark

# Compare results
python3 compare_benchmarks.py
```

## Container Configuration

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `DISPLAY` | `$DISPLAY` | X11 display server (GUI mode) |
| `HEADLESS` | `0` | Set to 1 for headless/benchmark mode |
| `RAY_memory` | `1000000000` | Ray memory allocation (bytes) |
| `RAY_object_store_memory` | `500000000` | Ray object store (bytes) |
| `QT_X11_NO_MITSHM` | `1` | X11 shared memory setting |

### Resource Limits (docker-compose.yml)
- **Memory**: 2GB (adjust for your system)
- **CPU**: 3 cores (GUI mode), 3.5 cores (benchmark)

For systems with more resources, adjust in `docker-compose.yml`:
```yaml
mem_limit: 4g      # Increase for larger simulations
cpus: 4            # Increase for more cores
```

For systems with limited resources:
```yaml
mem_limit: 1g      # Reduce memory
cpus: 1.5          # Reduce CPU allocation
```

## Troubleshooting

### Docker Not Running
```bash
# Windows: Start Docker Desktop
# macOS/Linux: Start Docker daemon
sudo systemctl start docker

# Verify it's running
docker ps
```

### Docker Command Not Found
```bash
# Add Docker to PATH (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# Windows: Restart PowerShell after Docker Desktop installation
```

### Display Issues (Linux/WSL2)
```bash
# Allow X11 access from container
xhost +local:docker

# Set correct DISPLAY variable
echo $DISPLAY  # Note the value, usually :0
export DISPLAY=:0
docker-compose up drone-swarm
```

### Out of Memory
Increase memory limit in docker-compose.yml:
```yaml
mem_limit: 4g  # Increase from 2g
```

Check system memory:
```bash
# Windows PowerShell
Get-ComputerInfo | Select-Object CsPhyicallyInstalledSystemMemory

# Linux
free -h
```

### High CPU Usage
- Increase `cpus` limit in docker-compose.yml (up to host CPU count minus 1)
- Reduce simulation parameters in config.yaml (num_boids, perception_radius)
- Check for other applications consuming CPU

### Slow Performance
1. Check current resource usage: `docker stats drone-swarm-sim`
2. Check logs: `docker logs drone-swarm-sim`
3. Review benchmark report in `benchmarks/` directory
4. Reduce simulation complexity or increase container resources

## Performance Optimization Tips

### For Maximum Performance on CPU-Only Systems
1. **Dedicated CPU Cores**: Use `--cpuset-cpus` to bind specific cores
2. **Memory Allocation**: Allocate sufficient memory to avoid swapping
3. **System Isolation**: Close other applications during benchmarking
4. **CPU Affinity**: Pin container to high-performance cores

### Container Runtime Performance
```bash
# Run with CPU pinning (benchmark on specific cores)
docker run --rm \
  --cpuset-cpus=0-3 \
  --mem 2g \
  -e HEADLESS=1 \
  drone-swarm:latest python3 benchmark_container.py
```

### Verify Minimal Overhead
Host performance vs containerized performance should be within 10-15% for:
- Compute time per frame
- CPU utilization
- Memory usage
- Overall simulation throughput

If overhead exceeds 20%, investigate:
- CPU isolation/pinning
- Memory pressure and swapping
- Competing processes
- Container memory/CPU limits

### CPU-Specific Optimizations
- Enable CPU affinity for better cache locality
- Use performance CPU governors: `sudo cpupower frequency-set -g performance`
- Disable unnecessary system services during benchmarking
- Monitor thermal throttling: High temps reduce CPU performance

## Advanced: Multi-Container Orchestration

### Run Multiple Simulations in Parallel
```bash
docker-compose up -d --scale drone-swarm=3
```

This enables:
- Parallel benchmarking across different random seeds
- Distributed testing
- Scalability analysis

## Cleanup

### Stop All Containers
```bash
docker-compose down
```

### Remove Everything
```bash
docker-compose down -v
docker system prune -a
```

### Remove Specific Image
```bash
docker rmi drone-swarm:latest
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test Drone Swarm in Container

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: docker/build-push-action@v2
        with:
          context: .
          push: false
      - name: Run Benchmark
        run: |
          docker-compose --profile benchmark up --abort-on-container-exit
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/
```

## Performance Benchmarking Results

Expected baseline (native system):
- **FPS**: 60-120 (depends on hardware)
- **Frame Time**: 8-16 ms
- **CPU Usage**: 40-60%
- **Memory**: 800-1200 MiB
- **GPU Usage**: 60-80% (if available)

After containerization, expect:
- **Performance Loss**: 5-10% maximum
- **Memory Overhead**: +50-100 MiB
- **GPU Overhead**: Minimal (0-5%)

If containerized performance is significantly worse, verify:
1. GPU passthrough is working (`nvidia-smi` in container)
2. CPU allocation is adequate
3. No memory swapping occurring
4. Filesystem not mounted with `noexec` flag

---

## Performance Benchmarking Results

All metrics are logged to CSV and JSON in the `benchmarks/` directory:

```json
{
  "fps_avg": 45.5,
  "fps_min": 38.3,
  "fps_max": 52.2,
  "frame_time_avg_ms": 22.0,
  "compute_time_avg_ms": 18.5,
  "cpu_avg": 75.3,
  "memory_peak_mb": 512.5
}
```

### Expected Baseline (CPU-Only Systems)
- **FPS**: 30-60 (depends on CPU and simulation complexity)
- **Frame Time**: 16-33 ms
- **CPU Usage**: 60-85%
- **Memory**: 400-800 MiB
- **Thermal**: Monitor for throttling

### After Containerization, Expect
- **Performance Loss**: 10-15% maximum (higher than GPU due to container overhead)
- **Memory Overhead**: +100-200 MiB
- **CPU Overhead**: Minimal (5-10%)
- **I/O Overhead**: Minimal

### If performance is worse than expected:
1. Check CPU thermal throttling
2. Verify memory isn't swapping
3. Increase CPU allocation in docker-compose.yml
4. Close unneeded applications
5. Run native benchmark to establish baseline

Use these metrics for:
- Performance regression testing
- Hardware comparison
- Optimization validation
- SLA verification

---

For additional support, check the logs in `logs/` directory or run:
```bash
docker logs drone-swarm-sim
```
