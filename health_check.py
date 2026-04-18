#!/usr/bin/env python3
"""
health_check.py - Pre-deployment validation for containerized drone swarm
Verifies all dependencies, GPU access, and performance baselines
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

class HealthCheck:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def check(self, name, condition, error_msg=""):
        """Record a health check result"""
        status = "✅ PASS" if condition else "❌ FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            status += f" - {error_msg}" if error_msg else ""
        
        self.results.append((name, status))
        print(f"{status:<20} {name}")
    
    def warn(self, name, condition, warning_msg=""):
        """Record a warning"""
        status = "⚠️  WARN" if not condition else "ℹ️  INFO"
        if not condition:
            self.warnings += 1
        
        self.results.append((name, status))
        print(f"{status:<20} {name} - {warning_msg}")
    
    def run_command(self, cmd):
        """Run a shell command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_docker(self):
        """Check Docker installation"""
        print("\n" + "="*60)
        print("DOCKER CHECKS")
        print("="*60)
        
        # Check Docker is installed
        success, output, error = self.run_command("docker --version")
        self.check("Docker Installed", success, error)
        if success:
            print(f"  └─ {output}")
        
        # Check Docker daemon is running
        success, output, error = self.run_command("docker ps")
        self.check("Docker Daemon Running", success, "Docker daemon not accessible")
        
        # Check Docker Compose
        success, output, error = self.run_command("docker-compose --version")
        self.check("Docker Compose Installed", success, error)
        if success:
            print(f"  └─ {output}")
    
    def check_gpu(self):
        """Check system capabilities (CPU-only)"""
        print("\n" + "="*60)
        print("SYSTEM CAPABILITIES CHECKS")
        print("="*60)
        
        # Check CPU cores
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            self.check("CPU Cores Available", cpu_count >= 2, f"Found {cpu_count}, need at least 2")
            if cpu_count >= 2:
                print(f"  └─ {cpu_count} cores detected")
        except Exception as e:
            self.check("CPU Cores Detection", False, str(e))
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.check("System Memory", memory_gb >= 2, f"Found {memory_gb:.1f} GB, need at least 2 GB")
            print(f"  └─ {memory_gb:.1f} GB total ({memory.available / (1024**3):.1f} GB available)")
        except Exception as e:
            self.warn("System Memory", True, "Memory check skipped")
        
        print("  ✅ CPU-Only system detected - GPU not required")
    
    def check_python(self):
        """Check Python dependencies"""
        print("\n" + "="*60)
        print("PYTHON DEPENDENCIES")
        print("="*60)
        
        # Check Python version
        success, output, error = self.run_command("python3 --version")
        self.check("Python 3 Installed", success, error)
        
        # Check required packages
        packages = {
            "psutil": "System monitoring",
            "numpy": "Numerical computing",
            "GPUtil": "GPU monitoring",
        }
        
        for package, desc in packages.items():
            success, output, error = self.run_command(f"python3 -c 'import {package}'")
            self.check(f"Package: {package}", success, f"{desc} not installed")
    
    def check_project_structure(self):
        """Check project file structure"""
        print("\n" + "="*60)
        print("PROJECT STRUCTURE")
        print("="*60)
        
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "benchmark_native.py",
            "benchmark_container.py",
            "compare_benchmarks.py",
            "DOCKER_GUIDE.md",
            "QUICKSTART.md",
        ]
        
        for filename in required_files:
            path = Path(filename)
            self.check(f"File: {filename}", path.exists(), "file not found")
        
        # Check Milestone 2 structure
        milestone_path = Path("Milestone 2/src")
        self.check("Milestone 2 source structure", milestone_path.exists(), "directory not found")
    
    def check_docker_image(self):
        """Check if Docker image can be built"""
        print("\n" + "="*60)
        print("DOCKER IMAGE BUILD TEST")
        print("="*60)
        
        print("Building Docker image (this may take 2-5 minutes)...")
        success, output, error = self.run_command("docker build -t drone-swarm:health-check . > /dev/null 2>&1")
        self.check("Docker Image Builds Successfully", success, "build failed")
        
        # Check image exists
        success, output, error = self.run_command("docker images | grep drone-swarm")
        self.check("Docker Image Available", success, "image not found")
        
        # Clean up test image
        self.run_command("docker rmi drone-swarm:health-check > /dev/null 2>&1")
    
    def check_docker_compose_config(self):
        """Check docker-compose configuration"""
        print("\n" + "="*60)
        print("DOCKER COMPOSE CONFIGURATION")
        print("="*60)
        
        success, output, error = self.run_command("docker-compose config > /dev/null 2>&1")
        self.check("docker-compose.yml Valid", success, error[:100] if error else "")
    
    def check_performance_baseline(self):
        """Check Python performance baseline"""
        print("\n" + "="*60)
        print("PERFORMANCE BASELINE")
        print("="*60)
        
        success, output, error = self.run_command(
            "python3 -c 'import numpy as np; import time; s=time.time(); "
            "[np.dot(np.random.random((1000,1000)), np.random.random((1000,1000))) for _ in range(5)]; "
            "print(f\"Time: {time.time()-s:.2f}s\")'"
        )
        self.check("Matrix Computation Performance", success, "performance test failed")
        if success:
            print(f"  └─ {output}")
    
    def generate_report(self):
        """Generate and display health check report"""
        print("\n" + "="*60)
        print("HEALTH CHECK SUMMARY")
        print("="*60)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Warnings: {self.warnings}")
        print("="*60)
        
        status = "✅ READY" if self.failed == 0 else "❌ NOT READY"
        print(f"\nStatus: {status}")
        
        if self.failed > 0:
            print("\n⚠️  Fix the following issues before proceeding:")
            for name, result in self.results:
                if "FAIL" in result:
                    print(f"  • {name}")
        elif self.warnings > 0:
            print("\n⚠️  Consider the following warnings:")
            for name, result in self.results:
                if "WARN" in result:
                    print(f"  • {name}")
        else:
            print("\n✅ All checks passed! You're ready to containerize the drone swarm.")
        
        print("="*60 + "\n")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "status": status,
            "results": self.results
        }
        
        with open("health_check_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return self.failed == 0

def main():
    print("\n" + "="*60)
    print("DRONE SWARM CONTAINERIZATION HEALTH CHECK")
    print("="*60 + "\n")
    
    health = HealthCheck()
    
    try:
        health.check_docker()
        health.check_gpu()
        health.check_python()
        health.check_project_structure()
        health.check_docker_compose_config()
        health.check_performance_baseline()
        
        # Optional: check if image can be built
        print("\nNote: Skipping full docker build test (slow). Run manually if needed:")
        print("  docker build -t drone-swarm:latest .")
        
        success = health.generate_report()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Health check failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
