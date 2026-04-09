.PHONY: lint test test-cov ci check setup cleanup gb200-fp8 gb200-fp4

NATS_VERSION ?= v2.10.28
ETCD_VERSION ?= v3.5.21
LOGS_DIR ?= logs
ARCH ?= $(shell uname -m)

default:
	./run_dashboard.sh

# === CI targets ===
lint:
	uv run ruff check src/srtctl/
	uv run ruff format src/srtctl/
	uv run ty check src/srtctl/ || true

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=srtctl --cov-report=term-missing --cov-report=html

# Run lint + tests in one command
check: lint test
	@echo "✓ All checks passed"

# Runners
gb200-fp8:
	srtctl apply -f recipes/gb200-fp8/1k1k/low-latency.yaml
	srtctl apply -f recipes/gb200-fp8/1k1k/max-tpt-2p1d.yaml
	srtctl apply -f recipes/gb200-fp8/1k1k/mid-curve-3p1d.yaml
	srtctl apply -f recipes/gb200-fp8/8k1k/low-latency.yaml
	srtctl apply -f recipes/gb200-fp8/8k1k/mid-curve-5p1d.yaml

gb200-fp4:
	srtctl apply -f recipes/gb200-fp4/1k1k/low-latency.yaml
	srtctl apply -f recipes/gb200-fp4/1k1k/max-tpt.yaml
	srtctl apply -f recipes/gb200-fp4/1k1k/mid-curve.yaml
	srtctl apply -f recipes/gb200-fp4/8k1k/low-latency.yaml
	srtctl apply -f recipes/gb200-fp4/8k1k/max-tpt.yaml
	srtctl apply -f recipes/gb200-fp4/8k1k/mid-curve.yaml

setup:
	@echo "📦 Setting up configs and logs directories..."
	@mkdir -p logs
	@echo "🖥️  Using architecture: $(ARCH)"
	@case "$(ARCH)" in \
		x86_64)  ARCH_SHORT="amd64" ;; \
		aarch64) ARCH_SHORT="arm64" ;; \
		*) echo "❌ Unsupported architecture: $(ARCH)"; exit 1 ;; \
	esac; \
	echo "ℹ️  Dynamo will be installed from PyPI when workers start"; \
	echo "⬇️  Downloading NATS ($(NATS_VERSION)) for $$ARCH_SHORT..."; \
	NATS_DEB="nats-server-$(NATS_VERSION)-$$ARCH_SHORT.deb"; \
	NATS_URL="https://github.com/nats-io/nats-server/releases/download/$(NATS_VERSION)/$$NATS_DEB"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$NATS_URL" -O "configs/$$NATS_DEB"; \
	echo "📁 Extracting NATS binary..."; \
	TMP_DIR=$$(mktemp -d); \
	dpkg-deb -x "configs/$$NATS_DEB" "$$TMP_DIR"; \
	if [ -f "$$TMP_DIR/usr/local/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/local/bin/nats-server" configs/; \
	elif [ -f "$$TMP_DIR/usr/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/bin/nats-server" configs/; \
	else \
		echo "❌ Could not find nats-server binary inside the .deb package"; \
		ls -R "$$TMP_DIR" | head -n 50; \
		exit 1; \
	fi; \
	chmod +x configs/nats-server; \
	rm -rf "$$TMP_DIR" "configs/$$NATS_DEB"; \
	echo "⬇️  Downloading ETCD ($(ETCD_VERSION)) for $$ARCH_SHORT..."; \
	ETCD_TAR="etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT.tar.gz"; \
	ETCD_URL="https://github.com/etcd-io/etcd/releases/download/$(ETCD_VERSION)/$$ETCD_TAR"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$ETCD_URL" -O "configs/$$ETCD_TAR"; \
	echo "📁 Extracting ETCD binaries..."; \
	tar -xzf "configs/$$ETCD_TAR" --strip-components=1 -C configs etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcd etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcdctl; \
	chmod +x configs/etcd configs/etcdctl; \
	rm "configs/$$ETCD_TAR"; \
	echo "✅ Done. Contents of configs directory:"; \
	ls -lh configs/; \
	echo ""; \
	echo "⚙️  Setting up srtslurm.yaml..."; \
	if [ -f srtslurm.yaml ]; then \
		echo "ℹ️  srtslurm.yaml already exists, skipping..."; \
	else \
		echo "Creating srtslurm.yaml with your cluster settings..."; \
		echo ""; \
		SRTCTL_ROOT=$$(pwd); \
		echo "📍 Auto-detected srtctl root: $$SRTCTL_ROOT"; \
		echo ""; \
		read -p "Enter SLURM account [restricted]: " account; \
		account=$${account:-restricted}; \
		read -p "Enter SLURM partition [batch]: " partition; \
		partition=$${partition:-batch}; \
		read -p "Enter GPUs per node [4]: " gpus_per_node; \
		gpus_per_node=$${gpus_per_node:-4}; \
		read -p "Enter time limit [4:00:00]: " time_limit; \
		time_limit=$${time_limit:-4:00:00}; \
		echo ""; \
		echo "# SRT SLURM Configuration" > srtslurm.yaml; \
		echo "# This file provides cluster-specific defaults and settings for srtctl" >> srtslurm.yaml; \
		echo "" >> srtslurm.yaml; \
		echo "# Default SLURM settings" >> srtslurm.yaml; \
		echo "default_account: \"$$account\"" >> srtslurm.yaml; \
		echo "default_partition: \"$$partition\"" >> srtslurm.yaml; \
		echo "default_time_limit: \"$$time_limit\"" >> srtslurm.yaml; \
		echo "" >> srtslurm.yaml; \
		echo "# Resource defaults" >> srtslurm.yaml; \
		echo "gpus_per_node: $$gpus_per_node" >> srtslurm.yaml; \
		echo "network_interface: \"\"" >> srtslurm.yaml; \
		echo "" >> srtslurm.yaml; \
		echo "# Path to srtctl repo root (where scripts/templates/ lives)" >> srtslurm.yaml; \
		echo "# Auto-detected from current directory" >> srtslurm.yaml; \
		echo "srtctl_root: \"$$SRTCTL_ROOT\"" >> srtslurm.yaml; \
		echo "✅ Created srtslurm.yaml"; \
		echo "   You can edit it anytime to add model_paths, containers, etc."; \
	fi

cleanup:
	@echo "🧹 Scanning logs directory for runs without benchmark results..."
	@EMPTY_DIRS=""; \
	if [ ! -d "$(LOGS_DIR)" ]; then \
		echo "❌ Logs directory $(LOGS_DIR) does not exist"; \
		exit 1; \
	fi; \
	for dir in $(LOGS_DIR)/*/; do \
		if [ -d "$$dir" ]; then \
			run_name=$$(basename "$$dir"); \
			has_subdirs=$$(find "$$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l); \
			if [ "$$has_subdirs" -eq 0 ]; then \
				EMPTY_DIRS="$$EMPTY_DIRS$$dir\n"; \
			fi; \
		fi; \
	done; \
	if [ -z "$$EMPTY_DIRS" ]; then \
		echo "✅ No empty run directories found!"; \
		exit 0; \
	fi; \
	echo ""; \
	echo "Found the following run directories without benchmark results:"; \
	echo ""; \
	echo "$$EMPTY_DIRS" | grep -v '^$$'; \
	echo ""; \
	read -p "❗ Delete these directories? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$$EMPTY_DIRS" | grep -v '^$$' | while read -r dir; do \
			if [ -n "$$dir" ]; then \
				echo "🗑️  Removing $$dir"; \
				rm -rf "$$dir"; \
			fi; \
		done; \
		echo "✅ Cleanup complete!"; \
	else \
		echo "❌ Cleanup cancelled."; \
	fi
