#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# IP address resolution utilities for SLURM environments.
# Provides robust IP discovery across different cluster configurations (GB200, H100, etc.)

# Core IP resolution logic - tries multiple methods
# Usage: _resolve_ip "network_interface"
# Returns: IP address on stdout, exits with code 1 on failure
_resolve_ip() {
    local network_interface=$1

    _is_bad_ip() {
        local ip=$1
        case "$ip" in
            ""|0.0.0.0|127.*|169.254.*) return 0 ;;
            *) return 1 ;;
        esac
    }

    _is_private_ip() {
        local ip=$1
        case "$ip" in
            10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) return 0 ;;
            *) return 1 ;;
        esac
    }

    _select_best_ip() {
        # Prefer RFC1918 IPs, avoid loopback and link-local.
        local ip
        for ip in "$@"; do
            if _is_bad_ip "$ip"; then
                continue
            fi
            if _is_private_ip "$ip"; then
                echo "$ip"
                return 0
            fi
        done
        for ip in "$@"; do
            if _is_bad_ip "$ip"; then
                continue
            fi
            echo "$ip"
            return 0
        done
        return 1
    }

    # Method 1: Use specific interface if provided
    if [ -n "$network_interface" ]; then
        ips=$(ip addr show "$network_interface" 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
        if [ -n "$ips" ]; then
            ip=$(_select_best_ip $ips)
            if [ -n "$ip" ]; then
                echo "$ip"
                return 0
            fi
        fi
    fi

    # Method 2: Use ip route to find default source IP
    ip=$(ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split($2,a," ");print a[1]}')
    if [ -n "$ip" ] && ! _is_bad_ip "$ip"; then
        echo "$ip"
        return 0
    fi

    # Method 3: Use hostname -I (prefer RFC1918, avoid loopback/link-local)
    ips=$(hostname -I 2>/dev/null)
    if [ -n "$ips" ]; then
        ip=$(_select_best_ip $ips)
        if [ -n "$ip" ]; then
            echo "$ip"
            return 0
        fi
    fi

    return 1
}

# Get local IP address
# Usage: get_local_ip "network_interface"
# Returns: IP address on stdout, or "127.0.0.1" if all methods fail
get_local_ip() {
    local network_interface=$1
    
    local result
    result=$(_resolve_ip "$network_interface")
    
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "127.0.0.1"
    fi
}

# Get IP address of a remote SLURM node via srun
# Usage: get_node_ip "node_name" "slurm_job_id" "network_interface"
# Returns: IP address on stdout, exits with code 1 on failure
get_node_ip() {
    local node=$1
    local slurm_job_id=$2
    local network_interface=$3

    # Create inline script with the resolution logic
    local ip_script="
        _is_bad_ip() {
            ip=\$1
            case \"\$ip\" in
                \"\"|0.0.0.0|127.*|169.254.*) return 0 ;;
                *) return 1 ;;
            esac
        }

        _is_private_ip() {
            ip=\$1
            case \"\$ip\" in
                10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) return 0 ;;
                *) return 1 ;;
            esac
        }

        _select_best_ip() {
            for ip in \"\$@\"; do
                if _is_bad_ip \"\$ip\"; then
                    continue
                fi
                if _is_private_ip \"\$ip\"; then
                    echo \"\$ip\"
                    return 0
                fi
            done
            for ip in \"\$@\"; do
                if _is_bad_ip \"\$ip\"; then
                    continue
                fi
                echo \"\$ip\"
                return 0
            done
            return 1
        }

        # Method 1: Use specific interface if provided
        if [ -n \"$network_interface\" ]; then
            ips=\$(ip addr show $network_interface 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1)
            if [ -n \"\$ips\" ]; then
                ip=\$(_select_best_ip \$ips)
                if [ -n \"\$ip\" ]; then
                    echo \"\$ip\"
                    exit 0
                fi
            fi
        fi

        # Method 2: Use ip route to find default source IP
        ip=\$(ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split(\$2,a,\" \");print a[1]}')
        if [ -n \"\$ip\" ] && ! _is_bad_ip \"\$ip\"; then
            echo \"\$ip\"
            exit 0
        fi

        # Method 3: Use hostname -I (prefer RFC1918, avoid loopback/link-local)
        ips=\$(hostname -I 2>/dev/null)
        if [ -n \"\$ips\" ]; then
            ip=\$(_select_best_ip \$ips)
            if [ -n \"\$ip\" ]; then
                echo \"\$ip\"
                exit 0
            fi
        fi

        exit 1
    "

    local result
    result=$(srun --quiet --jobid "$slurm_job_id" --nodes=1 --ntasks=1 --nodelist="$node" bash -c "$ip_script" 2>&1)
    local rc=$?
    local ip
    ip=$(printf '%s\n' "$result" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | tail -n 1)

    if [ $rc -eq 0 ] && [ -n "$ip" ]; then
        echo "$ip"
        return 0
    else
        echo "Error: Could not retrieve IP address for node $node" >&2
        return 1
    fi
}
