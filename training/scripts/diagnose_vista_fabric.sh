#!/usr/bin/env bash
# =============================================================================
# diagnose_vista_fabric.sh
#
# Diagnose Vista compute-node networking before running 30B multi-node
# training. Run this from an idev allocation with N>=2 nodes:
#
#   salloc -N 2 -p gh-dev -t 00:30:00 -A AST24021
#   bash training/scripts/diagnose_vista_fabric.sh
#
# Prints, for every node in the allocation:
#   1. All UP interfaces and their IPv4 addresses
#   2. Which one the sbatch IB regex would pick (and what override to use)
#   3. The mgmt IP from `hostname --ip-address` (what ray default-advertises)
#
# Then runs a cross-node connectivity matrix on the fabric IPs to confirm
# TCP between IB IPs actually works (the failure mode behind every multi-node
# 30B job to date — see project_multinode_networking_blocker.md).
# =============================================================================

set -u

if [[ -z "${SLURM_NODELIST:-}" ]]; then
    echo "ERROR: no SLURM_NODELIST — run this inside salloc / sbatch." >&2
    exit 1
fi

nodes=$(scontrol show hostnames "$SLURM_NODELIST")
N=$(echo "$nodes" | wc -l)
echo "================================================================"
echo "Vista fabric diagnosis — $N node(s): $(echo "$nodes" | tr '\n' ' ')"
echo "================================================================"

# Same regex the sbatch uses to auto-pick DIST_IFACE
IBRE='^(ib|ibp|ibs|ibo|hsn|opa|mlx|bond)[0-9]'

declare -A fabric_ip
declare -A mgmt_ip
declare -A iface

for node in $nodes; do
    echo
    echo "---- $node ----"

    # All UP interfaces with their IPv4 (one block per node)
    echo "[interfaces UP]"
    srun --nodes=1 --ntasks=1 -w "$node" bash -c '
        ip -o link show up | awk -F": " "{print \$2}" | grep -v "^lo$" | while read iface; do
            ip4=$(ip -o -4 addr show "$iface" 2>/dev/null | awk "{print \$4}" | head -1)
            mtu=$(ip -o link show "$iface" 2>/dev/null | grep -oE "mtu [0-9]+" | awk "{print \$2}")
            printf "  %-12s  ipv4=%-20s  mtu=%s\n" "$iface" "${ip4:--}" "${mtu:--}"
        done
    '

    # What the sbatch regex would pick
    iface[$node]=$(srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        ip -o link show up | awk -F': ' '{print \$2}' | grep -E '$IBRE' | head -1
    " 2>/dev/null | tr -d '[:space:]')

    if [[ -n "${iface[$node]}" ]]; then
        fabric_ip[$node]=$(srun --nodes=1 --ntasks=1 -w "$node" bash -c "
            ip -o -4 addr show '${iface[$node]}' | awk '{print \$4}' | cut -d/ -f1 | head -1
        " 2>/dev/null | tr -d '[:space:]')
        echo "[sbatch would pick] iface=${iface[$node]}  fabric_ip=${fabric_ip[$node]}"
    else
        echo "[sbatch would pick] NONE — regex '$IBRE' matched no UP interface!" >&2
        echo "                    ACTION: pick one above manually, export DIST_IFACE=<name>"
    fi

    mgmt_ip[$node]=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address 2>/dev/null | awk '{print $1}')
    echo "[hostname --ip-address] $node => ${mgmt_ip[$node]}  (this is the BAD IP ray uses by default)"
done

if (( N < 2 )); then
    echo
    echo "Only $N node — skipping connectivity matrix. Re-run with -N 2+ to confirm fabric routes."
    exit 0
fi

echo
echo "================================================================"
echo "Cross-node connectivity matrix (fabric IPs only)"
echo "================================================================"
echo "Listening on port 47777 of each node's fabric IP, then nc -zv from"
echo "every other node. A '[OK]' means the IP+port is reachable across"
echo "chassis on the fabric — the SGLang TCPStore traffic will too."
echo

# Start a listener on each node's fabric IP
declare -A listener_pid
for node in $nodes; do
    ip="${fabric_ip[$node]:-}"
    if [[ -z "$ip" ]]; then
        echo "[skip] $node has no fabric_ip; cannot run matrix row" >&2
        continue
    fi
    # Bind explicitly to the fabric IP (not 0.0.0.0) to make sure routing is
    # actually via the fabric interface, not mgmt.
    srun --nodes=1 --ntasks=1 -w "$node" \
        bash -c "timeout 60 nc -l -s '$ip' -p 47777 >/dev/null 2>&1 &" &
    listener_pid[$node]=$!
done
sleep 3

# Probe every (src, dst) pair where src != dst
for src in $nodes; do
    for dst in $nodes; do
        [[ "$src" == "$dst" ]] && continue
        dst_ip="${fabric_ip[$dst]:-}"
        [[ -z "$dst_ip" ]] && continue
        result=$(srun --nodes=1 --ntasks=1 -w "$src" \
            bash -c "timeout 5 nc -zv '$dst_ip' 47777 2>&1; echo EXIT=\$?" \
            2>/dev/null)
        if echo "$result" | grep -qE 'succeeded|open|EXIT=0'; then
            printf "  [OK]   %-12s -> %-12s (%s:47777)\n" "$src" "$dst" "$dst_ip"
        else
            printf "  [FAIL] %-12s -> %-12s (%s:47777)  -- %s\n" \
                "$src" "$dst" "$dst_ip" "$(echo "$result" | tr '\n' ' ' | head -c 80)"
        fi
    done
done

# Cleanup listeners
for pid in "${listener_pid[@]:-}"; do
    kill "$pid" 2>/dev/null || true
done

echo
echo "================================================================"
echo "DONE. If [FAIL] rows appear with fabric IPs, the fabric interface"
echo "is NOT routable across chassis. In that case the only options are:"
echo "  - co-locate all 8 nodes on the same chassis (Slurm topology hint)"
echo "  - escalate to TACC support to open inter-chassis TCP on this fabric"
echo "If matrix is all [OK], the sbatch fix should work — submit a real"
echo "multi-node job and verify the 'Ports for engine N:' log lines show"
echo "fabric IPs, not 129.114.x.x."
echo "================================================================"
