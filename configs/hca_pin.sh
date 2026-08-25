# sa-cluster worker preamble, sourced (NOT executed) as srtctl's cluster-level
# default_bash_preamble, so exports land in the worker's own shell.
#
# 1) unset UCX_TLS  -- the sa site injects UCX_TLS=rc via srun --export=ALL, which
#    drops cuda_copy/gdr_copy so UCX cannot register VRAM and NIXL asserts.
#    bia's logs show UCX_TLS unset, so unsetting is the faithful state.
#
# 2) per-rank UCX_NET_DEVICES on PREFILL RANKS ONLY -- byte-for-byte the same
#    behaviour as BTK's btk_711a4565_b300.patch (plugins/servers/trtllm_server.py),
#    which every 0819 point enables via TRTLLM_CTX_LOCAL_HCA_PIN=1. That patch:
#      - fires only when disagg_mode == "prefill"; GEN stays unpinned "so its GPU
#        memory is registered on every usable HCA"  <-- pinning BOTH sides leaves
#        same-node peers with only self/sysv, which cannot satisfy NIXL's peer
#        error handling ("no active messages transport"), and workers die.
#      - keys on the PHYSICAL gpu = CUDA_VISIBLE_DEVICES[SLURM_LOCALID], not on
#        LOCALID, because compact packing puts a second ctx worker on GPUs 4-7.
#    BTK measured on bia: 75908/76084 requests succeed pinned vs 6/714 unpinned.
#    The HCA table below is sa's, measured on b300 (nvidia-smi topo -m PXB columns
#    cross-checked against /sys/class/infiniband/*/device PCI ids); bia's table is
#    different hardware and must not be copied.
unset UCX_TLS

# ---------------------------------------------------------------------------
# Keep only devices whose port 1 is ACTIVE. Existence is NOT enough: measured
# 2026-08-25 on the six nodes of job 46702,
#     b300-007  mlx5_2 mlx5_3  state="1: DOWN"  phys_state="3: Disabled"
#     b300-017  mlx5_4 mlx5_5  state="1: DOWN"  phys_state="3: Disabled"
# both present in /sys/class/infiniband but dead. The old filter only tested for the
# directory, so a rank whose own rail pair happened to be one of those got pinned to
# two dead NICs and had no RDMA path at all. Symmetric mode never exposed this because
# it only ever used mlx5_0/1/10/11, which are up on every node; switching to the
# per-GPU table is what surfaced it (job 46701 logged 32 "are not available" warnings,
# 46702 logged 35 plus a "Destination is unreachable"; 46586 and 46600 logged none).
# Echoes the surviving list, empty if none survive. Fails OPEN: if /sys is not
# readable at all, the input is returned unchanged.
_srt_live_devices() {
    set -- /sys/class/infiniband/mlx5_*
    if [ ! -e "$1" ]; then printf '%s' "$_srt_in"; return 0; fi
    _srt_out=""
    _srt_oIFS="$IFS"; IFS=,
    for _srt_dv in $_srt_in; do
        _srt_bare="${_srt_dv%%:*}"
        case "$(cat "/sys/class/infiniband/$_srt_bare/ports/1/state" 2>/dev/null)" in
            *ACTIVE*) _srt_out="${_srt_out:+$_srt_out,}$_srt_dv" ;;
            *) echo "SRT_HCA_SKIP localid=${SLURM_LOCALID:-0} dropping non-ACTIVE device $_srt_dv" >&2 ;;
        esac
    done
    IFS="$_srt_oIFS"
    printf '%s' "$_srt_out"
}


# NOTE on UCX keepalive: an earlier attempt set UCX_KEEPALIVE_INTERVAL=inf to dodge a
# segfault in ucp_ep_do_uct_ep_am_keepalive (job 46389, gen rank-0 died mid-decode,
# exit 139). That was the wrong fix and is deliberately NOT here: without keepalive
# UCX warns "endpoint with error handling but without keepalive" and a dead peer is
# never detected, so job 46442 turned the crash into an indefinite hang -- MoE
# all-to-all reported "Rank 0/1/2 timed out waiting for completion flag from rank 3"
# and the spin-wait kernel died as "CUDA error: unspecified launch failure".
# Both runs lacked rail pinning, which is what actually causes the endpoint flood
# (dozens of new endpoints per second, all crowded onto rc_mlx5/mlx5_0). bia runs
# keepalive at its default with pinning on, so that is what is reproduced here.

# ---------------------------------------------------------------------------
# SYMMETRIC FABRIC MODE (opt-in per recipe via SRT_FABRIC_MODE=symmetric).
#
# Measured on sa: a run fails if and only if BOTH sides have >1 worker.
#   ctx=1 gen=1 (p01,p03) OK | ctx=1 gen=4 (p02) OK | ctx=6 gen=1 (p07) OK
#   ctx=3 gen=4 (p04) FAIL x4 | ctx=5 gen=2 (p05) FAIL x4 | ctx=8 gen=2 (p06) FAIL
# BTK's per-GPU ctx pinning leaves ctx ranks with 4-5 devices while gen keeps all
# 16. On bia every device is mutually routable so that asymmetry is harmless. sa
# is rail-based (each NIC a /31 to the switch plus a static route for its own /18,
# no cross-rail routing), so in a many-to-many ctx<->gen topology some (ctx rank,
# gen rank) pairs land on rail sets that cannot reach each other; the KV transfer
# wedges and aiperf aborts on the hung root warmup request.
# Symmetric mode gives EVERY rank the identical 4-rail set (one pair per socket:
# mlx5_0/1 = GPU3 on socket 0, mlx5_10/11 = GPU7 on socket 1) plus bond0, so all
# peers are mutually reachable while registration stays far below the site's 16.
# The four points already reproduced keep the per-GPU scheme untouched.
# SITE-DEFAULT MODE (opt-in per recipe via SRT_FABRIC_MODE=none): skip ctx pinning
# entirely, leaving every rank with the site's full 16-rail view (bond0 is still
# appended at the bottom). RATIONALE: pinning ctx has a cost on a rail-based fabric
# that it does not have on bia. bia's fabric is fully routable, so a pinned ctx rank
# can still reach whichever gen NIC is PCIe-local to the destination GPU. On sa a
# pinned ctx rank can only reach the peer's SAME rail, so with ctx TP4 -> gen TP8
# most (ctx rank, gen rank) pairs land on a gen NIC that is not local to the target
# GPU and the write crosses the socket. Measured on job 46540 (b300-002, 30 s):
#   mlx5_0 22.27 GB/s / 37,696 adp_retrans / 37,696 slow_restart_cnps  (56 per GB)
#   mlx5_1 22.26 GB/s / 42,086 adp_retrans                             (63 per GB)
#   mlx5_10 2.15 GB/s / 0 retrans        mlx5_11 2.15 GB/s / 0 retrans
# retrans == slow_restart_cnps exactly, i.e. congestion control throttling, and ctx
# TX 44.5 GB/s against gen RX 31.3 GB/s means ~30% of the wire traffic is retransmit.
# Unpinned, UCX can pick a rail that is PCIe-local at BOTH ends. The cost is the
# endpoint/registration flood that pinning exists to prevent, so this is an
# experiment, not a default.
case "${BASH_EXECUTION_STRING:-}" in
    *SRT_FABRIC_MODE=none*)
        echo "SRT_FABRIC_SITE_DEFAULT localid=${SLURM_LOCALID:-0} (no ctx pinning)"
        ;;
esac

# BIA-FAITHFUL MODE (opt-in per recipe via SRT_FABRIC_MODE=bia_faithful).
# (Earlier drafts called this ctx_only; there is no such mode -- use bia_faithful.)
#
# This is bia's ACTUAL configuration, and closing a fidelity gap rather than tuning.
# Verified against the archive on 2026-08-24:
#   pareto_06/results/ctx/btk_autogen_ctx__launch_server_slurm_instance*.sh
#       export TRTLLM_CTX_LOCAL_HCA_PIN=1
#       if [ "${TRTLLM_CTX_LOCAL_HCA_PIN:-0}" = "1" ]; then ... export UCX_NET_DEVICES ...
#   pareto_06/results/gen/btk_autogen_gen__launch_server_slurm_instance*.sh
#       export TRTLLM_CTX_LOCAL_HCA_PIN=1          <-- and NOTHING else; no if-block
# TRTLLM_CTX_LOCAL_HCA_PIN is not implemented anywhere in TensorRT-LLM (grep of the
# pinned a25a71 tree: zero hits), so on bia that export is INERT on the gen side and
# gen runs with UCX_NET_DEVICES unset -- all 24 HCAs usable.
#
# This preamble is cluster-level, so it fires for every worker, and its gate below is
#     TRTLLM_CTX_LOCAL_HCA_PIN=1  OR  cmdline matches trtllm_config_prefill
# The recipe copies bia's env verbatim, which means the gen workers also carry
# TRTLLM_CTX_LOCAL_HCA_PIN=1 -- so the left arm fires and we pin the receiver that bia
# leaves open. The symmetric branch does the same with no role gate at all. Header
# lines 8-16 of this file already state the intended contract ("PREFILL RANKS ONLY",
# "GEN stays unpinned"); the code never enforced it.
#
# Why it matters most on pareto06, and note the two modes fail differently:
#   * default per-GPU mode: ctx and gen hold DISJOINT rail sets, and sa's rails do not
#     cross-route, so the receiver is truly unreachable over RDMA and AUX drops to
#     tcp/bond0. The shared-pair and bond0 blocks exist to paper over exactly this.
#   * symmetric mode: reachability is fine (identical rails everywhere); the deviation
#     is receive-side CAPACITY -- gen serves 8 ctx workers over 4 of 24 rails where
#     bia's gen has all 24, with UCX_MAX_RNDV_RAILS=2 choosing the best 2 per peer.
# pareto06 is the worst-exposed point either way: the only 8 ctx x 2 gen point
# (01/03 are 1x1, 07 is 6x1).
#
# FALSIFIER: if pareto06 still fails at ~285s from warmup start with gen unpinned,
# receiver pinning is not the cause.
# BIA-FAITHFUL MODE (opt-in per recipe via SRT_FABRIC_MODE=bia_faithful).
#
# Structurally identical to bia's BTK preamble, on sa's rail table:
#   ctx rank -> the two rails of ITS OWN physical GPU, nothing else.
#               No shared pair. No bond0. bia adds neither.
#   gen / orchestrator -> untouched; UCX_NET_DEVICES stays unset.
# Union over the 8 ctx ranks is 16 devices, the same count and shape bia's gen shows
# (mlx5_0,1,6,7,10..17,20..23 there; mlx5_0..5,8..11,16,17,20..23 here).
#
# Held in reserve behind symmetric_ctx_only so that only ONE variable moves per run:
# symmetric_ctx_only changes the receiver alone, this also changes the sender from
# four shared rails to its own pair. Use it only if the receiver-only change is not
# sufficient.
case "${BASH_EXECUTION_STRING:-}" in
    *SRT_FABRIC_MODE=bia_faithful*)
        case "${BASH_EXECUTION_STRING:-}" in
            *trtllm_config_prefill*) ;;
            *)  echo "SRT_FABRIC_BIA localid=${SLURM_LOCALID:-0} role=non-prefill UCX_NET_DEVICES=<unset, matches bia gen>"
                return 0 2>/dev/null || true ;;
        esac
        _srt_cvd=$(printf '%s' "${BASH_EXECUTION_STRING:-}" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)
        if [ -n "$_srt_cvd" ]; then
            IFS=, read -r -a _srt_g <<< "$_srt_cvd"
            _srt_phys="${_srt_g[${SLURM_LOCALID:-0}]}"
            case "$_srt_phys" in
                0) _srt_hca="mlx5_2:1,mlx5_3:1"   ;;
                1) _srt_hca="mlx5_8:1,mlx5_9:1"   ;;
                2) _srt_hca="mlx5_4:1,mlx5_5:1"   ;;
                3) _srt_hca="mlx5_0:1,mlx5_1:1"   ;;
                4) _srt_hca="mlx5_16:1,mlx5_17:1" ;;
                5) _srt_hca="mlx5_22:1,mlx5_23:1" ;;
                6) _srt_hca="mlx5_20:1,mlx5_21:1" ;;
                7) _srt_hca="mlx5_10:1,mlx5_11:1" ;;
                *) echo "SRT_FABRIC_BIA: no mapping for physical GPU $_srt_phys" >&2; _srt_hca="" ;;
            esac
            if [ -n "$_srt_hca" ]; then
                _srt_in="$_srt_hca"; _srt_hca="$(_srt_live_devices)"
            fi
            if [ -n "$_srt_hca" ]; then
                export UCX_NET_DEVICES="$_srt_hca"
                echo "SRT_FABRIC_BIA localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=$UCX_NET_DEVICES"
            else
                # Both of this GPU's rails are down. Pinning to nothing is worse than not
                # pinning: leave it unset so UCX picks a live rail itself.
                echo "SRT_FABRIC_BIA localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=<unset, own pair is DOWN>"
            fi
        fi
        return 0 2>/dev/null || true
        ;;
esac

# ONE VARIABLE vs job 46600 (which ran SRT_FABRIC_MODE=symmetric and failed): the ctx
# side keeps symmetric's identical four rails; only the GEN side changes, from those
# same four rails to UNSET. Nothing else moves -- not the ctx device list, not bond0
# (the tail block already no-ops on an unset list), not any engine or env value.
case "${BASH_EXECUTION_STRING:-}" in
    *SRT_FABRIC_MODE=symmetric_ctx_only*)
        case "${BASH_EXECUTION_STRING:-}" in
            *trtllm_config_prefill*)
                export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_10:1,mlx5_11:1"
                echo "SRT_FABRIC_SYMCTX localid=${SLURM_LOCALID:-0} role=prefill UCX_NET_DEVICES=$UCX_NET_DEVICES"
                ;;
            *)
                echo "SRT_FABRIC_SYMCTX localid=${SLURM_LOCALID:-0} role=non-prefill UCX_NET_DEVICES=<unset, matches bia gen>"
                ;;
        esac
        return 0 2>/dev/null || true
        ;;
esac

case "${BASH_EXECUTION_STRING:-}" in
    *SRT_FABRIC_MODE=symmetric*)
        # 4 rails, one pair per socket (mlx5_0/1 = GPU3 on socket 0, mlx5_10/11 = GPU7 on
        # socket 1). Widening this pool to 8 rails was TRIED AND WAS WORSE: pareto06 job
        # 46476 aborted at request 257 versus ~2,560 with 4 rails. UCX_MAX_RNDV_RAILS=2
        # caps a transfer at 2 rails anyway, so the wider pool bought no bandwidth and
        # only enlarged the endpoint/registration matrix. Keep 4.
        # bond0 REMOVED from this branch 2026-08-24. It was copied here from the
        # per-GPU branch, where it is genuinely needed: with per-rank rail pinning two
        # same-node siblings hold DISJOINT device sets and UCX finds no transport that
        # satisfies the peer error handling NIXL requires. Symmetric mode gives EVERY
        # rank the identical four rails, so all peers are already mutually reachable
        # over rc_mlx5 and bond0 adds nothing but a TCP lane.
        #
        # That lane is not free. Comparing the UCX lane selection against bia:
        #   ours  am(rc_mlx5/mlx5_0 ... rc_mlx5/mlx5_11  tcp/bond0)
        #   bia   am(rc_mlx5/mlx5_0 rc_mlx5/mlx5_1 rc_mlx5/mlx5_0 rc_mlx5/mlx5_1)
        # bia's active-message path is pure RDMA; ours can fall onto tcp/bond0, a
        # shared 192.168.95.0/24 ethernet. AUX transfers -- the small CPU metadata
        # writes that carry a request from ctx to gen -- ride the AM path, and EVERY
        # observed pareto06 failure was an AUX-sized write: across eleven attempts the
        # failed src_size values run 742 B to 25,682 B and not one KV-sized transfer
        # ever failed. The failures also arrive in bursts of 2-8 within a few seconds
        # and always in the last 1-2% of warmup (2,526-2,568 of bia's 2,587), i.e.
        # exactly when AUX traffic across 48 agents peaks -- the signature of a
        # congested shared TCP lane, not of resource exhaustion (ctx kv_util was 0.692
        # and gen 0.125 at the moment of failure).
        #
        # CORRECTION 2026-08-25: the src_size reading above is WRONG, and the
        # conclusion drawn from it does not hold. The error line prints
        # src_size={int(write_meta.src_ptrs.size)}, and WriteMeta.src_ptrs is declared
        # "np.ndarray  # dtype=np.int64" -- so that number is a DESCRIPTOR COUNT, not a
        # byte count. 742-25,682 descriptors is the signature of the per-fragment KV
        # path (an AUX request carries a handful), so the failures ARE the oversize KV
        # transfers, not small metadata writes, and they follow a bounce-arena
        # overflow: job 46701 overflowed 9x at 00:09:51-54 (up to 9,682 MiB) and failed
        # at 00:10:16 with 13,958 descriptors. The 'congested shared TCP lane' reading
        # is therefore unsupported; bond0 was separately falsified on job 46600, where
        # removing it made the AM lane identical to bia and the failures continued.
        export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_10:1,mlx5_11:1"
        echo "SRT_FABRIC_SYMMETRIC localid=${SLURM_LOCALID:-0} UCX_NET_DEVICES=$UCX_NET_DEVICES"
        return 0 2>/dev/null || true
        ;;
esac

_srt_nopin=0
case "${BASH_EXECUTION_STRING:-}" in *SRT_FABRIC_MODE=none*) _srt_nopin=1 ;; esac
if [ "$_srt_nopin" = "0" ] && { [ "${TRTLLM_CTX_LOCAL_HCA_PIN:-0}" = "1" ] || \
   case "${BASH_EXECUTION_STRING:-}" in *trtllm_config_prefill*) true;; *) false;; esac ; }
then
    _srt_cvd=$(printf '%s' "${BASH_EXECUTION_STRING:-}" | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)
    if [ -n "$_srt_cvd" ]; then
        IFS=, read -r -a _srt_g <<< "$_srt_cvd"
        _srt_phys="${_srt_g[${SLURM_LOCALID:-0}]}"
        case "$_srt_phys" in
            # REVERTED 2026-08-24 after measurement. This table is RAIL-GROUP based
            # and that is the correct axis on sa; an attempt to re-pair it by PCIe
            # card produced 28 KV transfer failures (NIXL_ERR_NOT_FOUND on
            # createXferReq) on job 46573 where the baseline had zero.
            #
            # sa has TWO different, equally real pairings, and for GPU2/GPU3 they
            # CONFLICT because the cards at bus 63 and bus 73 are cross-cabled --
            # each of those cards puts one port on rail group 192 and the other on
            # rail group 128:
            #   rail groups, from the /31 addresses on each port
            #     group   0: mlx5_2 172.16.0.x    mlx5_3 172.20.0.x     <- gpu0
            #     group  64: mlx5_8 172.16.64.x   mlx5_9 172.20.64.x    <- gpu1
            #     group 128: mlx5_4 172.16.128.x  mlx5_5 172.20.128.x   <- gpu2
            #     group 192: mlx5_0 172.16.192.x  mlx5_1 172.20.192.x   <- gpu3
            #   PCI cards, from the bus of each device
            #     bus 03: mlx5_2,3   bus 13: mlx5_8,9
            #     bus 63: mlx5_0,5   bus 73: mlx5_1,4
            # gpu0 and gpu1 get both properties. gpu2 and gpu3 can have rail-group
            # coherence OR card locality, never both. Rail-group coherence wins:
            # each NIC is a /31 to its switch with a static route only for its own
            # /18 and there is no cross-rail routing, so a rank whose two rails sit
            # in different groups cannot stripe a rendezvous across them the way
            # UCX_MAX_RNDV_RAILS=2 expects.
            #
            # Keeping the PCI grouping here for future reference, since it is real
            # and `nvidia-smi topo -m` will report it as PXB/PIX (its NIC Legend is
            # numeric on this box, NICn == mlx5_n):
            #     bus 03: mlx5_2,mlx5_3    bus 13: mlx5_8,mlx5_9
            #     bus 63: mlx5_0,mlx5_5    bus 73: mlx5_1,mlx5_4
            #     bus 83: mlx5_16,mlx5_17  bus 93: mlx5_22,mlx5_23
            #     bus e3: mlx5_20,mlx5_21  bus f3: mlx5_10,mlx5_11
            # and every GPU sits exactly 3 buses above its own card
            #     gpu0 06  gpu1 16  gpu2 66  gpu3 76
            #     gpu4 86  gpu5 96  gpu6 e6  gpu7 f6
            # so mlx5_0/mlx5_5 are ports 0 and 1 of ONE card (63:00.0 / 63:00.1) and
            # mlx5_4/mlx5_1 are ports 0 and 1 of another (73:00.0 / 73:00.1). The old
            # table paired them by adjacent NUMBER, {0,1} and {4,5}, which split both
            # cards and handed GPU2 and GPU3 one rail belonging to the other GPU's
            # PCIe complex -- every RDMA on that rail crossed the root complex (SYS).
            # Confirmed twice, independently: `nvidia-smi topo -m` PXB/PIX columns
            # (NIC Legend is numeric here, NICn == mlx5_n) and the PCI bus grouping
            # above. It also explains the throughput spread first seen on job 46528,
            # where the mlx5_0/1 pair was the slowest at 3.66 GB/s against 7.74 GB/s
            # on mlx5_8/9: mlx5_0 was never GPU3's rail.
            0) _srt_hca="mlx5_2:1,mlx5_3:1"   ;;
            1) _srt_hca="mlx5_8:1,mlx5_9:1"   ;;
            2) _srt_hca="mlx5_4:1,mlx5_5:1"   ;;
            3) _srt_hca="mlx5_0:1,mlx5_1:1"   ;;
            4) _srt_hca="mlx5_16:1,mlx5_17:1" ;;
            5) _srt_hca="mlx5_22:1,mlx5_23:1" ;;
            6) _srt_hca="mlx5_20:1,mlx5_21:1" ;;
            7) _srt_hca="mlx5_10:1,mlx5_11:1" ;;
            *) _srt_hca="" ;;
        esac
        # sa is a RAIL-BASED fabric: each NIC has a /31 to the switch plus a static
        # route for its own /18 (e.g. "172.16.0.0/18 via 172.16.0.33 dev enp3s0f0np0"),
        # and there is NO cross-rail routing. ctx->gen is fine because gen is unpinned
        # and therefore has an address on every rail, but two ctx ranks on the same
        # node pinned to different rails cannot reach each other at all -- UCX says
        # "no route to 172.20.64.32 from enp99s0f0np0", the KV WRITE fails and the ctx
        # executor wedges (job 46451: ctx stopped at 46/70 requests, gen went idle,
        # no crash). Adding one COMMON rail pair to every ctx rank restores sibling
        # reachability over RDMA. It keeps registration at 4 devices instead of the
        # site's 16, so the endpoint flood the pinning exists to prevent stays fixed,
        # and UCX_IB_PREFER_NEAREST_DEVICE (default y) + UCX_MAX_RNDV_RAILS=2 keep the
        # bulk transfer on the GPU-local pair. mlx5_0:1,mlx5_1:1 is GPU3's own pair,
        # so that rank is unchanged.
        # STRICT-LOCAL MODE (opt-in per recipe via SRT_FABRIC_MODE=strict_local):
        # skip the common pair entirely, giving each ctx rank exactly its own two
        # rails -- byte-for-byte bia's pattern -- plus the bond0 fallback appended
        # below. RATIONALE: the common pair predates bond0. It was added when a ctx
        # rank had NO routable path to a same-node sibling; bond0 now supplies one
        # universally, so the RDMA common pair is likely redundant. It is not free:
        # mlx5_0/1 is GPU3's own pair, so loading it into ranks 0-2 as well makes
        # rank3 the straggler. Measured on job 46528 (ctx node b300-014, 20 s of
        # port_xmit_data): mlx5_8,9 = 7.74 GB/s, mlx5_4,5 = 7.32, mlx5_2,3 = 5.60,
        # mlx5_0,1 = 3.66 -- a 2.1x spread. A request is ready only when ALL four
        # ranks finish, so rank3 gates every transfer, and 2.1x is the same order as
        # the 2.8x TTFT gap versus bia (bia p50 4,221 ms, ours ~10,900 ms).
        # NB: match on BASH_EXECUTION_STRING, not \$SRT_FABRIC_MODE -- the recipe's
        # exports are emitted AFTER this preamble, so the variable is not yet set
        # here. This is the same mechanism the symmetric branch above uses.
        # The common pair is ON by default in this branch. (An earlier revision made
        # it opt-in via a 'common_pair' mode; NO SUCH MODE EXISTS -- to switch it off
        # use SRT_FABRIC_MODE=strict_local.) It was introduced when a ctx rank had no routable path at all to
        # a same-node sibling, before bond0 was appended to EVERY rank at the bottom
        # of this file; bond0 is universally routable and carries the active-message
        # lane NIXL needs, so the RDMA common pair is redundant. It is not free: it
        # injects a rail from another GPU's card into every rank, which is exactly
        # the cross-PCIe traversal the corrected per-GPU table above exists to avoid,
        # and after that correction the old mlx5_0/mlx5_1 choice was no longer even
        # self-consistent (mlx5_0 lives on GPU2's card at bus 63, mlx5_1 on GPU3's at
        # bus 73, so GPU2 silently kept 2 rails while GPU3 got a duplicate).
        # Two local rails plus bond0 is also byte-for-byte bia's own pattern.
        # KNOWN RISK: if sibling reachability really does need RDMA, ctx wedges
        # partway through (job 46451 stopped at 46/70 requests, gen idle, no crash).
        # That run had bond0 on the prefill side ONLY, which UCX rejects with
        # "tcp/bond0 - not available", so the wedge may well not reproduce.
        # DEFAULT ON again. Removing it is opt-in via SRT_FABRIC_MODE=strict_local
        # and is NOT yet validated in isolation: job 46573 removed it AND re-paired
        # the rail table at the same time and produced 28 KV transfer failures, so
        # neither change can be blamed individually. Until a clean single-variable
        # run exists, the default stays on the configuration that measured zero
        # kvfail across jobs 46455 and 46528.
        case "${BASH_EXECUTION_STRING:-}" in
            *SRT_FABRIC_MODE=strict_local*) ;;            # opt-in: no common pair
            *)
                case "$_srt_hca" in
                    mlx5_0:*) ;;                          # gpu3 already owns it
                    *) _srt_hca="$_srt_hca,mlx5_0:1,mlx5_1:1" ;;
                esac
                ;;
        esac
        # sa nodes are NOT homogeneous -- b300-017 is missing mlx5_4 and mlx5_5 --
        # so drop any device in the table that this particular node does not have.
        # A non-existent entry in UCX_NET_DEVICES is not benign: UCX either errors
        # out or silently falls back to a device that is not GPU-local, which is the
        # very thing the table exists to prevent. If filtering empties the list we
        # leave UCX_NET_DEVICES unset so the rank inherits the site default rather
        # than being pinned to nothing.
        # FAIL OPEN: only filter when the sysfs tree is actually visible here. It is
        # absent on the login node and could in principle be absent inside the
        # container; dropping every device in that case would silently switch the
        # pinning off, which is worse than the missing-device case it guards.
        # The test is "are there any mlx5 devices visible", NOT "does the directory
        # exist". The login node has an EMPTY /sys/class/infiniband, so a -d test
        # passes there and would drop every device -- silently disabling the pinning.
        set -- /sys/class/infiniband/mlx5_*
        if [ ! -e "$1" ]; then
            _srt_keep="$_srt_hca"
        else
        _srt_keep=""
        _srt_IFS="$IFS"; IFS=,
        for _srt_d in $_srt_hca; do
            # Existence is not enough -- some nodes carry devices whose port is
            # administratively Disabled (measured 2026-08-25: b300-007 mlx5_2/3 and
            # b300-017 mlx5_4/5 are present in /sys but state="1: DOWN"). Require ACTIVE.
            if [ ! -d "/sys/class/infiniband/${_srt_d%%:*}" ]; then
                echo "SRT_CTX_LOCAL_HCA_PIN localid=${SLURM_LOCALID:-0} dropping absent device $_srt_d"
            else
                case "$(cat "/sys/class/infiniband/${_srt_d%%:*}/ports/1/state" 2>/dev/null)" in
                    *ACTIVE*) _srt_keep="${_srt_keep:+$_srt_keep,}$_srt_d" ;;
                    *) echo "SRT_CTX_LOCAL_HCA_PIN localid=${SLURM_LOCALID:-0} dropping non-ACTIVE device $_srt_d" ;;
                esac
            fi
        done
        IFS="$_srt_IFS"
        fi
        _srt_hca="$_srt_keep"

        if [ -n "$_srt_hca" ]; then
            # bond0 appended as an ACTIVE-MESSAGE fallback only. With a strictly
            # per-rank IB pair, two ctx ranks on the same node have disjoint
            # device sets, so UCX can find no AM lane that supports the peer
            # error handling NIXL requests -- it reports
            #   "no active messages transport ...: self/memory - no peer failure
            #    handler, sysv/memory - no peer failure handler"
            # and the rank dies ("MPI Comm server exit"). Seen on job 46449 and
            # NOT on 46445 with byte-identical config, i.e. it is a race in
            # endpoint setup, not a deterministic misconfiguration.
            # UCX's tcp transport DOES support peer error handling, so bond0
            # gives a universally reachable AM lane. It cannot capture the bulk
            # KV path: UCX_RNDV_SCHEME=put_zcopy needs RDMA, which tcp has not,
            # so rendezvous still rides the two pinned IB rails.
            export UCX_NET_DEVICES="$_srt_hca"
            echo "SRT_CTX_LOCAL_HCA_PIN localid=${SLURM_LOCALID:-0} phys_gpu=$_srt_phys UCX_NET_DEVICES=$UCX_NET_DEVICES"
        fi
    fi
fi

# Every rank -- pinned or not -- needs bond0 in UCX_NET_DEVICES.
# sa's RoCE NICs each sit on their own /31 point-to-point subnet
# (e.g. 172.20.64.28/31), so a rank pinned to mlx5_2 has NO L3 route to a
# same-node sibling pinned to mlx5_4: UCX reports
#   "rc_verbs/mlx5_4:1 - no route to 172.20.64.32:0 from enp99s0f0np0 or lo".
# bond0 (192.168.95.0/24, measured 73 Gb/s, 0% loss) is the one routable link
# every node shares, and UCX's tcp transport supports the peer error handling
# NIXL asks for. It must be present on BOTH ends or UCX reports
# "tcp/bond0 - not available" -- which is exactly what happened when only the
# prefill side carried it (job 46451). It cannot steal the bulk KV path:
# UCX_RNDV_SCHEME=put_zcopy requires RDMA, which tcp does not provide.
case ",${UCX_NET_DEVICES:-}," in
    *,bond0,*) ;;
    ,,)        ;;                       # unset: leave UCX defaults alone
    *)         export UCX_NET_DEVICES="${UCX_NET_DEVICES},bond0" ;;
esac
true
