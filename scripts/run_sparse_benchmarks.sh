#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_file="${SPMV_CASE_FILE:-${repo_root}/data/sparse_matrices/benchmark_cases.csv}"
backend="${SPMV_BACKEND:-host}"
case "${backend}" in
    host) default_result="sparse_benchmark_results.csv" ;;
    pim) default_result="sparse_pim_results.csv" ;;
    *) echo "SPMV_BACKEND must be host or pim" >&2; exit 1 ;;
esac
result_file="${1:-${repo_root}/results/${default_result}}"
mkdir -p "$(dirname "${result_file}")"

if [[ "${SPMV_SKIP_BUILD:-0}" != 1 ]]; then
    echo "Building sparse benchmark executable" >&2
    build_args=(-j"${SPMV_BUILD_JOBS:-8}")
    [[ -z "${SPMV_GTEST_ROOT:-}" ]] || build_args+=("GTEST_ROOT=${SPMV_GTEST_ROOT}")
    (cd "${repo_root}" && scons "${build_args[@]}")
fi

if [[ "${backend}" == host ]]; then
    header="group,name,pattern,format,system_config,rows,cols,nnz,density,stored_entries,cycles,reads,writes,active_channels,max_channel_transactions,average_channel_transactions,channel_imbalance,energy"
    modes=(host)
    test_name="SparseMatrixBenchmark.configured_case"
    prefix="SPMV_RESULT,"
else
    header="group,name,pattern,case_format,execution_format,system_config,rows,cols,nnz,control,seed,mac_latency_cycles,physical_blocks,cycles,runtime_ns,reads,writes,macs,memory_wait_block_cycles,barrier_block_cycles,max_row_lead,estimated_dram_energy_pj,estimated_pim_energy_pj,estimated_total_energy_pj,estimated_average_power_mw,energy_model"
    case "${SPMV_PIM_CONTROL:-both}" in
        both) modes=(lockstep independent) ;;
        lockstep|independent) modes=("${SPMV_PIM_CONTROL}") ;;
        *) echo "SPMV_PIM_CONTROL must be both, lockstep, or independent" >&2; exit 1 ;;
    esac
    test_name="SparsePIMBenchmark.configured_case"
    prefix="SPMV_PIM_RESULT,"
fi
printf '%s\n' "${header}" > "${result_file}"

while IFS=, read -r group name rows cols pattern nnz_per_row format system_config block_size seed; do
    [[ "${group}" == "group" || -z "${group}" ]] && continue
    for mode in "${modes[@]}"; do
        echo "Running ${group}/${name} (${mode})" >&2
        run_output="$(
            cd "${repo_root}"
            SPMV_GROUP="${group}" \
            SPMV_NAME="${name}" \
            SPMV_ROWS="${rows}" \
            SPMV_COLS="${cols}" \
            SPMV_PATTERN="${pattern}" \
            SPMV_NNZ_PER_ROW="${nnz_per_row}" \
            SPMV_FORMAT="${format}" \
            SPMV_SYSTEM_CONFIG="${system_config}" \
            SPMV_BLOCK_SIZE="${block_size}" \
            SPMV_SEED="${seed}" \
            SPMV_PIM_CONTROL="${mode}" \
            ./sim --gtest_filter="${test_name}"
        )" || { printf '%s\n' "${run_output}" >&2; exit 1; }
        result_line="$(printf '%s\n' "${run_output}" | sed -n "s/^${prefix}//p")"
        if [[ -z "${result_line}" ]]; then
            printf '%s\n' "${run_output}" >&2
            echo "No benchmark result was produced for ${name}" >&2
            exit 1
        fi
        printf '%s\n' "${result_line}" >> "${result_file}"
    done
done < "${case_file}"
echo "Wrote sparse benchmark results to ${result_file}"
