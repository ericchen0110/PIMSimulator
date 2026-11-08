#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_file="${repo_root}/data/sparse_matrices/benchmark_cases.csv"
result_file="${1:-${repo_root}/results/sparse_benchmark_results.csv}"

mkdir -p "$(dirname "${result_file}")"

echo "Building sparse benchmark executable" >&2
(cd "${repo_root}" && scons -j8)

header="group,name,pattern,format,system_config,rows,cols,nnz,density,stored_entries,cycles,reads,writes,active_channels,max_channel_transactions,average_channel_transactions,channel_imbalance,energy,scheduler,sparse_scheduled,reordered,row_hit_admissions,starvation_promotions,average_queue_wait,activates,precharges"
printf '%s\n' "${header}" > "${result_file}"

while IFS=, read -r group name rows cols pattern nnz_per_row format system_config block_size seed; do
    [[ "${group}" == "group" || -z "${group}" ]] && continue
    echo "Running ${group}/${name}" >&2

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
        SPMV_SCHEDULER="${SPMV_SCHEDULER:-sparse}" \
        ./sim --gtest_filter=SparseMatrixBenchmark.configured_case
    )"

    result_line="$(printf '%s\n' "${run_output}" | sed -n 's/^SPMV_RESULT,//p')"
    if [[ -z "${result_line}" ]]; then
        printf '%s\n' "${run_output}" >&2
        echo "No benchmark result was produced for ${name}" >&2
        exit 1
    fi
    printf '%s\n' "${result_line}" >> "${result_file}"
done < "${case_file}"

echo "Wrote sparse benchmark results to ${result_file}"
