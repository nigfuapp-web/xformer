/**
 * @Description  : Xoron MoE kernel implementation for kt-kernel
 * @Author       : kt-kernel team
 * @Date         : 2024
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_XORON_MOE_H
#define CPUINFER_OPERATOR_XORON_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "../amx/la/amx.hpp"

namespace xoron {

/**
 * Xoron MoE Configuration
 * 
 * Extends GeneralMOEConfig with Xoron-specific parameters:
 * - Aux-Lossless routing (no auxiliary loss)
 * - Shared expert isolation
 * - MLA integration
 */
struct XoronMOEConfig {
    // Basic MoE parameters
    int layer_idx = 0;
    int num_experts = 8;
    int num_experts_per_tok = 2;
    int hidden_size = 1024;
    int intermediate_size = 2048;
    int max_len = 8192;
    
    // Xoron-specific
    bool use_shared_expert = true;
    bool use_aux_lossless = true;
    float moe_capacity_factor = 1.25f;
    
    // Ring attention chunk size
    int ring_attention_chunk_size = 4096;
    
    // Thread pool configuration
    WorkerPool* pool = nullptr;
    int threadpool_count = 2;
    
    // Weight loading
    bool load = true;
    std::string path = "";
    
    // Expert skip mask (for CPU offloading)
    std::vector<bool> gpu_experts_mask;
    
    bool should_skip_expert(int expert_id) const {
        if (gpu_experts_mask.empty()) return false;
        return expert_id < gpu_experts_mask.size() && gpu_experts_mask[expert_id];
    }
};

/**
 * SiLU activation function (Swish)
 * Used by Xoron for gate activation
 */
inline __m512 silu_avx512(__m512 x) {
    // SiLU(x) = x * sigmoid(x)
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    
    // Fast exp approximation for sigmoid
    // sigmoid(x) = 1 / (1 + exp(-x))
    __m512 exp_neg_x = _mm512_exp_ps(neg_x);
    __m512 sigmoid = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));
    
    return _mm512_mul_ps(x, sigmoid);
}

/**
 * Xoron MoE Layer with Aux-Lossless routing
 * 
 * Features:
 * - Top-K expert selection without auxiliary loss
 * - Shared expert processing
 * - AMX-optimized matrix operations
 * - Ring attention integration
 */
template <typename T>
class XoronMoE {
public:
    using BufferA = typename T::BufferA;
    using BufferB = typename T::BufferB;
    using BufferC = typename T::BufferC;
    
    XoronMOEConfig config;
    int tp_part_idx = 0;
    
    // Local buffers
    ggml_bf16_t* local_input = nullptr;
    ggml_bf16_t* local_gate_output = nullptr;
    ggml_bf16_t* local_up_output = nullptr;
    ggml_bf16_t* local_down_output = nullptr;
    
    // Expert buffers
    std::vector<std::vector<int>> local_pos;
    std::vector<int> local_num;
    std::vector<int> expert_id_map;
    std::vector<ggml_bf16_t*> local_input_ptr;
    std::vector<ggml_bf16_t*> local_gate_output_ptr;
    std::vector<ggml_bf16_t*> local_up_output_ptr;
    std::vector<ggml_bf16_t*> local_down_output_ptr;
    
    // Weight buffers
    std::vector<std::shared_ptr<BufferA>> gate_up_ba;
    std::vector<std::shared_ptr<BufferB>> gate_bb;
    std::vector<std::shared_ptr<BufferC>> gate_bc;
    std::vector<std::shared_ptr<BufferB>> up_bb;
    std::vector<std::shared_ptr<BufferC>> up_bc;
    std::vector<std::shared_ptr<BufferA>> down_ba;
    std::vector<std::shared_ptr<BufferB>> down_bb;
    std::vector<std::shared_ptr<BufferC>> down_bc;
    
    // Shared expert buffers
    std::shared_ptr<BufferB> shared_gate_bb;
    std::shared_ptr<BufferB> shared_up_bb;
    std::shared_ptr<BufferB> shared_down_bb;
    
    XoronMoE(XoronMOEConfig cfg, int tp_idx = 0) 
        : config(cfg), tp_part_idx(tp_idx) {
        init();
    }
    
    ~XoronMoE() = default;
    
    void init() {
        // Allocate local buffers
        MemoryRequest mem_requests;
        mem_requests.append_pointer(
            &local_input, 
            sizeof(ggml_bf16_t) * config.num_experts_per_tok * config.max_len * config.hidden_size
        );
        mem_requests.append_pointer(
            &local_gate_output,
            sizeof(ggml_bf16_t) * config.num_experts_per_tok * config.max_len * config.intermediate_size
        );
        mem_requests.append_pointer(
            &local_up_output,
            sizeof(ggml_bf16_t) * config.num_experts_per_tok * config.max_len * config.intermediate_size
        );
        mem_requests.append_pointer(
            &local_down_output,
            sizeof(ggml_bf16_t) * config.num_experts_per_tok * config.max_len * config.hidden_size
        );
        
        // Initialize position tracking
        local_pos.resize(config.max_len);
        for (int i = 0; i < config.max_len; i++) {
            local_pos[i].resize(config.num_experts_per_tok);
        }
        
        expert_id_map.resize(config.num_experts);
        local_num.resize(config.num_experts);
        local_input_ptr.resize(config.num_experts);
        local_gate_output_ptr.resize(config.num_experts);
        local_up_output_ptr.resize(config.num_experts);
        local_down_output_ptr.resize(config.num_experts);
        
        // Initialize weight buffers for each expert
        for (int i = 0; i < config.num_experts; i++) {
            gate_up_ba.push_back(make_buffer_a(config.max_len, config.hidden_size, nullptr));
            gate_bc.push_back(make_buffer_c(config.max_len, config.intermediate_size, nullptr));
            up_bc.push_back(make_buffer_c(config.max_len, config.intermediate_size, nullptr));
            down_ba.push_back(make_buffer_a(config.max_len, config.intermediate_size, nullptr));
            down_bc.push_back(make_buffer_c(config.max_len, config.hidden_size, nullptr));
            
            void* gate_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config.intermediate_size, config.hidden_size));
            gate_bb.push_back(make_buffer_b(config.intermediate_size, config.hidden_size, gate_bb_ptr));
            
            void* up_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config.intermediate_size, config.hidden_size));
            up_bb.push_back(make_buffer_b(config.intermediate_size, config.hidden_size, up_bb_ptr));
            
            void* down_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config.hidden_size, config.intermediate_size));
            down_bb.push_back(make_buffer_b(config.hidden_size, config.intermediate_size, down_bb_ptr));
        }
        
        // Initialize shared expert if enabled
        if (config.use_shared_expert) {
            void* shared_gate_ptr = std::aligned_alloc(64, buffer_b_required_size(config.intermediate_size, config.hidden_size));
            shared_gate_bb = make_buffer_b(config.intermediate_size, config.hidden_size, shared_gate_ptr);
            
            void* shared_up_ptr = std::aligned_alloc(64, buffer_b_required_size(config.intermediate_size, config.hidden_size));
            shared_up_bb = make_buffer_b(config.intermediate_size, config.hidden_size, shared_up_ptr);
            
            void* shared_down_ptr = std::aligned_alloc(64, buffer_b_required_size(config.hidden_size, config.intermediate_size));
            shared_down_bb = make_buffer_b(config.hidden_size, config.intermediate_size, shared_down_ptr);
        }
        
        shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    }
    
    /**
     * Forward pass through MoE layer
     * 
     * @param qlen Sequence length
     * @param k Number of experts per token
     * @param expert_ids Expert assignments [qlen * k]
     * @param weights Expert weights [qlen * k]
     * @param input Input tensor [qlen, hidden_size]
     * @param output Output tensor [qlen, hidden_size]
     */
    void forward(int qlen, int k, const int64_t* expert_ids, const float* weights,
                 const void* input, void* output) {
        if (qlen > 1) {
            forward_prefill(qlen, k, expert_ids, weights, input, output);
        } else {
            forward_decode(k, expert_ids, weights, input, output);
        }
    }
    
    /**
     * Prefill forward pass (parallel processing)
     */
    void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights,
                         const void* input, void* output) {
        auto pool = config.pool->get_subpool(tp_part_idx);
        
        // Reset expert counts
        std::fill(local_num.begin(), local_num.end(), 0);
        
        // Count tokens per expert and record positions
        int activated_expert = 0;
        for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
                int expert_id = expert_ids[i * k + j];
                if (config.should_skip_expert(expert_id)) continue;
                
                if (local_num[expert_id] == 0) {
                    expert_id_map[activated_expert] = expert_id;
                    activated_expert++;
                }
                local_pos[i][j] = local_num[expert_id];
                local_num[expert_id]++;
            }
        }
        
        // Setup pointers for each expert
        int offset = 0;
        for (int i = 0; i < activated_expert; i++) {
            int expert_id = expert_id_map[i];
            local_input_ptr[expert_id] = local_input + offset * config.hidden_size;
            local_gate_output_ptr[expert_id] = local_gate_output + offset * config.intermediate_size;
            local_up_output_ptr[expert_id] = local_up_output + offset * config.intermediate_size;
            local_down_output_ptr[expert_id] = local_down_output + offset * config.hidden_size;
            offset += local_num[expert_id];
        }
        
        // Gather inputs for each expert
        auto gather_fn = [this, qlen, k, expert_ids, input](int task_id) {
            int token_idx = task_id / k;
            int expert_slot = task_id % k;
            int expert_id = expert_ids[token_idx * k + expert_slot];
            
            if (config.should_skip_expert(expert_id)) return;
            
            int pos = local_pos[token_idx][expert_slot];
            memcpy(
                local_input_ptr[expert_id] + pos * config.hidden_size,
                (const ggml_bf16_t*)input + token_idx * config.hidden_size,
                sizeof(ggml_bf16_t) * config.hidden_size
            );
        };
        
        pool->do_work_stealing_job(qlen * k, nullptr, gather_fn, nullptr);
        
        // Process experts in parallel
        int nth = pool->get_thread_count();
        
        // Gate projection
        auto gate_fn = [this, activated_expert, nth](int task_id) {
            int expert_idx = task_id / nth;
            int ith = task_id % nth;
            int expert_id = expert_id_map[expert_idx];
            int m = local_num[expert_id];
            
            if (m == 0) return;
            
            T::matmul(
                gate_up_ba[expert_id].get(),
                gate_bb[expert_id].get(),
                gate_bc[expert_id].get(),
                m, config.intermediate_size, config.hidden_size,
                local_input_ptr[expert_id],
                local_gate_output_ptr[expert_id],
                ith, nth
            );
        };
        
        if (activated_expert > 0) {
            pool->do_work_stealing_job(activated_expert * nth, nullptr, gate_fn, nullptr);
        }
        
        // Up projection
        auto up_fn = [this, activated_expert, nth](int task_id) {
            int expert_idx = task_id / nth;
            int ith = task_id % nth;
            int expert_id = expert_id_map[expert_idx];
            int m = local_num[expert_id];
            
            if (m == 0) return;
            
            T::matmul(
                gate_up_ba[expert_id].get(),
                up_bb[expert_id].get(),
                up_bc[expert_id].get(),
                m, config.intermediate_size, config.hidden_size,
                local_input_ptr[expert_id],
                local_up_output_ptr[expert_id],
                ith, nth
            );
        };
        
        if (activated_expert > 0) {
            pool->do_work_stealing_job(activated_expert * nth, nullptr, up_fn, nullptr);
        }
        
        // Apply SiLU activation and element-wise multiply
        apply_activation(activated_expert, nth, qlen);
        
        // Down projection
        auto down_fn = [this, activated_expert, nth](int task_id) {
            int expert_idx = task_id / nth;
            int ith = task_id % nth;
            int expert_id = expert_id_map[expert_idx];
            int m = local_num[expert_id];
            
            if (m == 0) return;
            
            T::matmul(
                down_ba[expert_id].get(),
                down_bb[expert_id].get(),
                down_bc[expert_id].get(),
                m, config.hidden_size, config.intermediate_size,
                local_gate_output_ptr[expert_id],
                local_down_output_ptr[expert_id],
                ith, nth
            );
        };
        
        if (activated_expert > 0) {
            pool->do_work_stealing_job(activated_expert * nth, nullptr, down_fn, nullptr);
        }
        
        // Weighted scatter and accumulate output
        auto scatter_fn = [this, qlen, k, expert_ids, weights, output](int token_idx) {
            for (int e = 0; e < config.hidden_size; e += 32) {
                __m512 x0 = _mm512_setzero_ps();
                __m512 x1 = _mm512_setzero_ps();
                
                for (int j = 0; j < k; j++) {
                    int expert_id = expert_ids[token_idx * k + j];
                    if (config.should_skip_expert(expert_id)) continue;
                    
                    __m512 weight = _mm512_set1_ps(weights[token_idx * k + j]);
                    __m512 down_output0, down_output1;
                    
                    avx512_32xbf16_to_32xfp32(
                        (__m512i*)(local_down_output_ptr[expert_id] + local_pos[token_idx][j] * config.hidden_size + e),
                        &down_output0, &down_output1
                    );
                    
                    x0 = _mm512_fmadd_ps(down_output0, weight, x0);
                    x1 = _mm512_fmadd_ps(down_output1, weight, x1);
                }
                
                // Add shared expert output if enabled
                if (config.use_shared_expert) {
                    // Shared expert contribution is added with weight 1.0
                    // (processed separately in forward_shared_expert)
                }
                
                auto f32out = (__m512*)((float*)output + token_idx * config.hidden_size + e);
                f32out[0] = x0;
                f32out[1] = x1;
            }
        };
        
        pool->do_work_stealing_job(qlen, nullptr, scatter_fn, nullptr);
        
        // Add shared expert if enabled
        if (config.use_shared_expert) {
            forward_shared_expert(qlen, input, output);
        }
    }
    
    /**
     * Decode forward pass (single token)
     */
    void forward_decode(int k, const int64_t* expert_ids, const float* weights,
                        const void* input, void* output) {
        // For single token, run sequentially
        std::fill(local_num.begin(), local_num.end(), 0);
        
        for (int j = 0; j < k; j++) {
            int expert_id = expert_ids[j];
            if (config.should_skip_expert(expert_id)) continue;
            
            local_pos[0][j] = local_num[expert_id];
            local_num[expert_id]++;
            local_input_ptr[expert_id] = local_input + j * config.hidden_size;
            local_gate_output_ptr[expert_id] = local_gate_output + j * config.intermediate_size;
            local_up_output_ptr[expert_id] = local_up_output + j * config.intermediate_size;
            local_down_output_ptr[expert_id] = local_down_output + j * config.hidden_size;
        }
        
        // Copy input
        for (int j = 0; j < k; j++) {
            int expert_id = expert_ids[j];
            if (config.should_skip_expert(expert_id)) continue;
            memcpy(
                local_input_ptr[expert_id] + local_pos[0][j] * config.hidden_size,
                (const ggml_bf16_t*)input,
                sizeof(ggml_bf16_t) * config.hidden_size
            );
        }
        
        // Process each active expert
        for (int j = 0; j < k; j++) {
            int expert_id = expert_ids[j];
            if (config.should_skip_expert(expert_id)) continue;
            
            // Gate projection
            T::matmul(
                gate_up_ba[expert_id].get(),
                gate_bb[expert_id].get(),
                gate_bc[expert_id].get(),
                1, config.intermediate_size, config.hidden_size,
                local_input_ptr[expert_id],
                local_gate_output_ptr[expert_id],
                0, 1
            );
            
            // Up projection
            T::matmul(
                gate_up_ba[expert_id].get(),
                up_bb[expert_id].get(),
                up_bc[expert_id].get(),
                1, config.intermediate_size, config.hidden_size,
                local_input_ptr[expert_id],
                local_up_output_ptr[expert_id],
                0, 1
            );
            
            // Apply activation
            for (int i = 0; i < config.intermediate_size; i += 32) {
                __m512 gate_val0, gate_val1, up_val0, up_val1;
                avx512_32xbf16_to_32xfp32((__m512i*)(local_gate_output_ptr[expert_id] + i), &gate_val0, &gate_val1);
                avx512_32xbf16_to_32xfp32((__m512i*)(local_up_output_ptr[expert_id] + i), &up_val0, &up_val1);
                
                __m512 result0 = _mm512_mul_ps(silu_avx512(gate_val0), up_val0);
                __m512 result1 = _mm512_mul_ps(silu_avx512(gate_val1), up_val1);
                
                avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(local_gate_output_ptr[expert_id] + i));
            }
            
            // Down projection
            T::matmul(
                down_ba[expert_id].get(),
                down_bb[expert_id].get(),
                down_bc[expert_id].get(),
                1, config.hidden_size, config.intermediate_size,
                local_gate_output_ptr[expert_id],
                local_down_output_ptr[expert_id],
                0, 1
            );
        }
        
        // Weighted sum
        for (int e = 0; e < config.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            
            for (int j = 0; j < k; j++) {
                int expert_id = expert_ids[j];
                if (config.should_skip_expert(expert_id)) continue;
                
                __m512 weight = _mm512_set1_ps(weights[j]);
                __m512 down_output0, down_output1;
                
                avx512_32xbf16_to_32xfp32(
                    (__m512i*)(local_down_output_ptr[expert_id] + local_pos[0][j] * config.hidden_size + e),
                    &down_output0, &down_output1
                );
                
                x0 = _mm512_fmadd_ps(down_output0, weight, x0);
                x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            
            auto f32out = (__m512*)((float*)output + e);
            f32out[0] = x0;
            f32out[1] = x1;
        }
        
        // Add shared expert if enabled
        if (config.use_shared_expert) {
            forward_shared_expert(1, input, output);
        }
    }
    
    /**
     * Process shared expert
     * In Xoron, the shared expert processes all tokens and adds to output
     */
    void forward_shared_expert(int qlen, const void* input, void* output) {
        if (!config.use_shared_expert || !shared_gate_bb) return;
        
        // Allocate temporary buffers for shared expert
        std::vector<ggml_bf16_t> shared_gate(qlen * config.intermediate_size);
        std::vector<ggml_bf16_t> shared_up(qlen * config.intermediate_size);
        std::vector<ggml_bf16_t> shared_down(qlen * config.hidden_size);
        
        auto shared_ba = make_buffer_a(qlen, config.hidden_size, nullptr);
        auto shared_bc_gate = make_buffer_c(qlen, config.intermediate_size, nullptr);
        auto shared_bc_up = make_buffer_c(qlen, config.intermediate_size, nullptr);
        auto shared_bc_down = make_buffer_c(qlen, config.hidden_size, nullptr);
        
        // Gate projection
        T::matmul(
            shared_ba.get(),
            shared_gate_bb.get(),
            shared_bc_gate.get(),
            qlen, config.intermediate_size, config.hidden_size,
            (const ggml_bf16_t*)input,
            shared_gate.data(),
            0, 1
        );
        
        // Up projection
        T::matmul(
            shared_ba.get(),
            shared_up_bb.get(),
            shared_bc_up.get(),
            qlen, config.intermediate_size, config.hidden_size,
            (const ggml_bf16_t*)input,
            shared_up.data(),
            0, 1
        );
        
        // Apply activation
        for (int i = 0; i < qlen * config.intermediate_size; i += 32) {
            __m512 gate_val0, gate_val1, up_val0, up_val1;
            avx512_32xbf16_to_32xfp32((__m512i*)(shared_gate.data() + i), &gate_val0, &gate_val1);
            avx512_32xbf16_to_32xfp32((__m512i*)(shared_up.data() + i), &up_val0, &up_val1);
            
            __m512 result0 = _mm512_mul_ps(silu_avx512(gate_val0), up_val0);
            __m512 result1 = _mm512_mul_ps(silu_avx512(gate_val1), up_val1);
            
            avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(shared_gate.data() + i));
        }
        
        // Down projection
        auto shared_ba_down = make_buffer_a(qlen, config.intermediate_size, nullptr);
        T::matmul(
            shared_ba_down.get(),
            shared_down_bb.get(),
            shared_bc_down.get(),
            qlen, config.hidden_size, config.intermediate_size,
            shared_gate.data(),
            shared_down.data(),
            0, 1
        );
        
        // Add to output
        for (int i = 0; i < qlen; i++) {
            for (int e = 0; e < config.hidden_size; e += 32) {
                __m512 shared0, shared1, out0, out1;
                avx512_32xbf16_to_32xfp32((__m512i*)(shared_down.data() + i * config.hidden_size + e), &shared0, &shared1);
                
                auto f32out = (__m512*)((float*)output + i * config.hidden_size + e);
                out0 = _mm512_add_ps(f32out[0], shared0);
                out1 = _mm512_add_ps(f32out[1], shared1);
                
                f32out[0] = out0;
                f32out[1] = out1;
            }
        }
    }
    
    /**
     * Apply SiLU activation and element-wise multiply: gate * up
     */
    void apply_activation(int activated_expert, int nth, int qlen) {
        auto pool = config.pool->get_subpool(tp_part_idx);
        
        auto fn = [this, nth](int task_id) {
            int expert_idx = task_id / nth;
            int ith = task_id % nth;
            int expert_id = expert_id_map[expert_idx];
            
            auto [n_start, n_end] = T::split_range_n(config.intermediate_size, ith, nth);
            
            for (int i = 0; i < local_num[expert_id]; i++) {
                ggml_bf16_t* gate_ptr = &local_gate_output_ptr[expert_id][i * config.intermediate_size];
                ggml_bf16_t* up_ptr = &local_up_output_ptr[expert_id][i * config.intermediate_size];
                
                for (int j = n_start; j < n_end; j += 32) {
                    __m512 gate_val0, gate_val1, up_val0, up_val1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(gate_ptr + j), &gate_val0, &gate_val1);
                    avx512_32xbf16_to_32xfp32((__m512i*)(up_ptr + j), &up_val0, &up_val1);
                    
                    __m512 result0 = _mm512_mul_ps(silu_avx512(gate_val0), up_val0);
                    __m512 result1 = _mm512_mul_ps(silu_avx512(gate_val1), up_val1);
                    
                    avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_ptr + j));
                }
            }
        };
        
        if (activated_expert == 0) return;
        
        if (qlen < 10) {
            for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
                fn(task_id);
            }
        } else {
            pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
        }
    }
    
    /**
     * Load weights from file
     */
    void load_weights(const std::string& path) {
        // Implementation depends on weight format (safetensors, bin, etc.)
        // This is a placeholder for the actual loading logic
        if (path.empty() || !config.load) return;
        
        // Load expert weights
        for (int i = 0; i < config.num_experts; i++) {
            std::string expert_prefix = path + "/expert_" + std::to_string(i);
            // Load gate_proj, up_proj, down_proj for each expert
        }
        
        // Load shared expert weights if enabled
        if (config.use_shared_expert) {
            std::string shared_prefix = path + "/shared_expert";
            // Load shared expert weights
        }
    }

protected:
    std::shared_ptr<BufferA> make_buffer_a(size_t m, size_t k, void* data) {
        return std::make_shared<BufferA>(m, k, data);
    }
    
    std::shared_ptr<BufferB> make_buffer_b(size_t n, size_t k, void* data) {
        return std::make_shared<BufferB>(n, k, data);
    }
    
    std::shared_ptr<BufferC> make_buffer_c(size_t m, size_t n, void* data) {
        return std::make_shared<BufferC>(m, n, data);
    }
    
    size_t buffer_b_required_size(size_t n, size_t k) {
        return T::buffer_b_required_size(n, k);
    }
};

} // namespace xoron

#endif // CPUINFER_OPERATOR_XORON_MOE_H
