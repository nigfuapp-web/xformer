/**
 * @Description  : Xoron MLA (Multi-Head Latent Attention) kernel implementation
 * @Author       : kt-kernel team
 * @Date         : 2024
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_XORON_MLA_H
#define CPUINFER_OPERATOR_XORON_MLA_H

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"

namespace xoron {

/**
 * Xoron MLA Configuration
 * 
 * Multi-Head Latent Attention compresses KV cache while maintaining quality.
 * Key features:
 * - Latent compression of K and V
 * - Decoupled RoPE for positional encoding
 * - Compatible with YaRN for long-context
 */
struct XoronMLAConfig {
    int hidden_size = 1024;
    int num_heads = 16;
    int num_kv_heads = 4;  // GQA-style
    int kv_lora_rank = 512;
    int q_lora_rank = 0;  // 0 = no Q compression
    int rope_dim = 64;
    int max_position_embeddings = 131072;
    
    // Derived
    int head_dim() const { return hidden_size / num_heads; }
    int kv_head_dim() const { return head_dim() - rope_dim; }
    
    // Thread pool
    WorkerPool* pool = nullptr;
    int threadpool_count = 2;
};

/**
 * YaRN Rotary Position Embedding for long-context
 */
class YaRNRoPE {
public:
    int dim;
    int max_position_embeddings;
    float base;
    int original_max_position;
    float beta_fast;
    float beta_slow;
    float mscale;
    float scaling_factor;
    
    std::vector<float> inv_freq;
    
    YaRNRoPE(
        int dim = 64,
        int max_pos = 131072,
        float base = 500000.0f,
        int orig_max_pos = 8192,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f,
        float mscale = 1.0f
    ) : dim(dim), max_position_embeddings(max_pos), base(base),
        original_max_position(orig_max_pos), beta_fast(beta_fast),
        beta_slow(beta_slow), mscale(mscale) {
        
        scaling_factor = static_cast<float>(max_pos) / orig_max_pos;
        compute_yarn_inv_freq();
    }
    
    void compute_yarn_inv_freq() {
        inv_freq.resize(dim / 2);
        
        for (int i = 0; i < dim / 2; i++) {
            float pos_freq = std::pow(base, static_cast<float>(2 * i) / dim);
            float inv_freq_extrapolation = 1.0f / pos_freq;
            float inv_freq_interpolation = 1.0f / (scaling_factor * pos_freq);
            
            int low = std::max(0, static_cast<int>(std::floor(
                dim * std::log(original_max_position / (beta_fast * 2 * M_PI)) / (2 * std::log(base))
            )));
            int high = std::min(dim - 1, static_cast<int>(std::ceil(
                dim * std::log(original_max_position / (beta_slow * 2 * M_PI)) / (2 * std::log(base))
            )));
            
            if (i < low) {
                inv_freq[i] = inv_freq_interpolation;
            } else if (i > high) {
                inv_freq[i] = inv_freq_extrapolation;
            } else {
                float smooth = static_cast<float>(i - low) / std::max(high - low, 1);
                inv_freq[i] = (1 - smooth) * inv_freq_interpolation + smooth * inv_freq_extrapolation;
            }
        }
    }
    
    float get_mscale(float scale) const {
        if (scale <= 1.0f) return 1.0f;
        return 0.1f * std::log(scale) + 1.0f;
    }
    
    /**
     * Apply rotary position embedding
     * 
     * @param x Input tensor [batch, seq_len, dim]
     * @param position_ids Position indices [batch, seq_len]
     * @param cos Output cosine buffer
     * @param sin Output sine buffer
     */
    void compute_cos_sin(
        int seq_len,
        const int64_t* position_ids,
        float* cos,
        float* sin
    ) const {
        float scale = get_mscale(scaling_factor) * mscale;
        
        for (int i = 0; i < seq_len; i++) {
            int64_t pos = position_ids[i];
            
            for (int j = 0; j < dim / 2; j++) {
                float freq = pos * inv_freq[j];
                cos[i * dim + j] = std::cos(freq) * scale;
                cos[i * dim + j + dim / 2] = std::cos(freq) * scale;
                sin[i * dim + j] = std::sin(freq) * scale;
                sin[i * dim + j + dim / 2] = std::sin(freq) * scale;
            }
        }
    }
};

/**
 * Apply RoPE to a vector using AVX-512
 */
inline void apply_rope_avx512(
    const float* x,
    const float* cos,
    const float* sin,
    float* out,
    int dim
) {
    int half_dim = dim / 2;
    
    for (int i = 0; i < dim; i += 32) {
        if (i + 32 <= half_dim) {
            // First half: x * cos + rotate(x) * sin
            __m512 x0 = _mm512_loadu_ps(x + i);
            __m512 x1 = _mm512_loadu_ps(x + i + 16);
            __m512 cos0 = _mm512_loadu_ps(cos + i);
            __m512 cos1 = _mm512_loadu_ps(cos + i + 16);
            __m512 sin0 = _mm512_loadu_ps(sin + i);
            __m512 sin1 = _mm512_loadu_ps(sin + i + 16);
            
            // Rotated x: (-x[half:], x[:half])
            __m512 x_rot0 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_loadu_ps(x + i + half_dim));
            __m512 x_rot1 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_loadu_ps(x + i + half_dim + 16));
            
            __m512 out0 = _mm512_fmadd_ps(x0, cos0, _mm512_mul_ps(x_rot0, sin0));
            __m512 out1 = _mm512_fmadd_ps(x1, cos1, _mm512_mul_ps(x_rot1, sin1));
            
            _mm512_storeu_ps(out + i, out0);
            _mm512_storeu_ps(out + i + 16, out1);
        } else if (i >= half_dim) {
            // Second half: x * cos + rotate(x) * sin
            __m512 x0 = _mm512_loadu_ps(x + i);
            __m512 x1 = _mm512_loadu_ps(x + i + 16);
            __m512 cos0 = _mm512_loadu_ps(cos + i);
            __m512 cos1 = _mm512_loadu_ps(cos + i + 16);
            __m512 sin0 = _mm512_loadu_ps(sin + i);
            __m512 sin1 = _mm512_loadu_ps(sin + i + 16);
            
            // Rotated x: (x[half:], -x[:half])
            __m512 x_rot0 = _mm512_loadu_ps(x + i - half_dim);
            __m512 x_rot1 = _mm512_loadu_ps(x + i - half_dim + 16);
            
            __m512 out0 = _mm512_fmadd_ps(x0, cos0, _mm512_mul_ps(x_rot0, sin0));
            __m512 out1 = _mm512_fmadd_ps(x1, cos1, _mm512_mul_ps(x_rot1, sin1));
            
            _mm512_storeu_ps(out + i, out0);
            _mm512_storeu_ps(out + i + 16, out1);
        }
    }
}

/**
 * Xoron MLA Layer
 * 
 * Multi-Head Latent Attention with compressed KV cache.
 * 
 * Architecture:
 * - Q projection (optionally with LoRA)
 * - KV compression projection
 * - KV decompression
 * - Decoupled RoPE
 * - Attention computation
 */
template <typename T>
class XoronMLA {
public:
    XoronMLAConfig config;
    YaRNRoPE rope;
    int tp_part_idx = 0;
    
    // Projection weights (stored as BufferB for matmul)
    std::shared_ptr<typename T::BufferB> q_proj;     // hidden -> num_heads * head_dim
    std::shared_ptr<typename T::BufferB> kv_a_proj;  // hidden -> kv_lora_rank + rope_dim
    std::shared_ptr<typename T::BufferB> kv_b_proj;  // kv_lora_rank -> num_kv_heads * (head_dim - rope_dim) * 2
    std::shared_ptr<typename T::BufferB> o_proj;     // num_heads * head_dim -> hidden
    
    // KV norm
    std::vector<float> kv_norm_weight;
    
    // Temporary buffers
    float* q_buffer = nullptr;
    float* k_buffer = nullptr;
    float* v_buffer = nullptr;
    float* kv_latent = nullptr;
    float* kv_pe = nullptr;
    float* attn_output = nullptr;
    
    // RoPE buffers
    std::vector<float> cos_cache;
    std::vector<float> sin_cache;
    
    XoronMLA(XoronMLAConfig cfg, int tp_idx = 0)
        : config(cfg), rope(cfg.rope_dim, cfg.max_position_embeddings), tp_part_idx(tp_idx) {
        init();
    }
    
    ~XoronMLA() = default;
    
    void init() {
        int head_dim = config.head_dim();
        int kv_head_dim = config.kv_head_dim();
        
        // Allocate projection weights
        // These will be loaded later
        q_proj = std::make_shared<typename T::BufferB>(
            config.num_heads * head_dim, config.hidden_size, nullptr
        );
        
        kv_a_proj = std::make_shared<typename T::BufferB>(
            config.kv_lora_rank + config.rope_dim, config.hidden_size, nullptr
        );
        
        kv_b_proj = std::make_shared<typename T::BufferB>(
            config.num_kv_heads * kv_head_dim * 2, config.kv_lora_rank, nullptr
        );
        
        o_proj = std::make_shared<typename T::BufferB>(
            config.hidden_size, config.num_heads * head_dim, nullptr
        );
        
        // Initialize KV norm
        kv_norm_weight.resize(config.kv_lora_rank, 1.0f);
        
        // Pre-allocate RoPE caches
        cos_cache.resize(config.max_position_embeddings * config.rope_dim);
        sin_cache.resize(config.max_position_embeddings * config.rope_dim);
    }
    
    /**
     * Forward pass through MLA layer
     * 
     * @param hidden_states Input [batch, seq_len, hidden_size]
     * @param attention_mask Attention mask [batch, 1, seq_len, kv_len]
     * @param position_ids Position indices [batch, seq_len]
     * @param past_key_value Cached KV tuple
     * @param output Output tensor [batch, seq_len, hidden_size]
     */
    void forward(
        int batch_size,
        int seq_len,
        const float* hidden_states,
        const float* attention_mask,
        const int64_t* position_ids,
        const float* past_k,
        const float* past_v,
        int past_len,
        float* output,
        float* new_k,
        float* new_v,
        bool use_cache
    ) {
        int head_dim = config.head_dim();
        int kv_head_dim = config.kv_head_dim();
        int kv_len = past_len + seq_len;
        
        auto pool = config.pool->get_subpool(tp_part_idx);
        
        // Allocate temporary buffers
        std::vector<float> q(batch_size * seq_len * config.num_heads * head_dim);
        std::vector<float> kv_compress(batch_size * seq_len * (config.kv_lora_rank + config.rope_dim));
        std::vector<float> kv_latent(batch_size * seq_len * config.kv_lora_rank);
        std::vector<float> kv_pe(batch_size * seq_len * config.rope_dim);
        std::vector<float> kv_decompressed(batch_size * seq_len * config.num_kv_heads * kv_head_dim * 2);
        std::vector<float> k(batch_size * seq_len * config.num_kv_heads * head_dim);
        std::vector<float> v(batch_size * seq_len * config.num_kv_heads * kv_head_dim);
        
        // Q projection
        // hidden_states [batch * seq_len, hidden_size] -> q [batch * seq_len, num_heads * head_dim]
        T::matmul(
            nullptr, q_proj.get(), nullptr,
            batch_size * seq_len, config.num_heads * head_dim, config.hidden_size,
            hidden_states, q.data(), 0, 1
        );
        
        // KV compression projection
        // hidden_states -> [kv_lora_rank + rope_dim]
        T::matmul(
            nullptr, kv_a_proj.get(), nullptr,
            batch_size * seq_len, config.kv_lora_rank + config.rope_dim, config.hidden_size,
            hidden_states, kv_compress.data(), 0, 1
        );
        
        // Split KV compression into PE and latent parts
        for (int i = 0; i < batch_size * seq_len; i++) {
            // First rope_dim elements are PE
            memcpy(
                kv_pe.data() + i * config.rope_dim,
                kv_compress.data() + i * (config.kv_lora_rank + config.rope_dim),
                config.rope_dim * sizeof(float)
            );
            // Rest is latent
            memcpy(
                kv_latent.data() + i * config.kv_lora_rank,
                kv_compress.data() + i * (config.kv_lora_rank + config.rope_dim) + config.rope_dim,
                config.kv_lora_rank * sizeof(float)
            );
        }
        
        // Apply RMS norm to KV latent
        apply_rms_norm(kv_latent.data(), kv_norm_weight.data(), batch_size * seq_len, config.kv_lora_rank);
        
        // KV decompression
        // kv_latent [batch * seq_len, kv_lora_rank] -> [batch * seq_len, num_kv_heads * kv_head_dim * 2]
        T::matmul(
            nullptr, kv_b_proj.get(), nullptr,
            batch_size * seq_len, config.num_kv_heads * kv_head_dim * 2, config.kv_lora_rank,
            kv_latent.data(), kv_decompressed.data(), 0, 1
        );
        
        // Split into K content and V
        for (int i = 0; i < batch_size * seq_len; i++) {
            for (int h = 0; h < config.num_kv_heads; h++) {
                // K content (without rope_dim)
                memcpy(
                    k.data() + i * config.num_kv_heads * head_dim + h * head_dim + config.rope_dim,
                    kv_decompressed.data() + i * config.num_kv_heads * kv_head_dim * 2 + h * kv_head_dim * 2,
                    kv_head_dim * sizeof(float)
                );
                
                // V
                memcpy(
                    v.data() + i * config.num_kv_heads * kv_head_dim + h * kv_head_dim,
                    kv_decompressed.data() + i * config.num_kv_heads * kv_head_dim * 2 + h * kv_head_dim * 2 + kv_head_dim,
                    kv_head_dim * sizeof(float)
                );
            }
        }
        
        // Add PE to K (first rope_dim dimensions)
        for (int i = 0; i < batch_size * seq_len; i++) {
            for (int h = 0; h < config.num_kv_heads; h++) {
                memcpy(
                    k.data() + i * config.num_kv_heads * head_dim + h * head_dim,
                    kv_pe.data() + i * config.rope_dim,
                    config.rope_dim * sizeof(float)
                );
            }
        }
        
        // Compute RoPE
        rope.compute_cos_sin(seq_len, position_ids, cos_cache.data(), sin_cache.data());
        
        // Apply RoPE to Q (only rope_dim portion)
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                for (int h = 0; h < config.num_heads; h++) {
                    float* q_head = q.data() + (b * seq_len + s) * config.num_heads * head_dim + h * head_dim;
                    apply_rope_avx512(
                        q_head,
                        cos_cache.data() + s * config.rope_dim,
                        sin_cache.data() + s * config.rope_dim,
                        q_head,
                        config.rope_dim
                    );
                }
            }
        }
        
        // Apply RoPE to K (only rope_dim portion)
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                for (int h = 0; h < config.num_kv_heads; h++) {
                    float* k_head = k.data() + (b * seq_len + s) * config.num_kv_heads * head_dim + h * head_dim;
                    apply_rope_avx512(
                        k_head,
                        cos_cache.data() + s * config.rope_dim,
                        sin_cache.data() + s * config.rope_dim,
                        k_head,
                        config.rope_dim
                    );
                }
            }
        }
        
        // Handle KV cache
        std::vector<float> full_k, full_v;
        if (past_len > 0 && past_k && past_v) {
            // Concatenate past and current KV
            full_k.resize(batch_size * kv_len * config.num_kv_heads * head_dim);
            full_v.resize(batch_size * kv_len * config.num_kv_heads * kv_head_dim);
            
            for (int b = 0; b < batch_size; b++) {
                // Copy past K
                memcpy(
                    full_k.data() + b * kv_len * config.num_kv_heads * head_dim,
                    past_k + b * past_len * config.num_kv_heads * head_dim,
                    past_len * config.num_kv_heads * head_dim * sizeof(float)
                );
                // Copy current K
                memcpy(
                    full_k.data() + b * kv_len * config.num_kv_heads * head_dim + past_len * config.num_kv_heads * head_dim,
                    k.data() + b * seq_len * config.num_kv_heads * head_dim,
                    seq_len * config.num_kv_heads * head_dim * sizeof(float)
                );
                
                // Copy past V
                memcpy(
                    full_v.data() + b * kv_len * config.num_kv_heads * kv_head_dim,
                    past_v + b * past_len * config.num_kv_heads * kv_head_dim,
                    past_len * config.num_kv_heads * kv_head_dim * sizeof(float)
                );
                // Copy current V
                memcpy(
                    full_v.data() + b * kv_len * config.num_kv_heads * kv_head_dim + past_len * config.num_kv_heads * kv_head_dim,
                    v.data() + b * seq_len * config.num_kv_heads * kv_head_dim,
                    seq_len * config.num_kv_heads * kv_head_dim * sizeof(float)
                );
            }
        } else {
            full_k = k;
            full_v = v;
        }
        
        // Save new KV for cache if requested
        if (use_cache && new_k && new_v) {
            memcpy(new_k, full_k.data(), full_k.size() * sizeof(float));
            memcpy(new_v, full_v.data(), full_v.size() * sizeof(float));
        }
        
        // GQA: Repeat KV heads to match Q heads
        int repeat_factor = config.num_heads / config.num_kv_heads;
        std::vector<float> expanded_k(batch_size * kv_len * config.num_heads * head_dim);
        std::vector<float> expanded_v(batch_size * kv_len * config.num_heads * kv_head_dim);
        
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < kv_len; s++) {
                for (int h = 0; h < config.num_heads; h++) {
                    int kv_h = h / repeat_factor;
                    
                    memcpy(
                        expanded_k.data() + (b * kv_len + s) * config.num_heads * head_dim + h * head_dim,
                        full_k.data() + (b * kv_len + s) * config.num_kv_heads * head_dim + kv_h * head_dim,
                        head_dim * sizeof(float)
                    );
                    
                    memcpy(
                        expanded_v.data() + (b * kv_len + s) * config.num_heads * kv_head_dim + h * kv_head_dim,
                        full_v.data() + (b * kv_len + s) * config.num_kv_heads * kv_head_dim + kv_h * kv_head_dim,
                        kv_head_dim * sizeof(float)
                    );
                }
            }
        }
        
        // Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> attn_weights(batch_size * config.num_heads * seq_len * kv_len);
        std::vector<float> attn_out(batch_size * seq_len * config.num_heads * head_dim);
        
        // Q @ K^T
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < config.num_heads; h++) {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < kv_len; j++) {
                        float sum = 0.0f;
                        
                        // Dot product Q[b,i,h] with K[b,j,h]
                        const float* q_ptr = q.data() + (b * seq_len + i) * config.num_heads * head_dim + h * head_dim;
                        const float* k_ptr = expanded_k.data() + (b * kv_len + j) * config.num_heads * head_dim + h * head_dim;
                        
                        // AVX-512 dot product
                        for (int d = 0; d < head_dim; d += 16) {
                            __m512 q_vec = _mm512_loadu_ps(q_ptr + d);
                            __m512 k_vec = _mm512_loadu_ps(k_ptr + d);
                            sum += _mm512_reduce_add_ps(_mm512_mul_ps(q_vec, k_vec));
                        }
                        
                        attn_weights[(b * config.num_heads + h) * seq_len * kv_len + i * kv_len + j] = sum * scale;
                    }
                }
            }
        }
        
        // Apply attention mask and softmax
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < config.num_heads; h++) {
                for (int i = 0; i < seq_len; i++) {
                    float* row = attn_weights.data() + (b * config.num_heads + h) * seq_len * kv_len + i * kv_len;
                    
                    // Apply mask (causal)
                    for (int j = i + past_len + 1; j < kv_len; j++) {
                        row[j] = -1e9f;  // Mask future positions
                    }
                    
                    // Softmax
                    float max_val = -1e9f;
                    for (int j = 0; j < kv_len; j++) {
                        max_val = std::max(max_val, row[j]);
                    }
                    
                    float sum = 0.0f;
                    for (int j = 0; j < kv_len; j++) {
                        row[j] = std::exp(row[j] - max_val);
                        sum += row[j];
                    }
                    
                    for (int j = 0; j < kv_len; j++) {
                        row[j] /= sum;
                    }
                }
            }
        }
        
        // Attention @ V
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < config.num_heads; h++) {
                for (int i = 0; i < seq_len; i++) {
                    float* out_ptr = attn_out.data() + (b * seq_len + i) * config.num_heads * head_dim + h * head_dim;
                    const float* attn_ptr = attn_weights.data() + (b * config.num_heads + h) * seq_len * kv_len + i * kv_len;
                    
                    // Initialize output to zero
                    for (int d = 0; d < head_dim; d++) {
                        out_ptr[d] = 0.0f;
                    }
                    
                    // Weighted sum of V
                    for (int j = 0; j < kv_len; j++) {
                        const float* v_ptr = expanded_v.data() + (b * kv_len + j) * config.num_heads * kv_head_dim + h * kv_head_dim;
                        float weight = attn_ptr[j];
                        
                        for (int d = 0; d < kv_head_dim; d++) {
                            out_ptr[d] += weight * v_ptr[d];
                        }
                    }
                }
            }
        }
        
        // Output projection
        T::matmul(
            nullptr, o_proj.get(), nullptr,
            batch_size * seq_len, config.hidden_size, config.num_heads * head_dim,
            attn_out.data(), output, 0, 1
        );
    }
    
    void apply_rms_norm(float* x, const float* weight, int num_tokens, int dim) {
        const float eps = 1e-6f;
        
        for (int i = 0; i < num_tokens; i++) {
            float* row = x + i * dim;
            
            // Compute RMS
            float sum_sq = 0.0f;
            for (int j = 0; j < dim; j += 16) {
                __m512 v = _mm512_loadu_ps(row + j);
                sum_sq += _mm512_reduce_add_ps(_mm512_mul_ps(v, v));
            }
            float rms = std::sqrt(sum_sq / dim + eps);
            float scale = 1.0f / rms;
            
            // Apply norm and weight
            for (int j = 0; j < dim; j += 16) {
                __m512 v = _mm512_loadu_ps(row + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                __m512 s = _mm512_set1_ps(scale);
                _mm512_storeu_ps(row + j, _mm512_mul_ps(_mm512_mul_ps(v, s), w));
            }
        }
    }
    
    void load_weights(const std::string& path) {
        // Load projection weights from file
        // Implementation depends on weight format
    }
};

} // namespace xoron

#endif // CPUINFER_OPERATOR_XORON_MLA_H
