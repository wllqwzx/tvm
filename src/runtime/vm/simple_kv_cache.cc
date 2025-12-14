/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/runtime/vm/simple_kv_cache.cc
 * \brief Simple KV cache object for edge-side inference.
 */
#include "simple_kv_cache.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

SimpleAttentionKVCacheObj::SimpleAttentionKVCacheObj(
    int64_t max_batch_size, int64_t max_seq_len, int64_t num_layers, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t head_dim, RoPEMode rope_mode, double rotary_scale,
    double rotary_theta, Optional<NDArray> rope_ext_factors, DLDataType dtype, Device device,
    ffi::Function f_append_kv, ffi::Function f_attention_prefill, ffi::Function f_attention_decode,
    ffi::Function f_split_rotary)
    : max_batch_size_(max_batch_size),
      max_seq_len_(max_seq_len),
      num_layers_(num_layers),
      num_qo_heads_(num_qo_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      rope_mode_(rope_mode),
      rotary_scale_(rotary_scale),
      rotary_theta_(rotary_theta),
      rope_ext_factors_(std::move(rope_ext_factors)),
      kv_dtype_(DataType(dtype)),
      f_append_kv_(std::move(f_append_kv)),
      f_attention_prefill_(std::move(f_attention_prefill)),
      f_attention_decode_(std::move(f_attention_decode)),
      f_split_rotary_(std::move(f_split_rotary)),
      device_(device),
      cur_batch_size_(0),
      current_phase_(ForwardPhase::kPrefill) {
  // Initialize cache storage
  cache_.reserve(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    cache_.push_back(NDArray::Empty({max_batch_size_, max_seq_len_, 2, num_kv_heads_, head_dim_},
                                    dtype, device));
  }

  // Initialize sequence management
  cur_seq_lens_host_.resize(max_batch_size_, 0);

  // Initialize query positions
  cur_query_positions_ = NDArray::Empty({max_seq_len_}, dtype_aux_int64_, device);

  // Init temporary buffers
  q_data_tmp_ =
      NDArray::Empty({max_seq_len_ * max_batch_size_, num_qo_heads_, head_dim_}, dtype, device_);
  k_data_tmp_ =
      NDArray::Empty({max_seq_len_ * max_batch_size_, num_kv_heads_, head_dim_}, dtype, device_);
  v_data_tmp_ =
      NDArray::Empty({max_seq_len_ * max_batch_size_, num_kv_heads_, head_dim_}, dtype, device_);
}

void SimpleAttentionKVCacheObj::Clear() {
  cur_batch_size_ = 0;
  seq_id_to_idx_.clear();
  std::fill(cur_seq_lens_host_.begin(), cur_seq_lens_host_.end(), 0);
  current_phase_ = ForwardPhase::kPrefill;
}

/************** Sequence Management **************/

void SimpleAttentionKVCacheObj::AddSequence(int64_t seq_id) {
  CHECK(seq_id_to_idx_.find(seq_id) == seq_id_to_idx_.end())
      << "The sequence \"" << seq_id << "\" is already in the KV cache.";
  CHECK_LT(cur_batch_size_, max_batch_size_)
      << "Cannot add more sequences. Maximum batch size " << max_batch_size_ << " reached.";

  seq_id_to_idx_[seq_id] = cur_batch_size_;
  cur_seq_lens_host_[cur_batch_size_] = 0;
  cur_batch_size_++;
}

void SimpleAttentionKVCacheObj::RemoveSequence(int64_t seq_id) {
  auto it = seq_id_to_idx_.find(seq_id);
  CHECK(it != seq_id_to_idx_.end())
      << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";

  int64_t seq_idx = it->second;

  // Move the last sequence to fill the gap
  if (seq_idx < cur_batch_size_ - 1) {
    // Find the sequence that was at the last position
    int64_t last_seq_id = -1;
    for (const auto& pair : seq_id_to_idx_) {
      if (pair.second == cur_batch_size_ - 1) {
        last_seq_id = pair.first;
        break;
      }
    }

    if (last_seq_id != -1) {
      seq_id_to_idx_[last_seq_id] = seq_idx;
      cur_seq_lens_host_[seq_idx] = cur_seq_lens_host_[cur_batch_size_ - 1];

      // Copy cache data for all layers
      for (int layer = 0; layer < num_layers_; ++layer) {
        NDArray src_view = cache_[layer].CreateView(
            {1, max_seq_len_, 2, num_kv_heads_, head_dim_}, cache_[layer]->dtype,
            (cur_batch_size_ - 1) * max_seq_len_ * 2 * num_kv_heads_ * head_dim_ *
                cache_[layer].DataType().bytes());
        NDArray dst_view = cache_[layer].CreateView(
            {1, max_seq_len_, 2, num_kv_heads_, head_dim_}, cache_[layer]->dtype,
            seq_idx * max_seq_len_ * 2 * num_kv_heads_ * head_dim_ *
                cache_[layer].DataType().bytes());
        dst_view.CopyFrom(src_view);
      }
    }
  }

  seq_id_to_idx_.erase(it);
  cur_batch_size_--;
  cur_seq_lens_host_[cur_batch_size_] = 0;
}

void SimpleAttentionKVCacheObj::ForkSequence(int64_t parent_seq_id, int64_t child_seq_id,
                                             int64_t fork_pos) {
  auto parent_it = seq_id_to_idx_.find(parent_seq_id);
  CHECK(parent_it != seq_id_to_idx_.end())
      << "The parent sequence \"" << parent_seq_id << "\" cannot be found in KV cache.";
  CHECK(seq_id_to_idx_.find(child_seq_id) == seq_id_to_idx_.end())
      << "The child sequence \"" << child_seq_id << "\" is already in the KV cache.";
  CHECK_LT(cur_batch_size_, max_batch_size_)
      << "Cannot fork sequence. Maximum batch size " << max_batch_size_ << " reached.";

  int64_t parent_idx = parent_it->second;
  int64_t parent_seq_len = cur_seq_lens_host_[parent_idx];

  if (fork_pos == -1) {
    fork_pos = parent_seq_len;
  }

  CHECK_GE(fork_pos, 0) << "Fork position must be non-negative.";
  CHECK_LE(fork_pos, parent_seq_len)
      << "Fork position " << fork_pos << " exceeds parent sequence length " << parent_seq_len;

  // Add child sequence
  seq_id_to_idx_[child_seq_id] = cur_batch_size_;
  cur_seq_lens_host_[cur_batch_size_] = fork_pos;

  // Copy KV data from parent to child up to fork position
  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int pos = 0; pos < fork_pos; ++pos) {
      for (int kv = 0; kv < 2; ++kv) {  // 0=K, 1=V
        for (int head = 0; head < num_kv_heads_; ++head) {
          for (int dim = 0; dim < head_dim_; ++dim) {
            // Copy from parent[parent_idx, pos, kv, head, dim] to child[cur_batch_size_, pos, kv,
            // head, dim]
            NDArray parent_view = cache_[layer].CreateView(
                {1}, cache_[layer]->dtype,
                ((parent_idx * max_seq_len_ + pos) * 2 + kv) * num_kv_heads_ * head_dim_ *
                        cache_[layer].DataType().bytes() +
                    (head * head_dim_ + dim) * cache_[layer].DataType().bytes());
            NDArray child_view = cache_[layer].CreateView(
                {1}, cache_[layer]->dtype,
                ((cur_batch_size_ * max_seq_len_ + pos) * 2 + kv) * num_kv_heads_ * head_dim_ *
                        cache_[layer].DataType().bytes() +
                    (head * head_dim_ + dim) * cache_[layer].DataType().bytes());
            child_view.CopyFrom(parent_view);
          }
        }
      }
    }
  }

  cur_batch_size_++;
}

void SimpleAttentionKVCacheObj::EnableSlidingWindowForSeq(int64_t seq_id,
                                                          int32_t sliding_window_size,
                                                          int32_t attn_sink_size) {
  // Simple KV cache doesn't support sliding window yet
  LOG(WARNING) << "Sliding window is not supported in SimpleKVCache";
}

void SimpleAttentionKVCacheObj::PopN(int64_t seq_id, int32_t n) {
  auto it = seq_id_to_idx_.find(seq_id);
  CHECK(it != seq_id_to_idx_.end())
      << "The sequence \"" << seq_id << "\" cannot be found in KV cache.";

  int64_t seq_idx = it->second;
  CHECK_GE(n, 0) << "The length of popping " << n << " cannot be negative.";
  CHECK_LE(n, cur_seq_lens_host_[seq_idx])
      << "The sequence only has length " << cur_seq_lens_host_[seq_idx]
      << ", while the length of pop is " << n << " which exceeds the sequence length.";

  cur_seq_lens_host_[seq_idx] -= n;
}

/************** Raw Info Query **************/

bool SimpleAttentionKVCacheObj::Empty() const { return cur_batch_size_ == 0; }

int32_t SimpleAttentionKVCacheObj::GetNumAvailablePages() const {
  // In simple cache, we calculate remaining capacity
  int64_t total_used = 0;
  for (int i = 0; i < cur_batch_size_; ++i) {
    total_used += cur_seq_lens_host_[i];
  }
  int64_t total_capacity = max_batch_size_ * max_seq_len_;
  return static_cast<int32_t>(total_capacity - total_used);
}

int32_t SimpleAttentionKVCacheObj::GetTotalSequenceLength() const {
  int32_t total_length = 0;
  for (int i = 0; i < cur_batch_size_; ++i) {
    total_length += cur_seq_lens_host_[i];
  }
  return total_length;
}

/************** Attention **************/

ForwardPhase SimpleAttentionKVCacheObj::DetermineForwardPhase(
    const IntTuple& append_lengths) const {
  // Simple rule for edge inference:
  // - Prefill: Any sequence appends > 1 token (processing input prompt)
  // - Decode: All sequences append exactly 1 token (generating output)

  bool all_single_token = true;
  for (int i = 0; i < append_lengths.size(); ++i) {
    if (append_lengths[i] != 1) {
      all_single_token = false;
      break;
    }
  }

  return all_single_token ? ForwardPhase::kDecode : ForwardPhase::kPrefill;
}

void SimpleAttentionKVCacheObj::BeginForward(const IntTuple& seq_ids,
                                             const IntTuple& append_lengths,
                                             const Optional<IntTuple>& opt_token_tree_parent_ptr) {
  CHECK_EQ(seq_ids.size(), append_lengths.size())
      << "seq_ids and append_lengths must have the same size.";
  CHECK_EQ(seq_ids.size(), cur_batch_size_)
      << "Number of sequences in forward must match current batch size.";
  CHECK(!opt_token_tree_parent_ptr.defined())
      << "Tree attention is not supported in SimpleKVCache.";

  // Determine current forward phase based on append lengths
  current_phase_ = DetermineForwardPhase(append_lengths);

  // Update sequence lengths and prepare query positions
  std::vector<int32_t> q_positions;
  cur_seq_idx_host_.clear();
  batch_seq_ind_host_.clear();
  batch_seq_ind_host_.push_back(0);
  for (int i = 0; i < seq_ids.size(); ++i) {
    auto it = seq_id_to_idx_.find(seq_ids[i]);
    CHECK(it != seq_id_to_idx_.end()) << "Sequence " << seq_ids[i] << " not found in cache.";

    int64_t seq_idx = it->second;
    int64_t current_len = cur_seq_lens_host_[seq_idx];
    int64_t append_len = append_lengths[i];
    cur_seq_idx_host_.push_back(static_cast<int64_t>(seq_idx));
    batch_seq_ind_host_.push_back(batch_seq_ind_host_.back() + append_len);

    CHECK_LE(current_len + append_len, max_seq_len_)
        << "Sequence length " << (current_len + append_len) << " exceeds maximum " << max_seq_len_;

    // Add query positions for this sequence
    for (int64_t pos = 0; pos < append_len; ++pos) {
      q_positions.push_back(current_len + pos);
    }

    // Update sequence length
    cur_seq_lens_host_[seq_idx] += append_len;
  }
  batch_seq_ind_device_ = NDArray::Empty({cur_batch_size_ + 1}, dtype_aux_int64_, device_);
  batch_seq_ind_device_.CopyFromBytes(batch_seq_ind_host_.data(),
                                      (cur_batch_size_ + 1) * sizeof(int64_t));
  cur_seq_idx_device_ = NDArray::Empty({cur_batch_size_}, dtype_aux_int64_, device_);
  cur_seq_lens_device_ = NDArray::Empty({cur_batch_size_}, dtype_aux_int64_, device_);
  cur_seq_idx_device_.CopyFromBytes(cur_seq_idx_host_.data(), cur_batch_size_ * sizeof(int64_t));
  cur_seq_lens_device_.CopyFromBytes(cur_seq_lens_host_.data(), cur_batch_size_ * sizeof(int64_t));
  // Copy query positions to device
  cur_query_positions_ =
      NDArray::Empty({static_cast<int32_t>(q_positions.size())}, dtype_aux_int32_, device_);
  cur_query_positions_.CopyFromBytes(q_positions.data(), q_positions.size() * sizeof(int32_t));
}

void SimpleAttentionKVCacheObj::EndForward() {
  // No special cleanup needed for simple cache
}

IntTuple SimpleAttentionKVCacheObj::DisaggPrepareRecv(int64_t seq_id, int length) {
  LOG(FATAL) << "Disaggregation is not supported in SimpleKVCache.";
  return IntTuple{};
}

void SimpleAttentionKVCacheObj::DisaggMarkSend(int64_t seq_id, int64_t begin,
                                               const IntTuple& compressed_remote_position_map,
                                               int32_t recver_pe_offset) {
  LOG(FATAL) << "Disaggregation is not supported in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::AttentionWithFusedQKV(int64_t layer_id, NDArray qkv_data,
                                                      Optional<NDArray> mask, NDArray o_data,
                                                      double sm_scale) {
  CHECK_GE(layer_id, 0) << "Layer ID must be non-negative.";
  CHECK_LT(layer_id, num_layers_) << "Layer ID " << layer_id << " exceeds number of layers "
                                  << num_layers_;
  // Phase-aware attention computation
  const char* phase_name = (current_phase_ == ForwardPhase::kPrefill) ? "Prefill" : "Decode";
  int64_t total_len = qkv_data->shape[0];
  CHECK_EQ(qkv_data->shape[1], num_qo_heads_ + 2 * num_kv_heads_);
  CHECK_EQ(qkv_data->shape[2], head_dim_);
  // Copy Q, K, V data from fused QKV tensor
  NDArray q_view = q_data_tmp_.CreateView({total_len, num_qo_heads_, head_dim_}, qkv_data->dtype);
  NDArray k_view = k_data_tmp_.CreateView({total_len, num_kv_heads_, head_dim_}, qkv_data->dtype);
  NDArray v_view = v_data_tmp_.CreateView({total_len, num_kv_heads_, head_dim_}, qkv_data->dtype);

  f_split_rotary_(qkv_data, cur_query_positions_, q_view, k_view, v_view, rope_mode_);
  // Append KV to cache with correct TIR interface
  CHECK(f_append_kv_.defined()) << "Append KV function not defined.";
  f_append_kv_(cache_[layer_id], k_view, v_view, batch_seq_ind_device_, cur_seq_idx_device_,
               cur_seq_lens_device_);
  // Phase-aware attention computation - select appropriate function
  ffi::Function& attention_func =
      (current_phase_ == ForwardPhase::kPrefill) ? f_attention_prefill_ : f_attention_decode_;

  CHECK(attention_func.defined()) << "Attention compute function not defined for phase: "
                                  << (current_phase_ == ForwardPhase::kPrefill ? "Prefill"
                                                                               : "Decode");
  // Determine causal mode based on current phase:
  // Prefill: causal = 0 (no causal masking for initial prompt processing)
  // Decode: causal = 1 (causal masking for autoregressive generation)
  int32_t causal = (current_phase_ == ForwardPhase::kDecode) ? 0 : 1;
  // Calculate the maximum query length in the batch
  int64_t max_query_len = 0;
  for (int i = 0; i < cur_batch_size_; ++i) {
    max_query_len = std::max(max_query_len, batch_seq_ind_host_[i + 1] - batch_seq_ind_host_[i]);
  }
  attention_func(q_view, cache_[layer_id], batch_seq_ind_device_, cur_seq_idx_device_,
                 cur_seq_lens_device_, o_data, sm_scale, max_query_len, causal);
}

void SimpleAttentionKVCacheObj::SelfAttention(int64_t layer_id, NDArray q_data, NDArray k_data,
                                              NDArray v_data, NDArray o_data, NDArray lse_data,
                                              double sm_scale) {
  LOG(FATAL) << "SelfAttention is not implemented in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::CrossAttention(int64_t layer_id, NDArray q_data, NDArray o_data,
                                               NDArray lse_data, double sm_scale) {
  LOG(FATAL) << "Cross attention is not implemented in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::AppendMLAKV(int64_t layer_id, NDArray kv_data) {
  LOG(FATAL) << "MLA is not supported in SimpleKVCache.";
}

Array<NDArray> SimpleAttentionKVCacheObj::MergeAttnOutputInplace(NDArray o_self_attn,
                                                                 NDArray lse_self_attn,
                                                                 NDArray o_cross_attn,
                                                                 NDArray lse_cross_attn) {
  LOG(FATAL) << "MergeAttnOutputInplace is not implemented in SimpleKVCache.";
  return Array<NDArray>{};
}

void SimpleAttentionKVCacheObj::CommitAcceptedTokenTreeNodes(const IntTuple& seq_ids,
                                                             const IntTuple& leaf_indices) {
  LOG(WARNING) << "Tree attention is not supported in SimpleKVCache. Ignoring commit operation.";
}

NDArray SimpleAttentionKVCacheObj::GetQueryPositions() { return cur_query_positions_; }

ForwardPhase SimpleAttentionKVCacheObj::GetCurrentPhase() const { return current_phase_; }

void SimpleAttentionKVCacheObj::DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos,
                                           NDArray k_data, NDArray v_data) {
  auto it = seq_id_to_idx_.find(seq_id);
  CHECK(it != seq_id_to_idx_.end()) << "Sequence " << seq_id << " not found.";

  int64_t seq_idx = it->second;
  CHECK_GE(start_pos, 0) << "Start position must be non-negative.";
  CHECK_LE(end_pos, cur_seq_lens_host_[seq_idx]) << "End position exceeds sequence length.";
  CHECK_LT(start_pos, end_pos) << "Start position must be less than end position.";

  // Copy KV data for all layers
  for (int layer = 0; layer < num_layers_; ++layer) {
    for (int64_t pos = start_pos; pos < end_pos; ++pos) {
      // Copy K data
      NDArray k_src =
          cache_[layer].CreateView({1, num_kv_heads_, head_dim_}, cache_[layer]->dtype,
                                   ((seq_idx * max_seq_len_ + pos) * 2) * num_kv_heads_ *
                                       head_dim_ * cache_[layer].DataType().bytes());
      NDArray k_dst = k_data.CreateView({1, num_kv_heads_, head_dim_}, k_data->dtype,
                                        (layer * (end_pos - start_pos) + (pos - start_pos)) *
                                            num_kv_heads_ * head_dim_ * k_data.DataType().bytes());
      k_dst.CopyFrom(k_src);

      // Copy V data
      NDArray v_src =
          cache_[layer].CreateView({1, num_kv_heads_, head_dim_}, cache_[layer]->dtype,
                                   ((seq_idx * max_seq_len_ + pos) * 2 + 1) * num_kv_heads_ *
                                       head_dim_ * cache_[layer].DataType().bytes());
      NDArray v_dst = v_data.CreateView({1, num_kv_heads_, head_dim_}, v_data->dtype,
                                        (layer * (end_pos - start_pos) + (pos - start_pos)) *
                                            num_kv_heads_ * head_dim_ * v_data.DataType().bytes());
      v_dst.CopyFrom(v_src);
    }
  }
}

void SimpleAttentionKVCacheObj::DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos,
                                              NDArray kv_data) {
  LOG(FATAL) << "MLA debug is not supported in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::DebugSetKV(int64_t seq_id, int64_t start_pos, NDArray k_data,
                                           NDArray v_data) {
  LOG(FATAL) << "DebugSetKV is not implemented in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::LinearAttention(int64_t layer_id, NDArray q_data, NDArray k_data,
                                                NDArray v_data, double sm_scale) {
  LOG(FATAL) << "Linear attention is not supported in SimpleKVCache.";
}

void SimpleAttentionKVCacheObj::UpdateSequenceLength(int64_t seq_idx, int64_t new_length) {
  CHECK_GE(seq_idx, 0) << "Sequence index must be non-negative.";
  CHECK_LT(seq_idx, cur_batch_size_) << "Sequence index exceeds current batch size.";
  CHECK_GE(new_length, 0) << "New length must be non-negative.";
  CHECK_LE(new_length, max_seq_len_) << "New length exceeds maximum sequence length.";

  cur_seq_lens_host_[seq_idx] = new_length;
}

TVM_REGISTER_OBJECT_TYPE(SimpleAttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "vm.builtin.simple_attention_kv_cache_create", [](ffi::PackedArgs args, ffi::Any* rv) {
        CHECK_EQ(args.size(), 14) << "Invalid number of arguments for SimpleKVCache constructor.";

        ffi::Shape cache_config = args[0].cast<ffi::Shape>();
        CHECK_EQ(cache_config.size(), 2)
            << "Cache config should have 2 elements: [max_batch_size, max_seq_len].";

        int64_t max_batch_size = cache_config[0];
        int64_t max_seq_len = cache_config[1];
        int64_t num_layers = args[1].cast<int64_t>();
        int64_t num_qo_heads = args[2].cast<int64_t>();
        int64_t num_kv_heads = args[3].cast<int64_t>();
        int64_t head_dim = args[4].cast<int64_t>();
        int rope_mode = args[5].cast<int>();
        double rotary_scale = args[6].cast<double>();
        double rotary_theta = args[7].cast<double>();
        Optional<NDArray> rope_ext_factors = std::nullopt;  // args[8]
        NDArray init = args[9].cast<NDArray>();
        ffi::Function f_append_kv = args[10].cast<ffi::Function>();
        ffi::Function f_attention_prefill = args[11].cast<ffi::Function>();
        ffi::Function f_attention_decode = args[12].cast<ffi::Function>();
        ffi::Function f_split_rotary = args[13].cast<ffi::Function>();

        if (auto opt_nd = args[8].as<NDArray>()) {
          rope_ext_factors = opt_nd.value();
        }

        ObjectPtr<SimpleAttentionKVCacheObj> n = make_object<SimpleAttentionKVCacheObj>(
            max_batch_size, max_seq_len, num_layers, num_qo_heads, num_kv_heads, head_dim,
            RoPEMode(rope_mode), rotary_scale, rotary_theta, std::move(rope_ext_factors),
            init->dtype, init->device, std::move(f_append_kv), std::move(f_attention_prefill),
            std::move(f_attention_decode), std::move(f_split_rotary));

        *rv = AttentionKVCache(std::move(n));
      });

  refl::GlobalDef().def_packed("vm.builtin.simple_kv_cache_get_query_positions",
                               [](ffi::PackedArgs args, ffi::Any* rv) {
                                 CHECK_EQ(args.size(), 1) << "Invalid number of arguments.";
                                 AttentionKVCache cache = args[0].cast<AttentionKVCache>();
                                 *rv = cache->GetQueryPositions();
                               });

  refl::GlobalDef().def_packed(
      "vm.builtin.attention_kv_cache_attention_with_fused_qkv_simple",
      [](ffi::PackedArgs args, ffi::Any* rv) {
        CHECK_EQ(args.size(), 5) << "Invalid number of arguments.";
        // NDArray output = args[0].cast<NDArray>();  // DPS output tensor (first argument)
        AttentionKVCache cache = args[0].cast<AttentionKVCache>();
        int64_t layer_id = args[1].cast<int64_t>();
        double sm_scale = args[2].cast<double>();
        NDArray qkv_data = args[3].cast<NDArray>();
        NDArray output = args[4].cast<NDArray>();  // DPS output tensor (first argument)

        // Use the provided output tensor instead of creating a new one
        cache->AttentionWithFusedQKV(layer_id, qkv_data, std::nullopt, output, sm_scale);
        // No need to set *rv since we're using DPS (destination passing style)
      });
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm