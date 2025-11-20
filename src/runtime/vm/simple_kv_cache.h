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
 * \file src/runtime/vm/simple_kv_cache.h
 * \brief Header file for Simple KV cache object for edge-side inference.
 */
#ifndef TVM_RUNTIME_VM_SIMPLE_KV_CACHE_H_
#define TVM_RUNTIME_VM_SIMPLE_KV_CACHE_H_

#include <tvm/runtime/tensor.h>
#include <tvm/runtime/object.h>

#include <vector>

#include "attn_backend.h"
#include "kv_state.h"

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief Enumeration for different forward phases */
enum class ForwardPhase {
  kPrefill = 0,  // Processing input prompt (multiple tokens)
  kDecode = 1    // Generating output tokens (single token at a time)
};

/*!
 * \brief Simple KV cache for single-sequence edge-side inference.
 *
 * Unlike PagedKVCache which uses complex paging for memory efficiency in server scenarios,
 * SimpleKVCache uses contiguous memory allocation for simpler management and better
 * performance on edge devices where memory management complexity is not justified.
 *
 * This implementation is optimized for single-sequence inference with fixed maximum length,
 * which is the typical use case for edge devices.
 */
class SimpleAttentionKVCacheObj : public AttentionKVCacheObj {
 private:
  /********************* Configuration *********************/

  /*! \brief Maximum batch size supported by the cache. */
  const int64_t max_batch_size_;
  /*! \brief Maximum sequence length supported by the cache. */
  const int64_t max_seq_len_;
  /*! \brief The number of layers in the model. */
  const int64_t num_layers_;
  /*! \brief The number of query/output heads in the model. */
  const int64_t num_qo_heads_;
  /*! \brief The number of key/value heads in the model. */
  const int64_t num_kv_heads_;
  /*! \brief The number of features each head has. */
  const int64_t head_dim_;

  /*! \brief The RoPE application mode of KV cache.*/
  const RoPEMode rope_mode_;
  /*! \brief The RoPE scale. */
  const double rotary_scale_;
  /*! \brief The RoPE theta. */
  const double rotary_theta_;
  /*! \brief The optional RoPE extension factors for RoPE scaling. */
  const ffi::Optional<Tensor> rope_ext_factors_;

  /*! \brief The KV cache dtype. */
  const DataType kv_dtype_;
  /*! \brief We fix int32 to be the index dtype of auxiliary data. */
  const DLDataType dtype_aux_int64_ = DLDataType(DataType::Int(64, 1));
  const DLDataType dtype_aux_int32_ = DLDataType(DataType::Int(32, 1));

  /********************* Cache Storage *********************/

  /*!
   * \brief The KV data managed by the KV cache.
   * cache_ has `num_layers` NDArrays, each of them has layout
   * (max_batch_size, max_seq_len, 2, num_kv_heads, head_dim).
   * Along the "2" dimension, index 0 stands for K and 1 stands for V.
   */
  std::vector<Tensor> cache_;

  /********************* Sequence Management *********************/

  /*! \brief Current sequence lengths for each batch item. */
  std::vector<int64_t> cur_seq_lens_host_;
  Tensor cur_seq_lens_device_;
  /*! \brief Current batch size. */
  int64_t cur_batch_size_;
  std::vector<int64_t> batch_seq_ind_host_;
  Tensor batch_seq_ind_device_;

  /*! \brief Sequence id to index mapping. */
  std::unordered_map<int64_t, int64_t> seq_id_to_idx_;
  std::vector<int64_t> cur_seq_idx_host_;
  Tensor cur_seq_idx_device_;
  /*! \brief Current query positions for RoPE. */
  Tensor cur_query_positions_;
  /*! \brief Current forward phase for optimization. */
  ForwardPhase current_phase_;

  /********************* Kernel Functions *********************/

  ffi::Function f_append_kv_;
  ffi::Function f_attention_prefill_;  // Attention function for prefill phase
  ffi::Function f_attention_decode_;   // Attention function for decode phase
  ffi::Function f_split_rotary_;       // Function to split rotary embeddings

  /********************* Tempory Buffers *********************/
  Tensor q_data_tmp_;
  Tensor k_data_tmp_;
  Tensor v_data_tmp_;
  /*! \brief The device this SimpleKVCache runs on. */
  Device device_;

 public:
  /*! \brief Constructor. Take the cache configuration and initialize the NDArrays. */
  explicit SimpleAttentionKVCacheObj(int64_t max_batch_size, int64_t max_seq_len,
                                     int64_t num_layers, int64_t num_qo_heads, int64_t num_kv_heads,
                                     int64_t head_dim, RoPEMode rope_mode, double rotary_scale,
                                     double rotary_theta, ffi::Optional<Tensor> rope_ext_factors,
                                     DLDataType dtype, Device device, ffi::Function f_append_kv,
                                     ffi::Function f_attention_prefill,
                                     ffi::Function f_attention_decode,
                                     ffi::Function f_split_rotary);

  /*! \brief Reset the KV cache. */
  void Clear() final;

  /************** Sequence Management **************/

  void AddSequence(int64_t seq_id) final;
  void RemoveSequence(int64_t seq_id) final;
  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos = -1) final;
  void EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                 int32_t attn_sink_size) final;
  void PopN(int64_t seq_id, int32_t n) final;

  /************** Raw Info Query **************/

  bool Empty() const final;
  int32_t GetNumAvailablePages() const final;
  int32_t GetTotalSequenceLength() const final;

  /************** Attention **************/

  void BeginForward(const IntTuple& seq_ids, const IntTuple& append_lengths,
                    const ffi::Optional<IntTuple>& opt_token_tree_parent_ptr = std::nullopt) final;
  void EndForward() final;
  IntTuple DisaggPrepareRecv(int64_t seq_id, int length) final;
  void DisaggMarkSend(int64_t seq_id, int64_t begin, const IntTuple& compressed_remote_position_map,
                      int32_t recver_pe_offset) final;

  void AttentionWithFusedQKV(int64_t layer_id, Tensor qkv_data, ffi::Optional<Tensor> mask,
                             Tensor o_data, double sm_scale) final;
  void SelfAttention(int64_t layer_id, Tensor q_data, Tensor k_data, Tensor v_data,
                     Tensor o_data, Tensor lse_data, double sm_scale) final;
  void CrossAttention(int64_t layer_id, Tensor q_data, Tensor o_data, Tensor lse_data,
                      double sm_scale) final;
  void AppendMLAKV(int64_t layer_id, Tensor kv_data) final;
  ffi::Array<Tensor> MergeAttnOutputInplace(Tensor o_self_attn, Tensor lse_self_attn,
                                            Tensor o_cross_attn, Tensor lse_cross_attn) final;

  void CommitAcceptedTokenTreeNodes(const IntTuple& seq_ids, const IntTuple& leaf_indices) final;
  Tensor GetQueryPositions() final;

  /*! \brief Get current forward phase (prefill or decode). */
  ForwardPhase GetCurrentPhase() const;

  void DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos, Tensor k_data,
                  Tensor v_data) final;
  void DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos, Tensor kv_data) final;
  void DebugSetKV(int64_t seq_id, int64_t start_pos, Tensor k_data, Tensor v_data) final;

  void LinearAttention(int64_t layer_id, Tensor q_data, Tensor k_data, Tensor v_data,
                       double sm_scale) final;

  // static constexpr const char* _type_key = "relax.vm.SimpleAttentionKVCache";
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.vm.SimpleAttentionKVCache", SimpleAttentionKVCacheObj, AttentionKVCacheObj);

  /*! \brief Internal method to update sequence lengths. */
  void UpdateSequenceLength(int64_t seq_idx, int64_t new_length);
  /*! \brief Determine forward phase based on append lengths (simplified). */
  ForwardPhase DetermineForwardPhase(const IntTuple& append_lengths) const;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_SIMPLE_KV_CACHE_H_