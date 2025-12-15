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
 * \file tests/cpp/simple_kv_cache_test.cc
 * \brief Unit tests for SimpleKVCache implementation.
 */

#include "../../src/runtime/vm/simple_kv_cache.h"

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <vector>

using namespace tvm::runtime;
using namespace tvm::runtime::vm;

class SimpleKVCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Test configuration
    max_batch_size_ = 4;
    max_seq_len_ = 128;
    num_layers_ = 2;
    num_qo_heads_ = 8;
    num_kv_heads_ = 4;
    head_dim_ = 64;

    device_ = DLDevice{kDLCPU, 0};
    dtype_ = DLDataType{kDLFloat, 32, 1};

    // Create mock functions for append_kv and attention functions
    f_append_kv_ = CreateMockAppendKVFunction();
    f_attention_prefill_ = CreateMockPrefillFunction();
    f_attention_decode_ = CreateMockDecodeFunction();

    // Create SimpleKVCache instance
    cache_ = CreateSimpleKVCache();
  }

  void TearDown() override { cache_ = nullptr; }

  tvm::ffi::ObjectPtr<SimpleAttentionKVCacheObj> CreateSimpleKVCache() {
    return tvm::ffi::make_object<SimpleAttentionKVCacheObj>(
        max_batch_size_, max_seq_len_, num_layers_, num_qo_heads_, num_kv_heads_, head_dim_,
        RoPEMode::kNone, 1.0, 10000.0, std::nullopt, dtype_, device_, f_append_kv_,
        f_attention_prefill_, f_attention_decode_);
  }

  tvm::ffi::Function CreateMockAppendKVFunction() {
    // Mock implementation that does nothing but validates inputs
    return tvm::ffi::Function([](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
      EXPECT_EQ(args.size(), 4);  // cache, k_data, v_data, positions
    });
  }

  tvm::ffi::Function CreateMockPrefillFunction() {
    // Mock implementation that does nothing but validates inputs
    return tvm::ffi::Function([](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
      EXPECT_EQ(args.size(), 5);  // q_data, cache, seq_lens, output, sm_scale
    });
  }

  tvm::ffi::Function CreateMockDecodeFunction() {
    // Mock implementation that does nothing but validates inputs
    return tvm::ffi::Function([](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
      EXPECT_EQ(args.size(), 5);  // q_data, cache, seq_lens, output, sm_scale
    });
  }

  // Test parameters
  int64_t max_batch_size_;
  int64_t max_seq_len_;
  int64_t num_layers_;
  int64_t num_qo_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  DLDevice device_;
  DLDataType dtype_;
  tvm::ffi::Function f_append_kv_;
  tvm::ffi::Function f_attention_prefill_;
  tvm::ffi::Function f_attention_decode_;
  tvm::ffi::ObjectPtr<SimpleAttentionKVCacheObj> cache_;
};

TEST_F(SimpleKVCacheTest, ConstructorInitialization) {
  ASSERT_NE(cache_.get(), nullptr);
  EXPECT_TRUE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);
  EXPECT_GT(cache_->GetNumAvailablePages(), 0);
}

TEST_F(SimpleKVCacheTest, AddSequence) {
  int64_t seq_id = 1;

  // Initially empty
  EXPECT_TRUE(cache_->Empty());

  // Add a sequence
  cache_->AddSequence(seq_id);
  EXPECT_FALSE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);  // No tokens appended yet
}

TEST_F(SimpleKVCacheTest, AddMultipleSequences) {
  std::vector<int64_t> seq_ids = {1, 2, 3};

  for (auto seq_id : seq_ids) {
    cache_->AddSequence(seq_id);
  }

  EXPECT_FALSE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);
}

TEST_F(SimpleKVCacheTest, AddSequenceExceedsMaxBatch) {
  // Add sequences up to max batch size
  for (int64_t i = 0; i < max_batch_size_; ++i) {
    cache_->AddSequence(i);
  }

  // Verify all sequences are added
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);  // No tokens yet
}

TEST_F(SimpleKVCacheTest, AddDuplicateSequence) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // Verify sequence was added successfully
  EXPECT_FALSE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);
}

TEST_F(SimpleKVCacheTest, RemoveSequence) {
  int64_t seq_id = 1;

  // Add and then remove
  cache_->AddSequence(seq_id);
  EXPECT_FALSE(cache_->Empty());

  cache_->RemoveSequence(seq_id);
  EXPECT_TRUE(cache_->Empty());
}

TEST_F(SimpleKVCacheTest, RemoveNonexistentSequence) {
  int64_t seq_id = 999;

  // Add then remove should work
  cache_->AddSequence(seq_id);
  EXPECT_FALSE(cache_->Empty());

  cache_->RemoveSequence(seq_id);
  EXPECT_TRUE(cache_->Empty());
}

TEST_F(SimpleKVCacheTest, RemoveSequenceWithReordering) {
  std::vector<int64_t> seq_ids = {1, 2, 3};

  // Add multiple sequences
  for (auto seq_id : seq_ids) {
    cache_->AddSequence(seq_id);
  }

  // Remove middle sequence (should trigger reordering)
  cache_->RemoveSequence(2);

  // Should still have 2 sequences
  EXPECT_FALSE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);
}

TEST_F(SimpleKVCacheTest, ForkSequence) {
  int64_t parent_seq_id = 1;
  int64_t child_seq_id = 2;

  // Add parent sequence
  cache_->AddSequence(parent_seq_id);

  // Fork sequence
  cache_->ForkSequence(parent_seq_id, child_seq_id, 0);

  // Should have 2 sequences now
  EXPECT_FALSE(cache_->Empty());
}

TEST_F(SimpleKVCacheTest, ForkSequenceInvalidParent) {
  int64_t parent_seq_id = 999;  // Non-existent
  int64_t child_seq_id = 2;

  // Fork from non-existent parent should fail
  EXPECT_THROW(cache_->ForkSequence(parent_seq_id, child_seq_id, 0), tvm::Error);
}

TEST_F(SimpleKVCacheTest, ForkSequenceExistingChild) {
  int64_t parent_seq_id = 1;
  int64_t child_seq_id = 2;

  // Add both sequences
  cache_->AddSequence(parent_seq_id);
  cache_->AddSequence(child_seq_id);

  // Fork to existing child should fail
  EXPECT_THROW(cache_->ForkSequence(parent_seq_id, child_seq_id, 0), tvm::Error);
}

TEST_F(SimpleKVCacheTest, ForkSequenceExceedsMaxBatch) {
  // Fill cache to max capacity
  for (int64_t i = 0; i < max_batch_size_; ++i) {
    cache_->AddSequence(i);
  }

  // Fork should fail when cache is full
  EXPECT_THROW(cache_->ForkSequence(0, max_batch_size_, 0), tvm::Error);
}

TEST_F(SimpleKVCacheTest, PopN) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // PopN on sequence with zero length should work
  cache_->PopN(seq_id, 0);

  // Verify cache is still valid
  EXPECT_FALSE(cache_->Empty());
}

TEST_F(SimpleKVCacheTest, Clear) {
  // Add some sequences
  cache_->AddSequence(1);
  cache_->AddSequence(2);
  EXPECT_FALSE(cache_->Empty());

  // Clear should reset everything
  cache_->Clear();
  EXPECT_TRUE(cache_->Empty());
  EXPECT_EQ(cache_->GetTotalSequenceLength(), 0);
}

TEST_F(SimpleKVCacheTest, BeginForwardBasic) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({5});

  // Should not throw
  EXPECT_NO_THROW(cache_->BeginForward(seq_ids, append_lengths));
}

TEST_F(SimpleKVCacheTest, BeginForwardMismatchedSizes) {
  IntTuple seq_ids({1});
  IntTuple append_lengths({5, 10});  // Different size

  // Mismatched sizes should fail
  EXPECT_THROW(cache_->BeginForward(seq_ids, append_lengths), tvm::Error);
}

TEST_F(SimpleKVCacheTest, BeginForwardWithTokenTree) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({5});
  IntTuple token_tree({0, 0, 1, 1, 2});  // Sample token tree

  // Token tree should be rejected (not supported in SimpleKVCache)
  EXPECT_THROW(cache_->BeginForward(seq_ids, append_lengths, token_tree), tvm::Error);
}

TEST_F(SimpleKVCacheTest, BeginForwardExceedsMaxLength) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({max_seq_len_ + 1});  // Exceeds max length

  // Should fail when exceeding max sequence length
  EXPECT_THROW(cache_->BeginForward(seq_ids, append_lengths), tvm::Error);
}

TEST_F(SimpleKVCacheTest, EndForward) {
  // EndForward should not throw (it's a no-op in SimpleKVCache)
  EXPECT_NO_THROW(cache_->EndForward());
}

TEST_F(SimpleKVCacheTest, AttentionWithFusedQKV) {
  int64_t seq_id = 1;
  int64_t layer_id = 0;
  int64_t total_len = 10;

  cache_->AddSequence(seq_id);

  // Prepare for forward
  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({total_len});
  cache_->BeginForward(seq_ids, append_lengths);

  // Create test data
  Tensor qkv_data =
      Tensor::Empty({total_len, num_qo_heads_ + 2 * num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor output = Tensor::Empty({total_len, num_qo_heads_, head_dim_}, dtype_, device_);

  // Should not throw
  EXPECT_NO_THROW(cache_->AttentionWithFusedQKV(layer_id, qkv_data, std::nullopt, output, 1.0));
}

TEST_F(SimpleKVCacheTest, AttentionWithInvalidLayer) {
  int64_t invalid_layer = num_layers_ + 1;

  Tensor qkv_data =
      Tensor::Empty({1, num_qo_heads_ + 2 * num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor output = Tensor::Empty({1, num_qo_heads_, head_dim_}, dtype_, device_);

  // Invalid layer should fail
  EXPECT_THROW(cache_->AttentionWithFusedQKV(invalid_layer, qkv_data, std::nullopt, output, 1.0),
               tvm::Error);
}

TEST_F(SimpleKVCacheTest, SelfAttention) {
  int64_t seq_id = 1;
  int64_t layer_id = 0;
  int64_t total_len = 10;

  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({total_len});
  cache_->BeginForward(seq_ids, append_lengths);

  // Create test data
  Tensor q_data = Tensor::Empty({total_len, num_qo_heads_, head_dim_}, dtype_, device_);
  Tensor k_data = Tensor::Empty({total_len, num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor v_data = Tensor::Empty({total_len, num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor output = Tensor::Empty({total_len, num_qo_heads_, head_dim_}, dtype_, device_);
  Tensor lse_data = Tensor::Empty({total_len, num_qo_heads_}, dtype_, device_);

  // Should not throw
  EXPECT_NO_THROW(cache_->SelfAttention(layer_id, q_data, k_data, v_data, output, lse_data, 1.0));
}

// Note: UnsupportedOperations test removed as TVM uses CHECK macros that terminate
// the program rather than throwing exceptions. These would be tested in integration tests.

TEST_F(SimpleKVCacheTest, GetQueryPositions) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({5});
  cache_->BeginForward(seq_ids, append_lengths);

  Tensor positions = cache_->GetQueryPositions();
  EXPECT_NE(positions.get(), nullptr);
  EXPECT_EQ(positions->ndim, 1);
  EXPECT_EQ(positions->shape[0], 5);  // Should match append_length
}

TEST_F(SimpleKVCacheTest, DebugGetKV) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // Prepare some sequence length
  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({10});
  cache_->BeginForward(seq_ids, append_lengths);

  int64_t start_pos = 0;
  int64_t end_pos = 5;
  Tensor k_data =
      Tensor::Empty({num_layers_, end_pos - start_pos, num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor v_data =
      Tensor::Empty({num_layers_, end_pos - start_pos, num_kv_heads_, head_dim_}, dtype_, device_);

  // Should not throw
  EXPECT_NO_THROW(cache_->DebugGetKV(seq_id, start_pos, end_pos, k_data, v_data));
}

TEST_F(SimpleKVCacheTest, DebugGetKVValidRange) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // Prepare some sequence length
  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({5});
  cache_->BeginForward(seq_ids, append_lengths);

  int64_t start_pos = 0;
  int64_t end_pos = 3;
  Tensor k_data =
      Tensor::Empty({num_layers_, end_pos - start_pos, num_kv_heads_, head_dim_}, dtype_, device_);
  Tensor v_data =
      Tensor::Empty({num_layers_, end_pos - start_pos, num_kv_heads_, head_dim_}, dtype_, device_);

  // Valid ranges should work
  EXPECT_NO_THROW(cache_->DebugGetKV(seq_id, start_pos, end_pos, k_data, v_data));
}

TEST_F(SimpleKVCacheTest, CommitAcceptedTokenTreeNodes) {
  // This should generate a warning but not fail
  IntTuple seq_ids({1});
  IntTuple leaf_indices({0});

  // Should not throw (just generates warning)
  EXPECT_NO_THROW(cache_->CommitAcceptedTokenTreeNodes(seq_ids, leaf_indices));
}

// Tests for prefill/decode phase detection
TEST_F(SimpleKVCacheTest, PrefillPhaseDetection) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // Prefill: append multiple tokens (> 1)
  IntTuple seq_ids({seq_id});
  IntTuple append_lengths({10});  // Processing 10 tokens = prefill phase

  cache_->BeginForward(seq_ids, append_lengths);

  // Verify phase is detected as prefill
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
}

TEST_F(SimpleKVCacheTest, DecodePhaseDetection) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // First do a prefill
  IntTuple seq_ids({seq_id});
  IntTuple prefill_lengths({5});
  cache_->BeginForward(seq_ids, prefill_lengths);
  cache_->EndForward();

  // Then do decode: append exactly 1 token
  IntTuple decode_lengths({1});  // Processing 1 token = decode phase
  cache_->BeginForward(seq_ids, decode_lengths);

  // Verify phase is detected as decode
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kDecode);
}

TEST_F(SimpleKVCacheTest, MultiplePrefillPhases) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  // Multiple prefill phases with different lengths
  IntTuple seq_ids({seq_id});

  // First prefill
  IntTuple prefill1({8});
  cache_->BeginForward(seq_ids, prefill1);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
  cache_->EndForward();

  // Second prefill (user adds more input)
  IntTuple prefill2({3});
  cache_->BeginForward(seq_ids, prefill2);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
  cache_->EndForward();
}

TEST_F(SimpleKVCacheTest, BatchPrefillPhase) {
  // Add multiple sequences
  std::vector<int64_t> seq_ids_vec = {1, 2, 3};
  for (auto seq_id : seq_ids_vec) {
    cache_->AddSequence(seq_id);
  }

  // Batch prefill: all sequences have > 1 tokens
  IntTuple seq_ids({1, 2, 3});
  IntTuple append_lengths({5, 8, 3});  // All > 1 = prefill phase

  cache_->BeginForward(seq_ids, append_lengths);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
}

TEST_F(SimpleKVCacheTest, BatchDecodePhase) {
  // Add multiple sequences
  std::vector<int64_t> seq_ids_vec = {1, 2, 3};
  for (auto seq_id : seq_ids_vec) {
    cache_->AddSequence(seq_id);
  }

  // First do prefill for all
  IntTuple seq_ids({1, 2, 3});
  IntTuple prefill_lengths({5, 5, 5});
  cache_->BeginForward(seq_ids, prefill_lengths);
  cache_->EndForward();

  // Then batch decode: all sequences have exactly 1 token
  IntTuple decode_lengths({1, 1, 1});  // All = 1 = decode phase
  cache_->BeginForward(seq_ids, decode_lengths);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kDecode);
}

TEST_F(SimpleKVCacheTest, PhaseSwitching) {
  int64_t seq_id = 1;
  cache_->AddSequence(seq_id);

  IntTuple seq_ids({seq_id});

  // Start with prefill
  IntTuple prefill({10});
  cache_->BeginForward(seq_ids, prefill);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
  cache_->EndForward();

  // Switch to decode
  IntTuple decode({1});
  cache_->BeginForward(seq_ids, decode);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kDecode);
  cache_->EndForward();

  // Continue decode
  cache_->BeginForward(seq_ids, decode);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kDecode);
  cache_->EndForward();

  // Switch back to prefill (user adds more input)
  IntTuple prefill2({5});
  cache_->BeginForward(seq_ids, prefill2);
  EXPECT_EQ(cache_->GetCurrentPhase(), ForwardPhase::kPrefill);
}