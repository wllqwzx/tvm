# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Test cases for Simple KV Cache Kernel(non-paged implementation)."""

import numpy as np
import math
import unittest
import numpy as np
import tvm
from tvm.relax.frontend.nn.llm.kv_cache import _simple_attention_cpu


def simple_kvcache_attention_cpu_numpy(
    q: np.ndarray,
    kvcache: np.ndarray,
    qo_seq_lens: np.ndarray,
    kv_seq_lens: np.ndarray,
    output: np.ndarray,
    sm_scale: float,
):
    """
    A simple numpy implementation of attention with kv cache for testing.
    q: (total_seq_len, num_qo_heads, head_dim)
    kvcache: (batch_size, max_seq_len, 2, num_kv_heads, head_dim)
    qo_seq_lens: (batch_size+1,)
    kv_seq_lens: (batch_size,)
    output: (total_seq_len, num_qo_heads, head_dim)
    sm_scale: float
    """
    _, num_qo_heads, head_dim = q.shape
    batch_size = kv_seq_lens.shape[0]
    num_kv_heads = kvcache.shape[3]
    group_size = num_qo_heads // num_kv_heads
    for h_idx in range(num_qo_heads):
        for b_idx in range(batch_size):
            q_local = np.empty(shape=(head_dim,), dtype=q.dtype)
            k_local = np.empty(shape=(head_dim,), dtype=q.dtype)
            v_local = np.empty(shape=(head_dim,), dtype=q.dtype)
            o_local = np.empty(shape=(head_dim,), dtype=q.dtype)

            for qo_s_idx in range(qo_seq_lens[b_idx + 1] - qo_seq_lens[b_idx]):
                S = np.empty(shape=(1,), dtype=q.dtype)
                # online softmax
                D = np.zeros(shape=(1,), dtype=q.dtype)
                M = np.empty(shape=(1,), dtype=q.dtype)
                M[0] = -np.inf
                NEW_M = np.empty(shape=(1,), dtype=q.dtype)
                factor1 = np.empty(shape=(1,), dtype=q.dtype)
                factor2 = np.empty(shape=(1,), dtype=q.dtype)
                cur_qo_s_idx = qo_seq_lens[b_idx] + qo_s_idx
                # load q
                for d_idx in range(head_dim):
                    q_local[d_idx] = q[cur_qo_s_idx, h_idx, d_idx]
                    o_local[d_idx] = 0.0
                for kv_s_idx in range(kv_seq_lens[b_idx]):
                    S[0] = 0
                    # load k, v
                    for d_idx in range(head_dim):
                        k_local[d_idx] = kvcache[b_idx, kv_s_idx, 0, h_idx // group_size, d_idx]
                        v_local[d_idx] = kvcache[b_idx, kv_s_idx, 1, h_idx // group_size, d_idx]
                        # compute S
                        S[0] += q_local[d_idx] * k_local[d_idx]
                    S[0] *= sm_scale

                    NEW_M[0] = max(M[0], S[0])
                    factor1[0] = np.exp(M[0] - NEW_M[0])
                    factor2[0] = np.exp(S[0] - NEW_M[0])
                    D[0] *= factor1[0]
                    D[0] += factor2[0]
                    M[0] = NEW_M[0]

                    for d_idx in range(head_dim):
                        o_local[d_idx] = o_local[d_idx] * factor1[0] + v_local[d_idx] * factor2[0]
                for d_idx in range(head_dim):
                    output[cur_qo_s_idx, h_idx, d_idx] = o_local[d_idx] / D[0]


def run_simple_attention_cpu_tir(
    q: np.ndarray,
    kvcache: np.ndarray,
    qo_seq_lens: np.ndarray,
    kv_seq_lens: np.ndarray,
    output: np.ndarray,
    sm_scale: float,
):
    num_kv_heads = kvcache.shape[3]
    num_qo_heads = output.shape[1]
    head_dim = output.shape[2]
    dtype = "float32" if q.dtype == np.float32 else "float16"
    attn_func = _simple_attention_cpu(num_kv_heads, num_qo_heads, head_dim, dtype)
    mod = tvm.IRModule.from_expr(attn_func)
    ex = tvm.build(mod, target="llvm")
    q_tvm = tvm.nd.array(q)
    kvcache_tvm = tvm.nd.array(kvcache)
    qo_seq_lens_tvm = tvm.nd.array(qo_seq_lens)
    kv_seq_lens_tvm = tvm.nd.array(kv_seq_lens)
    cur_seq_idx = tvm.nd.array(np.array(list(range(kv_seq_lens.shape[0]))).astype("int64"))
    output_tvm = tvm.nd.array(output)
    ex(
        q_tvm,
        kvcache_tvm,
        qo_seq_lens_tvm,
        cur_seq_idx,
        kv_seq_lens_tvm,
        output_tvm,
        sm_scale,
        0,
    )
    return output_tvm.numpy()


class TestSimpleKVCacheAttentionCPUNumpy(unittest.TestCase):
    def run_attention_test(self, batch_size, max_seq_len, num_heads, head_dim):
        np.random.seed(0)

        # 随机生成每个 batch 的 kv 长度
        kv_seq_lens = np.random.randint(1, max_seq_len + 1, size=batch_size, dtype=np.int64)

        # 随机生成 query 长度，并转成 prefix sum
        qo_lengths = np.random.randint(1, max_seq_len + 1, size=batch_size, dtype=np.int64)
        qo_seq_lens = np.zeros(batch_size + 1, dtype=np.int64)
        qo_seq_lens[1:] = np.cumsum(qo_lengths)
        total_seq_len = qo_seq_lens[-1]

        q = np.random.randn(total_seq_len, num_heads, head_dim).astype(np.float32)
        kvcache = np.random.randn(batch_size, max_seq_len, 2, num_heads, head_dim).astype(
            np.float32
        )
        output = np.zeros((total_seq_len, num_heads, head_dim), dtype=np.float32)
        output_tir = np.zeros((total_seq_len, num_heads, head_dim), dtype=np.float32)
        sm_scale = 1.0 / np.sqrt(head_dim)

        # 调用待测函数
        simple_kvcache_attention_cpu_numpy(q, kvcache, qo_seq_lens, kv_seq_lens, output, sm_scale)
        output_tir = run_simple_attention_cpu_tir(
            q, kvcache, qo_seq_lens, kv_seq_lens, output_tir, sm_scale
        )

        # 用朴素实现计算期望输出
        output_expected = np.zeros_like(output)
        for b in range(batch_size):
            q_start, q_end = qo_seq_lens[b], qo_seq_lens[b + 1]
            kv_len = kv_seq_lens[b]
            for h in range(num_heads):
                Q = q[q_start:q_end, h]  # (q_len, head_dim)
                K = kvcache[b, :kv_len, 0, h]  # (kv_len, head_dim)
                V = kvcache[b, :kv_len, 1, h]  # (kv_len, head_dim)
                scores = (Q @ K.T) * sm_scale  # (q_len, kv_len)
                probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True)
                output_expected[q_start:q_end, h] = probs @ V
        np.testing.assert_allclose(output, output_expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(output_tir, output_expected, rtol=1e-5, atol=1e-6)

    def test_various_configs(self):
        configs = [
            (1, 4, 1, 2),  # small case
            (2, 5, 2, 4),  # medium case
            (3, 6, 2, 8),  # larger head_dim
            (2, 8, 4, 16),  # multiple heads
        ]
        for cfg in configs:
            with self.subTest(cfg=cfg):
                self.run_attention_test(*cfg)

    def test_extreme_cases(self):
        extreme_configs = [
            (1, 1, 1, 1),  # 最小规模，单 query 单 key 单 head 单 dim
            (1, 8, 1, 1),  # 单 query，多 key
            (1, 1, 1, 8),  # 单 key，多维度
            (1, 1, 4, 4),  # 单 query 单 key，多 head
            (2, 1, 1, 1),  # 多 batch，每个只有 1 个 kv
        ]
        for cfg in extreme_configs:
            with self.subTest(extreme_cfg=cfg):
                self.run_attention_test(*cfg)


if __name__ == "__main__":
    unittest.main()
