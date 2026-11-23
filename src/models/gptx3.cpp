#include "models.h"

#include "llama-memory-recurrent.h"

// GPT-X3 (Axiomic Labs): hybrid of GDN2 (gated delta net v2) layers and gated GQA layers in a 3:1 layout.
//
// GDN2 recurrence (per head, S is [d_k, d_v]):
//   S' = diag(exp(g)) S
//   S  = S' + k (w * v - (b * k)^T S')^T
//   o  = (q / sqrt(d))^T S
// with per-channel decay g, erase gate b and write gate w.
//
// This maps exactly onto ggml_rwkv_wkv7 (state stored transposed, [d_v][d_k]):
//   state = state * w_rwkv + v_rwkv k^T + (state a) b_rwkv^T
// using r = q, w_rwkv = exp(g), v_rwkv = w * v, a = b * k * exp(g), b_rwkv = -k.

llm_build_gptx3::llm_build_gptx3(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_gdn2(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_gptx3::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // q_gate_proj outputs [q | gate] per head
    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);

    ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    cb(Qcur, "Qcur_view", il);

    ggml_tensor * gate = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * ggml_element_size(Qcur_full));
    cb(gate, "gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur);
    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur);

    // n_head_kv is per-layer (0 on GDN2 layers)
    const int64_t n_head_kv_l = hparams.n_head_kv(il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_l, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_l, n_tokens);

    // OffsetRMSNorm: the converter stores (1 + w)
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    // partial RoPE (n_rot < head_dim), consecutive pairs
    Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_attn(inp,
            nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    gate = ggml_sigmoid(ctx0, gate);
    cb(gate, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(layer.wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

// projection + causal depthwise conv1d + SiLU, using the per-layer conv state
// conv state layout per sequence: [Q state | K state | V state], each (d_conv - 1) * d_inner
ggml_tensor * llm_build_gptx3::build_gdn2_conv(
        ggml_tensor * conv_states_all,
        ggml_tensor * conv_state_all,
        int64_t       qkv,
        ggml_tensor * x,
        ggml_tensor * proj_w,
        ggml_tensor * conv_w,
        int64_t       kv_head,
        int           il) {
    const int64_t d_conv          = hparams.ssm_d_conv;
    const int64_t d_inner         = proj_w->ne[1];
    const int64_t n_seqs          = ubatch.n_seqs;
    const int64_t n_seq_tokens    = ubatch.n_seq_tokens;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
            (d_conv - 1) * ggml_element_size(conv_state_all),
            n_embd_r_total * ggml_element_size(conv_state_all),
            qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = build_lora_mm(proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    // {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // save the last (d_conv - 1) columns back into the cache
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, last_conv_x,
                ggml_view_3d(ctx0, conv_states_all,
                    d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1) * ggml_element_size(conv_states_all),
                    n_embd_r_total * ggml_element_size(conv_states_all),
                    (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * y = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    y = ggml_silu(ctx0, y);
    cb(y, "gdn2_conv", il);

    return y;
}

ggml_tensor * llm_build_gptx3::build_layer_gdn2(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto & layer = model.layers[il];

    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t n_gdn_head   = n_embd / head_dim;
    const int64_t d_inner      = head_dim * n_gdn_head;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const auto * mctx_cur = inp->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * q = build_gdn2_conv(conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, kv_head, il);
    ggml_tensor * k = build_gdn2_conv(conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, kv_head, il);
    ggml_tensor * v = build_gdn2_conv(conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, kv_head, il);

    q = ggml_reshape_3d(ctx0, q, head_dim, n_gdn_head, n_tokens);
    k = ggml_reshape_3d(ctx0, k, head_dim, n_gdn_head, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_gdn_head, n_tokens);

    // L2 normalization of q and k, then 1/sqrt(d) query scaling
    q = ggml_l2_norm(ctx0, q, 1e-6f);
    k = ggml_l2_norm(ctx0, k, 1e-6f);
    q = ggml_scale(ctx0, q, 1.0f / sqrtf((float) head_dim));
    cb(q, "gdn2_q", il);
    cb(k, "gdn2_k", il);

    // per-channel log decay: g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_b, ggml_mul_mat(ctx0, layer.ssm_f_a, cur));
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_softplus(ctx0, g);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_gdn_head, n_tokens);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_gdn_head, 1)); // ssm_a = -exp(A_log)
    ggml_tensor * decay = ggml_exp(ctx0, g);
    cb(decay, "gdn2_decay", il);

    // per-channel erase (beta) and write gates
    ggml_tensor * beta = ggml_sigmoid(ctx0, build_lora_mm(layer.ssm_beta, cur));
    beta = ggml_reshape_3d(ctx0, beta, head_dim, n_gdn_head, n_tokens);

    ggml_tensor * wgate = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur));
    wgate = ggml_reshape_3d(ctx0, wgate, head_dim, n_gdn_head, n_tokens);

    ggml_tensor * v_in = ggml_mul(ctx0, v, wgate);
    ggml_tensor * a_in = ggml_mul(ctx0, ggml_mul(ctx0, beta, k), decay);
    ggml_tensor * b_in = ggml_neg(ctx0, k);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);

    ggml_tensor * wkv_out = ggml_rwkv_wkv7(ctx0, q, decay, k, v_in, a_in, b_in, state);

    ggml_tensor * o = ggml_view_1d(ctx0, wkv_out, d_inner * n_tokens, 0);
    ggml_tensor * new_state = ggml_view_1d(ctx0, wkv_out, d_inner * head_dim * n_seqs,
            d_inner * n_tokens * ggml_element_size(wkv_out));

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    hparams.n_embd_s() * kv_head * ggml_element_size(ssm_states_all))));

    // output: per-head RMSNorm * sigmoid(g_b(g_a(x)) + bias)
    ggml_tensor * ogate = ggml_mul_mat(ctx0, layer.ssm_g_b, ggml_mul_mat(ctx0, layer.ssm_g_a, cur));
    ogate = ggml_add(ctx0, ogate, layer.ssm_g_b_b);
    ogate = ggml_sigmoid(ctx0, ogate);
    ogate = ggml_reshape_3d(ctx0, ogate, head_dim, n_gdn_head, n_tokens);

    o = ggml_reshape_3d(ctx0, o, head_dim, n_gdn_head, n_tokens);
    o = ggml_rms_norm(ctx0, o, hparams.f_norm_group_eps);
    o = ggml_mul(ctx0, o, layer.ssm_o_norm);
    o = ggml_mul(ctx0, o, ogate);
    cb(o, "gdn2_gated", il);

    o = ggml_reshape_2d(ctx0, o, d_inner, n_tokens);
    cur = build_lora_mm(layer.wo, o);
    cb(cur, "gdn2_out", il);

    return cur;
}
