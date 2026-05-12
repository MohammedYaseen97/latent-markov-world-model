PHASE 0 — ONLINE VAE PRETRAINING (v3 design)

Backbone frozen for weight updates; gradients still pass through activations.
VAE, ZInjector, and OutcomeHead are trained jointly via live 3-chunk generation.
No pre-saved rollout file. This loop IS the generation AND training loop.

LOAD pretrained backbone (frozen params — no .step() on backbone optimizer)
LOAD problems from data/math_easy_pool.jsonl  (L1-L5, 4974 problems)

INIT VAEStateEncoder  (encoder, decoder, transition)
INIT ZInjector        (Linear 64→1536, std=0.01 near-zero init)
INIT OutcomeHead      (2-layer MLP, z_3 → sigmoid → P(correct))
INIT AdamW optimizer  (VAE + ZInjector + OutcomeHead params only)

total_steps      = phase0.n_steps
kl_warmup_steps  = 0.5 × total_steps     # kl_weight: 0 → 1 over first 50%

FOR step in range(total_steps):

    batch = sample(problems, batch_size)   # B problems, G=8 rollouts each

    kl_weight = min(1.0, step / kl_warmup_steps)   # 0.0 at step 0, 1.0 halfway

    FOR each problem in batch:

        # ── CHUNK 1 (no z prefix) ────────────────────────────────────────
        WITH no_grad:
            chunk1_ids = backbone.generate(prompt, max_new_tokens=341)

        WITH grad:
            forward([prompt | chunk1_ids]) → last_hidden
            repr_1 = mean_pool(last_hidden[prompt_len : prompt_len+L1])  # LIVE
            z_1, mu_1, lv_1 = vae.encode_reparameterize(repr_1)
            prefix_1 = z_injector(z_1)       # in computation graph

        # ── CHUNK 2 (prefix_1 prepended) ─────────────────────────────────
        WITH no_grad:
            chunk2_ids = backbone.generate(
                inputs_embeds=[prefix_1_emb | prompt_emb | chunk1_emb],
                max_new_tokens=341
            )

        WITH grad:
            forward([prefix_1_emb | prompt | chunk1 | chunk2]) → last_hidden
            repr_2 = mean_pool(last_hidden over chunk2 positions)   # LIVE
            # grad chain: repr_2 ← backbone activations ← prefix_1 ← z_injector
            z_2, mu_2, lv_2 = vae.encode_reparameterize(repr_2)
            prefix_2 = z_injector(z_2)

        # ── CHUNK 3 (prefix_2 prepended) ─────────────────────────────────
        WITH no_grad:
            chunk3_ids = backbone.generate(
                inputs_embeds=[prefix_2_emb | chunk2_emb],
                max_new_tokens=342
            )
            reward = grade(chunk1 + chunk2 + chunk3)

        WITH grad:
            forward([prefix_2_emb | ... | chunk3]) → last_hidden
            repr_3 = mean_pool(last_hidden over chunk3 positions)   # LIVE
            z_3, mu_3, lv_3 = vae.encode_reparameterize(repr_3)

        # ── LOSSES ───────────────────────────────────────────────────────
        L_ELBO = sum over h in {1,2,3}:
            MSE(vae.decode(z_h), repr_h)                             # reconstruction
          + kl_weight × KL(N(mu_h, exp(lv_h)) || N(0,1))           # annealed KL

        L_transition = MSE(vae.transition(z_1), z_2)
                     + MSE(vae.transition(z_2), z_3)

        L_outcome = BCE(outcome_head(z_3), reward)

        L_total = λ_elbo × L_ELBO  +  λ_trans × L_transition  +  λ_out × L_outcome

    # ── BACKWARD ─────────────────────────────────────────────────────────
    L_total.backward()
    # Grad routes:
    #   L_ELBO    → encoder, decoder
    #   L_trans   → transition, encoder, z_injector
    #              (z_{h+1} ← repr_{h+1}[LIVE] ← backbone ← prefix_h ← z_injector)
    #   L_outcome → outcome_head, encoder
    #   All       → backbone activations [passthrough — backbone optimizer NOT stepped]
    clip_grad_norm(vae + z_injector + outcome_head, max_norm=1.0)
    optimizer.step()   # updates VAE + ZInjector + OutcomeHead only

SAVE {vae.state_dict(), z_injector.state_dict()} → runs/latent_grpo/phase0_vae.pt
