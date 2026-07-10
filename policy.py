import os
import csv
import glob
import random
import threading
import numpy as np
import torch

from dreamer import Dreamer, BatchPrefetcher


class Policy:
    def __init__(self, envs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), visualize_dreams=False):
        self.device = device
        self.envs = envs
        self.action_dim = envs.action_space.n

        # Run-control parameters.
        self.total_num_episodes = 10000
        self.training_per_episodes = 500
        self.seed = 42
        self.checkpoint_interval = 2   # Save every N episodes
        self.dream_horizon = 15        # number of env steps imagined per dream

        # How many dreams to visualize at the end of each training session.
        self.num_max_dreams = 10       # top combined-advantage ("max") dreams to save
        self.num_random_dreams = 10    # randomly sampled dreams to save

        # ==========================================================
        # DREAMER CONFIGURATION (every Dreamer.__init__ variable, in one place)
        # ==========================================================
        self.dreamer_config = dict(
        action_dim=self.action_dim,

        recurrent_dim=1024,        # size of recurrent state (h) in TSSM
        tssm_layers=8,             # 4 -> 8: depth helps long-range dynamics most
        tssm_heads=8,              # head_dim=128 at d_model=1024 (leave as-is)
        tssm_kv_heads=2,           # 4:1 GQA
        tssm_ffn=2816,             # ~8/3 * 1024 (correct for this width)
        context_length=80,         
        rows=40, cols=40,
        number_of_sequences=64,    
        steps_per_sequence=96,
        dreams_per_sequence=8,
        buffer_size=1000000,
        team_dim=6, item_dim=2,
        teamitem_out=128, ltm_reward_out=512, grid_out=128,
        mlp_dim=1024,
        curiosity_scale=0.25, # scale of curiosity reward relative to environment reward
        pmpo_alpha=0.5,
        entropy_scale=0.1,
        critic_ema_decay=0.90,
        bt_alpha=5e-4,             # off-diagonal (redundancy) weight, R2-Dreamer Table 2
        decoder_train_frames=512,  # detached viz-decoder frames per update (visualization only)
        continue_discount=0.997,
        dream_priority_fraction=0.025,
        dream_reward_priority_fraction=0.025,
        reward_sample_fraction=0.025,
        curiosity_sample_fraction=0.025,
        dream_lead_steps=10,
        ltm_gate_threshold=0.4,
        grid_gate_threshold=0.5,
        grid_zero_weight=5.0,  # extra BCE cost for missing an unexplored grid cell
    )

        self.visualize_dreams = visualize_dreams
        self.seedMeDaddy(self.seed)

        self.dreamer = Dreamer(device=self.device, envs=self.envs, **self.dreamer_config)

    def train(self):
        self.dreamer.loadCheckpoints()

        csv_filename = "pokemon_training_metrics.csv"
        
        headers =[
            "envSteps", "gradientSteps", "totalReward", "totalCuriosity",
            "worldModelLoss", "barlowTwinsLoss", "rewardPredictorLoss", "klLoss", "teamItemLoss", "ltmRewardLoss", "gridLoss", "varLoss", "curiosityLoss",
            "actorLoss", "entropies", "criticLoss", "curiosityCriticLoss", "advantages", "curiosityAdvantages", "criticValues", "curiosityCriticValues"
        ]
        
        # --- Initialize or Load CSV ---
        if not os.path.exists(csv_filename):
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        else:
            try:
                with open(csv_filename, 'r') as f:
                    lines =[line for line in f.read().splitlines() if line.strip()]
                    if len(lines) > 1: # Ignore if just the header
                        last_line = lines[-1].split(',')
                        self.dreamer.total_num_steps = int(float(last_line[0]))
                        self.dreamer.total_num_updates = int(float(last_line[1]))
                        print(f"[*] Recovered CSV Progress: {self.dreamer.total_num_steps} Env Steps | {self.dreamer.total_num_updates} Gradient Steps")
            except Exception as e:
                print(f"[!] Warning: Could not recover steps from CSV: {e}")

        # --- Initial Buffer Fill ---
        buffer_has_transitions = self.dreamer.buffer.full or (self.dreamer.buffer.index > 0)
        if not buffer_has_transitions:
            print("\n" + "="*50)
            print("[*] Replay buffer is empty. Gathering initial data from environment...")
            initial_score, initial_curiosity = self.dreamer.Play_the_game(number_of_episodes_per_env=1)
            print(f"[*] Initial Collection Score: {initial_score} | Curiosity: {initial_curiosity}")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("[*] Replay buffer already contains data. Skipping initial environment collection.")
            print("="*50 + "\n")

        # --- Main Training Loop ---
        print("[*] Starting Training Loop...")
        for episode in range(self.total_num_episodes):
            print(f"\n" + "-"*50)
            print(f"--- Training Episode {episode + 1} / {self.total_num_episodes} ---")
            
            best_dreams_of_episode = []
            random_dreams_of_episode = []
            wm_metrics = {}

            # --- Environment collection overlapped with training ---
            collect_result = {}

            def _collect():
                try:
                    collect_result["out"] = self.dreamer.Play_the_game(number_of_episodes_per_env=1)
                except Exception as e:
                    print(f"[!] Collector thread failed: {e!r}")
                    collect_result["out"] = (0.0, 0.0)

            collector = threading.Thread(target=_collect, daemon=True)
            print("[*] Stepping environment in the background (overlapped with training)...")
            collector.start()

            # Prefetch batches on a background thread (concurrent flushes are
            # handled by buffer.lock).
            prefetcher = BatchPrefetcher(
                self.dreamer.buffer,
                batch_size=self.dreamer.number_of_sequences,
                sequence_size=self.dreamer.steps_per_sequence,
            )
            try:
                for step in range(self.training_per_episodes):
                    if step % 50 == 0:
                        print(f"    [Training] Step {step} / {self.training_per_episodes}")
                    # Batch already sampled+pinned in the background; just move it to the GPU.
                    sample = self.dreamer._batch_to_device(prefetcher.get())

                    # Only the last (logged) step needs the sync-heavy .item() metric pulls.
                    compute_metrics = (step == self.training_per_episodes - 1)

                    # Update Networks
                    full_states, dream_priorities, kv_context, wm_metrics = self.dreamer.TrainWorldModel(
                        sample, compute_metrics=compute_metrics)
                    # Two dreams: max combined-advantage and a random one.
                    dream_metrics, best_dream, rand_dream = self.dreamer.Dream(
                        full_states, batch_data=sample, horizon=self.dream_horizon,
                        dream_priorities=dream_priorities,
                        compute_metrics=compute_metrics,
                        kv_context=kv_context,
                    )

                    best_dreams_of_episode.append(best_dream)
                    random_dreams_of_episode.append(rand_dream)
                    self.dreamer.total_num_updates += 1
            finally:
                prefetcher.close()

            if self.visualize_dreams:
                best_dreams_of_episode.sort(key=lambda x: x[0], reverse=True)
                vis_best = best_dreams_of_episode[:self.num_max_dreams]
                vis_rand = random.sample(
                    random_dreams_of_episode,
                    min(self.num_random_dreams, len(random_dreams_of_episode)),
                )

                dreams_to_visualize = vis_best + vis_rand

                # Print a quick summary
                best_adv_str = ", ".join([f"{d[0]:+.2f}" for d in vis_best])
                rand_adv_str = ", ".join([f"{d[0]:+.2f}" for d in vis_rand])
                print(f"[*] Saving {len(vis_best)} MAX ADVANTAGE Dreams (Combined Advs: [{best_adv_str}])")
                print(f"[*] Saving {len(vis_rand)} RANDOM Dreams (Combined Advs: [{rand_adv_str}])")

                # --- Save this training phase's dreams to a single file in dreams/ ---
                # Keep at most 10 dream files; the oldest is replaced first.
                dream_path = self._save_dreams_to_file(dreams_to_visualize)
                print(f"[*] Saved {len(dreams_to_visualize)} dreams to {dream_path}")
            # --- Wait for the background episode started at the top of this phase ---
            print(f"[*] Waiting for the background episode to finish...")
            collector.join()
            avg_score, avg_curiosity = collect_result["out"]

            self.dreamer.buffer.print_diagnostics()
            
            # Print cleanly formatted metrics
            print(f"    > Total Env Steps : {self.dreamer.total_num_steps}")
            print(f"    > Gradient Steps  : {self.dreamer.total_num_updates}")
            print(f"    > Total Reward    : {avg_score:.2f}")
            print(f"    > Total Curiosity : {avg_curiosity:.4f}")
            print(f"    > Actor Entropy   : {dream_metrics.get('entropies', 0):.4f}")
            print(f"    > Mean dream curiosity: {dream_metrics.get('dream_mean_curiosity', 0):.4f}")

            # --- CSV Logging ---
            row_data =[
                self.dreamer.total_num_steps,                     # envSteps
                self.dreamer.total_num_updates,                   # gradientSteps
                avg_score,                                        # totalReward
                avg_curiosity,                                    # totalCuriosity (tiered tile curiosity, summed over episode)
                wm_metrics.get('world_model_loss', 0),            # worldModelLoss
                wm_metrics.get('bt_loss', 0),                     # barlowTwinsLoss (R2-Dreamer repr. loss)
                wm_metrics.get('reward_loss', 0),                 # rewardPredictorLoss
                wm_metrics.get('kl_loss', 0),                     # klLoss
                wm_metrics.get('teamitem_loss', 0),               # teamItemLoss
                wm_metrics.get('ltm_reward_loss', 0),             # ltmRewardLoss (whole-game LTM)
                wm_metrics.get('grid_loss', 0),                   # gridLoss (5x5 explored-grid recon)
                wm_metrics.get('var_loss', 0),                    # varLoss (latent value-alignment)
                wm_metrics.get('curiosity_loss', 0),              # curiosityLoss
                dream_metrics.get('actor_loss', 0),               # actorLoss
                dream_metrics.get('entropies', 0),                # entropies
                dream_metrics.get('critic_loss', 0),              # criticLoss
                dream_metrics.get('curiosity_critic_loss', 0),    # curiosityCriticLoss
                dream_metrics.get('advantages', 0),               # advantages
                dream_metrics.get('curiosity_advantages', 0),     # curiosityAdvantages
                dream_metrics.get('critic_values', 0),            # criticValues
                dream_metrics.get('curiosity_critic_values', 0),  # curiosityCriticValues
            ]
            
            # Round floats for a cleaner CSV
            row_data =[round(x, 4) if isinstance(x, float) else x for x in row_data]
            
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            # --- Checkpoint Saving ---
            if episode > 0 and episode % self.checkpoint_interval == 0:
                score_int = int(avg_score) if avg_score is not None else 0
                filename = f"pokemon_model_R{score_int}_G{self.dreamer.total_num_updates}.pt"
                path = os.path.join("checkpoints", filename)
                self.dreamer.saveCheckpoints(path)

                # Rotation: keep only the 15 most recent checkpoints (oldest out first).
                ckpts = sorted(
                    glob.glob(os.path.join("checkpoints", "pokemon_model_R*_G*.pt")),
                    key=os.path.getmtime,
                )
                while len(ckpts) > 15:
                    oldest = ckpts.pop(0)
                    try:
                        os.remove(oldest)
                    except OSError as e:
                        print(f"[!] Warning: could not remove old checkpoint {oldest}: {e}")

    def _save_dreams_to_file(self, dreams_to_visualize):
        """Save all dreams from one training phase into a single PDF inside the
        `dreams/` folder. Keeps at most 10 files, replacing the oldest first."""
        from matplotlib.backends.backend_pdf import PdfPages

        dreams_dir = "dreams"
        os.makedirs(dreams_dir, exist_ok=True)

        # Named by gradient-step count, e.g. dream_g5000.pdf.
        filename = f"dream_g{self.dreamer.total_num_updates}.pdf"
        file_path = os.path.join(dreams_dir, filename)

        with PdfPages(file_path) as pdf:
            for dream_data in dreams_to_visualize:
                (metric_val, b_states, b_rewards, b_values, b_actions,
                 b_adv_r, b_adv_c, b_adv_combined, label) = dream_data

                self.dreamer.visualize_single_dream(
                    b_states.to(self.device),
                    b_rewards,
                    b_values,
                    b_actions,
                    b_adv_r,
                    b_adv_c,
                    combined_advantages=b_adv_combined,
                    label=label,
                    max_advantage=metric_val,
                    pdf=pdf,
                )

        # --- Rotation: keep only the 10 most recent dream files ---
        existing = sorted(
            glob.glob(os.path.join(dreams_dir, "*.pdf")),
            key=os.path.getmtime,
        )
        while len(existing) > 10:
            oldest = existing.pop(0)
            try:
                os.remove(oldest)
            except OSError as e:
                print(f"[!] Warning: could not remove old dream file {oldest}: {e}")

        return file_path

    def evaluate(self, num_episodes=1):
        """ Evaluates BOTH agents simultaneously """
        self.dreamer.loadCheckpoints()
        print(f"Starting Parallel Evaluation for {num_episodes} episodes per agent...")
        
        num_envs = len(self.envs)
        episodes_completed = [0] * num_envs
        current_rewards =[0.0] * num_envs
        steps = [0] * num_envs
        recurrent_state = torch.zeros((num_envs, self.dreamer.recurrent_dim), device=self.device)
        context = self.dreamer.recurrentModel.make_cache(num_envs, self.dreamer.context_length, self.device)
        
        observations = []
        ltm_rewards = []
        grids = []
        team_levels = []
        item_counts = []
        for obs, info in self.envs.reset():  # all emulators reset in parallel
            observations.append(obs)
            ltm_rewards.append(np.array(info["ltm_reward"], dtype=np.float32))
            grids.append(np.array(info["grid"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))

        while min(episodes_completed) < num_episodes:
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ltm_reward_tensor = torch.from_numpy(np.array(ltm_rewards)).float().to(self.device)
            grid_tensor = torch.from_numpy(np.array(grids)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)

            with torch.no_grad():
                enc_img, enc_teamitem, enc_ltm_reward, enc_grid = self.dreamer._encode_components(
                    obs_tensor, ltm_reward_tensor, grid_tensor, team_tensor, item_tensor)
                encoded_obs = torch.cat([enc_img, enc_teamitem, enc_ltm_reward, enc_grid], dim=-1)

                latent_state, _ = self.dreamer.posteriorNet(encoded_obs)

                full_state = torch.cat((recurrent_state, latent_state), -1)
                action_onehot, _, _ = self.dreamer.actor(full_state)
                # Append the executed token u_t=(z_t, a_t); the transformer returns h_{t+1}.
                recurrent_state = self.dreamer.recurrentModel.forward_step(latent_state, action_onehot, context)
                action_idxs = torch.argmax(action_onehot, dim=-1).cpu().numpy()
                
            active = [episodes_completed[i] < num_episodes for i in range(num_envs)]
            results = self.envs.step(action_idxs, active=active)

            envs_to_reset = []
            for i in range(num_envs):
                if not active[i]:
                    continue
                obs, reward, terminated, truncated, info = results[i]
                done = terminated or truncated

                observations[i] = obs
                ltm_rewards[i] = np.array(info["ltm_reward"], dtype=np.float32)
                grids[i] = np.array(info["grid"], dtype=np.float32)
                team_levels[i] = np.array(info["team_levels"], dtype=np.float32)
                item_counts[i] = np.array(info["item_counts"], dtype=np.float32)
                current_rewards[i] += reward
                steps[i] += 1

                if steps[i] % 500 == 0:
                    print(f"[Agent {i+1}] Eval Step: {steps[i]}/{info['limit']} | Reward: {current_rewards[i]:.3f}")

                if done:
                    print(f"--- [Agent {i+1}] Episode Finished! Reward: {current_rewards[i]:.3f} | Steps: {steps[i]} ---")
                    episodes_completed[i] += 1

                    if episodes_completed[i] < num_episodes:
                        envs_to_reset.append(i)
                        current_rewards[i] = 0.0
                        steps[i] = 0
                        recurrent_state[i] = torch.zeros(self.dreamer.recurrent_dim, device=self.device)
                        context.reset_rows(i)  # clear the transformer's token history

            for i in envs_to_reset:
                obs, info = self.envs.reset_one(i)
                observations[i] = obs
                ltm_rewards[i] = np.array(info["ltm_reward"], dtype=np.float32)
                grids[i] = np.array(info["grid"], dtype=np.float32)
                team_levels[i] = np.array(info["team_levels"], dtype=np.float32)
                item_counts[i] = np.array(info["item_counts"], dtype=np.float32)

    def seedMeDaddy(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True   # your input shapes are fixed (64x64), so autotuning pays off