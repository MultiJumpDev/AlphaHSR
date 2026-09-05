import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # ---- Cell: automated environment & dependency setup (idempotent) --------
    # Headlessly installs RL deps (gymnasium, stable-baselines3, sb3-contrib,
    # tensorboard, torch) plus huggingface_hub for the auto-export to the HF
    # Hub, and the cli-hsr package itself, with no user input.
    # Prefers `uv pip` (molab uses uv); falls back to plain pip.
    # NOTE: `os` is imported under an alias on purpose. On molab the
    # Hugging Face connector cell (placed upstream) already does `import os`,
    # and marimo forbids re-importing the same module in another cell.
    import importlib.util
    import os as _os
    import subprocess
    import sys

    def _has(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    def _run(cmd: list[str]) -> None:
        print("$", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write((res.stderr or "")[-2000:])
            raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")

    def _uv_pip(*args: str) -> bool:
        try:
            _run(["uv", "pip", "install", "--quiet", *args])
            return True
        except Exception:
            return False

    def _pip_install(*args: str) -> None:
        _run([sys.executable, "-m", "pip", "install", "--quiet", *args])

    def _ensure(*pkgs: str) -> None:
        missing = [p for p in pkgs if not _has(p)]
        if not missing:
            print(f"[setup] already installed: {', '.join(pkgs)}")
            return
        print(f"[setup] installing: {', '.join(missing)}")
        if not _uv_pip(*missing):  # uv first (molab), pip fallback (local)
            _pip_install(*missing)

    # Work from a checkout of the repo so `pip install -e .` and ./data
    # resolve (db.py locates data/ relative to the *source* file, so a
    # non-editable install would break). On molab, a synced workspace mirrors
    # ONLY this notebook file, so we clone the repository ourselves when no
    # pyproject.toml is around.
    _repo_url = "https://github.com/MultiJumpDev/AlphaHSR.git"
    _repo_tgz = "https://codeload.github.com/MultiJumpDev/AlphaHSR/tar.gz/refs/heads/main"

    def _find_repo_root() -> str:
        for _cand in (_os.getcwd(), _os.path.dirname(_os.getcwd())):
            if _os.path.exists(_os.path.join(_cand, "pyproject.toml")):
                return _cand
        return ""

    _root = _find_repo_root()
    if not _root:
        _dest = _os.path.join(_os.getcwd(), "AlphaHSR")
        if not _os.path.exists(_os.path.join(_dest, "pyproject.toml")):
            print(f"[setup] workspace has no checkout; cloning {_repo_url}")
            try:
                _run(["git", "clone", "--depth", "1", _repo_url, _dest])
            except Exception:
                print("[setup] git unavailable, falling back to tarball download")
                import io as _io
                import tarfile as _tf
                import urllib.request as _urlreq

                with _urlreq.urlopen(_repo_tgz) as _resp:
                    _buf = _io.BytesIO(_resp.read())
                _tmp = _dest + "_tmp"
                with _tf.open(fileobj=_buf, mode="r:gz") as _tar:
                    _tar.extractall(_tmp)
                _os.rename(_os.path.join(_tmp, "AlphaHSR-main"), _dest)
                import shutil as _shutil

                _shutil.rmtree(_tmp, ignore_errors=True)
        _root = _dest
    _os.chdir(_root)
    print(f"[setup] cwd -> {_os.getcwd()}")

    _ensure("gymnasium", "stable_baselines3", "sb3_contrib", "tensorboard",
            "huggingface_hub")
    if not _has("torch"):
        print("[setup] installing torch (platform default: CUDA on Linux, CPU elsewhere)...")
        if not _uv_pip("torch"):
            _pip_install("torch")

    if _has("cli_hsr"):
        print(f"[setup] cli_hsr importable at {importlib.util.find_spec('cli_hsr').origin}")
    else:
        print("[setup] installing cli_hsr (editable) from the workspace...")
        if not _uv_pip("-e", "."):
            _pip_install("-e", ".")

    print("[setup] done.")
    setup_ok = True
    return (setup_ok,)


@app.cell
def _(setup_ok):
    # ---- Cell: hardware detection & environment-scaled defaults -------------
    # cuda available (e.g. molab GPU)  -> full production run
    # cpu only (local machine)         -> short dry-run
    assert setup_ok

    import torch

    from cli_hsr import db

    try:
        db.ensure_seeded()  # builds data/game.db from data/*.json if needed
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "cli_hsr data files not found. Upload the repository (src/, data/, "
            "pyproject.toml) to the notebook workspace root so the package and "
            "game database resolve."
        ) from e

    cuda = bool(torch.cuda.is_available())
    device = "cuda" if cuda else "cpu"
    runtime_env = "molab (GPU)" if cuda else "local (CPU)"
    gpu_name = torch.cuda.get_device_properties(0).name if cuda else ""
    torch_version = torch.__version__

    if cuda:
        default_timesteps = 3_000_000
        default_ckpt_freq = 200_000
        default_n_envs = 2
    else:
        default_timesteps = 50_000
        default_ckpt_freq = 25_000
        default_n_envs = 1
    return (
        default_ckpt_freq,
        default_n_envs,
        default_timesteps,
        device,
        gpu_name,
        runtime_env,
        torch_version,
    )


@app.cell
def _(
    default_ckpt_freq,
    default_timesteps,
    device,
    gpu_name,
    mo,
    runtime_env,
    torch_version,
):
    mo.md(
        f"""
        ### 🖥️ Runtime detected

        | | |
        |---|---|
        | **Environment** | {runtime_env} |
        | **Training device** | `{device}`{(" — " + gpu_name) if gpu_name else ""} |
        | **torch** | {torch_version} |
        | **Default timesteps** | {default_timesteps:,} |
        | **Checkpoint every** | {default_ckpt_freq:,} steps |

        Adjust the controls below if you want a different run — defaults are
        already tuned for this machine.
        """
    )
    return


@app.cell
def _(default_ckpt_freq, default_timesteps, mo):
    # ---- Cell: run configuration UI (defaults scale with the environment) ---
    team_a = mo.ui.text(
        value="seele,march_7th,bronya,fu_xuan",
        label="Side A (your learning agent's team)",
        full_width=True,
    )
    team_b = mo.ui.text(
        value="kafka,black_swan,luocha,himeko",
        label="Side B (opponent team)",
        full_width=True,
    )
    algo_choice = mo.ui.dropdown(options=["ppo", "dqn"], value="ppo", label="Algorithm")
    device_choice = mo.ui.dropdown(options=["auto", "cuda", "cpu"], value="auto", label="Device")
    timesteps_ui = mo.ui.number(
        10_000, 10_000_000, step=10_000, value=default_timesteps, label="Total timesteps"
    )
    ckpt_ui = mo.ui.number(
        5_000, 1_000_000, step=5_000, value=default_ckpt_freq, label="Checkpoint every N steps"
    )
    hf_repo_ui = mo.ui.text(
        value="",
        placeholder="username/cli-hsr-ppo",
        label="🤗 Hugging Face repo (optional — created private, e.g. username/cli-hsr-ppo)",
        full_width=True,
    )
    hf_ckpts_ui = mo.ui.checkbox(
        value=False,
        label="Also upload intermediate checkpoints (*_steps.zip)",
    )
    run_btn = mo.ui.run_button(label="▶  Run training")

    mo.vstack(
        [
            mo.hstack([team_a, team_b], justify="start", widths="equal"),
            mo.hstack([algo_choice, device_choice], justify="start", widths="equal"),
            mo.hstack([timesteps_ui, ckpt_ui], justify="start", widths="equal"),
            mo.hstack([hf_repo_ui], justify="start"),
            mo.hstack([hf_ckpts_ui, run_btn], justify="start"),
        ]
    )
    return (
        algo_choice,
        ckpt_ui,
        device_choice,
        hf_ckpts_ui,
        hf_repo_ui,
        run_btn,
        team_a,
        team_b,
        timesteps_ui,
    )


@app.cell
def _(mo, team_a, team_b):
    # ---- Cell: environment sanity probe -------------------------------------
    # Builds HSREnv once, resets it, and verifies the masked action space and
    # Gymnasium API before any training is attempted.
    import numpy as np

    from cli_hsr.agents import random_action
    from cli_hsr.env import HSREnv, MAX_ACTIONS, OBS_DIM

    probe = HSREnv(
        team_a=[s.strip() for s in team_a.value.split(",") if s.strip()],
        team_b=[s.strip() for s in team_b.value.split(",") if s.strip()],
        opponent_policy=random_action,
        level=80, max_av=3000.0, max_rounds=80, seed=0,
    )
    probe_obs, _probe_info = probe.reset(seed=0)
    probe_legal = int(np.asarray(probe.action_mask).sum())
    probe.close()
    mo.md(
        f"""
        ### 🧪 Environment probe

        - observation dim **{OBS_DIM}** (float32, shape ok: `{probe_obs.shape == (OBS_DIM,)}`)
        - action space `Discrete({MAX_ACTIONS})` with **{probe_legal}** legal actions after `reset()`
        - `env.action_masks()` (SB3 convention) ready for MaskablePPO
        """
    )
    return


@app.cell
def _(
    algo_choice,
    ckpt_ui,
    default_n_envs,
    device_choice,
    mo,
    run_btn,
    team_a,
    team_b,
    timesteps_ui,
):
    # ---- Cell: MaskablePPO training (runs on click) --------------------------
    # sb3-contrib MaskablePPO + MaskableActorCriticPolicy on HSREnv with action
    # masks, CheckpointCallback into ./checkpoints/ppo_molab/, then a final
    # registry bundle (model.zip + metadata.json).
    if not run_btn.value:
        train_md = mo.md("### ▶️ Configure the run above, then click **Run training**.")
        model = bundle_path = eval_stats = None
    else:
        import time as _time

        from cli_hsr.registry import ModelMetadata, save_model_bundle
        from cli_hsr.rl.train import TrainConfig, evaluate_policy, gpu_info, train

        run_ids_a = [s.strip() for s in team_a.value.split(",") if s.strip()]
        run_ids_b = [s.strip() for s in team_b.value.split(",") if s.strip()]
        n_envs = default_n_envs

        cfg = TrainConfig(
            algo=algo_choice.value,
            total_timesteps=int(timesteps_ui.value),
            n_envs=n_envs,
            device=device_choice.value,  # "auto" -> cuda when available
            checkpoint_dir="checkpoints/ppo_molab",
            checkpoint_freq=int(ckpt_ui.value),
            seed=0,
            verbose=1,
        )
        hw = gpu_info()
        t0 = _time.time()
        model, _env = train(run_ids_a, run_ids_b, cfg)
        elapsed = _time.time() - t0

        eval_stats = evaluate_policy(
            model, run_ids_a, run_ids_b, n_games=16, opponent="greedy", seed=1234
        )
        bundle_path = save_model_bundle(
            model,
            ModelMetadata(
                name="ppo_molab",
                algo=cfg.algo,
                team=run_ids_a,
                level=80,
                max_av=3000.0,
                max_rounds=80,
                trained_timesteps=int(timesteps_ui.value),
                eval_stats={k: v for k, v in eval_stats.items()
                            if isinstance(v, (int, float))},
                extra={"device": cfg.device, "n_envs": n_envs,
                       "hardware": hw.get("name", "cpu"),
                       "elapsed_s": round(elapsed, 1)},
            ),
            "checkpoints/ppo_molab",
        )

        train_md = mo.md(
            f"""
            ### ✅ Training complete

            | | |
            |---|---|
            | **Algorithm** | {cfg.algo.upper()} (`MaskableActorCriticPolicy`) on `{cfg.device}` |
            | **Timesteps** | {int(timesteps_ui.value):,} across {n_envs} env(s) |
            | **Elapsed** | {elapsed / 60:.1f} min |
            | **Checkpoints** | `checkpoints/ppo_molab/` every {int(ckpt_ui.value):,} steps |
            | **Bundle** | `{bundle_path}` (`model.zip` + `metadata.json`) |
            """
        )
    train_md
    return bundle_path, eval_stats, model, train_md


@app.cell
def _(bundle_path, hf_ckpts_ui, hf_repo_ui, mo, model):
    # ---- Cell: Hugging Face Hub export (auto-upload, local fallback) --------
    # Publishes the registry bundle to a *private* HF repo whenever a repo id
    # is set. Auth is resolved non-interactively: `HF_TOKEN` (molab Remote
    # Storage injects it automatically) or a cached `huggingface-cli login`
    # token. Every failure mode (no repo id, no token, hub not installed,
    # network/auth error) degrades to a warning here — the `mo.download`
    # widget below always remains the local fallback.
    import json

    if model is None:
        hf_md = mo.md(
            "🤗 *Optional: enter a Hugging Face repo id above — the trained "
            "bundle will be uploaded there automatically (private repo).*"
        )
    else:
        from cli_hsr.registry import ModelMetadata
        from cli_hsr.rl.train import build_model_card, hf_token, hf_upload_bundle

        repo_id = (hf_repo_ui.value or "").strip()
        if not repo_id:
            hf_md = mo.md(
                "🤗 **HF upload skipped** — no repo id configured. Enter one "
                "above (e.g. `username/cli-hsr-ppo`) to auto-upload on the next "
                "run; the local download below always works."
            )
        elif hf_token() is None:
            hf_md = mo.md(
                "⚠️ **HF upload skipped** — no Hugging Face token found. On "
                "molab, mount Remote Storage (it injects `HF_TOKEN`) or run "
                "`huggingface-cli login`. Falling back to the local download "
                "below."
            )
        else:
            meta_raw = json.loads(
                (bundle_path / "metadata.json").read_text(encoding="utf-8"))
            meta = ModelMetadata(**{k: v for k, v in meta_raw.items()
                                    if k in ModelMetadata.__dataclass_fields__})
            card = build_model_card(
                meta, repo_id,
                device=str(meta.extra.get("device", "auto")),
                hardware=str(meta.extra.get("hardware", "")),
                n_envs=int(meta.extra.get("n_envs", 1) or 1),
                elapsed_s=meta.extra.get("elapsed_s"),
            )
            res = hf_upload_bundle(
                bundle_path, repo_id,
                include_checkpoints=bool(hf_ckpts_ui.value),
                checkpoint_dir=bundle_path,  # SB3 *_steps.zip live here too
                card=card, private=True, verbose=True,
            )
            if res["status"] == "ok":
                hf_md = mo.md(
                    f"✅ **Uploaded {res['files']} file(s)** to 🤗 "
                    f"[{repo_id}]({res['url']}) — private repo (created if missing)."
                )
            else:
                hf_md = mo.md(
                    f"⚠️ **HF upload failed** — {res['reason']}. "
                    "Falling back to the local download below."
                )
    hf_md
    return (hf_md,)


@app.cell
def _(bundle_path, eval_stats, hf_md, mo, model, train_md):
    # ---- Cell: evaluation summary + one-click bundle download ---------------
    # Appears as soon as training finishes; the download widget is
    # non-blocking and serves the registry-compatible bundle (.zip).
    if model is None:
        download = None
        output = mo.md("⏳ Train a model above to unlock evaluation results and the download.")
    else:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(bundle_path.rglob("*")):
                if p.is_file() and p.name in ("model.zip", "metadata.json"):
                    zf.write(p, p.name)

        eval_md = mo.md(
            f"""
            ### 📊 Evaluation (deterministic, vs Greedy, 16 games)

            Win **{eval_stats["win_rate"]:.0%}** · Draw {eval_stats["draw_rate"]:.0%} ·
            Loss {eval_stats["loss_rate"]:.0%}
            """
        )
        download = mo.download(
            data=buf.getvalue(),
            filename="ppo_molab_bundle.zip",
            label="⬇️  Download trained model bundle (.zip)",
        )
        output = mo.vstack([train_md, hf_md, eval_md, download])
    output
    return (download,)


@app.cell
def _(bundle_path, mo, model, team_a, team_b):
    # ---- Cell: sample battle — trained agent vs Greedy -----------------------
    if model is None:
        battle_md = mo.md("⏳ Train a model above to see a sample battle.")
    else:
        from cli_hsr.agents import GreedyAgent, run_battle_between
        from cli_hsr.registry import CheckpointAgent

        battle_ids_a = [s.strip() for s in team_a.value.split(",") if s.strip()]
        battle_ids_b = [s.strip() for s in team_b.value.split(",") if s.strip()]
        agent = CheckpointAgent(model, "ppo_molab")
        battle = run_battle_between(agent, GreedyAgent(1),
                                    battle_ids_a, battle_ids_b, seed=7)
        mo.md(
            f"""
            ### ⚔️ Sample battle (seed 7)

            `{bundle_path.name}` vs Greedy → winner **{battle.winner}**
            after {battle.turn_count} turns ({battle.time:.0f} AV).
            Fight it from the CLI too:
            `python main.py fight --team-a {",".join(battle_ids_a)} --team-b {",".join(battle_ids_b)} --agent-a model:{bundle_path}`
            """
        )
    battle_md
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### 🏆 Next step: tournament registration

        The bundle under `checkpoints/ppo_molab/` is registry-compatible, so the
        trained policy can join a tournament straight from a checkpoint:

        ```bash
        python main.py tournament --contestants 4 \\
            --model "checkpoint:checkpoints/ppo_molab" \\
            --model "FireflyTeam=firefly,fu_xuan,luocha,himeko"
        ```

        Or programmatically: `contestant_from_checkpoint("checkpoints/ppo_molab")`
        returns a tournament-ready `Contestant` (team + gear come from
        `metadata.json`).
        """
    )
    return


if __name__ == "__main__":
    app.run()
