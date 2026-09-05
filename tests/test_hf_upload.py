"""HF Hub export: token detection, private-repo creation, graceful fallback.

``huggingface_hub`` is never required for these tests — a fake module stands
in for it, which also exercises the "hub not installed" degradation path.
"""

from __future__ import annotations

import json
import sys
import types

import importlib

import pytest

from cli_hsr.registry import ModelMetadata

# NB: `from cli_hsr.rl import train` would resolve to the re-exported train()
# function, not the submodule — import the module object explicitly.
rl_train = importlib.import_module("cli_hsr.rl.train")


def _make_bundle(tmp_path, *, with_ckpt=True):
    """Minimal registry bundle: model.zip + metadata.json (+ 1 SB3 ckpt)."""
    bundle = tmp_path / "ppo_molab"
    bundle.mkdir()
    (bundle / "model.zip").write_bytes(b"fake-zip")
    (bundle / "metadata.json").write_text(json.dumps({
        "name": "ppo_molab", "algo": "ppo", "team": ["seele"],
        "trained_timesteps": 1000, "eval_stats": {"win_rate": 0.5},
        "extra": {"device": "cpu", "n_envs": 1, "hardware": "cpu",
                  "elapsed_s": 12.3},
    }), encoding="utf-8")
    if with_ckpt:
        (bundle / "ppo_1000_steps.zip").write_bytes(b"ckpt")
    return bundle


class _FakeHfApi:
    """Records create_repo / upload_folder calls; can be told to explode."""

    def __init__(self, token=None, fail_on_upload=False):
        self.token = token
        self.created = []
        self.uploads = []
        self.fail_on_upload = fail_on_upload

    def create_repo(self, repo_id, private=False, exist_ok=False):
        self.created.append((repo_id, private, exist_ok))

    def upload_folder(self, *, repo_id, folder_path, path_in_repo="",
                      commit_message=None, **kwargs):
        if self.fail_on_upload:
            raise ConnectionError("no route to huggingface.co")
        self.uploads.append({"path_in_repo": path_in_repo, "kwargs": kwargs,
                             "commit_message": commit_message})


def _install_fake_hub(monkeypatch, *, token="tok-test", fail_on_upload=False):
    class Api(_FakeHfApi):
        def __init__(self, token=None):
            super().__init__(token, fail_on_upload=fail_on_upload)
            Api.instance = self

    Api.instance = None
    fake = types.SimpleNamespace(HfApi=Api, get_token=lambda: token)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(rl_train, "hf_token", lambda: token)
    return Api


# --------------------------------------------------------------------------- #
# token resolution                                                             #
# --------------------------------------------------------------------------- #
def test_hf_token_returns_none_when_hub_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError
    assert rl_train.hf_token() is None


def test_hf_token_from_module(monkeypatch):
    fake = types.SimpleNamespace(get_token=lambda: "tok-env")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert rl_train.hf_token() == "tok-env"


def test_hf_token_swallows_errors(monkeypatch):
    def _boom():
        raise RuntimeError("broken cache")

    fake = types.SimpleNamespace(get_token=_boom)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert rl_train.hf_token() is None


# --------------------------------------------------------------------------- #
# early exits                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("repo_id", [None, "", "   "])
def test_upload_skipped_without_repo_id(tmp_path, repo_id):
    res = rl_train.hf_upload_bundle(tmp_path, repo_id)
    assert res["status"] == "skipped"
    assert res["reason"] == "no-repo-id"


def test_upload_skipped_when_bundle_missing(tmp_path):
    res = rl_train.hf_upload_bundle(tmp_path / "nope", "user/model")
    assert res["status"] == "skipped"
    assert res["reason"].startswith("bundle not found")


def test_upload_skipped_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_train, "hf_token", lambda: None)
    res = rl_train.hf_upload_bundle(_make_bundle(tmp_path), "user/model")
    assert res["status"] == "skipped"
    assert res["reason"].startswith("no-token")


def test_upload_skipped_when_hub_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError
    res = rl_train.hf_upload_bundle(_make_bundle(tmp_path), "user/model",
                                    token="tok-explicit")
    assert res["status"] == "skipped"
    assert "not installed" in res["reason"]


# --------------------------------------------------------------------------- #
# happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_upload_creates_private_repo_and_uploads(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path)
    api_cls = _install_fake_hub(monkeypatch)

    res = rl_train.hf_upload_bundle(bundle, "user/cli-hsr-ppo",
                                    card="# card for model", verbose=True)

    assert res["status"] == "ok"
    assert res["url"] == "https://huggingface.co/user/cli-hsr-ppo"
    # repo created exactly once, private, idempotent
    assert api_cls.instance.created == [("user/cli-hsr-ppo", True, True)]
    # bundle uploaded at repo root; README.md card written next to the bundle
    assert len(api_cls.instance.uploads) == 1
    assert api_cls.instance.uploads[0]["path_in_repo"] == ""
    assert "ignore_patterns" in api_cls.instance.uploads[0]["kwargs"]
    assert (bundle / "README.md").read_text(encoding="utf-8") == "# card for model"
    # model.zip + metadata.json + README.md; the *_steps.zip ckpt is excluded
    assert res["files"] == 3


def test_upload_with_checkpoints(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path)
    api_cls = _install_fake_hub(monkeypatch)

    res = rl_train.hf_upload_bundle(bundle, "user/cli-hsr-ppo",
                                    include_checkpoints=True,
                                    checkpoint_dir=bundle)

    assert res["status"] == "ok"
    uploads = api_cls.instance.uploads
    assert [u["path_in_repo"] for u in uploads] == ["", "checkpoints"]
    # same-dir checkpoint upload only ships the *_steps.zip files (no re-upload
    # of model.zip / metadata.json / README.md)
    assert uploads[1]["kwargs"].get("allow_patterns") == ["*_steps.zip"]
    assert "ignore_patterns" not in uploads[0]["kwargs"]
    assert res["files"] == 4  # 3 bundle files + 1 checkpoint


def test_upload_separate_checkpoint_dir(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, with_ckpt=False)
    ckpts = bundle.parent / "separate_ckpts"
    ckpts.mkdir()
    (ckpts / "ppo_1000_steps.zip").write_bytes(b"ckpt")
    _install_fake_hub(monkeypatch)

    res = rl_train.hf_upload_bundle(bundle, "user/cli-hsr-ppo",
                                    include_checkpoints=True,
                                    checkpoint_dir=ckpts)

    assert res["status"] == "ok"
    assert res["files"] == 2 + 1  # model.zip + metadata.json + 1 checkpoint


# --------------------------------------------------------------------------- #
# failure handling                                                             #
# --------------------------------------------------------------------------- #
def test_upload_network_error_is_reported_not_raised(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path)
    _install_fake_hub(monkeypatch, fail_on_upload=True)

    res = rl_train.hf_upload_bundle(bundle, "user/cli-hsr-ppo", verbose=True)

    assert res["status"] == "error"
    assert "no route to huggingface.co" in res["reason"]
    assert res["url"] is None


# --------------------------------------------------------------------------- #
# model card                                                                   #
# --------------------------------------------------------------------------- #
def test_build_model_card_renders_repo_and_stats():
    meta = ModelMetadata(
        name="ppo_molab", algo="ppo", team=["seele", "bronya"],
        trained_timesteps=3_000_000,
        eval_stats={"win_rate": 0.75},
        extra={"device": "cuda"},
    )
    card = rl_train.build_model_card(meta, "user/cli-hsr-ppo", device="cuda",
                                     hardware="L4", n_envs=2, elapsed_s=60.0)
    assert isinstance(card, str)
    assert card.startswith("---\n")
    assert "library_name: stable-baselines3" in card
    assert "user/cli-hsr-ppo" in card
    assert "| win_rate | 0.75 |" in card
    assert "seele, bronya" in card
    assert "1.0 min" in card  # 60 s -> 1.0 min


def test_build_model_card_frontmatter_parses():
    yaml = pytest.importorskip("yaml")
    meta = ModelMetadata(name="m", algo="ppo", team=["seele"])
    card = rl_train.build_model_card(meta, "user/repo")
    front_matter = card.split("---", 2)[1]
    parsed = yaml.safe_load(front_matter)
    assert parsed["library_name"] == "stable-baselines3"
    assert "maskable-ppo" in parsed["tags"]
