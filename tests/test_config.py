"""Guards for artifact paths and the CLI's shared argument handling.

A wrong path here does not raise: it writes artifacts somewhere unexpected, or reads a
stale tree and produces plausible numbers from the wrong run. So the cases pinned are
the ones a naive reader silently gets wrong.
"""

from __future__ import annotations

import json

import pytest


def test_every_path_is_repo_relative_with_nothing_to_configure():
    """There is no .env: a fresh checkout must resolve every location on its own."""
    from innerj import config

    for name in ("DATA_ROOT", "LOG_DIR", "FIGURE_DIR", "PAPER_TEX"):
        value = getattr(config, name)
        assert value.is_absolute(), name
        assert config.REPO_ROOT in value.parents, f"{name} escapes the repo"


# --- shared CLI flags ---------------------------------------------------------


def test_parser_rejects_an_unknown_flag_name():
    """A typo used to add no flag at all and run on undeclared defaults."""
    from innerj.cli import common

    with pytest.raises(KeyError, match="unknown shared flag"):
        common.parser("x", needs=("seedz",))


def test_parser_rejects_a_bare_string():
    """`needs=("device")` is a string, and `"model" in "device"` is merely False.

    A missing comma silently dropped every shared flag; one CLI shipped that way.
    """
    from innerj.cli import common

    with pytest.raises(TypeError, match="must be a tuple"):
        common.parser("x", needs="device")


def test_a_cli_default_tag_survives_the_shared_flag():
    """--tag is shared but never required: each CLI names its own artifacts."""
    from innerj.cli import common

    parser = common.parser("x", needs=("tag",))
    parser.set_defaults(tag="ablate")
    assert parser.parse_args([]).tag == "ablate"


# --- instance selection -------------------------------------------------------


def _records(tmp_path, instances, arms):
    """A records file with ``instances`` present in every arm of ``arms``."""
    from innerj.tasks.base import Record, write_jsonl

    rows = [
        Record(
            id=f"{i}_{arm}", family="language", condition=arm,
            semantic_instance_id=i, template_id="t", context="x" * 20,
            instruction="Answer.", latent_name="language", latent_value="Spanish",
            latent_token_id=1, control_token_ids=[2, 3], gold_answer="7",
            candidate_answers=["7", "3"], candidate_token_ids=[4, 5],
        )
        for i in instances
        for arm in arms
    ]
    path = tmp_path / "records.jsonl"
    write_jsonl(rows, path)
    return path


def _args(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


def test_an_arm_absent_from_the_records_is_fatal(tmp_path):
    """Trap 11: an empty arm reports a clean null that looks like a finding."""
    from innerj.cli import common
    from innerj.tasks.base import Condition

    path = _records(tmp_path, ["a", "b"], [Condition.FLEXIBLE])
    with pytest.raises(SystemExit, match="no semantic instance covers"):
        common.instances(_args(records=str(path), pairs=None),
                         (Condition.FLEXIBLE, Condition.AUTOMATIC))


def test_the_instance_cap_is_deterministic_not_sampled(tmp_path):
    """Two CLIs run at the same cap must see the same instances.

    Selection is by sorted id. Sampling instead would make a screen and the
    counterfactual that follows it disagree about which instances they covered.
    """
    from innerj.cli import common
    from innerj.tasks.base import Condition

    arms = (Condition.FLEXIBLE, Condition.AUTOMATIC)
    path = _records(tmp_path, [f"i{n:02d}" for n in range(10)], arms)
    kept = common.instances(_args(records=str(path), pairs=3), arms)
    assert list(kept) == ["i00", "i01", "i02"]


def test_no_cap_keeps_everything(tmp_path):
    """`--limit 0` and an absent `--pairs` both mean all of them, not none."""
    from innerj.cli import common
    from innerj.tasks.base import Condition

    arms = (Condition.FLEXIBLE,)
    path = _records(tmp_path, [f"i{n}" for n in range(5)], arms)
    assert len(common.instances(_args(records=str(path), pairs=None), arms)) == 5
    assert len(common.instances(_args(records=str(path)), arms, limit=0)) == 5


def test_only_the_requested_arms_come_back(tmp_path):
    """`arms` means these arms, not at least these.

    The datasets carry all four conditions, so returning whatever an instance
    happens to have added a fourth row to a three-arm ablation --- a full extra
    condition in the results table, with no error anywhere.
    """
    from innerj.cli import common
    from innerj.tasks.base import Condition

    path = _records(tmp_path, ["a", "b"], tuple(Condition))
    asked = (Condition.FLEXIBLE, Condition.REPORT, Condition.CONTROL)
    kept = common.instances(_args(records=str(path), pairs=None), asked)
    assert all(set(group) == set(asked) for group in kept.values())


# --- artifacts ----------------------------------------------------------------


def test_the_patch_experiments_keep_the_filenames_they_already_wrote():
    """`figures/build.py` opens four artifacts by exact name.

    The patching CLIs have always written `tag_model`, without the dataset stem
    the readout CLIs use. A rename would leave the figure build reading the
    previous run's numbers, which is silent.
    """
    from innerj.cli import common

    args = _args(tag="D_necessity", model="Qwen/Qwen3.6-27B",
                 records="/data/language/Qwen3.6-27B_matched_n400_s0.jsonl")
    assert common.stem(args, dataset=False) == "D_necessity_Qwen3.6-27B"
    assert common.stem(args) == (
        "D_necessity_Qwen3.6-27B_Qwen3.6-27B_matched_n400_s0"
    )


def test_a_result_keeps_the_fields_its_to_dict_adds(tmp_path, monkeypatch):
    """Several results attach a verdict or a derived field in `to_dict`.

    Falling back to `asdict` would drop them, and the loss is invisible: the
    artifact still parses and still holds every number.
    """
    from dataclasses import dataclass

    from innerj.cli import common

    @dataclass
    class Result:
        point: float

        def to_dict(self):
            return {"point": self.point, "verdict": "NOT J-MEDIATED"}

    monkeypatch.setattr(common, "DATA_ROOT", tmp_path)
    path = common.save(
        "probe", _args(tag="t", model="m", records=None),
        observations=[{"instance": "a"}], results=[Result(0.5)],
    )
    written = json.loads(path.read_text())
    assert written["results"] == [{"point": 0.5, "verdict": "NOT J-MEDIATED"}]
    assert (tmp_path / "probe" / "t_m_observations.jsonl").exists()


def test_an_estimate_lands_as_numbers_not_as_prose(tmp_path, monkeypatch):
    """`str(Estimate)` is "+0.0891 [+0.0799, +0.0983]".

    Stringified, `innerj audit` cannot see those as numbers and no re-analysis can
    parse them, so nested dataclasses must serialise as objects.
    """
    from innerj.analysis.stats import Estimate
    from innerj.cli import common

    monkeypatch.setattr(common, "DATA_ROOT", tmp_path)
    path = common.write_lines(
        "screen", "s", [{"delta": Estimate(0.0891, 0.0799, 0.0983, 40)}]
    )
    assert json.loads(path.read_text())["delta"]["point"] == 0.0891


# --- console ------------------------------------------------------------------


def test_a_bracketed_path_does_not_crash_the_terminal_path(monkeypatch):
    """Rich parses markup only on a terminal, so this crash hid from every log.

    `wrote("/tmp/a [/tmp/b]")` raised MarkupError interactively and printed fine
    when piped, which is the worst shape for a bug: invisible in the artifact
    that gets reviewed.
    """
    from innerj import console

    monkeypatch.setattr(console, "interactive", lambda: True)
    console.wrote("/tmp/a [/tmp/b]")
    console.step("verdict: NOT J-MEDIATED [+0.11, +0.19]")
    console.warn("[CONFOUNDED]")


def test_log_output_is_plain_and_prefixed(capsys, monkeypatch):
    from innerj import console

    monkeypatch.setattr(console, "interactive", lambda: False)
    console.step("loading")
    console.detail("64 layers")
    console.warn("CONFOUNDED")
    console.wrote("/tmp/x.json")
    out = capsys.readouterr().out
    assert out == ":: loading\n   64 layers\n!! CONFOUNDED\n   wrote /tmp/x.json\n"
