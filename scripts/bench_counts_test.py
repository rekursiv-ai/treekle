from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, cast

import subprocess
import sys

import pytest

from scripts import bench_configs, bench_counts


if TYPE_CHECKING:
    from collections.abc import Sequence


def test_perf_output_gives_its_instruction_count() -> None:
    text = "# started on Tue\n\n522901234,,instructions:u,1000,100.00,,\n"

    assert bench_counts._read_perf(text) == 522_901_234


def test_a_hybrid_cpu_s_count_is_the_core_type_that_counted() -> None:
    text = (
        "<not counted>,,cpu_atom/instructions/u,0,0.00,,\n"
        "94500501,,cpu_core/instructions/u,19269312,100.00,,\n"
    )

    assert bench_counts._read_perf(text) == 94_500_501


def test_perf_output_that_counted_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no instruction count"):
        bench_counts._read_perf("<not counted>,,instructions:u,0,0.00,,\n")


def test_a_case_run_gives_its_calls_and_mismatches() -> None:
    assert bench_counts._read_child("noise\ncalls=265 mismatched=2\n") == (265, 2)


def test_a_case_run_that_reports_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no result"):
        bench_counts._read_child("Traceback ...\n")


def test_the_table_shows_each_count_the_ratio_and_the_mismatches() -> None:
    row = bench_counts._Row(
        case="decode cold",
        counts=[
            bench_counts._Count(instructions=500_000_000, calls=265, mismatched=0),
            bench_counts._Count(instructions=450_000_000, calls=265, mismatched=1),
        ],
    )

    table = bench_counts._render([row], refs=["aaaa", "bbbb"])

    assert table.splitlines()[1].split() == [
        "decode",
        "cold",
        "265",
        "500.0M",
        "450.0M",
        "0.900",
        "0/1",
    ]


@pytest.mark.parametrize(
    ("make", "target"),
    [
        (bench_counts._Slotted, list[bench_counts._Slotted]),
        (bench_counts._DictBacked, list[bench_counts._DictBacked]),
        (bench_counts._Derived, list[bench_counts._Derived]),
        (bench_counts._Plain, list[bench_counts._Plain]),
    ],
)
def test_rows_read_back_as_python_reads_them(make: type, target: object) -> None:
    rows = bench_counts._rows(make, count=3)

    for work in (
        bench_counts._decode(rows, rounds=2, warm=True, target=target),
        bench_counts._loads(rows, rounds=2, warm=False, target=target),
        bench_counts._encode(rows, rounds=2, warm=False),
        bench_counts._dumps(rows, rounds=2, warm=True),
    ):
        assert bench_counts._measure(work) == (2, 0)


@pytest.mark.parametrize("text", [False, True], ids=["from_plain", "json_loads"])
def test_defaulted_rows_read_back_as_python_reads_them(text: bool) -> None:
    trees = bench_counts._defaulted_rows(count=3)
    target = list[bench_counts._Defaulted]

    work = bench_counts._read(trees, target=target, rounds=2, text=text)
    rows = cast(list[bench_counts._Defaulted], work.calls[0]())

    assert bench_counts._measure(work) == (2, 0)
    assert rows[1] == bench_counts._Defaulted(a=1)


def test_a_result_python_does_not_give_is_counted_as_a_mismatch() -> None:
    work = bench_counts._Work(
        calls=[lambda: [1, 2]],
        references=[lambda: [1, 3]],
        rounds=3,
        warm=False,
    )

    assert bench_counts._measure(work) == (3, 3)


def test_perf_is_told_when_to_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, ack = tmp_path / "ctl", tmp_path / "ack"
    ack.write_bytes(b"ack\n\x00")
    monkeypatch.setenv("PERF_CTL", str(control))
    monkeypatch.setenv("PERF_ACK", str(ack))

    bench_counts._perf("enable")

    assert control.read_text() == "enable\n"


def test_counting_waits_for_perf_to_take_the_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``perf`` reads the command when it gets to it; the calls counted must
    # not start, nor the checks after them run, until it has.
    control, ack = tmp_path / "ctl", tmp_path / "ack"
    ack.write_bytes(b"")
    monkeypatch.setenv("PERF_CTL", str(control))
    monkeypatch.setenv("PERF_ACK", str(ack))

    with pytest.raises(RuntimeError, match="perf did not take 'disable'"):
        bench_counts._perf("disable")


def test_configs_are_mirrored_once_per_round(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def configs(package: str) -> tuple[list[tuple[str, object]], list[str]]:
        asked.append(package)
        return [("a", 1), ("b", 2)], []

    def fresh(config: object) -> object:
        return ("fresh", config)

    monkeypatch.setattr(bench_configs, "_configs", configs)
    monkeypatch.setattr(bench_configs, "_fresh", fresh)

    values = bench_counts._configs(2)

    assert values == [("fresh", 1), ("fresh", 2)] * 2
    assert asked == [f"{bench_counts._PREFIX}priml.baselines"]


def test_git_answers_from_the_repository_it_is_run_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    repo = tmp_path / "repo"
    repo.mkdir()
    bench_counts._git(repo, "init", "--quiet")

    root = bench_counts._git(repo, "rev-parse", "--show-toplevel")

    assert Path(root) == repo.resolve()


def test_a_ref_is_checked_out_once_and_built_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    package = Path("pkg")
    monkeypatch.setattr(bench_counts, "_CWD", repo / package / "scripts")
    monkeypatch.setattr(bench_counts, "_RUNS", tmp_path / "runs")
    gits: list[tuple[str, ...]] = []
    builds: list[Sequence[str]] = []

    def git(where: Path, *args: str) -> str:
        del where
        gits.append(args)
        if args[0] == "worktree":
            (Path(args[3]) / package).mkdir(parents=True)
        return ""

    def run(argv: Sequence[str], **_: object) -> CompletedProcess[str]:
        builds.append(argv)
        return CompletedProcess(argv, 0)

    monkeypatch.setattr(bench_counts, "_git", git)
    monkeypatch.setattr(subprocess, "run", run)

    roots = [
        bench_counts._prepared(repo, "a" * 40, "b" * 40, pgo=True) for _ in range(2)
    ]
    root = roots[-1]

    assert roots[0] == root
    assert root == tmp_path / "runs" / f"{'a' * 12}-on-{'b' * 12}"
    assert [args[0] for args in gits] == ["worktree", "checkout", "checkout"]
    assert gits[1] == ("checkout", "a" * 40, "--", str(package))
    assert [argv[-1] for argv in builds] == [str(root / package / "build_rust.py")]
    assert (root / ".bench-counts-built").exists()


def test_a_case_is_counted_in_its_own_process_under_perf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Sequence[str]] = []

    def run(argv: Sequence[str], **_: object) -> CompletedProcess[str]:
        seen.append(argv)
        out = Path(argv[list(argv).index("-o") + 1])
        out.write_text("123456,,instructions:u,1,100.00,,\n")
        return CompletedProcess(argv, 0, stdout="calls=4 mismatched=0\n")

    monkeypatch.setattr(subprocess, "run", run)

    count = bench_counts._count(tmp_path, "decode warm", "3")

    assert count == bench_counts._Count(instructions=123_456, calls=4, mismatched=0)
    assert seen[0][:3] == ["taskset", "-c", "3"]
    assert seen[0][-2:] == ["--case", "decode warm"]


def test_main_counts_every_case_at_both_refs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots = {"base": Path("/base"), "head": Path("/head")}
    monkeypatch.setattr(sys, "argv", ["bench_counts", "--cpu", "1"])

    def git(where: Path, *args: str) -> str:
        del where
        return args[-1]

    def prepared(repo: Path, ref: str, head: str, *, pgo: bool) -> Path:
        del repo, head, pgo
        return roots["base" if ref == "c4afb1c292" else "head"]

    monkeypatch.setattr(bench_counts, "_git", git)
    monkeypatch.setattr(bench_counts, "_prepared", prepared)

    def count(root: Path, case: str, cpu: str) -> bench_counts._Count:
        del case, cpu
        mismatched = int(root == roots["head"])
        return bench_counts._Count(instructions=100, calls=1, mismatched=mismatched)

    monkeypatch.setattr(bench_counts, "_count", count)

    assert bench_counts.main() == 1
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "PGO builds; user instructions on CPU 1; one process per case."
    assert len(printed) == 2 + len(bench_counts._CASES)


def test_the_ratio_is_the_last_ref_over_the_one_before_it() -> None:
    row = bench_counts._Row(
        case="encode cold",
        counts=[
            bench_counts._Count(instructions=90_000_000, calls=265, mismatched=0),
            bench_counts._Count(instructions=100_000_000, calls=265, mismatched=0),
            bench_counts._Count(instructions=110_000_000, calls=265, mismatched=0),
        ],
    )

    table = bench_counts._render([row], refs=["a", "b", "c"])

    assert table.splitlines()[1].split()[-2:] == ["1.100", "0/0/0"]


def _sha(ref: str) -> str:
    """Return a full-length stand-in commit id for ``ref``."""
    return (ref * 40)[:40]


def _fake_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> tuple[
    list[tuple[str, str, bool]],
    list[tuple[Path, bool]],
    list[tuple[Path, str]],
]:
    """Run ``main`` on ``argv`` with git, builds and counts faked; return what each saw."""
    prepared: list[tuple[str, str, bool]] = []
    built: list[tuple[Path, bool]] = []
    counted: list[tuple[Path, str]] = []
    monkeypatch.setattr(sys, "argv", ["bench_counts", "--cpu", "5", *argv])

    def git(where: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            assert where == bench_counts._CWD
            return "/repo"
        assert where == Path("/repo")
        assert args[0] == "rev-parse"
        assert len(args) == 2
        return _sha(args[1])

    def prepare(repo: Path, ref: str, head: str, *, pgo: bool) -> Path:
        assert repo == Path("/repo")
        prepared.append((ref, head, pgo))
        return Path(f"/runs/{ref[:3]}")

    def build(root: Path, package: Path, *, pgo: bool) -> None:
        assert package == Path("pkg")
        built.append((root, pgo))

    def count(root: Path, case: str, cpu: str) -> bench_counts._Count:
        assert cpu == "5"
        counted.append((root, case))
        return bench_counts._Count(instructions=100, calls=1, mismatched=0)

    monkeypatch.setattr(bench_counts, "_CWD", Path("/repo/pkg/scripts"))
    monkeypatch.setattr(bench_counts, "_git", git)
    monkeypatch.setattr(bench_counts, "_prepared", prepare)
    monkeypatch.setattr(bench_counts, "_build", build)
    monkeypatch.setattr(bench_counts, "_count", count)
    assert bench_counts.main() == 0
    return prepared, built, counted


def test_main_counts_each_ref_given_between_the_base_and_the_head(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared, _, counted = _fake_main(
        monkeypatch,
        ["--base", "a", "--ref", "b", "--head", "c", "--only", "encode cold"],
    )

    assert prepared == [(_sha(ref), _sha("c"), True) for ref in "abc"]
    assert counted == [(Path(f"/runs/{ref * 3}"), "encode cold") for ref in "abc"]
    header = capsys.readouterr().out.splitlines()[1].split()
    assert header[2:5] == [ref * 10 for ref in "abc"]


def test_a_worktree_head_counts_the_checkout_as_it_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, built, counted = _fake_main(
        monkeypatch,
        ["--base", "a", "--head", "WORKTREE", "--only", "decode warm"],
    )

    # The base reads the workload of the commit the checkout stands on.
    assert prepared == [(_sha("a"), _sha("HEAD"), True)]
    assert built == [(Path("/repo"), True)]
    assert counted[-1] == (Path("/repo"), "decode warm")


def test_no_pgo_builds_every_ref_without_a_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared, built, _ = _fake_main(
        monkeypatch,
        ["--head", "WORKTREE", "--no-pgo", "--only", "decode warm"],
    )

    assert [pgo for _, _, pgo in prepared] == [False]
    assert built == [(Path("/repo"), False)]
    assert capsys.readouterr().out.startswith("Builds without a profile;")


def test_a_ref_built_without_a_profile_is_kept_apart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    package = Path("pkg")
    monkeypatch.setattr(bench_counts, "_CWD", repo / package / "scripts")
    monkeypatch.setattr(bench_counts, "_RUNS", tmp_path / "runs")
    builds: list[Sequence[str]] = []

    def git(where: Path, *args: str) -> str:
        del where
        if args[0] == "worktree":
            (Path(args[3]) / package).mkdir(parents=True)
        return ""

    def run(argv: Sequence[str], **_: object) -> CompletedProcess[str]:
        builds.append(argv)
        return CompletedProcess(argv, 0)

    monkeypatch.setattr(bench_counts, "_git", git)
    monkeypatch.setattr(subprocess, "run", run)

    root = bench_counts._prepared(repo, "a" * 40, "b" * 40, pgo=False)

    assert root == tmp_path / "runs" / f"{'a' * 12}-on-{'b' * 12}-no-pgo"
    assert builds[0][-1] == "--no-pgo"


def test_main_runs_one_case_in_this_process(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work = bench_counts._Work(
        calls=[lambda: 1],
        references=[lambda: 1],
        rounds=2,
        warm=True,
    )
    monkeypatch.setitem(bench_counts._CASES, "decode warm", lambda: work)
    monkeypatch.setattr(sys, "argv", ["bench_counts", "--case", "decode warm"])

    assert bench_counts.main() == 0
    assert capsys.readouterr().out == "calls=2 mismatched=0\n"


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
