from evalsize.cli import main


def test_compare_command(capsys):
    assert main(["compare", "--baseline", "0.7", "--improvement", "0.05", "--disagreement", "0.15"]) == 0
    assert "469 questions" in capsys.readouterr().out


def test_bad_input_gives_a_clear_error(capsys):
    assert main(["compare", "--baseline", "0.7", "--improvement", "0.05", "--disagreement", "0.01"]) == 2
    assert "too low" in capsys.readouterr().err


def test_every_command_runs(capsys):
    commands = [
        ["margin", "--margin", "0.03"],
        ["detect", "--baseline", "0.7", "--n", "300"],
        ["diff", "--n", "300", "--only-a", "12", "--only-b", "28"],
        ["reruns", "--n", "200", "--share", "0.6"],
        ["judge"],
        ["judge", "--judge-rate", "0.82", "--sensitivity", "0.95", "--specificity", "0.7"],
    ]
    for command in commands:
        assert main(command) == 0
