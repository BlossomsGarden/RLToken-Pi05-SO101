import json
import sys

from scripts.summarize_online_metrics import main
from windows.deploy_rlt import save_episode


def test_summary_does_not_count_assisted_success_as_autonomous(tmp_path, monkeypatch, capsys):
    def record(*sources):
        steps = [{"state": [0.] * 6, "action": [0.] * 6, "leader": [0.] * 6,
                  "source": source, "critical": True, "critical_prob": .8,
                  "chunk_id": 5, "step_ms": 30.} for source in sources]
        return {"reward": 1, "steps": steps, "boundaries": [{"step": len(steps), "req_id": 5}]}
    save_episode(tmp_path, 1, record("ACTOR"))
    save_episode(tmp_path, 2, record("VLA", "HUMAN"))
    monkeypatch.setattr(sys, "argv", ["summarize_online_metrics", "--episodes-dir", str(tmp_path)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["autonomous_successes"] == 1
    assert result["assisted_successes"] == 1
