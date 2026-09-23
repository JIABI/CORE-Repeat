import numpy as np
import pytest


def test_prepared_query_wrapper_forwards_saved_inputs(tmp_path, monkeypatch):
    from scripts import score_prepared_queries as wrapper
    from opal2 import r4_final_model
    queries = tmp_path / "queries.npz"
    np.savez(queries, ids=np.array(["q"]), groups=np.array(["g"]),
             X=np.ones((1, 9)), chem=np.ones((1, 513)), chem_mask=np.array([True]))
    received = []
    monkeypatch.setattr(r4_final_model, "load", lambda model: "fitted")
    monkeypatch.setattr(r4_final_model, "score", lambda *args, **kw: received.append((args, kw)))
    wrapper.main(["--model", str(tmp_path / "model"), "--query", str(queries),
                  "--output", str(tmp_path / "output"), "--population-size", "10"])
    assert received[0][0][0] == "fitted"
    assert received[0][0][1]["ids"].tolist() == ["q"]
    assert received[0][1] == {"population_size": 10}


def test_prepared_query_wrapper_does_not_overwrite(tmp_path):
    from scripts import score_prepared_queries as wrapper
    with pytest.raises(FileExistsError):
        wrapper.main(["--model", "unused", "--query", "unused",
                      "--output", str(tmp_path), "--population-size", "10"])
