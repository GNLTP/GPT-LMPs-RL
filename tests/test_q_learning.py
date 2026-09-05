from pathlib import Path

import numpy as np

from gpt_lmps.rl.q_learning import StateDiscretizer, TabularQAgent


def test_discretizer_and_q_table_round_trip(tmp_path: Path) -> None:
    observation = np.zeros(20, dtype=np.float32)
    state = StateDiscretizer().encode(observation)
    assert len(state) == 13

    agent = TabularQAgent()
    next_state = tuple(value + 1 for value in state)
    agent.update(
        state,
        action=3,
        reward=2.0,
        next_state=next_state,
        learning_rate=0.5,
        discount=0.9,
        terminal=True,
    )
    checkpoint = agent.save(tmp_path / "q_table.npz")
    loaded = TabularQAgent.load(checkpoint)
    assert loaded.greedy_action(state) == 3
