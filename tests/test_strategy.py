"""Tests for CollaborationStrategy."""

from testbench.models.strategy import CollaborationStrategy


class TestCollaborationStrategy:
    def test_solo_defaults(self):
        s = CollaborationStrategy()
        assert s.config_id == "0000"
        assert s.is_solo
        assert s.active_phases == []

    def test_full_collab(self):
        s = CollaborationStrategy(
            spawn_subagents=True,
            parallel_execution=True,
            share_partial_results=True,
            synthesis_review=True,
        )
        assert s.config_id == "1111"
        assert not s.is_solo
        assert len(s.active_phases) == 4

    def test_solo_forces_other_flags_off(self):
        """When spawn_subagents=False, other flags are forced to False."""
        s = CollaborationStrategy(
            spawn_subagents=False,
            parallel_execution=True,
            share_partial_results=True,
            synthesis_review=True,
        )
        assert s.config_id == "0000"
        assert s.parallel_execution is False
        assert s.share_partial_results is False
        assert s.synthesis_review is False

    def test_spawn_only(self):
        s = CollaborationStrategy(spawn_subagents=True)
        assert s.config_id == "1000"
        assert s.active_phases == ["SPAWN_SUBAGENTS"]

    def test_from_config_id_roundtrip(self):
        for s in CollaborationStrategy.all_strategies():
            reconstructed = CollaborationStrategy.from_config_id(s.config_id)
            assert reconstructed == s

    def test_from_config_id_invalid(self):
        import pytest
        with pytest.raises(ValueError):
            CollaborationStrategy.from_config_id("abc")
        with pytest.raises(ValueError):
            CollaborationStrategy.from_config_id("00000")

    def test_all_strategies_count(self):
        """Should produce exactly 9 effective configurations."""
        strategies = CollaborationStrategy.all_strategies()
        assert len(strategies) == 9

    def test_all_strategies_unique(self):
        strategies = CollaborationStrategy.all_strategies()
        config_ids = [s.config_id for s in strategies]
        assert len(config_ids) == len(set(config_ids))

    def test_all_strategies_contains_solo(self):
        strategies = CollaborationStrategy.all_strategies()
        solo = [s for s in strategies if s.is_solo]
        assert len(solo) == 1
        assert solo[0].config_id == "0000"

    def test_describe_solo(self):
        s = CollaborationStrategy()
        assert "Solo" in s.describe()

    def test_describe_collab(self):
        s = CollaborationStrategy(spawn_subagents=True, parallel_execution=True)
        desc = s.describe()
        assert "Spawn: ON" in desc
        assert "Parallel: ON" in desc

    def test_frozen(self):
        import pytest
        s = CollaborationStrategy()
        with pytest.raises(AttributeError):
            s.spawn_subagents = True  # type: ignore
