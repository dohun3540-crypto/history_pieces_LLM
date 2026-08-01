from history_chatbot.dialogue.personalization_tags import TagScope, observations


def test_single_preference_is_only_candidate_not_profile() -> None:
    item = observations(("prefers_short",), turn_id="turn-1", user_message="짧게 말해줘")[0]
    assert item.scope == TagScope.PREFERENCE_CANDIDATE
    assert not item.profile_candidate
    assert item.evidence_turn_id == "turn-1"
    assert item.original_user_message == "짧게 말해줘"


def test_repeated_preference_can_become_profile_candidate() -> None:
    item = observations(("prefers_short",), turn_id="turn-2", user_message="또 짧게", repeated_tags=frozenset({"prefers_short"}))[0]
    assert item.profile_candidate


def test_fatigue_is_session_scoped() -> None:
    item = observations(("current_fatigue",), turn_id="turn-3", user_message="피곤해")[0]
    assert item.scope == TagScope.SESSION_OBSERVATION
    assert not item.profile_candidate


def test_unknown_or_sensitive_inference_is_not_stored() -> None:
    assert observations(("political_orientation", "nationality_guess"), turn_id="turn-4", user_message="원문") == ()
