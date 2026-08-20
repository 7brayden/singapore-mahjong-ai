"""API tests for the FastAPI backend."""

from fastapi.testclient import TestClient

from mahjong.server.app import app

client = TestClient(app)


def create(seed=5, bots=None, human_seat=0):
    response = client.post("/games", json={
        "seed": seed, "bots": bots, "human_seat": human_seat})
    assert response.status_code == 200, response.text
    return response.json()


def test_root_and_tiles():
    root = client.get("/").json()
    assert "hybrid" in root["bot_types"]

    tiles = client.get("/tiles").json()
    assert tiles["0"]["short"] == "1w"
    assert tiles["33"]["name"] == "White Dragon"
    assert tiles["42"]["suit"] == "animal"


def test_create_and_view():
    data = create()
    view = data["view"]
    # Human sits at seat 0 = dealer, so the first decision is theirs
    assert view["pending"]["type"] == "discard"
    assert len(view["hand"]) == 14

    again = client.get(f"/games/{data['game_id']}").json()
    assert again["hand"] == view["hand"]


def test_full_game_via_hints():
    data = create(seed=7, bots=["greedy"] * 4)
    game_id = data["game_id"]
    view = data["view"]
    for _ in range(300):
        if view["game_over"]:
            break
        suggestion = client.get(f"/games/{game_id}/hint").json()["suggestion"]
        response = client.post(f"/games/{game_id}/action",
                               json={"answer": suggestion})
        assert response.status_code == 200, response.text
        view = response.json()
    assert view["game_over"]
    assert sum(view["result"]["payments"]) == 0


def test_invalid_actions_return_400():
    data = create()
    game_id = data["game_id"]
    not_held = next(t for t in range(34) if t not in data["view"]["hand"])

    assert client.post(f"/games/{game_id}/action",
                       json={"answer": not_held}).status_code == 400
    assert client.post(f"/games/{game_id}/action",
                       json={"answer": True}).status_code == 400
    # Game is untouched and still playable
    assert client.get(f"/games/{game_id}").json()["pending"] is not None


def test_unknown_game_is_404():
    assert client.get("/games/nope").status_code == 404
    assert client.post("/games/nope/action",
                       json={"answer": 1}).status_code == 404
    assert client.delete("/games/nope").status_code == 404


def test_bad_create_requests_return_400():
    assert client.post("/games", json={
        "bots": ["hybrid", "hybrid", "alphazero", "greedy"]}).status_code == 400
    assert client.post("/games", json={"human_seat": 7}).status_code == 400
    assert client.post("/games", json={"tai_cap": 0}).status_code == 400
    assert client.post("/games", json={"base_unit": 0}).status_code == 400


def test_scoring_config_passthrough():
    from mahjong.server.app import manager

    data = create()
    game_id = client.post("/games", json={
        "seed": 1, "tai_cap": 5, "base_unit": 5}).json()["game_id"]
    config = manager.get(game_id).interactive.game.score_config
    assert config.tai_cap == 5
    assert config.base_unit == 5
    # Defaults: 6-tai limit, 1-chip base
    default_config = manager.get(data["game_id"]).interactive.game.score_config
    assert default_config.tai_cap == 6
    assert default_config.base_unit == 1


def test_analysis_payload():
    data = create(seed=9)
    analysis = client.get(f"/games/{data['game_id']}/analysis").json()
    assert "shanten" in analysis
    assert len(analysis["opponents"]) == 3
    first = analysis["discards"][0]
    assert set(first) >= {"tile", "shanten_after", "acceptance",
                          "danger", "danger_components"}
    assert set(first["danger_components"]) == {
        "visibility", "discard_absence", "opponent_threat", "suit_safety"}


def test_advisor_stars_the_agents_pick():
    """The starred discard must be the SAME tile the coach explains.

    /hint returns the seat agent's choice; /analysis must list that
    tile first and flag it, so the sidebar's recommendation can never
    contradict the coach's narration of the agent's pick.
    """
    data = create(seed=9)
    game_id = data["game_id"]
    suggestion = client.get(f"/games/{game_id}/hint").json()["suggestion"]
    analysis = client.get(f"/games/{game_id}/analysis").json()
    first = analysis["discards"][0]
    assert first["tile"] == suggestion
    assert first["agent_pick"] is True
    # Only the pick carries the flag
    assert all("agent_pick" not in d for d in analysis["discards"][1:])


def test_websocket_pushes_view_after_action():
    data = create(seed=4, bots=["greedy"] * 4)
    game_id = data["game_id"]
    with client.websocket_connect(f"/games/{game_id}/ws") as ws:
        first = ws.receive_json()
        assert first["seat"] == 0

        suggestion = client.get(f"/games/{game_id}/hint").json()["suggestion"]
        client.post(f"/games/{game_id}/action", json={"answer": suggestion})
        update = ws.receive_json()
        assert update["turn"] >= first["turn"]


def test_delete_game():
    game_id = create()["game_id"]
    assert client.delete(f"/games/{game_id}").status_code == 200
    assert client.get(f"/games/{game_id}").status_code == 404


def test_session_advances_with_dealer_rules():
    """Next-hand applies payments, rotates the dealership per table
    rules (dealer repeats on own win or draw), and labels the round."""
    from mahjong.server.manager import GameManager

    mgr = GameManager()
    managed = mgr.create(seed=5)
    assert managed.view()["session"]["round_label"] == "East 1"
    assert managed.view()["session"]["hand_number"] == 1

    # Drive the human seat with its own agent until the hand ends
    interactive = managed.interactive
    while interactive.pending is not None:
        interactive.submit(interactive.game.dispatch_to_agent(interactive.pending))
    result = interactive.result
    assert result is not None

    old_dealer = managed.dealer
    managed.next_hand()
    sess = managed.view()["session"]
    assert sess["hand_number"] == 2
    # Payments from hand 1 are banked in the running scores
    assert sess["scores"] == list(result.payments or [0, 0, 0, 0])
    # Dealer repeats on own win or draw, else passes
    if result.winner is None or result.winner == old_dealer:
        assert managed.dealer == old_dealer
        assert sess["round_label"] == "East 1"
    else:
        assert managed.dealer == (old_dealer + 1) % 4
        assert sess["round_label"] == "East 2"
    # A fresh wall was dealt with the right dealer
    assert managed.interactive.game.dealer == managed.dealer
    assert not managed.view()["game_over"]


def test_next_hand_refused_mid_hand():
    from mahjong.server.manager import GameManager
    import pytest as _pytest

    managed = GameManager().create(seed=6)
    with _pytest.raises(RuntimeError):
        managed.next_hand()


def test_hint_claim_context_and_headline():
    """The /hint machinery attaches engine tai facts to claim windows,
    and the one-liner carries the consequence — no fabricated reasons."""
    from mahjong.game import GameState, ClaimRequest
    from mahjong.agents import GreedyAgent
    from mahjong.hand import Hand
    from mahjong.server.app import _claim_context, _suggestion_text

    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    game.wall = [0] * 60
    hand = Hand()
    # The reproduction hand: clean concealed ping hu track, 5s pair
    for t in [1, 2, 3, 4, 8, 12, 15, 19, 20, 21, 22, 22, 24]:
        hand.add_tile(t)
    game.hands[0] = hand

    request = ClaimRequest(0, 22, "pong")
    ctx = _claim_context(game, request, False)
    assert ctx is not None
    assert ctx["dead_after"] is True
    assert ctx["kills_pinghu"] is True

    text = _suggestion_text(request, False, game=game)
    assert "pass" in text
    assert "cannot win" in text  # the engine's consequence, attached
