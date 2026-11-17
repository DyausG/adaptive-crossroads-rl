from types import SimpleNamespace
from environment import episode_metrics


def fake_episode(emergency=True, entry=300, clear=None):
    return SimpleNamespace(current_scenario={"vehicle_count": 3, "emergency": emergency,
        "id": "case", "seed": 1, "level": "low", "asymmetric": False, "ambulance_approach": "n" if emergency else None},
        reg_departures={"car1": 1}, reg_arrivals={}, reg_wait={"car1": 12}, reg_loss={"car1": 16},
        sim_step=1200, ambulance_entry=entry, ambulance_clear=clear, ambulance_depart_actual=290,
        collisions=0, red_crossings=[], num_teleported_vehicles=0,
        samples=[{"time": t, "total_queue": 0 if t < 300 else 10} for t in range(1, 1201)])


def test_censored_passage_and_unfinished_vehicles_are_retained():
    result = episode_metrics(fake_episode())
    assert result["ambulance_passage_status"] == "right_censored"
    assert result["ambulance_passage_seconds"] is None
    assert result["ambulance_observed_seconds"] == 900
    assert result["regular_unfinished"] == 1
    assert result["regular_not_inserted"] == 1
    assert result["regular_wait_mean_inserted"] == 12
    assert result["regular_wait_mean_completed"] is None


def test_no_detection_and_normal_episodes_are_distinguished():
    assert episode_metrics(fake_episode(entry=None))["ambulance_passage_status"] == "not_detected"
    result = episode_metrics(fake_episode(emergency=False, entry=None))
    assert result["ambulance_passage_status"] == "not_applicable"
    assert result["recovery_status"] == "not_applicable"


def test_completed_passage_with_no_queue_recovery_is_censored():
    result = episode_metrics(fake_episode(clear=350))
    assert result["ambulance_passage_seconds"] == 50
    assert result["recovery_status"] == "right_censored"
    assert result["queue_recovery_seconds"] is None
    assert result["recovery_observed_seconds"] == 850
