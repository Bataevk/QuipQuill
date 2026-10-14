from quipquill_cloud.mq.worker import CancelRegistry


def test_cancels_expire_when_nobody_forgets_them() -> None:
    now = [0.0]
    cancels = CancelRegistry(ttl=10, clock=lambda: now[0])

    cancels.cancel("u1", "finished-long-ago")  # the request ended before the cancel arrived
    cancels.cancel("u1", "running")
    assert cancels.is_cancelled("u1", "running")
    assert not cancels.is_cancelled("u2", "running")  # cancels are per user

    cancels.forget("u1", "running")
    now[0] = 11
    cancels.cancel("u1", "new")
    assert len(cancels) == 1
    assert cancels.is_cancelled("u1", "new")
