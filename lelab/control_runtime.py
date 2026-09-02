"""Server-owned lease watching and bounded control-session shutdown.

The runtime starts no work at import time and owns no hardware.  A server
lifespan starts its watchdog explicitly, while the active operation's worker
remains the only code allowed to finish teardown and release the global claim.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from dataclasses import dataclass
from numbers import Real

from .control_session import ControlSessionManager

DEFAULT_WATCHDOG_INTERVAL_S = 0.1


@dataclass(frozen=True, slots=True)
class RuntimeShutdownResult:
    """Truthful result of one bounded server-shutdown wait."""

    session_id: str | None
    teardown_complete: bool
    watchdog_stopped: bool
    quarantine_reason: str | None = None


class ControlRuntime:
    """Poll lease expiry and coordinate bounded process shutdown."""

    def __init__(
        self,
        manager: ControlSessionManager,
        *,
        watchdog_interval_s: float = DEFAULT_WATCHDOG_INTERVAL_S,
    ) -> None:
        if (
            isinstance(watchdog_interval_s, bool)
            or not isinstance(watchdog_interval_s, Real)
            or not math.isfinite(float(watchdog_interval_s))
            or watchdog_interval_s <= 0
        ):
            raise ValueError("watchdog_interval_s must be a finite positive number")
        if watchdog_interval_s >= manager.lease_ttl_s:
            raise ValueError("watchdog_interval_s must be shorter than the lease TTL")

        self.manager = manager
        self.watchdog_interval_s = float(watchdog_interval_s)
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog_error: BaseException | None = None
        self._shutdown_started = False

    @property
    def watchdog_error(self) -> BaseException | None:
        with self._lock:
            return self._watchdog_error

    @property
    def watchdog_alive(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the lease watchdog once for this server runtime."""

        with self._lock:
            if self._shutdown_started:
                raise RuntimeError("control runtime is shutting down")
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("control runtime watchdog has already exited")
            self._thread = threading.Thread(
                target=self._watchdog_loop,
                name="control-lease-watchdog",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def shutdown(
        self,
        *,
        timeout_s: float,
        reason: str = "server shutdown",
    ) -> RuntimeShutdownResult:
        """Signal the active owner, stop the watchdog, and wait boundedly.

        A timeout never marks a session terminal and never releases its claim.
        The operation worker remains responsible for calling
        :meth:`ControlSessionManager.finish_teardown` after real cleanup.
        """

        timeout = _finite_nonnegative_timeout(timeout_s)
        started_at = time.monotonic()

        with self._lock:
            self._shutdown_started = True

        shutdown_status = self.manager.begin_shutdown(reason=reason)
        session_id = (
            shutdown_status.session_id
            if shutdown_status is not None and not shutdown_status.terminal
            else None
        )
        self._stop_requested.set()

        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(_remaining(timeout, started_at))
        watchdog_stopped = thread is None or not thread.is_alive()

        teardown_complete = True
        if session_id is not None:
            teardown_complete = self.manager.wait_for_teardown(
                session_id,
                timeout=_remaining(timeout, started_at),
            )

        quarantine_reason = self.manager.quarantine_reason
        if quarantine_reason is not None:
            teardown_complete = False
            if session_id is None and shutdown_status is not None:
                session_id = shutdown_status.session_id
        return RuntimeShutdownResult(
            session_id=session_id,
            teardown_complete=teardown_complete,
            watchdog_stopped=watchdog_stopped,
            quarantine_reason=quarantine_reason,
        )

    def _watchdog_loop(self) -> None:
        try:
            while not self._stop_requested.wait(self.watchdog_interval_s):
                self.manager.check_lease_expiry()
        except BaseException as error:
            reason = f"control lease watchdog failed: {type(error).__name__}: {error}"
            try:
                # Loss of the lease enforcer is itself a fail-closed shutdown
                # event. The active owner retains the claim until its real
                # teardown calls finish_teardown().
                self.manager.begin_shutdown(reason=reason, terminal_error=True)
            except BaseException as shutdown_error:
                with contextlib.suppress(BaseException):
                    error.add_note(
                        "additionally failed to signal control shutdown: "
                        f"{type(shutdown_error).__name__}: {shutdown_error}"
                    )
            with self._lock:
                self._watchdog_error = error
            self._stop_requested.set()


def _finite_nonnegative_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
        raise ValueError("timeout_s must be a finite non-negative number")
    return float(value)


def _remaining(timeout_s: float, started_at: float) -> float:
    return max(0.0, timeout_s - (time.monotonic() - started_at))
