"""Owner-scoped browser recovery. No robot imports or device operations."""
import math
import threading
import time


class PlannerSupervision:
    def __init__(self, config, *, clock=time.monotonic, wall=time.time, track_parent=True):
        self.c, self.clock, self.wall = config, clock, wall
        self.track_parent = track_parent
        self.lock = threading.RLock()
        self.started = self.last_parent = self.last_browser = clock()
        self.terminal_reason = None
        self.pause_reason = None
        self.diagnostics = None
        self.health = None
        self.stable_since = self.stable_last = None
        self.stable_count = 0

    def cancel(self, reason):
        with self.lock:
            if self.terminal_reason is None:
                self.terminal_reason = reason

    def _pause(self):
        if not self.pause_reason:
            now = self.clock()
            self.pause_reason = 'browser_heartbeat_timeout'
            self.stable_since = self.stable_last = None
            self.stable_count = 0
            self.diagnostics = dict(
                reason=self.pause_reason, captured_unix=self.wall(),
                browser_heartbeat_age_s=now-self.last_browser,
                parent_heartbeat_age_s=now-self.last_parent,
                connection_health=self.health)

    def heartbeat(self, seen, health=None):
        with self.lock:
            now = self.clock()
            self.last_parent = now
            if isinstance(health, dict):
                self.health = dict(health)
            if type(seen) not in (int, float) or not math.isfinite(seen) or seen > now:
                self.cancel('invalid_browser_heartbeat')
                return
            if now-seen > self.c.browser_disconnect_timeout_s:
                self._pause()
            if seen <= self.last_browser:
                return  # Parent supervision messages are not browser responses.
            self.last_browser = seen
            if self.pause_reason and now-seen <= 1.5:
                if self.stable_last is None or seen-self.stable_last > 2.5:
                    self.stable_since, self.stable_count = seen, 0
                self.stable_last = seen
                self.stable_count += 1

    def interrupted(self):
        with self.lock:
            now = self.clock()
            if now-self.started >= self.c.max_task_duration_s:
                self.cancel('task_time_limit')
            if self.track_parent and now-self.last_parent > 2:
                self.cancel('parent_disconnected')
            elif self.track_parent and now-self.last_browser > self.c.browser_disconnect_timeout_s:
                self._pause()
            return bool(self.terminal_reason or self.pause_reason)

    def reason(self):
        with self.lock:
            return self.terminal_reason or self.pause_reason or 'user_cancelled'

    def recovery(self):
        with self.lock:
            self.interrupted()
            stable = (0. if self.stable_since is None else
                      max(0., (self.stable_last or self.stable_since)-self.stable_since))
            ready = (bool(self.pause_reason) and not self.terminal_reason
                     and self.clock()-self.last_browser <= 1.5
                     and stable >= 3. and self.stable_count >= 3
                     and bool(self.health and self.health.get('active_connections', 0) > 0))
            return dict(ready=ready, stable_s=stable, required_stable_s=3.,
                        heartbeat_age_s=self.clock()-self.last_browser,
                        responses=self.stable_count)

    def resume(self):
        with self.lock:
            if not self.recovery()['ready']:
                return False
            self.pause_reason = None
            self.stable_since = self.stable_last = None
            self.stable_count = 0
            return True
