"""Which window has focus, reported by a KWin script over the session bus.

Wayland does not tell an app which window is active. On KDE Plasma a KWin script can watch
workspace.windowActivated and call a D-Bus method of ours with the window's id, resource
class (the app, like "org.kde.konsole") and caption. FocusTracker owns that D-Bus service
on its own thread, loads the script on start and unloads it on stop.

Window ids are KWin's UUIDs, mapped to small integers because the rest of the app compares
windows as integers. Outside KDE, or when the script cannot be loaded, the tracker reports a
single pseudo window so delivery still pastes into whatever has focus.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BUS_NAME = "io.github.kapidolli.Spells"
OBJECT_PATH = "/Focus"
INTERFACE = "io.github.kapidolli.Spells.Focus"
SCRIPT_NAME = "spells-focus"
PSEUDO_WINDOW = 1

SCRIPT = """
var tracked = null;
function report(w) {
    var geometry = w && w.output ? w.output.geometry : null;
    callDBus("BUS", "PATH", "IFACE", "Activated",
        w ? String(w.internalId) : "",
        w ? String(w.resourceClass) : "",
        w ? String(w.caption) : "",
        w ? String(w.pid) : "0",
        geometry ? [geometry.x, geometry.y, geometry.width, geometry.height].join(",") : "");
}
function onCaption() { report(tracked); }
function follow(w) {
    if (tracked) {
        try { tracked.captionChanged.disconnect(onCaption); } catch (e) {}
    }
    tracked = w;
    if (w) { w.captionChanged.connect(onCaption); }
    report(w);
}
workspace.windowActivated.connect(follow);
follow(workspace.activeWindow);
"""
SCRIPT = SCRIPT.replace("BUS", BUS_NAME).replace("PATH", OBJECT_PATH).replace("IFACE", INTERFACE)


@dataclass(frozen=True)
class WindowInfo:
    app: str = ""
    title: str = ""
    pid: int = 0
    output: tuple[int, int, int, int] | None = None


def parse_output(text: str) -> tuple[int, int, int, int] | None:
    try:
        x, y, width, height = (int(float(part)) for part in text.split(","))
    except ValueError:
        return None
    return (x, y, x + width, y + height)


class FocusTracker:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._info: dict[int, WindowInfo] = {}
        self._active = 0
        self._live = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._script_path: Path | None = None

    @property
    def live(self) -> bool:
        return self._live

    @property
    def available(self) -> bool:
        """Whether tracking can work here: KDE Plasma with jeepney installed."""
        if "KDE" not in os.environ.get("XDG_CURRENT_DESKTOP", "").upper():
            return False
        try:
            import jeepney  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self) -> None:
        """Serve the D-Bus method and load the script, both on the tracker's own thread."""
        if self._thread is not None:
            return
        if not self.available:
            log.info("No KWin to ask; focus tracking is off and delivery targets any window")
            return
        self._thread = threading.Thread(target=self._serve, name="spells-focus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._live:
            self._unload_script()
        self._live = False

    def foreground(self) -> int:
        if not self._live:
            return PSEUDO_WINDOW
        return self._active

    def info(self, window: int) -> WindowInfo:
        with self._lock:
            return self._info.get(window, WindowInfo())

    def activated(self, uuid: str, app: str, title: str, pid: str, output: str) -> None:
        with self._lock:
            if not uuid:
                self._active = 0
                return
            window = self._ids.setdefault(uuid, len(self._ids) + 2)
            try:
                pid_value = int(float(pid))
            except ValueError:
                pid_value = 0
            self._info[window] = WindowInfo(app, title, pid_value, parse_output(output))
            self._active = window

    def _serve(self) -> None:
        try:
            from jeepney import HeaderFields, MessageType, new_error, new_method_return
            from jeepney.bus_messages import message_bus
            from jeepney.io.blocking import open_dbus_connection
        except ImportError:
            log.warning("jeepney is missing; focus tracking is off")
            return
        try:
            connection = open_dbus_connection(bus="SESSION")
            reply = connection.send_and_get_reply(message_bus.RequestName(BUS_NAME, 4))
            if reply.body[0] not in (1, 4):
                log.warning("The bus name %s is taken; focus tracking is off", BUS_NAME)
                return
        except Exception:
            log.exception("The session bus is unavailable; focus tracking is off")
            return
        self._live = True
        threading.Thread(target=self._load_script, name="spells-kwin-script", daemon=True).start()
        with connection:
            while not self._stop.is_set():
                try:
                    message = connection.receive(timeout=0.5)
                except TimeoutError:
                    continue
                except Exception:
                    log.exception("The focus service stopped")
                    self._live = False
                    return
                header = message.header
                if header.message_type is not MessageType.method_call:
                    continue
                fields = header.fields
                if (
                    fields.get(HeaderFields.path) == OBJECT_PATH
                    and fields.get(HeaderFields.member) == "Activated"
                ):
                    body = [str(item) for item in message.body] + [""] * 5
                    self.activated(*body[:5])
                    connection.send(new_method_return(message))
                else:
                    connection.send(
                        new_error(message, "org.freedesktop.DBus.Error.UnknownMethod")
                    )

    def _kwin_call(self, path: str, interface: str, member: str, signature: str, *args):
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection

        address = DBusAddress(path, bus_name="org.kde.KWin", interface=interface)
        with open_dbus_connection(bus="SESSION") as connection:
            reply = connection.send_and_get_reply(
                new_method_call(address, member, signature, args), timeout=3.0
            )
        return reply.body

    def _load_script(self) -> None:
        try:
            self._unload_script()
            handle, name = tempfile.mkstemp(prefix="spells-focus-", suffix=".js")
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(SCRIPT)
            self._script_path = Path(name)
            body = self._kwin_call(
                "/Scripting", "org.kde.kwin.Scripting", "loadScript", "ss", name, SCRIPT_NAME
            )
            script_id = int(body[0])
            if script_id < 0:
                raise RuntimeError(f"KWin refused the script ({script_id})")
            self._kwin_call(f"/Scripting/Script{script_id}", "org.kde.kwin.Script", "run", "")
            log.info("KWin focus script %d is running", script_id)
        except Exception:
            log.exception("The KWin focus script could not be loaded; focus tracking is off")
            self._live = False

    def _unload_script(self) -> None:
        try:
            self._kwin_call("/Scripting", "org.kde.kwin.Scripting", "unloadScript", "s", SCRIPT_NAME)
        except Exception:
            log.debug("Unloading the KWin focus script failed", exc_info=True)
        path, self._script_path = self._script_path, None
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass
