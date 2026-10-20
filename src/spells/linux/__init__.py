"""Linux system access for spells.platform.linux: keyboard, clipboard, focus and desktop glue.

Only spells.platform.linux imports these modules. They talk to the kernel's input devices
(evdev to read, uinput to send), the Wayland clipboard tools, KWin and the session bus.
"""
