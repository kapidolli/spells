"""The building blocks of the settings, welcome and diagnostics windows.

Surfaces and plain controls take their look from the application stylesheet (style.py)
through a `role` or `kind` property; the pieces a stylesheet cannot draw well (the toggle
switch, the segmented control, navigation items, keycaps, the status dot and the logo) are
painted here from the installed palette.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells.ui import brand, style

Qt = QtCore.Qt


def set_prop(widget: QtWidgets.QWidget, name: str, value: Any) -> None:
    """Set a dynamic property and repolish so the stylesheet picks it up."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def make_label(
    text: str = "",
    role: str = "body",
    tone: str | None = None,
    *,
    wrap: bool = False,
    parent: QtWidgets.QWidget | None = None,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text, parent)
    label.setFont(style.font(role))
    if tone:
        label.setProperty("tone", tone)
    label.setWordWrap(wrap)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def make_button(
    text: str,
    kind: str | None = None,
    *,
    glyph: str | None = None,
    parent: QtWidgets.QWidget | None = None,
) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text, parent)
    button.setFont(style.font("body"))
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind:
        button.setProperty("kind", kind)
    if glyph:
        colour = None
        current = style.palette()
        if kind == "primary":
            colour = current.on_accent
        elif kind in ("subtle", "chipadd"):
            colour = current.accent_text
        elif kind == "danger":
            colour = current.critical
        button.setIcon(style.glyph_icon(glyph, colour, gap=6))
        button.setIconSize(QtCore.QSize(20, 14))
    return button


def painter_for(widget: QtWidgets.QWidget) -> QtGui.QPainter:
    painter = QtGui.QPainter(widget)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
    return painter


# Toggle switch -----------------------------------------------------------------------------------


class ToggleSwitch(QtWidgets.QAbstractButton):
    """A Windows 11 toggle: a 40 by 20 track with a sliding knob and an On or Off label."""

    TRACK_WIDTH = 40
    TRACK_HEIGHT = 20
    LABEL_GAP = 12

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, state_text: bool = True) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFont(style.font("body"))
        self._state_text = state_text
        self._hover = False
        self._keyboard_focus = False
        self._animation = QtCore.QVariantAnimation(self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(lambda _value: self.update())
        self.toggled.connect(self._start_animation)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QtCore.QSize:
        width = self.TRACK_WIDTH + 4
        if self._state_text:
            metrics = QtGui.QFontMetrics(self.font())
            width += max(metrics.horizontalAdvance("On"), metrics.horizontalAdvance("Off")) + self.LABEL_GAP
        return QtCore.QSize(width, 28)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def position(self) -> float:
        target = 1.0 if self.isChecked() else 0.0
        if self._animation.state() == QtCore.QAbstractAnimation.State.Running:
            if float(self._animation.endValue()) != target:
                self._animation.stop()
                return target
            return float(self._animation.currentValue())
        return target

    def _start_animation(self, checked: bool) -> None:
        self._animation.stop()
        if not self.isVisible():
            return
        self._animation.setStartValue(0.0 if checked else 1.0)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def hitButton(self, pos: QtCore.QPoint) -> bool:
        return self.rect().contains(pos)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self._keyboard_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._keyboard_focus = False
        super().focusOutEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            enabled = self.isEnabled()
            pos = self.position()
            track = QtCore.QRectF(
                self.width() - self.TRACK_WIDTH - 2,
                (self.height() - self.TRACK_HEIGHT) / 2.0,
                self.TRACK_WIDTH,
                self.TRACK_HEIGHT,
            )
            if self._state_text:
                painter.setFont(self.font())
                painter.setPen(current.color("text" if enabled else "disabled"))
                text_rect = QtCore.QRectF(0, 0, track.left() - self.LABEL_GAP, self.height())
                painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), "On" if self.isChecked() else "Off")
            accent = current.color("accent_hover" if self._hover else "accent")
            off_fill = current.color("control_hover" if self._hover else "control")
            off_border = current.color("track_off_border")
            if not enabled:
                accent = current.color("disabled")
                off_border = current.color("disabled")
            fill = style.mix(accent, off_fill, pos)
            border = style.mix(accent, off_border, pos)
            radius = self.TRACK_HEIGHT / 2.0
            painter.setPen(QtGui.QPen(border, 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(track.adjusted(0.5, 0.5, -0.5, -0.5), radius - 0.5, radius - 0.5)
            knob_size = 14.0 if (self._hover and enabled) else 12.0
            if self.isDown():
                knob_size = 14.0
            left = track.left() + 4 + knob_size / 2
            right = track.right() - 4 - knob_size / 2
            cx = left + (right - left) * pos
            cy = track.center().y()
            knob_colour = style.mix(current.color("on_accent"), current.color("text2" if enabled else "disabled"), pos)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(knob_colour)
            painter.drawEllipse(QtCore.QPointF(cx, cy), knob_size / 2, knob_size / 2)
            if self._keyboard_focus and self.hasFocus():
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(current.color("text"), 1.5))
                ring = track.adjusted(-3, -3, 3, 3)
                painter.drawRoundedRect(ring, radius + 3, radius + 3)
        finally:
            painter.end()


# Segmented control -------------------------------------------------------------------------------


class SegmentedControl(QtWidgets.QWidget):
    """A row of mutually exclusive options with the QComboBox calls the pages use."""

    currentIndexChanged = QtCore.Signal(int)

    PAD = 16
    HEIGHT = 32

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, Any]] = []
        self._index = -1
        self._hover = -1
        self._show_focus = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(style.font("body"))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def addItem(self, text: str, data: Any = None) -> None:
        self._items.append((str(text), data))
        self.updateGeometry()
        if self._index < 0:
            self._index = 0
            self.currentIndexChanged.emit(0)
        self.update()

    def clear(self) -> None:
        self._items = []
        self._index = -1
        self.updateGeometry()
        self.update()

    def count(self) -> int:
        return len(self._items)

    def itemText(self, index: int) -> str:
        return self._items[index][0] if 0 <= index < len(self._items) else ""

    def itemData(self, index: int) -> Any:
        return self._items[index][1] if 0 <= index < len(self._items) else None

    def currentIndex(self) -> int:
        return self._index

    def currentData(self) -> Any:
        return self.itemData(self._index)

    def currentText(self) -> str:
        return self.itemText(self._index)

    def findData(self, data: Any) -> int:
        for index, (_text, value) in enumerate(self._items):
            if value == data:
                return index
        return -1

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items) or index == self._index:
            return
        self._index = index
        self.update()
        self.currentIndexChanged.emit(index)

    def _widths(self) -> list[int]:
        metrics = QtGui.QFontMetrics(style.font("body_strong"))
        return [max(56, metrics.horizontalAdvance(text) + 2 * self.PAD) for text, _data in self._items]

    def _rects(self) -> list[QtCore.QRectF]:
        rects = []
        x = 3.0
        for width in self._widths():
            rects.append(QtCore.QRectF(x, 3.0, width, self.HEIGHT - 6.0))
            x += width
        return rects

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(sum(self._widths()) + 6, self.HEIGHT)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def _index_at(self, pos: QtCore.QPointF) -> int:
        for index, rect in enumerate(self._rects()):
            if rect.contains(pos):
                return index
        return -1

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._index_at(event.position())
        if index >= 0:
            self.setCurrentIndex(index)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.setCurrentIndex(max(0, self._index - 1))
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.setCurrentIndex(min(len(self._items) - 1, self._index + 1))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            outer = QtCore.QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
            painter.setPen(QtGui.QPen(current.color("control_border"), 1.0))
            painter.setBrush(current.color("base"))
            painter.drawRoundedRect(outer, 8, 8)
            regular = style.font("body")
            strong = style.font("body_strong")
            for index, rect in enumerate(self._rects()):
                text = self._items[index][0]
                if index == self._index:
                    painter.setPen(QtGui.QPen(current.color("control_border"), 1.0))
                    painter.setBrush(current.color("control"))
                    painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(current.color("accent"))
                    bar = QtCore.QRectF(rect.center().x() - 8, rect.bottom() - 3.5, 16, 3)
                    painter.drawRoundedRect(bar, 1.5, 1.5)
                    painter.setFont(strong)
                    painter.setPen(current.color("text"))
                elif index == self._hover:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(current.color("nav_hover"))
                    painter.drawRoundedRect(rect, 6, 6)
                    painter.setFont(regular)
                    painter.setPen(current.color("text"))
                else:
                    painter.setFont(regular)
                    painter.setPen(current.color("text2"))
                painter.drawText(rect.adjusted(0, -1, 0, -1), int(Qt.AlignmentFlag.AlignCenter), text)
            if self.hasFocus() and self.focusPolicy() != Qt.FocusPolicy.NoFocus and self._index >= 0 and self._show_focus:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(current.color("text"), 1.5))
                painter.drawRoundedRect(outer.adjusted(1, 1, -1, -1), 7, 7)
        finally:
            painter.end()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self._show_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._show_focus = False
        super().focusOutEvent(event)


# Navigation --------------------------------------------------------------------------------------------


class NavItem(QtWidgets.QAbstractButton):
    """One entry of the navigation rail: a glyph and a label, with the accent bar when selected."""

    def __init__(self, key: str, text: str, glyph: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.glyph = glyph
        self.setText(text)
        self.setAccessibleName(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setFont(style.font("body"))
        self.setFixedHeight(style.NAV_ITEM_HEIGHT)
        self._hover = False
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(200, style.NAV_ITEM_HEIGHT)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            rect = QtCore.QRectF(self.rect())
            background = None
            if self.isDown() or self.isChecked():
                background = current.color("nav_selected")
            elif self._hover:
                background = current.color("nav_hover")
            if background is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(background)
                painter.drawRoundedRect(rect, style.RADIUS_CONTROL, style.RADIUS_CONTROL)
            if self.isChecked():
                painter.setBrush(current.color("accent"))
                bar = QtCore.QRectF(0, (rect.height() - 16) / 2, 3, 16)
                painter.drawRoundedRect(bar, 1.5, 1.5)
            if style.has_icon_font():
                painter.setFont(style.icon_font(16))
                painter.setPen(current.color("text"))
                painter.drawText(QtCore.QRectF(14, 0, 16, rect.height()), int(Qt.AlignmentFlag.AlignCenter), self.glyph)
            painter.setFont(self.font())
            painter.setPen(current.color("text"))
            painter.drawText(
                QtCore.QRectF(46, 0, rect.width() - 52, rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.text(),
            )
            if self.hasFocus():
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(current.color("text"), 1.5))
                painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 5, 5)
        finally:
            painter.end()


class NavRail(QtWidgets.QFrame):
    """The logo and name on top, the page entries, and a second group pinned to the bottom."""

    current_changed = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("spellsRail")
        self.setFixedWidth(style.NAV_WIDTH)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(2)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(12, 8, 12, 22)
        header.setSpacing(12)
        self.wordmark = Wordmark(30, self)
        if self.wordmark.available():
            header.addWidget(self.wordmark)
            self.logo = None
            self.brand_label = None
        else:
            self.wordmark.hide()
            self.logo = LogoMark(28, self)
            header.addWidget(self.logo)
            self.brand_label = make_label("Spells", "brand", parent=self)
            header.addWidget(self.brand_label)
        header.addStretch(1)
        layout.addLayout(header)
        self._top = QtWidgets.QVBoxLayout()
        self._top.setSpacing(2)
        layout.addLayout(self._top)
        layout.addStretch(1)
        self._bottom = QtWidgets.QVBoxLayout()
        self._bottom.setSpacing(2)
        layout.addLayout(self._bottom)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self.items: dict[str, NavItem] = {}

    def add_item(self, key: str, text: str, glyph: str, *, bottom: bool = False) -> NavItem:
        item = NavItem(key, text, glyph, self)
        self._group.addButton(item)
        (self._bottom if bottom else self._top).addWidget(item)
        item.clicked.connect(lambda _checked=False, k=key: self.current_changed.emit(k))
        self.items[key] = item
        return item

    def set_current(self, key: str) -> None:
        item = self.items.get(key)
        if item is not None:
            item.setChecked(True)

    def current(self) -> str | None:
        for key, item in self.items.items():
            if item.isChecked():
                return key
        return None

    def labels(self) -> list[str]:
        return [item.text() for item in self.items.values()]


# Surfaces --------------------------------------------------------------------------------------------


class Divider(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget | None = None, *, vertical: bool = False) -> None:
        super().__init__(parent)
        self.setProperty("role", "vdivider" if vertical else "divider")
        if vertical:
            self.setFixedWidth(1)
        else:
            self.setFixedHeight(1)


class Card(QtWidgets.QFrame):
    """A rounded surface; rows added with add_row are separated by dividers."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(0)
        self._rows: list[QtWidgets.QWidget] = []

    def add_row(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        if self._rows:
            self.body.addWidget(Divider(self))
        widget.setParent(self)
        self.body.addWidget(widget)
        self._rows.append(widget)
        return widget

    def add_widget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> QtWidgets.QWidget:
        widget.setParent(self)
        self.body.addWidget(widget, stretch)
        return widget

    def rows(self) -> list[QtWidgets.QWidget]:
        return list(self._rows)

    def clear(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self._rows = []


class SettingRow(QtWidgets.QWidget):
    """A title and a one-line description on the left, the control on the right."""

    def __init__(
        self,
        title: str,
        description: str = "",
        control: QtWidgets.QWidget | QtWidgets.QLayout | None = None,
        *,
        glyph: str | None = None,
        badge_glyph: bool = False,
        below: QtWidgets.QWidget | None = None,
        leading: QtWidgets.QWidget | None = None,
        wide_control: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)
        self.glyph_label: QtWidgets.QLabel | None = None
        if leading is not None:
            layout.addWidget(leading, 0, Qt.AlignmentFlag.AlignVCenter)
        if glyph is not None:
            self.glyph_label = GlyphLabel(glyph, 18 if badge_glyph else 20, parent=self)
            if badge_glyph:
                self.glyph_label.setProperty("role", "glyphbadge")
                self.glyph_label.setFixedSize(36, 36)
            else:
                self.glyph_label.setFixedSize(24, 24)
            layout.addWidget(self.glyph_label, 0, Qt.AlignmentFlag.AlignVCenter)
        texts = QtWidgets.QVBoxLayout()
        texts.setSpacing(2)
        texts.setContentsMargins(0, 0, 0, 0)
        self.title_label = make_label(title, "body", parent=self)
        self.title_label.setWordWrap(True)
        texts.addWidget(self.title_label)
        self.description_label = make_label(description, "caption", "secondary", wrap=True, parent=self)
        self.description_label.setVisible(bool(description))
        texts.addWidget(self.description_label)
        if below is not None:
            texts.addSpacing(8)
            texts.addWidget(below)
        self.texts = texts
        layout.addLayout(texts, 0 if wide_control else 1)
        if wide_control:
            self.title_label.setMinimumWidth(150)
        self.controls = QtWidgets.QHBoxLayout()
        self.controls.setSpacing(8)
        if isinstance(control, QtWidgets.QLayout):
            self.controls.addLayout(control)
        elif control is not None:
            self.controls.addWidget(control)
        layout.addLayout(self.controls, 1 if wide_control else 0)
        self.setMinimumHeight(style.ROW_MIN_HEIGHT if description or below is not None else 48)

    def add_control(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.controls.addWidget(widget)
        return widget

    def set_description(self, text: str) -> None:
        self.description_label.setText(text)
        self.description_label.setVisible(bool(text))


class SectionHeader(QtWidgets.QWidget):
    def __init__(self, title: str, description: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(2, 0, 0, 4)
        outer.setSpacing(2)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.title_label = make_label(title, "body_strong", parent=self)
        row.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        self.trailing = row
        outer.addLayout(row)
        self.description_label = make_label(description, "caption", "secondary", wrap=True, parent=self)
        self.description_label.setVisible(bool(description))
        outer.addWidget(self.description_label)

    def add_trailing(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.trailing.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        return widget


class ScrollPage(QtWidgets.QWidget):
    """A page title, an optional subtitle and sections in a vertical scroll area."""

    def __init__(self, title: str, subtitle: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        self.body = QtWidgets.QWidget(self.scroll)
        self.body.setObjectName("scrollBody")
        self.scroll.setWidget(self.body)
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(36, 28, 32, 36)
        self.body_layout.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(4)
        self.title_label = make_label(title, "title", parent=self.body)
        titles.addWidget(self.title_label)
        self.subtitle_label = make_label(subtitle, "body", "secondary", wrap=True, parent=self.body)
        self.subtitle_label.setVisible(bool(subtitle))
        titles.addWidget(self.subtitle_label)
        header.addLayout(titles, 1)
        self.header_actions = QtWidgets.QHBoxLayout()
        self.header_actions.setSpacing(8)
        header.addLayout(self.header_actions, 0)
        self.body_layout.addLayout(header)
        self.body_layout.addSpacing(20)
        self._sections = 0

    def add_header_action(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.header_actions.addWidget(widget, 0, Qt.AlignmentFlag.AlignBottom)
        return widget

    def add_section(
        self, title: str, widget: QtWidgets.QWidget | None = None, *, description: str = ""
    ) -> SectionHeader:
        if self._sections:
            self.body_layout.addSpacing(24)
        self._sections += 1
        header = SectionHeader(title, description, self.body)
        self.body_layout.addWidget(header)
        self.body_layout.addSpacing(4)
        if widget is not None:
            self.body_layout.addWidget(widget)
        return header

    def add_widget(self, widget: QtWidgets.QWidget, spacing_before: int = 8) -> QtWidgets.QWidget:
        if spacing_before:
            self.body_layout.addSpacing(spacing_before)
        self.body_layout.addWidget(widget)
        return widget

    def finish(self) -> None:
        self.body_layout.addStretch(1)


# Small pieces -------------------------------------------------------------------------------------------


class GlyphLabel(QtWidgets.QLabel):
    """A Segoe Fluent Icons glyph as a label; colour follows the tone property."""

    def __init__(self, glyph: str, pixels: int = 16, *, tone: str | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(glyph if style.has_icon_font() else "", parent)
        self.setFont(style.icon_font(pixels))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if tone:
            self.setProperty("tone", tone)


class IconButton(QtWidgets.QToolButton):
    def __init__(self, glyph: str, tooltip: str, *, pixels: int = 12, size: int = 28, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(glyph if style.has_icon_font() else "x")
        self.setFont(style.icon_font(pixels))
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(size, size)


class Badge(QtWidgets.QLabel):
    """A small rounded status label: ok, caution, critical, neutral or muted."""

    def __init__(self, text: str = "", kind: str = "muted", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setFont(style.font("caption_strong"))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("badge", kind)
        self.setFixedHeight(24)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_badge(self, kind: str, text: str) -> None:
        self.setText(text)
        set_prop(self, "badge", kind)
        self.setVisible(bool(text))


class CellHost(QtWidgets.QWidget):
    """Holds one widget inside a table cell, left aligned and vertically centred.

    A table stretches a cell widget over the whole cell, so a widget of fixed height sits at
    the top edge. The host fills the cell instead and its layout centres the content.
    """

    def __init__(
        self, content: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None, margin: int = 8
    ) -> None:
        super().__init__(parent)
        self.content = content
        content.setParent(self)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(margin, 0, margin, 0)
        layout.setSpacing(0)
        layout.addWidget(content, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

    def text(self) -> str:
        getter = getattr(self.content, "text", None)
        return str(getter()) if callable(getter) else ""

    def toolTip(self) -> str:
        return self.content.toolTip()

    def property(self, name: str) -> Any:
        own = super().property(name)
        return self.content.property(name) if own is None else own


class StatusDot(QtWidgets.QWidget):
    """A coloured dot in a soft halo: ok, caution, critical, busy or idle."""

    LEVELS = ("ok", "caution", "critical", "busy", "idle")

    def __init__(self, level: str = "idle", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.level = level
        self.setFixedSize(20, 20)

    def set_level(self, level: str) -> None:
        if level != self.level:
            self.level = level
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        colours = {
            "ok": ("success", "success_bg"),
            "caution": ("caution", "caution_bg"),
            "critical": ("critical", "critical_bg"),
            "busy": ("accent", "accent_soft"),
            "idle": ("text3", "nav_selected"),
        }
        dot, halo = colours.get(self.level, colours["idle"])
        painter = painter_for(self)
        try:
            centre = QtCore.QPointF(self.width() / 2, self.height() / 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(current.color(halo))
            painter.drawEllipse(centre, 9, 9)
            painter.setBrush(current.color(dot))
            painter.drawEllipse(centre, 5, 5)
        finally:
            painter.end()


class InfoMark(QtWidgets.QWidget):
    """The filled circle with an i or ! at the start of an info bar."""

    def __init__(self, kind: str = "info", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(20, 20)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        colour = {"info": "accent", "caution": "caution", "critical": "critical", "success": "success"}.get(self.kind, "accent")
        painter = painter_for(self)
        try:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(current.color(colour))
            painter.drawEllipse(QtCore.QRectF(2, 2, 16, 16))
            ink = current.color("card") if current.dark else QtGui.QColor("#FFFFFF")
            painter.setBrush(ink)
            if self.kind == "info":
                painter.drawEllipse(QtCore.QPointF(10, 6.6), 1.05, 1.05)
                painter.drawRoundedRect(QtCore.QRectF(9.1, 8.6, 1.8, 5.8), 0.9, 0.9)
            elif self.kind == "success":
                pen = QtGui.QPen(ink, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                path = QtGui.QPainterPath(QtCore.QPointF(6.6, 10.2))
                path.lineTo(9.0, 12.5)
                path.lineTo(13.4, 7.6)
                painter.drawPath(path)
            else:
                painter.drawRoundedRect(QtCore.QRectF(9.1, 5.4, 1.8, 5.8), 0.9, 0.9)
                painter.drawEllipse(QtCore.QPointF(10, 13.6), 1.05, 1.05)
        finally:
            painter.end()


class InfoBar(QtWidgets.QFrame):
    """A tinted message: info, caution, critical or success."""

    def __init__(
        self, text: str = "", kind: str = "info", *, flush: bool = False, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setProperty("infobar", kind)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        if flush:
            self.setProperty("flush", "true")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 16, 12)
        layout.setSpacing(12)
        self.mark = InfoMark(kind, self)
        layout.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        self.label = make_label(text, "body", wrap=True, parent=self)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.label, 1)
        self.actions = layout
        self.setVisible(bool(text))

    def set_text(self, text: str) -> None:
        self.label.setText(text)
        self.setVisible(bool(text))

    def text(self) -> str:
        return self.label.text()

    def set_kind(self, kind: str) -> None:
        self.mark.kind = kind
        self.mark.update()
        set_prop(self, "infobar", kind)


class KeycapRow(QtWidgets.QWidget):
    """The keys of a chord as keycaps; text() gives the familiar "Ctrl+Win" label."""

    def __init__(
        self,
        keys: Sequence[str] = (),
        *,
        large: bool = False,
        placeholder: str = "Not set",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._keys = [str(k) for k in keys]
        self._large = large
        self._placeholder = placeholder
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    @property
    def _cap_height(self) -> int:
        return 38 if self._large else 26

    @property
    def _gap(self) -> int:
        return 8 if self._large else 4

    def _font(self) -> QtGui.QFont:
        return style.font("body_strong" if self._large else "caption_strong")

    def set_keys(self, keys: Sequence[str]) -> None:
        self._keys = [str(k) for k in keys]
        self.updateGeometry()
        self.update()
        self.setAccessibleName(self.text() or self._placeholder)

    def keys(self) -> list[str]:
        return list(self._keys)

    def text(self) -> str:
        return "+".join(self._keys)

    def _widths(self) -> list[int]:
        metrics = QtGui.QFontMetrics(self._font())
        pad = 14 if self._large else 8
        minimum = 40 if self._large else 26
        return [max(minimum, metrics.horizontalAdvance(key) + 2 * pad) for key in self._keys]

    def sizeHint(self) -> QtCore.QSize:
        if not self._keys:
            metrics = QtGui.QFontMetrics(style.font("body"))
            return QtCore.QSize(metrics.horizontalAdvance(self._placeholder) + 4, self._cap_height)
        widths = self._widths()
        return QtCore.QSize(sum(widths) + self._gap * (len(widths) - 1) + 1, self._cap_height)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            if not self._keys:
                painter.setFont(style.font("body"))
                painter.setPen(current.color("text3"))
                painter.drawText(QtCore.QRectF(self.rect()), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), self._placeholder)
                return
            painter.setFont(self._font())
            height = float(self._cap_height)
            lip = 3.0 if self._large else 2.0
            radius = 7.0 if self._large else float(style.RADIUS_KEYCAP)
            top = (self.height() - height) / 2.0
            x = 0.5
            for key, width in zip(self._keys, self._widths(), strict=True):
                outer = QtCore.QRectF(x, top + 0.5, width - 1.0, height - 1.0)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(current.color("keycap_bottom"))
                painter.drawRoundedRect(outer, radius, radius)
                face = QtCore.QRectF(outer.x(), outer.y(), outer.width(), outer.height() - lip)
                painter.setPen(QtGui.QPen(current.color("keycap_border"), 1.0))
                painter.setBrush(current.color("keycap"))
                painter.drawRoundedRect(face, radius, radius)
                painter.setPen(current.color("text"))
                painter.drawText(face, int(Qt.AlignmentFlag.AlignCenter), key)
                x += width + self._gap
        finally:
            painter.end()


class LogoMark(QtWidgets.QWidget):
    """The logo from data/brand, or the drawn placeholder mark."""

    def __init__(self, size: int = 28, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAccessibleName("Spells")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            brand.paint_logo(painter, QtCore.QRectF(self.rect()), current.color("brand_a"), current.color("brand_b"))
        finally:
            painter.end()


class Wordmark(QtWidgets.QWidget):
    """The Spells wordmark from data/brand with its letters in the text colour."""

    def __init__(self, height: int = 30, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._height = height
        self.setAccessibleName("Spells")
        aspect = brand.wordmark_aspect(style.palette().color("text")) or 3.0
        self.setFixedSize(round(height * aspect), height)

    def available(self) -> bool:
        return brand.wordmark_renderer(style.palette().color("text")) is not None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = painter_for(self)
        try:
            brand.paint_wordmark(painter, QtCore.QRectF(self.rect()), style.palette().color("text"))
        finally:
            painter.end()


class Chip(QtWidgets.QFrame):
    """A rounded pill with an optional code badge and an optional remove button."""

    remove_clicked = QtCore.Signal()

    def __init__(
        self,
        text: str,
        *,
        code: str | None = None,
        removable: bool = True,
        muted: bool = False,
        remove_tooltip: str = "Remove",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("role", "chip")
        if muted:
            self.setProperty("muted", "true")
        self.setFixedHeight(32)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4 if code else 12, 3, 6 if removable else 12, 3)
        layout.setSpacing(8)
        self.code_label: QtWidgets.QLabel | None = None
        if code:
            self.code_label = make_label(code.upper(), "caption_strong", parent=self)
            self.code_label.setProperty("role", "code")
            self.code_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.code_label.setFixedHeight(24)
            self.code_label.setMinimumWidth(34)
            layout.addWidget(self.code_label)
        self.text_label = make_label(text, "body", "secondary" if muted else None, parent=self)
        layout.addWidget(self.text_label)
        self.remove_button: IconButton | None = None
        if removable:
            self.remove_button = IconButton(style.Glyph.CLOSE, remove_tooltip, pixels=9, size=20, parent=self)
            self.remove_button.setProperty("role", "chipclose")
            self.remove_button.clicked.connect(self.remove_clicked.emit)
            layout.addWidget(self.remove_button)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def text(self) -> str:
        return self.text_label.text()


class ComboBox(QtWidgets.QComboBox):
    """A combo box that elides a long current text and draws a Windows 11 chevron."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFont(style.font("body"))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setMaxVisibleItems(12)
        view = QtWidgets.QListView(self)
        view.setUniformItemSizes(True)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setView(view)

    def showPopup(self) -> None:
        view = self.view()
        widest = max((self.fontMetrics().horizontalAdvance(self.itemText(i)) for i in range(self.count())), default=0)
        view.setMinimumWidth(min(max(self.width(), widest + 48), self.width() * 2))
        super().showPopup()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, option)
        field = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox, option, QtWidgets.QStyle.SubControl.SC_ComboBoxEditField, self
        )
        option.currentText = QtGui.QFontMetrics(self.font()).elidedText(
            option.currentText, Qt.TextElideMode.ElideRight, max(0, field.width() - 4)
        )
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, option)
        current = style.palette()
        arrow = QtCore.QRectF(self.width() - 30, 0, 22, self.height())
        style.draw_chevron(painter, arrow, current.color("text2" if self.isEnabled() else "disabled"), "down", 9.0)
        painter.end()


class SpinBox(QtWidgets.QSpinBox):
    """A spin box with small chevrons in its step buttons."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFont(style.font("body"))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        option = QtWidgets.QStyleOptionSpinBox()
        self.initStyleOption(option)
        current = style.palette()
        colour = current.color("text2" if self.isEnabled() else "disabled")
        painter = QtGui.QPainter(self)
        try:
            for control, direction in (
                (QtWidgets.QStyle.SubControl.SC_SpinBoxUp, "up"),
                (QtWidgets.QStyle.SubControl.SC_SpinBoxDown, "down"),
            ):
                rect = self.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_SpinBox, option, control, self)
                style.draw_chevron(painter, QtCore.QRectF(rect), colour, direction, 7.0)
        finally:
            painter.end()


class LevelBar(QtWidgets.QWidget):
    """A thin rounded level meter in the accent colour."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(12)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def level(self) -> float:
        return self._level

    def set_level(self, level: float) -> None:
        self._level = min(1.0, max(0.0, float(level)))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            track = QtCore.QRectF(0, (self.height() - 6) / 2, self.width(), 6)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(current.color("nav_selected"))
            painter.drawRoundedRect(track, 3, 3)
            if self._level > 0:
                fill = QtCore.QRectF(track.x(), track.y(), max(6.0, track.width() * self._level), track.height())
                painter.setBrush(current.color("accent"))
                painter.drawRoundedRect(fill, 3, 3)
        finally:
            painter.end()


class StepDots(QtWidgets.QWidget):
    """Progress dots for a short wizard; the current step is a wider accent pill."""

    def __init__(self, count: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._count = count
        self._current = 0
        self.setFixedSize(18 + (count - 1) * 12 + 4, 12)

    def set_current(self, index: int) -> None:
        self._current = index
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        current = style.palette()
        painter = painter_for(self)
        try:
            painter.setPen(Qt.PenStyle.NoPen)
            x = 0.0
            for index in range(self._count):
                if index == self._current:
                    painter.setBrush(current.color("accent"))
                    painter.drawRoundedRect(QtCore.QRectF(x, 3, 18, 6), 3, 3)
                    x += 18 + 6
                else:
                    painter.setBrush(current.color("text3"))
                    painter.drawEllipse(QtCore.QRectF(x, 3, 6, 6))
                    x += 6 + 6
        finally:
            painter.end()


# Flow layout -------------------------------------------------------------------------------------------


class FlowLayout(QtWidgets.QLayout):
    """Lays its items out left to right and wraps to the next line."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def clear(self) -> None:
        while self._items:
            item = self._items.pop()
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _arrange(self, rect: QtCore.QRect, *, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = area.x()
        y = area.y()
        line = 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line > 0:
                x = area.x()
                y += line + self._spacing
                line = 0
            if apply:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y() + margins.bottom()


__all__ = [
    "Badge",
    "Card",
    "CellHost",
    "Chip",
    "ComboBox",
    "Divider",
    "FlowLayout",
    "GlyphLabel",
    "IconButton",
    "InfoBar",
    "KeycapRow",
    "LevelBar",
    "LogoMark",
    "NavItem",
    "NavRail",
    "ScrollPage",
    "SectionHeader",
    "SegmentedControl",
    "SettingRow",
    "SpinBox",
    "StatusDot",
    "StepDots",
    "ToggleSwitch",
    "Wordmark",
    "make_button",
    "make_label",
    "set_prop",
]
