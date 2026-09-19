"""spells.ui.widgets: the toggle, the segmented control, navigation, keycaps, chips and layout."""

from __future__ import annotations

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from spells.ui import brand
from spells.ui.widgets import (
    Badge,
    Card,
    Chip,
    ComboBox,
    FlowLayout,
    InfoBar,
    KeycapRow,
    LevelBar,
    LogoMark,
    NavRail,
    SegmentedControl,
    SettingRow,
    SpinBox,
    StatusDot,
    StepDots,
    ToggleSwitch,
    Wordmark,
)

from .test_ui_support import flush, qt_app


@pytest.fixture(scope="module")
def app():
    return qt_app()


def _rendered(widget: QtWidgets.QWidget) -> QtGui.QImage:
    widget.adjustSize()
    image = widget.grab().toImage()
    assert not image.isNull()
    return image


# Toggle switch ---------------------------------------------------------------------------------


def test_toggle_switch_is_a_checkable_button(app):
    switch = ToggleSwitch()
    seen: list[bool] = []
    switch.toggled.connect(seen.append)
    assert switch.isCheckable() and not switch.isChecked()
    switch.click()
    assert switch.isChecked() and seen == [True]
    switch.setChecked(False)
    assert seen == [True, False]
    switch.blockSignals(True)
    switch.setChecked(True)
    switch.blockSignals(False)
    assert seen == [True, False]
    assert switch.position() == 1.0
    switch.close()


def test_toggle_switch_keyboard_and_paint(app):
    switch = ToggleSwitch()
    switch.show()
    flush(app)
    QtWidgets.QApplication.sendEvent(
        switch, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Space, QtCore.Qt.KeyboardModifier.NoModifier, " ")
    )
    QtWidgets.QApplication.sendEvent(
        switch, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyRelease, QtCore.Qt.Key.Key_Space, QtCore.Qt.KeyboardModifier.NoModifier, " ")
    )
    assert switch.isChecked()
    off = ToggleSwitch(state_text=False)
    assert off.sizeHint().width() < switch.sizeHint().width()
    assert _rendered(switch).width() == switch.width()
    switch.setEnabled(False)
    _rendered(switch)
    switch.close()
    off.close()


# Segmented control -------------------------------------------------------------------------------


def test_segmented_control_offers_the_combo_calls(app):
    control = SegmentedControl()
    changes: list[int] = []
    control.currentIndexChanged.connect(changes.append)
    for code, label in (("100", "Last 100"), ("7d", "7 days"), ("off", "Off")):
        control.addItem(label, code)
    assert changes == [0]
    assert control.count() == 3
    assert [control.itemData(i) for i in range(3)] == ["100", "7d", "off"]
    assert control.findData("off") == 2 and control.findData("nope") == -1
    control.setCurrentIndex(2)
    assert control.currentData() == "off" and control.currentText() == "Off"
    control.setCurrentIndex(2)
    control.setCurrentIndex(9)
    assert changes == [0, 2]
    control.blockSignals(True)
    control.setCurrentIndex(1)
    control.blockSignals(False)
    assert changes == [0, 2] and control.currentIndex() == 1
    control.close()


def test_segmented_control_follows_clicks_and_arrow_keys(app):
    control = SegmentedControl()
    for label in ("Paste", "Type"):
        control.addItem(label, label.lower())
    control.resize(control.sizeHint())
    control.show()
    flush(app)
    rects = control._rects()
    centre = rects[1].center().toPoint()
    QtWidgets.QApplication.sendEvent(
        control,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(centre),
            QtCore.QPointF(control.mapToGlobal(centre)),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )
    assert control.currentData() == "type"
    QtWidgets.QApplication.sendEvent(
        control, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.NoModifier)
    )
    assert control.currentData() == "paste"
    _rendered(control)
    control.clear()
    assert control.count() == 0 and control.currentIndex() == -1
    control.close()


# Navigation -----------------------------------------------------------------------------------------


def test_nav_rail_selects_one_item_and_reports_clicks(app):
    rail = NavRail()
    rail.add_item("general", "General", "a")
    rail.add_item("history", "History", "b")
    rail.add_item("about", "About", "c", bottom=True)
    clicked: list[str] = []
    rail.current_changed.connect(clicked.append)
    rail.set_current("history")
    assert rail.current() == "history" and clicked == []
    rail.items["about"].click()
    assert clicked == ["about"]
    assert rail.current() == "about"
    assert not rail.items["history"].isChecked()
    assert rail.labels() == ["General", "History", "About"]
    _rendered(rail)
    rail.close()


# Keycaps, chips, badges ----------------------------------------------------------------------------


def test_keycaps_keep_the_familiar_label(app):
    keys = KeycapRow(["Ctrl", "Alt", "D"])
    assert keys.text() == "Ctrl+Alt+D"
    assert keys.keys() == ["Ctrl", "Alt", "D"]
    wide = keys.sizeHint().width()
    keys.set_keys(["Ctrl"])
    assert keys.sizeHint().width() < wide
    keys.set_keys([])
    assert keys.text() == "" and keys.accessibleName() == "Not set"
    large = KeycapRow(["Ctrl"], large=True)
    assert large.sizeHint().height() > KeycapRow(["Ctrl"]).sizeHint().height()
    _rendered(large)
    keys.close()
    large.close()


def test_chip_reports_remove_and_can_be_fixed(app):
    chip = Chip("German", code="de")
    removed: list[int] = []
    chip.remove_clicked.connect(lambda: removed.append(1))
    assert chip.code_label.text() == "DE" and chip.text() == "German"
    chip.remove_button.click()
    assert removed == [1]
    fixed = Chip("English", removable=False, muted=True)
    assert fixed.remove_button is None
    assert fixed.property("muted") == "true"
    _rendered(chip)
    chip.close()
    fixed.close()


def test_badge_info_bar_dot_and_meter(app):
    badge = Badge("Ready", "ok")
    badge.set_badge("caution", "On CPU")
    assert badge.text() == "On CPU" and badge.property("badge") == "caution"
    bar = InfoBar("", "caution")
    assert bar.isHidden()
    bar.set_text("Something to know")
    assert not bar.isHidden() and bar.text() == "Something to know"
    bar.set_kind("info")
    assert bar.property("infobar") == "info"
    dot = StatusDot()
    for level in StatusDot.LEVELS:
        dot.set_level(level)
        _rendered(dot)
    meter = LevelBar()
    meter.set_level(3.0)
    assert meter.level() == 1.0
    meter.set_level(-1.0)
    assert meter.level() == 0.0
    dots = StepDots(3)
    dots.set_current(2)
    _rendered(dots)
    for widget in (badge, bar, dot, meter, dots):
        widget.close()


# Cards, rows and layout ----------------------------------------------------------------------------


def test_card_rows_are_separated_and_cleared(app):
    card = Card()
    first = card.add_row(SettingRow("One", "First row"))
    card.add_row(SettingRow("Two"))
    assert card.rows()[0] is first
    assert card.body.count() == 3
    card.clear()
    flush(app)
    assert card.rows() == [] and card.body.count() == 0
    card.close()


def test_setting_row_description_hides_when_empty(app):
    row = SettingRow("Title", "", QtWidgets.QPushButton("Go"))
    assert row.description_label.isHidden()
    row.set_description("Now with words")
    assert not row.description_label.isHidden()
    assert row.minimumHeight() == 48
    row.close()


def test_flow_layout_wraps_when_narrow(app):
    host = QtWidgets.QWidget()
    flow = FlowLayout(host, spacing=8)
    for _index in range(6):
        flow.addItem(QtWidgets.QSpacerItem(80, 30, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed))
    assert flow.count() == 6
    assert flow.heightForWidth(1000) == 30
    assert flow.heightForWidth(180) > 60
    flow.clear()
    assert flow.count() == 0
    host.close()


def test_combo_and_spin_box_paint_their_chevrons(app):
    combo = ComboBox()
    combo.addItem("Automatic (NVIDIA GeForce RTX 5060 Laptop GPU with a very long name)")
    combo.resize(160, 32)
    spin = SpinBox()
    spin.resize(120, 32)
    for widget in (combo, spin):
        widget.show()
        flush(app)
        image = widget.grab().toImage()
        assert not image.isNull()
        widget.close()


# Brand marks ---------------------------------------------------------------------------------------


def test_logo_falls_back_to_the_drawn_mark(app, monkeypatch, tmp_path):
    monkeypatch.setattr(brand, "brand_path", lambda name: tmp_path / name)
    image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    used_svg = brand.paint_logo(painter, QtCore.QRectF(0, 0, 64, 64), QtGui.QColor("#5B3CF0"), QtGui.QColor("#16B4E8"))
    painter.end()
    assert used_svg is False
    assert image.pixelColor(32, 4).alpha() > 0
    assert brand.tinted_mark(32, QtGui.QColor("white")) is None
    assert brand.wordmark_renderer(QtGui.QColor("white")) is None
    rail = NavRail()
    assert rail.logo is not None and rail.brand_label.text() == "Spells"
    rail.close()


def test_logo_and_marks_load_from_svg_files(app, monkeypatch, tmp_path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100" color="#15212A">'
        '<rect x="0" y="0" width="200" height="100" fill="currentColor"/></svg>'
    )
    for name in (brand.LOGO, brand.MONO_MARK, brand.WORDMARK):
        (tmp_path / name).write_text(svg, encoding="utf-8")
    monkeypatch.setattr(brand, "brand_path", lambda name: tmp_path / name)
    image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    assert brand.paint_logo(painter, QtCore.QRectF(0, 0, 64, 64), QtGui.QColor("red"), QtGui.QColor("blue"))
    painter.end()
    mark = brand.tinted_mark(40, QtGui.QColor("#FFFFFF"))
    assert mark is not None and mark.pixelColor(20, 20).name() == "#ffffff"
    assert brand.wordmark_aspect(QtGui.QColor("#FFFFFF")) == pytest.approx(2.0)
    wordmark = Wordmark(30)
    assert wordmark.available() and wordmark.width() == 60
    _rendered(LogoMark(40))
    wordmark.close()


def test_a_broken_svg_is_treated_as_missing(app, monkeypatch, tmp_path):
    (tmp_path / brand.LOGO).write_text("not an svg", encoding="utf-8")
    monkeypatch.setattr(brand, "brand_path", lambda name: tmp_path / name)
    assert brand.renderer(brand.LOGO) is None
