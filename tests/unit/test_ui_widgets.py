"""spells.ui.widgets: the toggle, the segmented control, navigation, keycaps, chips and layout."""

from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from spells.ui import brand
from spells.ui.widgets import (
    Chip,
    FlowLayout,
    NavRail,
    SegmentedControl,
    ToggleSwitch,
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


def test_a_brand_file_elsewhere_with_the_same_mtime_is_read_afresh(app, monkeypatch, tmp_path):
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10"><rect width="20" height="10"/></svg>'
    first = tmp_path / "first"
    second = tmp_path / "second"
    for folder, text in ((first, svg), (second, "not an svg")):
        folder.mkdir()
        for name in (brand.LOGO, brand.WORDMARK):
            (folder / name).write_text(text, encoding="utf-8")
    for name in (brand.LOGO, brand.WORDMARK):
        stamp = (first / name).stat().st_mtime_ns
        os.utime(second / name, ns=(stamp, stamp))
    white = QtGui.QColor("#FFFFFF")
    monkeypatch.setattr(brand, "brand_path", lambda name: first / name)
    assert brand.renderer(brand.LOGO) is not None
    assert brand.wordmark_renderer(white) is not None
    monkeypatch.setattr(brand, "brand_path", lambda name: second / name)
    assert brand.renderer(brand.LOGO) is None
    assert brand.wordmark_renderer(white) is None
