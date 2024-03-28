import pathlib
from test.asammdf.gui.test_base import DragAndDrop
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets
from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt
from PySide6.QtGui import QInputDevice, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QWidget


class QMenuWrap(QtWidgets.QMenu):
    return_action = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exec(self, *args, **kwargs):
        if not self.return_action:
            return super().exec_(*args, **kwargs)
        return self.return_action


class TestPlotWidget(TestFileWidget):
    class Column:
        NAME = 0
        VALUE = 1
        UNIT = 2
        COMMON_AXIS = 3
        INDIVIDUAL_AXIS = 4

    measurement_file = str(pathlib.Path(TestFileWidget.resource, "ASAP2_Demo_V171.mf4"))

    def add_channel_to_plot(self, plot=None, channel_name=None, channel_index=None):
        if not plot and self.plot:
            plot = self.plot

        # Select channel
        channel_tree = self.widget.channels_tree
        channel_tree.clearSelection()

        selected_channel = self.find_channel(channel_tree, channel_name, channel_index)
        selected_channel.setSelected(True)
        if not channel_name:
            channel_name = selected_channel.text(self.Column.NAME)

        drag_position = channel_tree.visualItemRect(selected_channel).center()
        drop_position = plot.channel_selection.viewport().rect().center()

        # PreEvaluation
        DragAndDrop(
            src_widget=channel_tree,
            dst_widget=plot.channel_selection,
            src_pos=drag_position,
            dst_pos=drop_position,
        )
        self.processEvents(0.05)
        plot_channel = None
        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item and item.text(0) == channel_name:
                plot_channel = item
            iterator += 1

        return plot_channel

    def context_menu(self, action_text, position=None):
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap):
            mo_action = mock.MagicMock()
            mo_action.text.return_value = action_text
            QMenuWrap.return_action = mo_action

            if not position:
                QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            else:
                QtTest.QTest.mouseClick(
                    self.plot.channel_selection.viewport(),
                    QtCore.Qt.MouseButton.RightButton,
                    QtCore.Qt.KeyboardModifiers(),
                    position,
                )
            self.processEvents(0.01)

            while not mo_action.text.called:
                self.processEvents(0.02)

    def move_channel_to_group(self, plot=None, src=None, dst=None):
        if not plot and self.plot:
            plot = self.plot

        drag_position = plot.channel_selection.visualItemRect(src).center()
        drop_position = plot.channel_selection.visualItemRect(dst).center()

        DragAndDrop(
            src_widget=plot.channel_selection,
            dst_widget=plot.channel_selection,
            src_pos=drag_position,
            dst_pos=drop_position,
        )
        self.processEvents(0.05)

    def wheel_action(self, w: QWidget, x: float, y: float, angle_delta: int):
        """
        Used to simulate mouse wheel event.

        Parameters
        ----------
        w: widget - widget object
        x: float - x position of cursor
        y: float - y position of cursor
        angle_delta: int - physical wheel rotations units

        Returns
        -------

        """
        pos = QPointF(x, y)
        global_pos = self.widget.mapToGlobal(pos)
        x = global_pos.x()
        global_pos.setX(x + pos.x())

        pixel_d = QPoint(0, 0)
        angle_d = QPoint(0, angle_delta * 120)
        buttons = Qt.MouseButtons()
        modifiers = Qt.KeyboardModifiers()
        phase = Qt.ScrollPhase(0x0)
        inverted = False
        source = Qt.MouseEventSource(0x0)
        device = QPointingDevice(
            "core pointer",
            1,
            QInputDevice.DeviceType(0x1),
            QPointingDevice.PointerType(0x0),
            QInputDevice.Capabilities(),
            1,
            3,
        )
        # Create event
        event = QWheelEvent(pos, global_pos, pixel_d, angle_d, buttons, modifiers, phase, inverted, source, device)
        # Post event
        QCoreApplication.postEvent(w, event)
        self.assertTrue(event.isAccepted())
        QCoreApplication.processEvents()
