from functools import partial
import weakref

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PySide6 import QtCore, QtGui, QtWidgets


class ViewBoxMenu(QtWidgets.QMenu):
    pan = "Pan mode"
    cursor = "Cursor mode"

    def __init__(self, view):
        super().__init__()

        self._settings = QtCore.QSettings()

        self.view = weakref.ref(
            view
        )  ## keep weakref to view to avoid circular reference (don't know why, but this prevents the ViewBox from being collected)
        self.valid = False  ## tells us whether the ui needs to be updated
        self.viewMap = weakref.WeakValueDictionary()  ## weakrefs to all views listed in the link combos

        # mouse mode
        self.mouse_mode_menu = QtWidgets.QMenu("Mouse Mode")
        group = QtGui.QActionGroup(self)

        pan = QtGui.QAction(ViewBoxMenu.pan, self.mouse_mode_menu)
        cursor = QtGui.QAction(ViewBoxMenu.cursor, self.mouse_mode_menu)
        self.mouse_mode_menu.addAction(pan)
        self.mouse_mode_menu.addAction(cursor)
        pan.triggered.connect(partial(self.set_mouse_mode, "pan"))
        cursor.triggered.connect(partial(self.set_mouse_mode, "cursor"))
        pan.setCheckable(True)
        cursor.setCheckable(True)
        pan.setActionGroup(group)
        cursor.setActionGroup(group)
        self.mouseModes = [pan, cursor]
        self.addMenu(self.mouse_mode_menu)

        # X zoom mode
        self.x_zoom_mode_menu = QtWidgets.QMenu("X-axis zoom mode")
        group = QtGui.QActionGroup(self)

        center_on_cursor = QtGui.QAction("Center on cursor", self.x_zoom_mode_menu)
        center_on_cursor.setCheckable(True)
        center_on_cursor.setActionGroup(group)
        self.x_zoom_mode_menu.addAction(center_on_cursor)

        center_on_mouse = QtGui.QAction("Center on mouse position", self.x_zoom_mode_menu)
        center_on_mouse.setCheckable(True)
        center_on_mouse.setActionGroup(group)
        self.x_zoom_mode_menu.addAction(center_on_mouse)

        if self._settings.value("zoom_x_center_on_cursor", True, type=bool):
            center_on_cursor.setChecked(True)
        else:
            center_on_mouse.setChecked(True)
        center_on_cursor.triggered.connect(partial(self.set_x_zoom_mode, True))
        center_on_mouse.triggered.connect(partial(self.set_x_zoom_mode, False))

        self.addMenu(self.x_zoom_mode_menu)

        # Y zoom mode
        self.y_zoom_mode_menu = QtWidgets.QMenu("Y-axis zoom mode")
        group = QtGui.QActionGroup(self)

        center_on_cursor = QtGui.QAction("Center on cursor", self.y_zoom_mode_menu)
        center_on_cursor.setCheckable(True)
        center_on_cursor.setActionGroup(group)
        self.y_zoom_mode_menu.addAction(center_on_cursor)

        center_on_mouse = QtGui.QAction("Center on mouse position", self.y_zoom_mode_menu)
        center_on_mouse.setCheckable(True)
        center_on_mouse.setActionGroup(group)
        self.y_zoom_mode_menu.addAction(center_on_mouse)

        pin_zero_level = QtGui.QAction("Pin zero level", self.y_zoom_mode_menu)
        pin_zero_level.setCheckable(True)
        pin_zero_level.setActionGroup(group)
        self.y_zoom_mode_menu.addAction(pin_zero_level)

        zoom_y_mode = self._settings.value("zoom_y_mode", "")
        if not zoom_y_mode:
            if self._settings.value("zoom_y_center_on_cursor", True, type=bool):
                zoom_y_mode = "center_on_cursor"
            else:
                zoom_y_mode = "center_on_mouse"

        if zoom_y_mode == "pin_zero_level":
            pin_zero_level.setChecked(True)
        elif zoom_y_mode == "center_on_cursor":
            center_on_cursor.setChecked(True)
        elif zoom_y_mode == "center_on_mouse":
            center_on_mouse.setChecked(True)

        center_on_cursor.triggered.connect(partial(self.set_y_zoom_mode, "center_on_cursor"))
        center_on_mouse.triggered.connect(partial(self.set_y_zoom_mode, "center_on_mouse"))
        pin_zero_level.triggered.connect(partial(self.set_y_zoom_mode, "pin_zero_level"))

        self.addMenu(self.x_zoom_mode_menu)

        self.addMenu(self.y_zoom_mode_menu)

        self.view().sigStateChanged.connect(self.viewStateChanged)
        self.updateState()

    def viewStateChanged(self):
        self.valid = False
        self.updateState()

    def updateState(self):
        state = self.view().getState(copy=False)
        if state["mouseMode"] == ViewBoxWithCursor.PanMode:
            self.mouseModes[0].setChecked(True)
        elif state["mouseMode"] == ViewBoxWithCursor.CursorMode:
            self.mouseModes[1].setChecked(True)

        self.valid = True

    def popup(self, *args):
        if not self.valid:
            self.updateState()
        QtWidgets.QMenu.popup(self, *args)

    def set_mouse_mode(self, mode):
        self.view().setLeftButtonAction(mode)

    def set_x_zoom_mode(self, on_cursor=True):
        self._settings.setValue("zoom_x_center_on_cursor", on_cursor)

    def set_y_zoom_mode(self, zoom_mode="center_on_cursor"):
        if zoom_mode != "pin_zero_level":
            self._settings.setValue("zoom_y_center_on_cursor", zoom_mode == "center_on_cursor")
        self._settings.setValue("zoom_y_mode", zoom_mode)


class ViewBoxWithCursor(pg.ViewBox):
    PanMode = 3
    CursorMode = 2
    RectMode = 1

    sigCursorMoved = QtCore.Signal(object)
    sigZoomChanged = QtCore.Signal(object)
    sigZoomFinished = QtCore.Signal(object)
    hover_at = QtCore.Signal(object)

    X_zoom = QtCore.QKeyCombination(
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        QtCore.Qt.Key.Key_Shift,
    ).toCombined()

    Y_zoom = QtCore.QKeyCombination(
        QtCore.Qt.KeyboardModifier.AltModifier,
        QtCore.Qt.Key.Key_Alt,
    ).toCombined()

    XY_zoom = (
        QtCore.QKeyCombination(
            QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier,
            QtCore.Qt.Key.Key_Alt,
        ).toCombined(),
        QtCore.QKeyCombination(
            QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier,
            QtCore.Qt.Key.Key_Shift,
        ).toCombined(),
    )

    def __init__(self, plot, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setAcceptHoverEvents(True)

        self.menu.setParent(None)
        self.menu.deleteLater()
        self.menu = None
        self.menu = ViewBoxMenu(self)

        self.zoom = None
        self.zoom_start = None
        self._matrixNeedsUpdate = True
        self.updateMatrix()

        self.cursor = None
        self.plot = plot

        self._settings = QtCore.QSettings()

    def close(self):
        self.cursor = None
        self.plot = None
        super().close()

    def __repr__(self):
        return "ASAM ViewBox"

    def hoverMoveEvent(self, event):
        self.hover_at.emit(self.mapSceneToView(event.scenePos()))

    def setMouseMode(self, mode):
        """Set the mouse interaction mode. `mode` must be either
        ViewBoxWithCursor.PanMode, ViewBoxWithCursor.CursorMode or
        ViewBoxWithCursor.RectMode.
        In PanMode, the left mouse button pans the view and the right button
        scales.
        In RectMode, the left button draws a rectangle which updates the visible
        region (this mode is more suitable for single-button mice).
        """
        if mode not in [
            ViewBoxWithCursor.PanMode,
            ViewBoxWithCursor.CursorMode,
            ViewBoxWithCursor.RectMode,
        ]:
            raise Exception(
                "Mode must be ViewBoxWithCursor.PanMode, ViewBoxWithCursor.RectMode or ViewBoxWithCursor.CursorMode"
            )
        self.state["mouseMode"] = mode
        self.sigStateChanged.emit(self)

    def setLeftButtonAction(self, mode="rect"):  ## for backward compatibility
        if mode.lower() == "rect":
            self.setMouseMode(ViewBoxWithCursor.RectMode)
        elif mode.lower() == "pan":
            self.setMouseMode(ViewBoxWithCursor.PanMode)
        elif mode.lower() == "cursor":
            self.setMouseMode(ViewBoxWithCursor.CursorMode)
        else:
            raise Exception(
                f'graphicsItems:ViewBox:setLeftButtonAction: unknown mode = {mode} (Options are "pan", "cursor" and "rect")'
            )

    def mouseDragEvent(self, ev, axis=None, ignore_cursor=False):
        ## if axis is specified, event will only affect that axis.
        ev.accept()  ## we accept all buttons

        pos = ev.scenePos()
        dif = pos - ev.lastScenePos()
        dif = dif * -1

        ## Ignore axes if mouse is disabled
        mask = np.array(self.state["mouseEnabled"], dtype=np.float64)
        if axis is not None:
            mask[1 - axis] = 0.0

        if self.state["mouseMode"] == ViewBoxWithCursor.CursorMode and not ignore_cursor:
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                self.sigCursorMoved.emit(ev)
                if self.zoom_start is not None:
                    end = self.mapSceneToView(ev.scenePos())
                    self.sigZoomChanged.emit((self.zoom_start, end, self.zoom))
                    if ev.isFinish():
                        self.sigZoomFinished.emit((self.zoom_start, end, self.zoom))
                        self.zoom_start = None
                        self.sigZoomChanged.emit(None)

            else:
                tr = self.childGroup.transform()
                tr = fn.invertQTransform(tr)
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                mask[1] = 0

                self._resetTarget()
                if x is not None:
                    self.translateBy(x=x, y=0)
                self.sigRangeChangedManually.emit(mask)

        else:
            ## Scale or translate based on mouse button
            if ev.button() in [
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.MiddleButton,
            ]:
                tr = self.childGroup.transform()
                tr = fn.invertQTransform(tr)
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                self._resetTarget()
                if x is not None or y is not None:
                    self.translateBy(x=x, y=y)
                self.sigRangeChangedManually.emit(mask)

            elif ev.button() & QtCore.Qt.MouseButton.RightButton:
                if self.state["aspectLocked"] is not False:
                    mask[0] = 0

                dif = ev.screenPos() - ev.lastScreenPos()
                dif = np.array([dif.x(), dif.y()])
                dif[0] *= -1
                s = ((mask * 0.02) + 1) ** dif

                tr = self.childGroup.transform()
                tr = fn.invertQTransform(tr)

                x = s[0] if mask[0] == 1 else None
                y = s[1] if mask[1] == 1 else None

                center = pg.Point(tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton)))
                self._resetTarget()
                self.scaleBy(x=x, y=y, center=center)
                self.sigRangeChangedManually.emit(mask)

    def keyPressEvent(self, ev):
        if self.zoom_start is None:
            self.zoom = ev.keyCombination().toCombined()
        ev.ignore()

    def keyReleaseEvent(self, ev):
        if self.zoom_start is None:
            self.zoom = None
        ev.ignore()

    def mousePressEvent(self, ev):
        if self.state["mouseMode"] == ViewBoxWithCursor.CursorMode and self.zoom in (
            self.X_zoom,
            self.Y_zoom,
            *self.XY_zoom,
        ):
            self.zoom_start = self.mapSceneToView(ev.scenePos())

        ev.ignore()

    def setXRange(self, min, max, padding=None, update=True):
        min = round(min, 12)
        max = round(max, 12)
        return super().setXRange(min, max, padding, update)

    def updateScaleBox(self, p1, p2):
        r = QtCore.QRectF(p1, p2)
        r = self.childGroup.mapRectFromScene(r)
        self.rbScaleBox.setPos(r.topLeft())
        tr = QtGui.QTransform.fromScale(r.width(), r.height())
        self.rbScaleBox.setTransform(tr)
        self.rbScaleBox.show()

    def wheelEvent(self, ev, axis=None):
        if self.state["mouseMode"] == ViewBoxWithCursor.CursorMode:
            mask = [True, False]
        else:
            if axis in (0, 1):
                mask = [False, False]
                mask[axis] = self.state["mouseEnabled"][axis]
            else:
                mask = self.state["mouseEnabled"][:]

        pos = ev.pos()

        factor = self._settings.value("zoom_wheel_factor", 0.165, type=float)

        if ev.delta() > 0:
            s = [(None if m is False else 1 / (1 + factor)) for m in mask]
        else:
            s = [(None if m is False else 1 + factor) for m in mask]

        if any(np.isnan(v) for v in s if v is not None):
            return

        center = pg.Point(fn.invertQTransform(self.childGroup.transform()).map(pos))

        self._resetTarget()

        if s is not None:
            x, y = s[0], s[1]

        affect = [x is not None, y is not None]
        if not any(affect):
            return

        scale = pg.Point([1.0 if x is None else x, 1.0 if y is None else y])

        if self.state["aspectLocked"] is not False:
            scale[0] = scale[1]

        vr = self.targetRect()
        if center is None:
            center = pg.Point(vr.center())
        else:
            center = pg.Point(center)

        tl = center + (vr.topLeft() - center) * scale
        br = center + (vr.bottomRight() - center) * scale

        x_range = tl.x(), br.x()
        y_range = br.y(), tl.y()

        if (
            self._settings.value("zoom_x_center_on_cursor", True, type=bool)
            and self.cursor is not None
            and self.cursor.isVisible()
        ):
            delta = x_range[1] - x_range[0]

            pos = self.cursor.value()
            x_range = pos - delta / 2, pos + delta / 2

        zoom_y_mode = self._settings.value("zoom_y_mode", "")

        if not zoom_y_mode:
            if self._settings.value("zoom_y_center_on_cursor", False, type=bool):
                zoom_y_mode = "center_on_cursor"
            else:
                zoom_y_mode = "center_on_mouse"

        if zoom_y_mode == "pin_zero_level":
            y_pos_val, sig_y_bottom, sig_y_top = self.plot.value_at_cursor()
            delta_proc = sig_y_top / (sig_y_top - sig_y_bottom)

            delta = sig_y_top - sig_y_bottom

            if ev.delta() > 0:
                delta *= 1 / (1 + factor)
            else:
                delta *= 1 + factor

            end = delta_proc * delta
            start = end - delta

            y_range = start, end

        elif zoom_y_mode == "center_on_cursor":
            y_pos_val, sig_y_bottom, sig_y_top = self.plot.value_at_cursor()
            if isinstance(y_pos_val, (int, float)):
                delta = y_range[1] - y_range[0]
                y_range = y_pos_val - delta / 2, y_pos_val + delta / 2

        if not mask[0]:
            x_range = None

        if not mask[1]:
            y_range = None

        self.setRange(xRange=x_range, yRange=y_range, padding=0)

        ev.accept()
        self.sigRangeChangedManually.emit(mask)

    def vertical_zoom(self, zoom_in=True):
        if not self.state["mouseEnabled"][1]:
            return

        factor = self._settings.value("zoom_wheel_factor", 0.165, type=float)

        if zoom_in:
            scale = 1 / (1 + factor)
        else:
            scale = 1 + factor

        zoom_y_mode = self._settings.value("zoom_y_mode", "")

        if not zoom_y_mode:
            if self._settings.value("zoom_y_center_on_cursor", False, type=bool):
                zoom_y_mode = "center_on_cursor"
            else:
                zoom_y_mode = "center_on_mouse"

        selected_uuids, current_uuid = self.plot.selected_items()
        viewbox_y_range = None

        if zoom_y_mode == "pin_zero_level":
            for uuid in selected_uuids:
                sig, idx = self.plot.signal_by_uuid(uuid)
                y_pos_val, sig_y_bottom, sig_y_top = self.plot.value_at_cursor(uuid=uuid)
                if isinstance(y_pos_val, (int, float)):
                    delta_proc = sig_y_top / (sig_y_top - sig_y_bottom)

                    delta = (sig_y_top - sig_y_bottom) * scale

                    end = delta_proc * delta
                    start = end - delta
                    y_range = start, end

                    if uuid == current_uuid:
                        viewbox_y_range = y_range
                    else:
                        sig.y_range = y_range

        elif zoom_y_mode == "center_on_cursor":
            for uuid in selected_uuids:
                sig, idx = self.plot.signal_by_uuid(uuid)
                y_pos_val, sig_y_bottom, sig_y_top = self.plot.value_at_cursor(uuid=uuid)
                if isinstance(y_pos_val, (int, float)):
                    delta = (sig_y_top - sig_y_bottom) * scale
                    new_y_range = y_pos_val - delta / 2, y_pos_val + delta / 2

                    if uuid == current_uuid:
                        viewbox_y_range = new_y_range
                    else:
                        sig.y_range = new_y_range

        else:
            for uuid in selected_uuids:
                sig, idx = self.plot.signal_by_uuid(uuid)
                y_pos_val, sig_y_bottom, sig_y_top = self.plot.value_at_cursor(uuid=uuid)

                center = (sig_y_top + sig_y_bottom) / 2
                half_range = (sig_y_top - sig_y_bottom) * scale / 2

                new_y_range = center - half_range, center + half_range

                if uuid == current_uuid:
                    viewbox_y_range = new_y_range
                else:
                    sig.y_range = new_y_range

        if viewbox_y_range is not None:
            self.setRange(xRange=None, yRange=viewbox_y_range, padding=0)
            mask = [False, True]
            self.sigRangeChangedManually.emit(mask)
