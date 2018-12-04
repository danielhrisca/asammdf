# -*- coding: utf-8 -*-

try:
    from PyQt5.QtWidgets import QTreeWidgetItem
except ImportError:
    from PyQt4.QtGui import QTreeWidgetItem


class TreeItem(QTreeWidgetItem):
    def __init__(self, entry, *args, **kwargs):

        super(TreeItem, self).__init__(*args, **kwargs)

        self.entry = entry
