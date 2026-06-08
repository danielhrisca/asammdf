from time import perf_counter, sleep
from traceback import format_exc

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..ui.gps import Ui_GPSDisplay

try:
    from pyqtlet2 import L, MapWidget
    from PySide6.QtWebEngineCore import QWebEngineSettings

except:
    print(format_exc())


PROVIDERS = {
    "OpenStreetMap.DE": {
        "url": "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
    "OpenStreetMap.France": {
        "url": "https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        "attribution": '&copy; OpenStreetMap France | &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
    "OPNVKarte": {
        "url": "https://tileserver.memomaps.de/tilegen/{z}/{x}/{y}.png",
        "attribution": 'Map <a href="https://memomaps.de/">memomaps.de</a> <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
    "OpenTopoMap": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attribution": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
    },
    "Esri.WorldStreetMap": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012",
    },
    "Esri.WorldImagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    },
    "CartoDB.Positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    },
    "CartoDB.Voyager": {
        "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    },
    "TopPlusOpen.Color": {
        "url": "http://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/web/default/WEBMERCATOR/{z}/{y}/{x}.png",
        "attribution": 'Map data: &copy; <a href="http://www.govdata.de/dl-de/by-2-0">dl-de/by-2-0</a>',
    },
    "MtbMap": {
        "url": "http://tile.mtbmap.cz/mtbmap_tiles/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &amp; USGS',
    },
    "CyclOSM": {
        "url": "https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
        "attribution": '<a href="https://github.com/cyclosm/cyclosm-cartocss-style/releases" title="CyclOSM - Open Bicycle render">CyclOSM</a> | Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
}


class GPS(Ui_GPSDisplay, QtWidgets.QWidget):
    timestamp_changed_signal = QtCore.Signal(object, float)

    def __init__(self, latitude_channel, longitude_channel, zoom=15, *args, **kwargs):
        self.tile_provider = kwargs.pop("tile_provider", "TopPlusOpen.Color")
        self.tile_provider_url = PROVIDERS[self.tile_provider]["url"]
        self.tile_provider_attribution = PROVIDERS[self.tile_provider]["attribution"]

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        timebase = np.around(np.union1d(latitude_channel.timestamps, longitude_channel.timestamps), 9)
        self.latitude_signal = latitude_channel.interp(timebase)
        self.longitude_signal = longitude_channel.interp(timebase)

        if len(timebase):
            self.latitude = float(self.latitude_signal.samples[0])
            self.longitude = float(self.longitude_signal.samples[0])
        else:
            self.latitude = self.longitude = None

        self._min = self._max = 0

        self._inhibit = False

        if len(timebase):
            self._min = timebase[0]
            self._max = timebase[-1]
        else:
            self._min = float("inf")
            self._max = -float("inf")

        if self._min == float("inf"):
            self._min = self._max = 0

        self._timestamp = self._min

        self.timestamp.setRange(self._min, self._max)
        self.timestamp.setValue(self._min)
        self.min_t.setText(f"{self._min:.6f}s")
        self.max_t.setText(f"{self._max:.6f}s")

        self.mapWidget = MapWidget()
        self.mapWidget.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.map_layout.insertWidget(0, self.mapWidget)
        self.map_layout.setStretch(0, 1)

        self.map = L.map(self.mapWidget)
        self.map.setView([47.13698, 27.59774], zoom)

        L.tileLayer(self.tile_provider_url, {"attribution": self.tile_provider_attribution}).addTo(self.map)

        if len(timebase):
            line = L.polyline(np.column_stack([self.latitude_signal.samples, self.longitude_signal.samples]).tolist())
            line.addTo(self.map)

            self.map.setView([self.latitude, self.longitude], zoom)
            self.marker = L.marker([self.latitude, self.longitude])
            self.map.addLayer(self.marker)
        else:
            self.marker = None

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)
        self.set_timestamp()

        self.show()

    def _timestamp_changed(self, stamp):
        if not self._inhibit:
            self.set_timestamp(stamp)

    def _timestamp_slider_changed(self, stamp):
        if not self._inhibit:
            factor = stamp / 99999
            stamp = (self._max - self._min) * factor + self._min
            self.set_timestamp(stamp)

    def get_zoom(self):
        result = []

        def callback(*args):
            result.append(args[0])

        map_widget = self.map.getMapWidgetAtIndex(self.map.mapWidgetIndex)
        map_widget.page.runJavaScript("map.getZoom()", self.map.mapWidgetIndex, callback)

        app = QtWidgets.QApplication.instance()

        start = perf_counter()
        while not result and perf_counter() - start < 1:
            sleep(0.1)
            app.processEvents()

        return int(result[0]) if result else 15

    def set_timestamp(self, stamp=None):
        if stamp is None:
            stamp = self._timestamp

        if not (self._min <= stamp <= self._max):
            return

        try:
            self.latitude = float(self.latitude_signal.cut(stamp, stamp).samples[0])
            self.longitude = float(self.longitude_signal.cut(stamp, stamp).samples[0])
        except:
            return
        if self.marker is not None:
            self.marker.setLatLng([self.latitude, self.longitude])
            app = QtWidgets.QApplication.instance()
            app.processEvents()

        self._inhibit = True
        if self._min != self._max:
            val = int((stamp - self._min) / (self._max - self._min) * 99999)
            self.timestamp_slider.setValue(val)
        self.timestamp.setValue(stamp)
        self._inhibit = False
        self.timestamp_changed_signal.emit(self, stamp)

    def to_config(self):
        config = {
            "latitude_channel": self.latitude_signal.name,
            "longitude_channel": self.longitude_signal.name,
            "zoom": self.get_zoom(),
            "tile_provider": self.tile_provider,
            "tile_provider_url": self.tile_provider_url,
            "tile_provider_attribution": self.tile_provider_attribution,
        }

        return config
