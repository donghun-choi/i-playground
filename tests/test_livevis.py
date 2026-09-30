import http.client
import json
import unittest
import urllib.request
from urllib.parse import urlparse

from livevis import LiveVis


class LiveVisTest(unittest.TestCase):
    def start(self, **kwargs) -> LiveVis:
        viz = LiveVis("test", port=0, **kwargs).start()
        self.addCleanup(viz.stop)
        return viz

    def get_json(self, viz: LiveVis, path: str) -> dict:
        with urllib.request.urlopen(viz.url + path, timeout=5) as resp:
            return json.load(resp)

    def test_snapshot_keeps_logged_points(self):
        viz = self.start()
        viz.log(loss=1.0)
        viz.log(loss=0.5, acc=0.9)
        viz.log(5, loss=float("nan"))

        series = self.get_json(viz, "snapshot")["series"]

        self.assertEqual(series["loss"], [[0, 1.0], [1, 0.5], [5, None]])
        self.assertEqual(series["acc"], [[1, 0.9]])

    def test_max_points_drops_oldest(self):
        viz = self.start(max_points=3)
        for step in range(5):
            viz.log(step, x=step)

        self.assertEqual(self.get_json(viz, "snapshot")["series"]["x"], [[2, 2.0], [3, 3.0], [4, 4.0]])

    def test_index_page_is_served(self):
        viz = self.start()
        with urllib.request.urlopen(viz.url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn(b"EventSource", resp.read())

    def test_events_stream_sends_history_then_live_values(self):
        viz = self.start()
        viz.log(0, loss=1.0)

        conn = http.client.HTTPConnection("127.0.0.1", urlparse(viz.url).port, timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")

        name, data = read_event(resp)
        self.assertEqual(name, "init")
        self.assertEqual(data["series"]["loss"], [[0, 1.0]])

        viz.log(7, loss=0.25)
        name, data = read_event(resp)
        self.assertEqual(name, "log")
        self.assertEqual(data, {"step": 7, "metrics": {"loss": 0.25}})


def read_event(resp) -> tuple[str, dict]:
    name, data = None, None
    while True:
        line = resp.readline().decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: "):])
        elif line == "" and name is not None:
            return name, data


if __name__ == "__main__":
    unittest.main()
