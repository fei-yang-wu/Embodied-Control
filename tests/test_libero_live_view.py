from __future__ import annotations

import json
from urllib.request import urlopen

from embodied_control.sim.libero_eval import LiberoLiveView


def test_libero_live_view_serves_page_status_and_snapshot():
    view = LiberoLiveView(port=0)
    try:
        url = view.start()
        jpeg = b"\xff\xd8fake-jpeg\xff\xd9"
        view._publish_encoded(
            {"agentview_image": jpeg},
            {"phase": "running", "episode": 2, "step": 17, "language": "pick up bowl"},
        )

        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert b"LIBERO live view" in response.read()
        with urlopen(f"{url}status.json", timeout=2) as response:
            status = json.load(response)
        assert status["phase"] == "running"
        assert status["episode"] == 2
        assert status["step"] == 17
        assert status["cameras"] == ["agentview_image"]
        with urlopen(f"{url}snapshot/agentview_image.jpg", timeout=2) as response:
            assert response.headers.get_content_type() == "image/jpeg"
            assert response.read() == jpeg
    finally:
        view.close()
