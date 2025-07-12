import argparse
import urllib.request

import main


def test_main_pipeline(tmp_path, monkeypatch):
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
        "Megamind.avi"
    )
    input_video = tmp_path / "test.avi"
    urllib.request.urlretrieve(url, str(input_video))
    out_video = tmp_path / 'out.mp4'
    log_file = tmp_path / 'log.txt'

    class DummyDetector:
        def __init__(self, *a, **k):
            pass
        def detect(self, frame):
            return [{"bbox": [10, 10, 20, 20], "conf": 0.9, "label": "car"}]

    class DummyCaption:
        def caption(self, frame):
            return "dummy"

    monkeypatch.setattr(main, 'YOLODetector', DummyDetector)
    monkeypatch.setattr(main, 'CaptionGenerator', lambda: DummyCaption())

    args = argparse.Namespace(
        input=str(input_video),
        output=str(out_video),
        log=str(log_file),
        model='fake.pt',
        device='cpu',
        caption=True,
        simple=False,
    )

    main.main(args)

    assert out_video.exists()
    assert log_file.exists()
    lines = log_file.read_text().splitlines()
    assert len(lines) > 0
    assert all(line.endswith('Free') for line in lines)
