
import argparse
import main
import demo


def test_run_demo(tmp_path, monkeypatch):
    video = tmp_path / "v.avi"
    video.write_bytes(b"0")
    out = tmp_path / "out.mp4"
    log = tmp_path / "log.txt"

    def dummy_main(args):
        out.touch()
        log.touch()

    monkeypatch.setattr(main, "main", dummy_main)

    args = argparse.Namespace(
        video=str(video),
        output=str(out),
        log=str(log),
        device="cpu",
        caption=False,
        flamingo=False,
        caption_model="nlpconnect/vit-gpt2-image-captioning",
    )
    demo.run_demo(args)
    assert out.exists()
    assert log.exists()
