from sample_splitter import manifest


def test_write_then_read_round_trips_slices_and_skipped(tmp_path):
    original = manifest.Manifest(
        slices=[
            manifest.SliceRecord(source="track.flac", start_s=0.1, end_s=0.3, output_path="track_01.flac"),
            manifest.SliceRecord(source="track.flac", start_s=0.8, end_s=1.0, output_path="track_02.flac"),
        ],
        skipped=[manifest.SkippedRecord(source="demo.flac", reason="montage")],
    )
    path = tmp_path / "manifest.json"

    manifest.write(path, original)
    loaded = manifest.read(path)

    assert loaded == original
