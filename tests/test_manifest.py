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


def test_write_then_read_round_trips_naming_records(tmp_path):
    original = manifest.NamingManifest(
        names=[
            manifest.NameRecord(
                source="track.wav_01.flac",
                category="drums",
                subtype="kick",
                confidence=0.82,
                review=False,
                output_path="drums/kick/kick_01.flac",
            ),
            manifest.NameRecord(
                source="track.wav_02.flac",
                category="fx",
                subtype="noise",
                confidence=0.1,
                review=True,
                output_path="_review/fx/noise/noise_01.flac",
            ),
        ]
    )
    path = tmp_path / "naming.json"

    manifest.write_naming(path, original)
    loaded = manifest.read_naming(path)

    assert loaded == original
