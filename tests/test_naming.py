from pathlib import Path

from sample_splitter import naming


def test_sanitize_lowercases_and_replaces_spaces():
    assert naming.sanitize("Kick Drum") == "kick_drum"


def test_sanitize_replaces_special_characters_with_a_separator():
    assert naming.sanitize("kick/snare!.wav") == "kick_snare_wav"


def test_sanitize_does_not_collapse_labels_differing_only_by_punctuation():
    assert naming.sanitize("kick.drum") != naming.sanitize("kickdrum")


def test_sanitize_collapses_leading_and_trailing_underscores():
    assert naming.sanitize("  -kick- ") == "kick"


def test_sanitize_falls_back_to_unknown_for_an_all_unsafe_string():
    assert naming.sanitize("!!!") == "unknown"


def test_next_index_starts_at_one_for_an_empty_bucket():
    assert naming.next_index(set()) == 1


def test_next_index_returns_the_smallest_unused_index():
    assert naming.next_index({1, 2, 4}) == 3


def test_relative_path_builds_category_subtype_filename():
    path = naming.relative_path("drums", "kick", 1, review=False)

    assert path == Path("drums/kick/kick_01.flac")


def test_relative_path_routes_review_results_under_review_root():
    path = naming.relative_path("drums", "kick", 1, review=True)

    assert path == Path("_review/drums/kick/kick_01.flac")


def test_relative_path_sanitizes_category_and_subtype():
    path = naming.relative_path("Drums!", "Kick Drum", 3, review=False)

    assert path == Path("drums/kick_drum/kick_drum_03.flac")


def test_relative_path_zero_pads_single_digit_indices():
    path = naming.relative_path("drums", "kick", 7, review=False)

    assert path.name == "kick_07.flac"


def test_relative_path_does_not_truncate_indices_past_99():
    path = naming.relative_path("drums", "kick", 123, review=False)

    assert path.name == "kick_123.flac"


def test_is_review_true_below_threshold():
    assert naming.is_review(0.4, threshold=0.5) is True


def test_is_review_false_at_threshold_boundary():
    assert naming.is_review(0.5, threshold=0.5) is False


def test_is_review_false_above_threshold():
    assert naming.is_review(0.9, threshold=0.5) is False


def test_parse_index_reads_the_trailing_number():
    assert naming.parse_index("drums/kick/kick_07.flac") == 7


def test_parse_index_reads_a_three_digit_number():
    assert naming.parse_index("_review/drums/kick/kick_123.flac") == 123


def test_parse_index_returns_none_for_an_unrecognised_name():
    assert naming.parse_index("drums/kick/not_a_match.flac") is None
