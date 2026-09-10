import json

import numpy as np

from raga_detector.compmusic import discover_recordings, load_pitch
from raga_detector.training import split_by_artist


def _recording(root, raga_id, raga_name, artist, title):
    info = root / "_info_"
    info.mkdir(parents=True, exist_ok=True)
    mapping_path = info / "ragaId_to_ragaName_mapping.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    mapping[raga_id] = raga_name
    mapping_path.write_text(json.dumps(mapping))

    base = root / "features" / raga_id / artist / "album" / title / title
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(base.with_suffix(".pitch"), [[0.0, 200.0], [0.01, 201.0]])
    base.with_suffix(".tonic").write_text("200.0")


def test_discovers_unicode_raga_and_loads_pitch(tmp_path):
    _recording(tmp_path, "r1", "Māyāmāḷavagauḷa", "Singer", "Song")
    recordings = discover_recordings(tmp_path)
    assert len(recordings) == 1
    assert recordings[0].raga == "mayamalavagowla"
    assert recordings[0].artist == "Singer"
    times, frequencies, tonic = load_pitch(recordings[0])
    assert times.tolist() == [0.0, 0.01]
    assert frequencies.tolist() == [200.0, 201.0]
    assert tonic == 200.0


def test_artist_split_has_no_leakage_and_covers_every_raga(tmp_path):
    for raga_id, raga_name in (("r1", "Kalyāṇi"), ("r2", "Mōhanaṁ")):
        for number in range(8):
            _recording(tmp_path, raga_id, raga_name, f"Singer-{number}", f"Song-{number}")
    recordings = discover_recordings(tmp_path)
    train, test = split_by_artist(recordings)
    assert {recording.artist for recording in train}.isdisjoint(
        {recording.artist for recording in test}
    )
    assert {recording.raga for recording in train} == {"kalyani", "mohanam"}
    assert {recording.raga for recording in test} == {"kalyani", "mohanam"}
