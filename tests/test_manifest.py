from peagle_q.manifest import TensorManifestEntry


def test_manifest_dict_round_trip() -> None:
    entry = TensorManifestEntry(
        prompt_id="sample-1",
        dataset_index=0,
        prompt_sha256="abc",
        teacher_model_path="teacher",
        teacher_revision=None,
        tokenizer_path="teacher",
        tokenizer_revision=None,
        quantization="none",
        layer_indices=[4, 16, 28],
        tensor_path="tensor.pt",
        prompt_length=32,
        created_at="2026-04-09T00:00:00+00:00",
    )
    loaded = TensorManifestEntry.from_dict(entry.to_dict())
    assert loaded == entry
