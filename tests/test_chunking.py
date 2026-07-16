from src.common import (
    MAX_JOINT_TOKENS,
    make_chunk_plan,
    partition_by_mass,
    split_sentences,
)


def ws_len(text: str) -> int:
    return len(text.split())


def test_short_segment_passes_through():
    plan = make_chunk_plan("A small source.", "Eine kleine Quelle.", ws_len, budget=220)
    assert plan.n_chunks == 1
    assert not plan.over_limit
    assert not plan.oversized_chunk


def test_long_segment_is_chunked_within_budget():
    src = " ".join(f"Sentence number {i} has exactly seven words here." for i in range(120))
    mt = " ".join(f"Satz nummer {i} hat genau sieben woerter hier." for i in range(120))
    budget = 100
    plan = make_chunk_plan(src, mt, ws_len, budget=budget)
    assert plan.over_limit
    assert plan.n_chunks > 1
    assert len(plan.src_chunks) == len(plan.mt_chunks)
    max_sent = 9  # longest single sentence in this synthetic text
    for chunk in plan.src_chunks + plan.mt_chunks:
        assert ws_len(chunk) <= budget + max_sent
    # nothing lost: chunk word counts sum back to the original
    assert sum(ws_len(c) for c in plan.src_chunks) == ws_len(src)
    assert sum(ws_len(c) for c in plan.mt_chunks) == ws_len(mt)


def test_unsplittable_giant_sentence_is_flagged_not_hidden():
    src = "word " * 900  # no sentence-final punctuation: one giant "sentence"
    mt = "wort " * 900
    plan = make_chunk_plan(src.strip(), mt.strip(), ws_len, budget=220)
    assert plan.over_limit
    assert plan.oversized_chunk
    assert plan.n_chunks == 1


def test_mismatched_sentence_counts_still_align():
    # source has 40 sentences, translation collapsed everything into 2
    src = " ".join(f"Source sentence {i} carries some meaningful payload." for i in range(40))
    mt = ("Une tres longue phrase " * 60).strip() + ". " + ("Encore une phrase " * 60).strip() + "."
    plan = make_chunk_plan(src, mt, ws_len, budget=60)
    assert len(plan.src_chunks) == len(plan.mt_chunks)
    assert plan.n_chunks <= 2  # clamped by the 2-sentence side
    assert plan.oversized_chunk  # and honestly flagged


def test_cjk_sentence_splitting():
    text = "这是第一句。这是第二句！最后一句？"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_partition_preserves_order_and_nonempty():
    items = [f"s{i}" for i in range(7)]
    weights = [1, 1, 50, 1, 1, 1, 1]
    groups = partition_by_mass(items, weights, 3)
    assert len(groups) == 3
    assert all(groups)
    assert [x for g in groups for x in g] == items


def test_max_joint_constant_sane():
    assert MAX_JOINT_TOKENS == 512
