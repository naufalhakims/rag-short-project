from app.db import keyword_query
from app.rag.retriever import reciprocal_rank_fusion
from tests.conftest import GUIDE

# ── Keyword index ────────────────────────────────────────────────


def test_keyword_query_drops_stopwords_and_quotes_terms():
    assert keyword_query("What is the Dead Drift penalty?") == '"dead" OR "drift" OR "penalty"'
    assert keyword_query("what is the") is None
    # Punctuation and FTS operators in user input must not reach the query parser.
    assert keyword_query('voss" NEAR( *thruster* )^2') == '"voss" OR "near" OR "thruster"'


def test_keyword_search_finds_exact_terms(library):
    hits = library.db.keyword_search("Kaelen Voss", limit=5)
    chunks = library.db.get_chunks(hits)
    assert "Kaelen Voss" in chunks[hits[0]].text
    assert library.db.keyword_search("zebra quantum", limit=5) == []


def test_keyword_search_can_be_limited_to_documents(library):
    paper = library.db.get_document_by_filename("swarm_paper.md")
    hits = library.db.keyword_search("points consensus", limit=10, doc_ids=[paper["id"]])
    assert hits and all(h.startswith(paper["id"]) for h in hits)


# ── Fusion ───────────────────────────────────────────────────────


def test_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"  # near the top of both lists
    assert set(fused) == {"a", "b", "c", "d"}
    assert reciprocal_rank_fusion([[], []]) == []


# ── Retriever ────────────────────────────────────────────────────


def test_retrieves_the_relevant_passage_first(library):
    result = library.retriever.search("Who is Kaelen Voss?")
    assert result.mode == "matched"
    assert "Kaelen Voss" in result.passages[0].chunk.text
    assert result.passages[0].chunk.filename == "gravity_ball_guide.txt"

    result = library.retriever.search("consensus accuracy when drones are malfunctioning")
    assert result.passages[0].chunk.filename == "swarm_paper.md"
    assert result.passages[0].chunk.heading == "Neuromorphic Swarms > Results"


def test_unrelated_query_falls_back_to_document_overviews(library):
    result = library.retriever.search("zzz qqq")
    assert result.mode == "overview"
    # The opening chunk of every document is included.
    assert {p.chunk.filename for p in result.passages} == {
        "gravity_ball_guide.txt", "swarm_paper.md",
    }
    assert any(p.chunk.ordinal == 0 for p in result.passages)


def test_empty_library(services):
    result = services.retriever.search("anything at all")
    assert result.mode == "empty" and result.passages == []


def test_search_can_be_limited_to_documents(library):
    paper = library.db.get_document_by_filename("swarm_paper.md")
    result = library.retriever.search("points penalty watts", doc_ids=[paper["id"]])
    assert result.passages
    assert all(p.chunk.doc_id == paper["id"] for p in result.passages)


def test_top_k_caps_the_number_of_passages(library):
    assert len(library.retriever.search("gravity ball players points", top_k=2).passages) <= 2


# ── Library maintenance ──────────────────────────────────────────


def test_reupload_replaces_and_identical_upload_is_a_noop(library):
    before = library.db.count_chunks()
    assert library.library.add("gravity_ball_guide.txt", GUIDE.encode())["status"] == "unchanged"

    updated = GUIDE.replace("Kaelen Voss", "Mira Okafor")
    assert library.library.add("gravity_ball_guide.txt", updated.encode())["status"] == "replaced"
    assert library.db.count_chunks() == before
    assert len(library.db.list_documents()) == 2
    assert library.db.keyword_search("Kaelen", limit=5) == []
    assert library.retriever.search("Who is Mira Okafor?").passages[0].chunk.text.count("Mira")


def test_delete_removes_chunks_everywhere(library):
    guide = library.db.get_document_by_filename("gravity_ball_guide.txt")
    assert library.library.remove(guide["id"]) is True
    assert library.library.remove(guide["id"]) is False
    assert library.db.keyword_search("Kaelen", limit=5) == []
    result = library.retriever.search("Who is Kaelen Voss?")
    assert all(p.chunk.filename == "swarm_paper.md" for p in result.passages)


def test_vector_index_is_rebuilt_when_out_of_sync(library):
    assert library.library.sync_index() == 0  # already consistent
    guide = library.db.get_document_by_filename("gravity_ball_guide.txt")
    library.library._index.delete_document(guide["id"])  # simulate a damaged index
    assert library.library.sync_index() == library.db.count_chunks()
    assert "Kaelen Voss" in library.retriever.search("Who is Kaelen Voss?").passages[0].chunk.text


def test_filenames_are_sanitised(services):
    result = services.library.add("..\\..\\secret/../My  Notes.txt", b"hello world")
    assert result["document"]["filename"] == "My Notes.txt"


def test_a_few_passages_are_kept_once_something_matches(library):
    """Two-part questions need the runner-up passages even if they score under the cut-off."""
    library.retriever._settings.rerank_min_score = 0.9  # only a near-perfect match passes
    result = library.retriever.search("Kaelen Voss")
    assert result.mode == "matched"
    scores = [p.score for p in result.passages]
    assert scores[0] >= 0.9 and len(scores) == 3 and min(scores) < 0.9
    assert scores == sorted(scores, reverse=True)


def test_empty_scope_searches_nothing(library):
    """A conversation with no documents must not fall back to searching everything."""
    result = library.retriever.search("Who is Kaelen Voss?", doc_ids=[])
    assert result.mode == "empty" and result.passages == []


def test_upgrade_links_old_conversations_to_the_documents_they_cited(tmp_path):
    """Version 1 databases had no per-conversation documents; the upgrade infers them."""
    import sqlite3

    from app.db import Database

    path = tmp_path / "old.db"
    db = Database(path)
    guide, _ = db.add_document(
        filename="guide.txt", content_hash="a", size_bytes=1, char_count=1, chunks=[]
    )
    paper, _ = db.add_document(
        filename="paper.md", content_hash="b", size_bytes=1, char_count=1, chunks=[]
    )
    # v1 imported answers name the file only; v2 answers carry ids and a cited flag.
    db.save_turn(session_id="s1", title="t", question="q", answer="a", meta={},
                 sources=[{"filename": "guide.txt", "doc_id": None, "cited": True}])
    db.save_turn(session_id="s2", title="t", question="q", answer="a", meta={},
                 sources=[{"filename": "paper.md", "doc_id": paper, "cited": True},
                          {"filename": "guide.txt", "doc_id": guide, "cited": False}])
    with sqlite3.connect(path) as conn:  # roll the database back to version 1
        conn.execute("DELETE FROM session_documents")
        conn.execute("PRAGMA user_version = 1")

    upgraded = Database(path)
    assert upgraded.session_document_ids("s1") == [guide]
    assert upgraded.session_document_ids("s2") == [paper]  # uncited documents are not linked
